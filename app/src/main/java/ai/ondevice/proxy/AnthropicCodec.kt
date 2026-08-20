package ai.ondevice.proxy

import ai.ondevice.core.SparseParams
import ai.ondevice.engine.EngineMessage
import ai.ondevice.engine.GenerationEvent
import ai.ondevice.engine.StopReason
import ai.ondevice.engine.ToolCallRequest
import ai.ondevice.engine.ToolSpec
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * The Anthropic Messages API, both directions.
 *
 * The inbound half is the fiddly one and is a close port of telecode's
 * `_decompose_anthropic_message`: Anthropic packs a tool call and its
 * accompanying prose into *one* assistant message, and a tool result and the
 * next question into *one* user message, where every chat template this app
 * renders wants them separate. The interesting failure it prevents is the
 * `tool_result` whose content is an array carrying an image: that image belongs
 * to the conversation but a `tool` message may not hold one, so it is lifted
 * out into a following user turn rather than dropped.
 *
 * The outbound half is much smaller than telecode's, and the reason is in
 * `docs/proxy-plan.md` 0: telecode reconstructs thinking blocks by scanning a
 * text stream for `<think>` tags because that is all llama-server gives it.
 * Here `GenerationEvent.ThinkingDelta` arrives already separated, so the
 * writer's whole job is block bookkeeping.
 */
object AnthropicCodec {

    // ── inbound ─────────────────────────────────────────────────────────

    fun decode(
        body: JsonObject,
        midSystemPolicy: String,
        stripBookkeeping: Boolean,
        media: MediaSink,
    ): ChatRequest {
        val model = body.str("model")
            ?: throw ProxyRefusal.badRequest("No `model` in the request.")

        val system = body["system"]?.asTextContent()?.takeIf { it.isNotBlank() }

        val raw = body.arr("messages")?.mapNotNull { it as? JsonObject }
            ?: throw ProxyRefusal.badRequest("No `messages` in the request.")

        val decomposed = raw.flatMap { decompose(it, media) }
        val messages = applyMidSystemPolicy(decomposed, midSystemPolicy)
            .let { if (stripBookkeeping) it.map(::strippedOf) else it }

        // Images on the final user turn are what a vision model is shown. They
        // are also carried per-message, so a multi-turn conversation keeps them
        // where they were said.
        val images = messages.flatMap { it.imagePaths }

        val tools = body.arr("tools")?.mapNotNull { element ->
            val tool = element as? JsonObject ?: return@mapNotNull null
            val name = tool.str("name") ?: return@mapNotNull null
            toolSpecFrom(name, tool.str("description").orEmpty(), tool["input_schema"])
        }.orEmpty()

        return ChatRequest(
            protocol = Protocol.ANTHROPIC,
            requestedModel = model,
            messages = messages,
            system = system,
            tools = tools,
            stream = body.b("stream") ?: false,
            params = params(body),
            imagePaths = images,
            forcedTool = forcedTool(body["tool_choice"]),
        )
    }

    /**
     * Anthropic's sampling keys, in this app's names.
     *
     * Only the keys the client actually sent — the map is sparse for the same
     * reason every override map here is, so a model's own stored parameters
     * still apply to everything the request did not mention.
     */
    private fun params(body: JsonObject): SparseParams {
        var out = SparseParams.EMPTY
        body.i("max_tokens")?.let { out = out.with("n_predict", it) }
        body.f("temperature")?.let { out = out.with("temp", it) }
        body.f("top_p")?.let { out = out.with("top_p", it) }
        body.i("top_k")?.let { out = out.with("top_k", it) }
        body.strings("stop_sequences").takeIf { it.isNotEmpty() }?.let { out = out.with("stop", it) }

        // Extended thinking, mapped onto `--reasoning-budget`.
        //
        // Two wrong turns before this one, both worth recording. The kwarg was
        // only written when `thinking` was present, which left the template's
        // own default in charge — and Qwen3.5's default is on, so a client that
        // never asked for reasoning had its whole `max_tokens` spent on it and
        // got an empty answer with `stop_reason: max_tokens`. Measured: 300
        // tokens, all 300 of them thinking.
        //
        // Then `enable_thinking: false` was written unconditionally, which the
        // template honours and this model answers badly: three tokens, EOS, no
        // content at all. Turning the block off is not the same request as
        // ending it immediately, and this model only does the second one well.
        //
        // `reasoning_budget` is the knob that actually says it — upstream's
        // `--reasoning-budget`, where 0 ends the thinking block straight away
        // and -1 lets it run. It also gives `budget_tokens` somewhere real to
        // go, which is what the protocol says it is for.
        val thinking = body.obj("thinking")
        val on = thinking?.str("type") == "enabled"
        out = out.with(
            "reasoning_budget",
            if (!on) NO_THINKING else thinking?.i("budget_tokens") ?: UNBOUNDED_THINKING,
        )
        return out
    }

    private fun forcedTool(choice: kotlinx.serialization.json.JsonElement?): String? =
        when (choice) {
            is JsonObject -> when (choice.str("type")) {
                "tool" -> choice.str("name")
                "any" -> ANY_TOOL
                else -> null
            }
            else -> null
        }

    /**
     * One Anthropic message becomes one or more engine messages.
     *
     * Assistant: prose and tool calls travel together in this protocol and
     * separately in every chat template, so they are split.
     *
     * User: each `tool_result` becomes its own `tool` message. Prose and images
     * that shared the turn with it are emitted *after*, as a user message, so
     * the order the model reads matches the order things happened.
     */
    private fun decompose(message: JsonObject, media: MediaSink): List<EngineMessage> {
        val role = message.str("role") ?: "user"
        val content = message["content"]

        if (content is JsonPrimitive) {
            return listOf(EngineMessage(role = role, content = content.content))
        }
        val blocks = (content as? JsonArray)?.mapNotNull { it as? JsonObject }
            ?: return listOf(EngineMessage(role = role, content = ""))

        if (role == "assistant") {
            val text = StringBuilder()
            val calls = mutableListOf<ToolCallRequest>()
            blocks.forEach { block ->
                when (block.str("type")) {
                    "text" -> text.append(block.str("text").orEmpty())
                    // Prior-turn thinking is dropped rather than replayed. It is
                    // not part of the answer, it re-enters the context at full
                    // price every turn, and no template this app renders has a
                    // slot for it.
                    "thinking", "redacted_thinking" -> Unit
                    "tool_use" -> calls += ToolCallRequest(
                        name = block.str("name").orEmpty(),
                        argumentsJson = block["input"]?.let { encode(it) } ?: "{}",
                        id = block.str("id") ?: newCallId(),
                    )
                }
            }
            return listOf(
                EngineMessage(role = "assistant", content = text.toString(), toolCalls = calls),
            )
        }

        val out = mutableListOf<EngineMessage>()
        val prose = StringBuilder()
        val images = mutableListOf<String>()

        blocks.forEach { block ->
            when (block.str("type")) {
                "text" -> {
                    if (prose.isNotEmpty()) prose.append("\n")
                    prose.append(block.str("text").orEmpty())
                }
                "image" -> imagePath(block.obj("source"), media)?.let { images += it }
                // No native PDF anywhere in this app. A document whose source is
                // already text is passed through as text; anything else is
                // named as unread rather than silently ignored, because a
                // question about a file that never arrived is unanswerable in a
                // way that looks like the model being stupid.
                "document" -> {
                    val source = block.obj("source")
                    when (source?.str("type")) {
                        "text" -> prose.append(source.str("data").orEmpty())
                        else -> prose.append(
                            "[a ${source?.str("media_type") ?: "document"} was attached; " +
                                "this device has no reader for it]",
                        )
                    }
                }
                "tool_result" -> {
                    val (text, lifted) = splitToolResult(block["content"], media)
                    out += EngineMessage(
                        role = "tool",
                        content = text,
                        toolCallId = block.str("tool_use_id"),
                        toolName = block.str("name"),
                    )
                    // A `tool` message cannot carry a picture, so a screenshot
                    // returned by a tool follows as its own user turn rather
                    // than being thrown away.
                    if (lifted.isNotEmpty()) {
                        out += EngineMessage(role = "user", content = "", imagePaths = lifted)
                    }
                }
            }
        }

        if (prose.isNotEmpty() || images.isNotEmpty()) {
            out += EngineMessage(role = "user", content = prose.toString(), imagePaths = images)
        }
        return out
    }

    private fun splitToolResult(
        content: kotlinx.serialization.json.JsonElement?,
        media: MediaSink,
    ): Pair<String, List<String>> {
        if (content is JsonPrimitive) return content.content to emptyList()
        val blocks = (content as? JsonArray)?.mapNotNull { it as? JsonObject }
            ?: return ("" to emptyList())
        val text = StringBuilder()
        val images = mutableListOf<String>()
        blocks.forEach { block ->
            when (block.str("type")) {
                "text" -> {
                    if (text.isNotEmpty()) text.append("\n")
                    text.append(block.str("text").orEmpty())
                }
                "image" -> imagePath(block.obj("source"), media)?.let { images += it }
            }
        }
        return text.toString() to images
    }

    private fun imagePath(source: JsonObject?, media: MediaSink): String? {
        source ?: return null
        return when (source.str("type")) {
            "base64" -> media.writeBase64(
                source.str("data").orEmpty(),
                source.str("media_type") ?: "image/png",
            )
            // A URL would be an outbound fetch this app did not choose to make,
            // on behalf of whoever is holding the socket. Refused by name.
            "url" -> null
            else -> null
        }
    }

    // ── outbound ────────────────────────────────────────────────────────

    /**
     * Assembles the Anthropic event stream.
     *
     * The block index is the whole of the bookkeeping and the whole of the
     * difficulty. Anthropic numbers content blocks within a message, a block
     * must be closed before a block of another kind opens, and a client that
     * sees an index it did not see opened will drop the message. Status lines
     * (4.1) take the low indices, which is why [statusEmitted] seeds the count.
     */
    class Writer(private val clientModel: String) {

        private val messageId = newMessageId()
        private var started = false
        private var nextIndex = 0
        private var openKind: String? = null
        private var openIndex = -1
        private var statusEmitted = 0

        private var inputTokens = 0
        private var cachedTokens = 0
        private var outputTokens = 0
        private var thinkingTokens = 0

        /**
         * Sent before anything else, exactly once.
         *
         * Clients buffer every event until they have seen `message_start`, so a
         * status line written before this one is invisible until the end of the
         * request — which is worse than not writing it at all. Telecode learned
         * this the same way and emits its initial frame at request start rather
         * than at first token.
         */
        fun start(): String {
            if (started) return ""
            started = true
            val message = buildJsonObject {
                put("type", "message_start")
                put(
                    "message",
                    buildJsonObject {
                        put("id", messageId)
                        put("type", "message")
                        put("role", "assistant")
                        put("model", clientModel)
                        put("content", JsonArray(emptyList()))
                        put("stop_reason", JsonPrimitive(null as String?))
                        put("stop_sequence", JsonPrimitive(null as String?))
                        put(
                            "usage",
                            buildJsonObject {
                                put("input_tokens", 0)
                                put("output_tokens", 0)
                            },
                        )
                    },
                )
            }
            return Sse.named("message_start", encode(message))
        }

        /** A round of tool work, rendered as a text block the person can read. */
        fun status(text: String): String {
            val index = statusEmitted++
            if (index >= nextIndex) nextIndex = index + 1
            return Sse.named(
                "content_block_start",
                encode(
                    buildJsonObject {
                        put("type", "content_block_start")
                        put("index", index)
                        put(
                            "content_block",
                            buildJsonObject {
                                put("type", "text")
                                put("text", "")
                            },
                        )
                    },
                ),
            ) + Sse.named(
                "content_block_delta",
                encode(
                    buildJsonObject {
                        put("type", "content_block_delta")
                        put("index", index)
                        put(
                            "delta",
                            buildJsonObject {
                                put("type", "text_delta")
                                put("text", text)
                            },
                        )
                    },
                ),
            ) + Sse.named(
                "content_block_stop",
                encode(
                    buildJsonObject {
                        put("type", "content_block_stop")
                        put("index", index)
                    },
                ),
            )
        }

        /** One engine event, as whatever it is worth on this protocol. */
        fun event(event: GenerationEvent): String = when (event) {
            is GenerationEvent.PromptProcessed -> {
                inputTokens = event.promptTokens
                cachedTokens = event.cachedTokens
                ""
            }
            is GenerationEvent.ThinkingDelta -> delta("thinking", event.text)
            is GenerationEvent.ThinkingDone -> {
                thinkingTokens = event.totalTokens
                closeOpen()
            }
            is GenerationEvent.Token -> delta("text", event.text)
            is GenerationEvent.Stats -> {
                outputTokens = event.generatedTokens
                ""
            }
            is GenerationEvent.ToolCall -> toolBlock(event)
            is GenerationEvent.Done -> finish(event.stopReason)
            is GenerationEvent.Failed -> error(event.message, event.suggestion)
        }

        fun error(message: String, suggestion: String?): String {
            val text = if (suggestion.isNullOrBlank()) message else "$message\n\n$suggestion"
            return closeOpen() + Sse.named(
                "error",
                encode(
                    buildJsonObject {
                        put("type", "error")
                        put(
                            "error",
                            buildJsonObject {
                                put("type", "api_error")
                                put("message", text)
                            },
                        )
                    },
                ),
            )
        }

        /**
         * A complete tool call, as an opened-filled-closed block.
         *
         * Complete rather than streamed because llama.cpp's own parser hands
         * the whole call over at once — `GenerationEvent.ToolCall` is emitted
         * from the `done` step, with the arguments already assembled. Emitting
         * one `input_json_delta` carrying the whole string is the honest
         * rendering of that; pretending to stream it would be theatre.
         */
        private fun toolBlock(call: GenerationEvent.ToolCall): String {
            val prefix = closeOpen()
            val index = nextIndex++
            return prefix + Sse.named(
                "content_block_start",
                encode(
                    buildJsonObject {
                        put("type", "content_block_start")
                        put("index", index)
                        put(
                            "content_block",
                            buildJsonObject {
                                put("type", "tool_use")
                                put("id", call.id.ifBlank { newCallId() })
                                put("name", call.name)
                                put("input", JsonObject(emptyMap()))
                            },
                        )
                    },
                ),
            ) + Sse.named(
                "content_block_delta",
                encode(
                    buildJsonObject {
                        put("type", "content_block_delta")
                        put("index", index)
                        put(
                            "delta",
                            buildJsonObject {
                                put("type", "input_json_delta")
                                put("partial_json", call.argumentsJson.ifBlank { "{}" })
                            },
                        )
                    },
                ),
            ) + Sse.named(
                "content_block_stop",
                encode(
                    buildJsonObject {
                        put("type", "content_block_stop")
                        put("index", index)
                    },
                ),
            )
        }

        private fun delta(kind: String, text: String): String {
            if (text.isEmpty()) return ""
            val prefix = if (openKind == kind) "" else closeOpen() + open(kind)
            val type = if (kind == "thinking") "thinking_delta" else "text_delta"
            val field = if (kind == "thinking") "thinking" else "text"
            return prefix + Sse.named(
                "content_block_delta",
                encode(
                    buildJsonObject {
                        put("type", "content_block_delta")
                        put("index", openIndex)
                        put(
                            "delta",
                            buildJsonObject {
                                put("type", type)
                                put(field, text)
                            },
                        )
                    },
                ),
            )
        }

        private fun open(kind: String): String {
            openIndex = nextIndex++
            openKind = kind
            val block = if (kind == "thinking") {
                buildJsonObject {
                    put("type", "thinking")
                    put("thinking", "")
                }
            } else {
                buildJsonObject {
                    put("type", "text")
                    put("text", "")
                }
            }
            return Sse.named(
                "content_block_start",
                encode(
                    buildJsonObject {
                        put("type", "content_block_start")
                        put("index", openIndex)
                        put("content_block", block)
                    },
                ),
            )
        }

        private fun closeOpen(): String {
            if (openIndex < 0) return ""
            val index = openIndex
            openIndex = -1
            openKind = null
            return Sse.named(
                "content_block_stop",
                encode(
                    buildJsonObject {
                        put("type", "content_block_stop")
                        put("index", index)
                    },
                ),
            )
        }

        /** Closed and stopped. Nothing may follow this on the stream. */
        fun finish(reason: StopReason): String = closeOpen() + Sse.named(
            "message_delta",
            encode(
                buildJsonObject {
                    put("type", "message_delta")
                    put(
                        "delta",
                        buildJsonObject {
                            put("stop_reason", stopReason(reason))
                            put("stop_sequence", JsonPrimitive(null as String?))
                        },
                    )
                    put("usage", usage())
                },
            ),
        ) + Sse.named(
            "message_stop",
            encode(buildJsonObject { put("type", "message_stop") }),
        )

        private fun usage() = buildJsonObject {
            // Anthropic counts cache reads separately from fresh input, and
            // llama.cpp's `cachedTokens` is exactly that number, so a client
            // showing a cache-hit rate shows a true one.
            put("input_tokens", (inputTokens - cachedTokens).coerceAtLeast(0))
            put("output_tokens", outputTokens)
            put("cache_read_input_tokens", cachedTokens)
            put("cache_creation_input_tokens", 0)
            if (thinkingTokens > 0) put("thinking_tokens", thinkingTokens)
        }

        /** The whole non-streaming body, for a client that asked for one. */
        fun whole(
            text: String,
            thinking: String,
            calls: List<ToolCallRequest>,
            reason: StopReason,
        ): String {
            val blocks = buildList {
                if (thinking.isNotBlank()) {
                    add(
                        buildJsonObject {
                            put("type", "thinking")
                            put("thinking", thinking)
                        },
                    )
                }
                if (text.isNotBlank()) {
                    add(
                        buildJsonObject {
                            put("type", "text")
                            put("text", text)
                        },
                    )
                }
                calls.forEach { call ->
                    add(
                        buildJsonObject {
                            put("type", "tool_use")
                            put("id", call.id.ifBlank { newCallId() })
                            put("name", call.name)
                            put(
                                "input",
                                runCatching {
                                    ProxyJson.parseToJsonElement(call.argumentsJson.ifBlank { "{}" })
                                }.getOrElse { JsonObject(emptyMap()) },
                            )
                        },
                    )
                }
            }
            return encode(
                buildJsonObject {
                    put("id", messageId)
                    put("type", "message")
                    put("role", "assistant")
                    put("model", clientModel)
                    put("content", JsonArray(blocks))
                    put("stop_reason", stopReason(reason))
                    put("stop_sequence", JsonPrimitive(null as String?))
                    put("usage", usage())
                },
            )
        }

        fun noteUsage(input: Int, cached: Int, output: Int, thinking: Int) {
            inputTokens = input
            cachedTokens = cached
            outputTokens = output
            thinkingTokens = thinking
        }

        private fun stopReason(reason: StopReason): String = when (reason) {
            StopReason.EOS -> "end_turn"
            StopReason.STOP_SEQUENCE -> "stop_sequence"
            StopReason.MAX_TOKENS, StopReason.CONTEXT_FULL -> "max_tokens"
            StopReason.CANCELLED -> "end_turn"
        }
    }

    /** `tool_choice:{type:"any"}` — some tool, any tool. */
    const val ANY_TOOL = " any"

    /** `--reasoning-budget 0`: end the thinking block at once and answer. */
    internal const val NO_THINKING = 0

    /** `-1`: think until it stops, which on a phone is a hot device. */
    internal const val UNBOUNDED_THINKING = -1
}

/**
 * Apply the mid-conversation system-message policy.
 *
 * Shared by both codecs because both protocols can express one and every chat
 * template refuses it in the same way. The four modes are telecode's, and the
 * default is `demote` for the reason written on its config: it is the only mode
 * that is both template-safe and cache-safe, because it leaves the prompt
 * prefix append-only. Hoisting to the top is template-safe and pins the cache —
 * a client that emits one system message per turn grows the front block every
 * turn, which re-prefills the entire history each time.
 */
internal fun applyMidSystemPolicy(
    messages: List<EngineMessage>,
    policy: String,
): List<EngineMessage> {
    val firstNonSystem = messages.indexOfFirst { it.role != "system" }
    if (firstNonSystem < 0) return messages
    val late = messages.withIndex().filter { it.index > firstNonSystem && it.value.role == "system" }
    if (late.isEmpty()) return messages

    return when (policy) {
        ProxySpecs.MID_KEEP -> messages
        ProxySpecs.MID_STRIP -> messages.filterIndexed { index, message ->
            !(index > firstNonSystem && message.role == "system")
        }
        ProxySpecs.MID_MERGE_TOP -> {
            val merged = late.joinToString("\n\n") { it.value.content }
            messages
                .filterIndexed { index, message -> !(index > firstNonSystem && message.role == "system") }
                .mapIndexed { index, message ->
                    if (index == 0 && message.role == "system") {
                        message.copy(content = message.content + "\n\n" + merged)
                    } else {
                        message
                    }
                }
                .let { out ->
                    if (out.firstOrNull()?.role == "system") {
                        out
                    } else {
                        listOf(EngineMessage("system", merged)) + out
                    }
                }
        }
        // demote, and anything unrecognised, which must not be a crash.
        else -> messages.mapIndexed { index, message ->
            if (index > firstNonSystem && message.role == "system") {
                message.copy(role = "user")
            } else {
                message
            }
        }
    }
}

internal fun strippedOf(message: EngineMessage): EngineMessage =
    if (message.content.isEmpty()) message
    else message.copy(content = stripClientBookkeeping(message.content))

/** Where a decoded base64 image goes, so the codecs never touch the filesystem. */
interface MediaSink {
    /** Returns an absolute path, or null when the payload could not be read. */
    fun writeBase64(data: String, mediaType: String): String?

    fun writeBytes(bytes: ByteArray, extension: String): String?
}
