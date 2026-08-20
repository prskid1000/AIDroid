package ai.ondevice.proxy

import ai.ondevice.core.SparseParams
import ai.ondevice.engine.EngineMessage
import ai.ondevice.engine.GenerationEvent
import ai.ondevice.engine.StopReason
import ai.ondevice.engine.ToolCallRequest
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * The OpenAI chat-completions API, both directions.
 *
 * Simpler than the Anthropic side in both directions, because this protocol
 * already separates what that one packs together: a tool call is a field on an
 * assistant message, and a tool result is a message with its own role. The only
 * decomposition needed is content *parts*, and only when a turn carries images.
 */
object OpenAiCodec {

    // What each effort tier is worth in thinking tokens. The manifest caps
    // `reasoning_budget` at 8192, and these sit under it on purpose.
    private const val LOW_EFFORT_TOKENS = 512
    private const val MEDIUM_EFFORT_TOKENS = 2048
    private const val HIGH_EFFORT_TOKENS = 8192

    // ── inbound ─────────────────────────────────────────────────────────

    fun decode(
        body: JsonObject,
        midSystemPolicy: String,
        stripBookkeeping: Boolean,
        media: MediaSink,
    ): ChatRequest {
        val model = body.str("model")
            ?: throw ProxyRefusal.badRequest("No `model` in the request.")

        val raw = body.arr("messages")?.mapNotNull { it as? JsonObject }
            ?: throw ProxyRefusal.badRequest("No `messages` in the request.")

        val all = raw.map { decodeMessage(it, media) }

        // A leading system message becomes the system prompt rather than a
        // message, because that is the slot every chat template here has for
        // it. Later ones are governed by the policy, exactly as on the other
        // protocol.
        val leading = all.takeWhile { it.role == "system" }
        val system = leading.joinToString("\n\n") { it.content }.takeIf { it.isNotBlank() }
        val rest = all.drop(leading.size)

        val messages = applyMidSystemPolicy(rest, midSystemPolicy)
            .let { if (stripBookkeeping) it.map(::strippedOf) else it }

        val tools = body.arr("tools")?.mapNotNull { element ->
            val tool = element as? JsonObject ?: return@mapNotNull null
            val fn = tool.obj("function") ?: return@mapNotNull null
            val name = fn.str("name") ?: return@mapNotNull null
            toolSpecFrom(name, fn.str("description").orEmpty(), fn["parameters"])
        }.orEmpty()

        return ChatRequest(
            protocol = Protocol.OPENAI,
            requestedModel = model,
            messages = messages,
            system = system,
            tools = tools,
            stream = body.b("stream") ?: false,
            params = params(body),
            imagePaths = messages.flatMap { it.imagePaths },
            forcedTool = forcedTool(body["tool_choice"]),
        )
    }

    private fun params(body: JsonObject): SparseParams {
        var out = SparseParams.EMPTY
        // `max_tokens` is the deprecated name and `max_completion_tokens` the
        // current one. Both are accepted; the newer wins when a client sends
        // both, which the SDKs do during their own migrations.
        (body.i("max_completion_tokens") ?: body.i("max_tokens"))
            ?.let { out = out.with("n_predict", it) }
        body.f("temperature")?.let { out = out.with("temp", it) }
        body.f("top_p")?.let { out = out.with("top_p", it) }
        body.f("frequency_penalty")?.let { out = out.with("frequency_penalty", it) }
        body.f("presence_penalty")?.let { out = out.with("presence_penalty", it) }
        body.i("seed")?.let { out = out.with("seed", it) }
        when (val stop = body["stop"]) {
            is JsonPrimitive -> out = out.with("stop", listOf(stop.content))
            is JsonArray -> out = out.with("stop", stop.mapNotNull { (it as? JsonPrimitive)?.content })
            else -> Unit
        }
        // `reasoning_effort` onto `--reasoning-budget`, which is the knob that
        // exists — see the Anthropic codec for the two wrong answers that came
        // before this one.
        //
        // The tiers are given real numbers rather than being collapsed to
        // on/off. "high" is deliberately not unbounded: a phone that thinks
        // until it stops on its own is a hot device and a flat battery, and a
        // client asking for high effort is asking for more thinking rather than
        // for all of it.
        val budget = when (body.str("reasoning_effort")) {
            null, "none", "minimal" -> AnthropicCodec.NO_THINKING
            "low" -> LOW_EFFORT_TOKENS
            "medium" -> MEDIUM_EFFORT_TOKENS
            else -> HIGH_EFFORT_TOKENS
        }
        // Only the budget. The template switch belongs to the model's own row —
        // see the note in AnthropicCodec for why a proxy must not write it.
        out = out.with("reasoning_budget", budget)
        return out
    }

    private fun forcedTool(choice: kotlinx.serialization.json.JsonElement?): String? =
        when (choice) {
            is JsonPrimitive -> when (choice.content) {
                "required" -> AnthropicCodec.ANY_TOOL
                else -> null
            }
            is JsonObject -> choice.obj("function")?.str("name")
            else -> null
        }

    private fun decodeMessage(message: JsonObject, media: MediaSink): EngineMessage {
        val role = message.str("role") ?: "user"
        val calls = message.arr("tool_calls")?.mapNotNull { element ->
            val call = element as? JsonObject ?: return@mapNotNull null
            val fn = call.obj("function") ?: return@mapNotNull null
            ToolCallRequest(
                name = fn.str("name").orEmpty(),
                argumentsJson = fn.str("arguments").orEmpty().ifBlank { "{}" },
                id = call.str("id") ?: newCallId(),
            )
        }.orEmpty()

        val images = mutableListOf<String>()
        val text = when (val content = message["content"]) {
            is JsonPrimitive -> content.content
            is JsonArray -> buildString {
                content.mapNotNull { it as? JsonObject }.forEach { part ->
                    when (part.str("type")) {
                        "text" -> {
                            if (isNotEmpty()) append("\n")
                            append(part.str("text").orEmpty())
                        }
                        "image_url" -> {
                            val url = part.obj("image_url")?.str("url").orEmpty()
                            dataUriToFile(url, media)?.let { images += it }
                        }
                        // Audio arriving inline on a chat turn has nowhere to go:
                        // no model this app loads for chat takes audio. Named
                        // rather than dropped.
                        "input_audio" -> {
                            if (isNotEmpty()) append("\n")
                            append("[audio was attached; this device's chat models do not read audio]")
                        }
                    }
                }
            }
            else -> ""
        }

        return EngineMessage(
            role = role,
            content = text,
            imagePaths = images,
            toolCalls = calls,
            toolCallId = message.str("tool_call_id"),
            toolName = message.str("name"),
        )
    }

    /**
     * `data:image/png;base64,…` becomes a file.
     *
     * An `http(s)` URL is refused for the reason the Anthropic side refuses
     * one: fetching it would be an outbound request this app did not choose to
     * make, aimed by whoever holds the socket.
     */
    private fun dataUriToFile(url: String, media: MediaSink): String? {
        if (!url.startsWith("data:")) return null
        val comma = url.indexOf(',')
        if (comma < 0) return null
        val header = url.substring(5, comma)
        if (!header.contains("base64")) return null
        val mediaType = header.substringBefore(';').ifBlank { "image/png" }
        return media.writeBase64(url.substring(comma + 1), mediaType)
    }

    // ── outbound ────────────────────────────────────────────────────────

    /** Assembles `chat.completion.chunk` frames, then the `[DONE]` sentinel. */
    class Writer(private val clientModel: String) {

        private val id = newCompletionId()
        private val created = System.currentTimeMillis() / 1000
        private var started = false

        private var inputTokens = 0
        private var cachedTokens = 0
        private var outputTokens = 0
        private var reasoningTokens = 0

        /**
         * The opener carrying `role: "assistant"`.
         *
         * Strict SDK parsers want the role in the first chunk they see. Sent up
         * front for the same reason the Anthropic side sends `message_start` up
         * front: a status line written before it can be discarded by a client
         * that has not yet decided what it is reading.
         */
        fun start(): String {
            if (started) return ""
            started = true
            return chunk(
                buildJsonObject {
                    put("role", "assistant")
                    put("content", "")
                },
                finish = null,
            )
        }

        fun status(text: String): String =
            chunk(buildJsonObject { put("content", text) }, finish = null)

        fun event(event: GenerationEvent): String = when (event) {
            is GenerationEvent.PromptProcessed -> {
                inputTokens = event.promptTokens
                cachedTokens = event.cachedTokens
                ""
            }
            // `reasoning_content` is the de-facto field: llama.cpp's own server
            // emits it, and every client that renders thinking at all reads it.
            is GenerationEvent.ThinkingDelta ->
                chunk(buildJsonObject { put("reasoning_content", event.text) }, finish = null)
            is GenerationEvent.ThinkingDone -> {
                reasoningTokens = event.totalTokens
                ""
            }
            is GenerationEvent.Token ->
                chunk(buildJsonObject { put("content", event.text) }, finish = null)
            is GenerationEvent.Stats -> {
                outputTokens = event.generatedTokens
                ""
            }
            is GenerationEvent.ToolCall -> chunk(
                buildJsonObject {
                    put(
                        "tool_calls",
                        buildJsonArray {
                            add(
                                buildJsonObject {
                                    put("index", 0)
                                    put("id", event.id.ifBlank { newCallId() })
                                    put("type", "function")
                                    put(
                                        "function",
                                        buildJsonObject {
                                            put("name", event.name)
                                            put("arguments", event.argumentsJson.ifBlank { "{}" })
                                        },
                                    )
                                },
                            )
                        },
                    )
                },
                finish = null,
            )
            is GenerationEvent.Done -> finish(event.stopReason, hadToolCalls = false)
            is GenerationEvent.Failed -> error(event.message, event.suggestion)
        }

        fun error(message: String, suggestion: String?): String {
            val text = if (suggestion.isNullOrBlank()) message else "$message\n\n$suggestion"
            return Sse.data(
                encode(
                    buildJsonObject {
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

        fun finish(reason: StopReason, hadToolCalls: Boolean): String =
            chunk(buildJsonObject { }, finish = finishReason(reason, hadToolCalls)) + usageChunk()

        /**
         * A final chunk carrying nothing but `usage`.
         *
         * Sent unconditionally rather than only under `stream_options`, because
         * a chunk with an empty `choices` array is what the spec says this is
         * and clients that do not want it ignore it — where a client that does
         * want it and never gets it reports zero tokens for every request.
         */
        private fun usageChunk(): String = Sse.data(
            encode(
                buildJsonObject {
                    put("id", id)
                    put("object", "chat.completion.chunk")
                    put("created", created)
                    put("model", clientModel)
                    put("choices", JsonArray(emptyList()))
                    put("usage", usage())
                },
            ),
        )

        private fun chunk(delta: JsonObject, finish: String?): String {
            if (delta.isEmpty() && finish == null) return ""
            return Sse.data(
                encode(
                    buildJsonObject {
                        put("id", id)
                        put("object", "chat.completion.chunk")
                        put("created", created)
                        put("model", clientModel)
                        put(
                            "choices",
                            buildJsonArray {
                                add(
                                    buildJsonObject {
                                        put("index", 0)
                                        put("delta", delta)
                                        put(
                                            "finish_reason",
                                            finish?.let { JsonPrimitive(it) }
                                                ?: JsonPrimitive(null as String?),
                                        )
                                    },
                                )
                            },
                        )
                    },
                ),
            )
        }

        private fun usage() = buildJsonObject {
            put("prompt_tokens", inputTokens)
            put("completion_tokens", outputTokens)
            put("total_tokens", inputTokens + outputTokens)
            put(
                "prompt_tokens_details",
                buildJsonObject { put("cached_tokens", cachedTokens) },
            )
            put(
                "completion_tokens_details",
                buildJsonObject { put("reasoning_tokens", reasoningTokens) },
            )
        }

        fun noteUsage(input: Int, cached: Int, output: Int, thinking: Int) {
            inputTokens = input
            cachedTokens = cached
            outputTokens = output
            reasoningTokens = thinking
        }

        fun whole(
            text: String,
            thinking: String,
            calls: List<ToolCallRequest>,
            reason: StopReason,
        ): String = encode(
            buildJsonObject {
                put("id", id)
                put("object", "chat.completion")
                put("created", created)
                put("model", clientModel)
                put(
                    "choices",
                    buildJsonArray {
                        add(
                            buildJsonObject {
                                put("index", 0)
                                put(
                                    "message",
                                    buildJsonObject {
                                        put("role", "assistant")
                                        put("content", text)
                                        if (thinking.isNotBlank()) {
                                            put("reasoning_content", thinking)
                                        }
                                        if (calls.isNotEmpty()) {
                                            put(
                                                "tool_calls",
                                                buildJsonArray {
                                                    calls.forEachIndexed { index, call ->
                                                        add(
                                                            buildJsonObject {
                                                                put("index", index)
                                                                put("id", call.id.ifBlank { newCallId() })
                                                                put("type", "function")
                                                                put(
                                                                    "function",
                                                                    buildJsonObject {
                                                                        put("name", call.name)
                                                                        put("arguments", call.argumentsJson)
                                                                    },
                                                                )
                                                            },
                                                        )
                                                    }
                                                },
                                            )
                                        }
                                    },
                                )
                                put("finish_reason", finishReason(reason, calls.isNotEmpty()))
                            },
                        )
                    },
                )
                put("usage", usage())
            },
        )

        private fun finishReason(reason: StopReason, hadToolCalls: Boolean): String = when {
            hadToolCalls -> "tool_calls"
            reason == StopReason.MAX_TOKENS || reason == StopReason.CONTEXT_FULL -> "length"
            else -> "stop"
        }
    }
}
