package ai.ondevice.proxy

import ai.ondevice.core.Modality
import ai.ondevice.engine.GenerateRequest
import ai.ondevice.engine.GenerationEvent
import ai.ondevice.engine.StopReason
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * `/v1/messages` and `/v1/chat/completions`.
 *
 * One implementation for both, which is the payoff for decoding into
 * [ChatRequest] and writing out of [GenerationEvent]: the two protocols differ
 * only in which codec decodes and which writer encodes, and everything between
 * those two points is shared.
 */
suspend fun ProxyCall.chat() {
    val body = body()
    val request = when (protocol) {
        Protocol.ANTHROPIC -> AnthropicCodec.decode(
            body, config.midSystem, config.stripReminders, media,
        )
        Protocol.OPENAI -> OpenAiCodec.decode(
            body, config.midSystem, config.stripReminders, media,
        )
    }

    val model = resolveModel(
        request.requestedModel,
        setOf(Modality.TEXT, Modality.VISION),
        "chat",
    )
    log.update(requestId) { it.copy(streaming = request.stream) }
    phase("Answering")

    // Refused rather than accepted and ignored.
    //
    // Both protocols can demand a particular tool, or any tool; llama.cpp's
    // chat handler cannot be told to require one -- there is no
    // `tool_choice` on `GenerateRequest` and no grammar here that would
    // constrain it honestly. Accepting the field and letting the model answer
    // in prose is the failure SPEC 1.2 exists to forbid: the client gets a
    // well-formed reply that quietly did not do what it asked for, and there
    // is nothing in it to notice.
    request.forcedTool?.let { forced ->
        throw ProxyRefusal.badRequest(
            if (forced == AnthropicCodec.ANY_TOOL) {
                "This device cannot be made to call a tool. `tool_choice` may only be `auto`."
            } else {
                "This device cannot be made to call `$forced`. `tool_choice` may only be `auto`."
            },
            "The runtime here has no way to constrain the model to a tool, so honouring this " +
                "would mean pretending. Ask in the prompt instead, and check the reply.",
        )
    }

    guardModelSwap(model.id)

    // Video is deliberately not offered on a chat turn: a clip is tens of
    // minutes on this hardware, and a tool call that pauses a sentence for
    // three quarters of an hour is not a capability, it is a hang.
    val registry = localTools(offerVideo = false)
    val pipeline = ChatPipeline(runner, config, log, requestId)
    val prepared = pipeline.prepare(request, model, registry)

    if (request.stream) {
        streamChat(pipeline, prepared, registry, request)
    } else {
        wholeChat(pipeline, prepared, registry, request)
    }
}

/**
 * What to do when the request wants a model other than the resident one.
 *
 * Three policies because there are three defensible answers and which is right
 * depends on where the phone is. Queue is the default: a conversation on the
 * screen is worth more than a request from the network, and waiting is the only
 * answer that does not throw one of them away.
 */
private suspend fun ProxyCall.guardModelSwap(wanted: String) {
    val resident = runner.residentRuntime ?: return
    if (!runner.busy) return
    when (config.modelPolicy) {
        ProxySpecs.POLICY_REFUSE -> throw ProxyRefusal.conflict(
            "This device is busy on its $resident runtime and is set to refuse rather than " +
                "interrupt it.",
            "Set `proxy.model_policy` to queue or swap, or try again shortly.",
        )
        ProxySpecs.POLICY_SWAP -> runner.evict("an HTTP request asked for $wanted")
        // queue: fall through and wait on the engine gate, which is what
        // ModelRunner.exclusive already does.
        else -> Unit
    }
}

private suspend fun ProxyCall.streamChat(
    pipeline: ChatPipeline,
    prepared: ChatPipeline.Prepared,
    registry: ai.ondevice.tools.ToolRegistry?,
    request: ChatRequest,
) {
    val anthropic = AnthropicCodec.Writer(request.requestedModel)
    val openai = OpenAiCodec.Writer(request.requestedModel)

    stream { emit ->
        var heartbeat: Job? = null
        try {
            val sink = object : ChatPipeline.Sink {
                override suspend fun opening() {
                    // Before anything else, and this ordering is load-bearing.
                    // Anthropic clients buffer every event until they see
                    // `message_start`; OpenAI SDK parsers want the role in the
                    // first chunk. A status line written before either arrives
                    // is invisible until the request ends, which is worse than
                    // no status line at all.
                    emit(
                        when (protocol) {
                            Protocol.ANTHROPIC -> anthropic.start()
                            Protocol.OPENAI -> openai.start()
                        },
                    )
                }

                override suspend fun status(text: String) {
                    emit(
                        when (protocol) {
                            Protocol.ANTHROPIC -> anthropic.status(text)
                            Protocol.OPENAI -> openai.status("\n$text\n")
                        },
                    )
                }

                override suspend fun event(event: GenerationEvent) {
                    // `Done` is swallowed here and re-emitted once at the end.
                    // Every round produces one, and a client that receives
                    // `message_stop` after the first round stops reading — so a
                    // tool loop would deliver only its opening move.
                    if (event is GenerationEvent.Done) return
                    emit(
                        when (protocol) {
                            Protocol.ANTHROPIC -> anthropic.event(event)
                            Protocol.OPENAI -> openai.event(event)
                        },
                    )
                }

                override suspend fun failed(message: String, suggestion: String?) {
                    emit(
                        when (protocol) {
                            Protocol.ANTHROPIC -> anthropic.error(message, suggestion)
                            Protocol.OPENAI -> openai.error(message, suggestion)
                        },
                    )
                }
            }

            heartbeat = scope.launch { beat(emit, anthropic) }

            val result = pipeline.run(prepared, registry, sink)
            heartbeat.cancel()

            anthropic.noteUsage(
                result.promptTokens, result.cachedTokens, result.generatedTokens, result.thinkingTokens,
            )
            openai.noteUsage(
                result.promptTokens, result.cachedTokens, result.generatedTokens, result.thinkingTokens,
            )

            emit(
                when (protocol) {
                    Protocol.ANTHROPIC -> anthropic.finish(result.stopReason)
                    Protocol.OPENAI ->
                        openai.finish(result.stopReason, result.toolCalls.isNotEmpty()) + Sse.DONE
                },
            )
            record(result)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Throwable) {
            val refusal = ChatPipeline.refusalFor(failure)
            emit(
                when (protocol) {
                    Protocol.ANTHROPIC -> anthropic.error(refusal.message, refusal.suggestion)
                    Protocol.OPENAI ->
                        openai.error(refusal.message, refusal.suggestion) + Sse.DONE
                },
            )
            log.finish(requestId, refusal.status, refusal.message)
        } finally {
            heartbeat?.cancel()
        }
    }
}

/**
 * Keep the connection alive while the model is still thinking.
 *
 * A prefill on this hardware can run for minutes before the first token, which
 * is longer than most clients' idle timeouts and much longer than most proxies
 * in between. Anthropic has a `ping` event for exactly this; OpenAI's SSE has
 * no such frame, so a comment line does the same job for anything that is
 * merely counting bytes.
 */
private suspend fun ProxyCall.beat(emit: ProxyCall.Emit, anthropic: AnthropicCodec.Writer) {
    val every = config.pingInterval.coerceAtLeast(1) * 1000L
    while (kotlinx.coroutines.currentCoroutineContext().isActive) {
        delay(every)
        runCatching {
            emit(
                when (protocol) {
                    Protocol.ANTHROPIC -> Sse.named(
                        "ping",
                        encode(buildJsonObject { put("type", "ping") }),
                    )
                    Protocol.OPENAI -> Sse.KEEPALIVE
                },
            )
        }.onFailure { return }
    }
}

private suspend fun ProxyCall.wholeChat(
    pipeline: ChatPipeline,
    prepared: ChatPipeline.Prepared,
    registry: ai.ondevice.tools.ToolRegistry?,
    request: ChatRequest,
) {
    val statuses = StringBuilder()
    var failure: Pair<String, String?>? = null

    val sink = object : ChatPipeline.Sink {
        override suspend fun opening() = Unit

        // A non-streaming client has nowhere to put a progress line as it
        // happens, so the rounds are collected and prefixed to the answer. Kept
        // rather than dropped: "why did that take four minutes" is answerable
        // from them and from nothing else.
        override suspend fun status(text: String) {
            statuses.append(text).append('\n')
        }

        override suspend fun event(event: GenerationEvent) = Unit

        override suspend fun failed(message: String, suggestion: String?) {
            failure = message to suggestion
        }
    }

    val result = pipeline.run(prepared, registry, sink)
    failure?.let { (message, suggestion) ->
        throw ProxyRefusal(500, "api_error", message, suggestion)
    }
    record(result)

    val text = if (statuses.isEmpty()) result.text else "$statuses\n${result.text}"

    when (protocol) {
        Protocol.ANTHROPIC -> {
            val writer = AnthropicCodec.Writer(request.requestedModel)
            writer.noteUsage(
                result.promptTokens, result.cachedTokens, result.generatedTokens, result.thinkingTokens,
            )
            json(writer.whole(text, result.thinking, result.toolCalls, result.stopReason))
        }
        Protocol.OPENAI -> {
            val writer = OpenAiCodec.Writer(request.requestedModel)
            writer.noteUsage(
                result.promptTokens, result.cachedTokens, result.generatedTokens, result.thinkingTokens,
            )
            json(writer.whole(text, result.thinking, result.toolCalls, result.stopReason))
        }
    }
}

private fun ProxyCall.record(result: ChatPipeline.Result) {
    log.update(requestId) {
        it.copy(
            rounds = result.rounds,
            promptTokens = result.promptTokens,
            generatedTokens = result.generatedTokens,
            tokensPerSecond = result.tokensPerSecond,
        )
    }
}

/**
 * `/v1/messages/count_tokens`.
 *
 * Exact rather than estimated, because it can be: the prompt is rendered
 * through the model's own chat template and tokenised by the model's own
 * vocabulary. Telecode has to ask llama-server over HTTP for the same answer.
 *
 * It does mean loading the model, which a client sending this before every
 * request may not expect. Worth it — an estimate from a character count is
 * wrong by tens of percent on a template-heavy conversation, and a client that
 * trusts it will overflow the context and be told so far too late.
 */
suspend fun ProxyCall.countTokens() {
    val body = body()
    val request = AnthropicCodec.decode(body, config.midSystem, config.stripReminders, media)
    val model = resolveModel(
        request.requestedModel,
        setOf(Modality.TEXT, Modality.VISION),
        "token counting",
    )

    val rendered = runner.exclusive(runner.runtimeFor(model)) {
        val engine = runner.loadText(model, runner.paramsFor(model, request.params))
        engine.renderPrompt(
            GenerateRequest(
                messages = request.messages,
                params = request.params,
                systemPrompt = request.system,
                imagePaths = request.imagePaths,
                tools = request.tools,
            ),
        )
    }

    json(
        encode(
            buildJsonObject {
                put("input_tokens", rendered.totalTokens + rendered.imageTokens)
                put("cache_read_input_tokens", rendered.cachedTokens)
                put("cache_creation_input_tokens", 0)
                // Not in the protocol, and the two most useful numbers a client
                // can have when deciding what to trim: what the limit is, and
                // how much of it the pictures took.
                put("context_limit", rendered.contextLimit)
                put("image_tokens", rendered.imageTokens)
            },
        ),
    )
}

/** Kept so the stop reason has one spelling across both files. */
internal fun StopReason.isTruncation(): Boolean =
    this == StopReason.MAX_TOKENS || this == StopReason.CONTEXT_FULL
