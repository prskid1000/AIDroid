package ai.ondevice.proxy

import ai.ondevice.core.Modality
import ai.ondevice.core.SparseParams
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.engine.EngineMessage
import ai.ondevice.engine.GenerateRequest
import ai.ondevice.engine.GenerationEvent
import ai.ondevice.engine.ModelRunFailure
import ai.ondevice.engine.ModelRunner
import ai.ondevice.engine.StopReason
import ai.ondevice.engine.ToolCallRequest
import ai.ondevice.engine.ToolSpec
import ai.ondevice.tools.ToolRegistry
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * One chat request, from a decoded body to the last byte on the wire.
 *
 * The intercept loop is telecode's and ports whole, because it is
 * protocol-independent: run the model, look at what it asks for first, and if
 * that is something *we* can answer — a tool search, a tool this device runs
 * itself, a schema it forgot to load, a name that does not exist — answer it,
 * write a line saying so, and go round again. Only a turn that asks for nothing
 * we own reaches the client.
 *
 * One thing is simpler here than there. llama.cpp's own parser hands a tool
 * call over complete, in the `done` step, so `GenerationEvent.ToolCall` arrives
 * whole. Telecode has to assemble one from SSE fragments and decide whether to
 * intercept before it has finished reading the name — which is why its loop has
 * a `just_decided` flag and a warning about deciding too early. None of that is
 * needed here.
 */
class ChatPipeline(
    private val runner: ModelRunner,
    private val config: ProxyConfig,
    private val log: RequestLog,
    private val requestId: String,
) {

    /** What the client is allowed to see, in whichever protocol it speaks. */
    interface Sink {
        /** Called before anything else, exactly once. */
        suspend fun opening(): Unit

        suspend fun status(text: String)

        suspend fun event(event: GenerationEvent)

        suspend fun failed(message: String, suggestion: String?)
    }

    data class Prepared(
        val model: ModelEntity,
        val messages: List<EngineMessage>,
        val system: String?,
        /** What the model is shown this turn. */
        val visibleTools: List<ToolSpec>,
        /** Held back behind ToolSearch, retrievable by name or by search. */
        val deferredTools: List<ToolSpec>,
        /** Names this device runs itself; a call to one never reaches the client. */
        val localToolNames: Set<String>,
        val params: SparseParams,
        val imagePaths: List<String>,
    )

    /**
     * Everything decided before the engine is touched.
     *
     * Kept separate from [run] so the whole of it is testable without an
     * Android device or a loaded model — which is what CLAUDE.md means by the
     * logic worth a unit test being the logic that is invisible on hardware.
     */
    suspend fun prepare(
        request: ChatRequest,
        model: ModelEntity,
        registry: ToolRegistry?,
    ): Prepared {
        val localSpecs = registry?.specs().orEmpty()
        val localNames = localSpecs.map { it.name }.toSet()

        // A tool this device runs itself replaces the client's same-named one.
        // That is the whole point of pointing a client here: its `WebSearch`
        // reaches a cloud API and ours reaches Brave without a key.
        val stripped = config.stripTools.toSet()
        val clientTools = request.tools
            .filterNot { it.name in localNames }
            .filterNot { it.name in stripped }
            // Never let a client define its own ToolSearch — the meta-tool is
            // ours, and a second one with different semantics would be
            // intercepted by name and answered wrongly.
            .filterNot { it.name == ToolSearchIndex.TOOL_SEARCH_SPEC.name }

        val offered = config.injectManaged
            ?.let { allowed -> localSpecs.filter { it.name in allowed } }
            ?: localSpecs

        val all = (offered + clientTools).let { tools ->
            if (config.sortTools) tools.sortedBy { it.name } else tools
        }

        val visible: List<ToolSpec>
        val deferred: List<ToolSpec>
        if (config.toolSearch && all.size > 1) {
            val core = config.coreTools.toSet()
            visible = all.filter { it.name in core }
            deferred = all.filterNot { it.name in core }
        } else {
            visible = all
            deferred = emptyList()
        }

        val withMeta = if (deferred.isEmpty()) {
            visible
        } else {
            listOf(ToolSearchIndex.TOOL_SEARCH_SPEC) + visible
        }

        return Prepared(
            model = model,
            messages = withDeferredNotice(request.messages, deferred),
            system = systemPrompt(request.system),
            visibleTools = withMeta,
            deferredTools = deferred,
            localToolNames = localNames,
            params = runner.paramsFor(model, request.params),
            imagePaths = request.imagePaths,
        )
    }

    /**
     * The system prompt the model actually sees.
     *
     * The date is injected from this device's clock. Telecode also looks up a
     * location over `ip-api.com`; that call is deliberately not ported — an
     * outbound request on every cold start, to learn something the person can
     * type once, is exactly what SPEC 13 rules out.
     */
    private fun systemPrompt(clientSystem: String?): String? {
        val parts = mutableListOf<String>()
        clientSystem?.takeIf { it.isNotBlank() }?.let(parts::add)
        config.instruction.takeIf { it.isNotBlank() }?.let(parts::add)
        if (config.injectDate) {
            val today = SimpleDateFormat("EEEE, d MMMM yyyy", Locale.UK).format(Date())
            val where = config.location.takeIf { it.isNotBlank() }
            parts += if (where != null) "Today is $today. The user is in $where." else "Today is $today."
        }
        return parts.joinToString("\n\n").takeIf { it.isNotBlank() }
    }

    /**
     * Tell the model what exists but is not loaded.
     *
     * Names only, not schemas — the names are what it needs to know a
     * capability is there, and the schemas are what it costs context to carry.
     * Appended to the first user turn rather than the system prompt, so it sits
     * after the cacheable prefix instead of changing it.
     */
    private fun withDeferredNotice(
        messages: List<EngineMessage>,
        deferred: List<ToolSpec>,
    ): List<EngineMessage> {
        if (deferred.isEmpty()) return messages
        val notice = "<system-reminder>\nTools available but not loaded — call " +
            "ToolSearch before using one: " + deferred.joinToString(", ") { it.name } +
            "\n</system-reminder>"
        val index = messages.indexOfFirst { it.role == "user" }
        if (index < 0) return messages
        return messages.mapIndexed { i, message ->
            if (i != index) {
                message
            } else {
                message.copy(
                    content = if (message.content.isBlank()) notice else "$notice\n\n${message.content}",
                )
            }
        }
    }

    /** What a finished turn amounted to, for the non-streaming answer and the log. */
    data class Result(
        val text: String,
        val thinking: String,
        val toolCalls: List<ToolCallRequest>,
        val stopReason: StopReason,
        val rounds: Int,
        val promptTokens: Int,
        val cachedTokens: Int,
        val generatedTokens: Int,
        val thinkingTokens: Int,
        val tokensPerSecond: Float,
    )

    /**
     * Run the loop.
     *
     * The engine gate is held for the whole of it, across every round. Letting
     * go between rounds would be correct for fairness and wrong for everything
     * else: the second round's prompt is the first round's prompt plus a tool
     * result, so a model evicted in between re-prefills the entire conversation
     * to answer a question it had already read.
     */
    suspend fun run(
        prepared: Prepared,
        registry: ToolRegistry?,
        sink: Sink,
    ): Result {
        val messages = prepared.messages.toMutableList()
        val visible = prepared.visibleTools.toMutableList()
        val loaded = visible.map { it.name }.toMutableSet()
        val index = ToolSearchIndex(prepared.deferredTools)
        val deferredByName = prepared.deferredTools.associateBy { it.name }

        val text = StringBuilder()
        val thinking = StringBuilder()
        var stop = StopReason.EOS
        var rounds = 0
        var promptTokens = 0
        var cachedTokens = 0
        var generatedTokens = 0
        var thinkingTokens = 0
        var rate = 0f
        var finalCalls: List<ToolCallRequest> = emptyList()

        sink.opening()

        val runtime = runner.runtimeFor(prepared.model)
        runner.exclusive(runtime) {
            runner.touch(prepared.model.id)

            while (rounds < config.maxRoundTrips) {
                rounds++
                // Loaded per round, not once before the loop.
                //
                // A tool can change what is resident: `generate_image` is a
                // provider like any other and it runs on the diffusion engine,
                // which on this hardware means letting go of the text model to
                // make room. Loading once left the next round generating
                // against a freed handle, which surfaces as "No model is
                // loaded" in the middle of an answer that was going fine.
                // `EngineManager.load` returns immediately when the model is
                // already resident, so this costs nothing in the ordinary case.
                val engine = runner.loadText(prepared.model, prepared.params)
                val calls = mutableListOf<ToolCallRequest>()
                val roundText = StringBuilder()
                var failure: Pair<String, String?>? = null

                runner.text(
                    engine,
                    GenerateRequest(
                        messages = messages.toList(),
                        params = prepared.params,
                        systemPrompt = prepared.system,
                        imagePaths = prepared.imagePaths,
                        tools = visible.toList(),
                    ),
                ).collect { event ->
                    when (event) {
                        is GenerationEvent.PromptProcessed -> {
                            promptTokens = event.promptTokens
                            cachedTokens = event.cachedTokens
                            sink.event(event)
                        }
                        is GenerationEvent.ThinkingDelta -> {
                            thinking.append(event.text)
                            sink.event(event)
                        }
                        is GenerationEvent.ThinkingDone -> {
                            thinkingTokens += event.totalTokens
                            sink.event(event)
                        }
                        is GenerationEvent.Token -> {
                            roundText.append(event.text)
                            sink.event(event)
                        }
                        is GenerationEvent.Stats -> {
                            generatedTokens = event.generatedTokens
                            rate = event.tokensPerSecond
                            // Pushed out as it arrives rather than kept for the
                            // summary. The notification is the only view of this
                            // run once the app is off screen, and a rate that
                            // only appears after the answer does is a rate
                            // nobody needed.
                            log.update(requestId) {
                                it.copy(
                                    rounds = rounds,
                                    generatedTokens = event.generatedTokens,
                                    tokensPerSecond = event.tokensPerSecond,
                                )
                            }
                        }
                        // Held back rather than forwarded: the loop has to see
                        // the name before the client does, or an intercepted
                        // call would already be on the wire by the time we knew
                        // it was ours to answer.
                        is GenerationEvent.ToolCall -> calls += ToolCallRequest(
                            name = event.name,
                            argumentsJson = event.argumentsJson.ifBlank { "{}" },
                            id = event.id.ifBlank { newCallId() },
                        )
                        is GenerationEvent.Done -> stop = event.stopReason
                        is GenerationEvent.Failed -> failure = event.message to event.suggestion
                    }
                }

                text.append(roundText)

                failure?.let { (message, suggestion) ->
                    sink.failed(message, suggestion)
                    log.intercept(
                        requestId,
                        InterceptRecord(InterceptRecord.Kind.REFUSED, "generate", message),
                    )
                    return@exclusive
                }

                if (calls.isEmpty()) break

                // Anything the client owns ends the loop and goes out as it is.
                val ours = calls.filter { handledHere(it, prepared, deferredByName, loaded) }
                if (ours.isEmpty()) {
                    finalCalls = calls
                    calls.forEach { sink.event(GenerationEvent.ToolCall(it.name, it.argumentsJson, it.id)) }
                    break
                }

                // A round handles one call, because each one changes what the
                // next round is allowed to see and running them together would
                // hide that from the model.
                val call = ours.first()
                val outcome = handle(call, prepared, index, deferredByName, loaded, visible, registry)
                sink.status(outcome.status)
                messages += EngineMessage(
                    role = "assistant",
                    content = roundText.toString(),
                    toolCalls = listOf(call),
                )
                messages += EngineMessage(
                    role = "tool",
                    content = outcome.result,
                    toolCallId = call.id,
                    toolName = call.name,
                )
            }

            if (rounds >= config.maxRoundTrips) {
                // Said out loud rather than returned as a bare stop: a loop that
                // hit its ceiling looks exactly like a model that finished, and
                // the difference matters to whoever is reading the answer.
                sink.status(
                    "● Gave up after ${config.maxRoundTrips} tool rounds\n" +
                        "└  Raise proxy.max_roundtrips if this was legitimate work",
                )
            }
        }

        // An answer that is empty because the whole budget went to reasoning
        // is a failure that looks exactly like a success: the frames are
        // well-formed, the stop reason is legal, and there is simply nothing
        // in it. The engine's own workflow runner refuses this case by name and
        // so does this one — said as text, because a client that has already
        // received `message_start` cannot be handed a 400 instead.
        if (text.isBlank() && finalCalls.isEmpty()) {
            val excuse: String = when {
                thinking.isNotEmpty() && stop == StopReason.MAX_TOKENS ->
                    "[This model spent all $generatedTokens tokens reasoning and produced no " +
                        "answer. Raise max_tokens, or leave `thinking` unset to turn reasoning off.]"
                stop == StopReason.MAX_TOKENS ->
                    "[This model hit the $generatedTokens-token limit before answering. " +
                        "Raise max_tokens.]"
                thinking.isNotEmpty() ->
                    "[This model reasoned for ${thinking.length} characters and then " +
                        "answered with nothing.]"
                // The branch that used to be `null`, and saying nothing here is
                // how an empty answer reached a client as a valid, silent,
                // contentless message. Measured: three tokens, EOS, no content,
                // and a stream that carried only its own opening frame.
                else ->
                    "[This model stopped after $generatedTokens tokens without producing " +
                        "an answer. Its template may be refusing the request as posed.]"
            }
            sink.event(GenerationEvent.Token(excuse, 0))
            text.append(excuse)
        }

        return Result(
            text = text.toString(),
            thinking = thinking.toString(),
            toolCalls = finalCalls,
            stopReason = stop,
            rounds = rounds,
            promptTokens = promptTokens,
            cachedTokens = cachedTokens,
            generatedTokens = generatedTokens,
            thinkingTokens = thinkingTokens,
            tokensPerSecond = rate,
        )
    }

    /** Whether this call is ours to answer rather than the client's. */
    private fun handledHere(
        call: ToolCallRequest,
        prepared: Prepared,
        deferred: Map<String, ToolSpec>,
        loaded: Set<String>,
    ): Boolean = when {
        call.name == ToolSearchIndex.TOOL_SEARCH_SPEC.name -> true
        call.name in prepared.localToolNames -> true
        call.name in deferred && call.name !in loaded -> true
        // A name nobody has: not the client's either, so answering it here with
        // a suggestion beats forwarding a call that can only come back as an
        // error the model cannot learn from.
        call.name !in loaded -> true
        else -> false
    }

    private class Outcome(val status: String, val result: String)

    private suspend fun handle(
        call: ToolCallRequest,
        prepared: Prepared,
        index: ToolSearchIndex,
        deferred: Map<String, ToolSpec>,
        loaded: MutableSet<String>,
        visible: MutableList<ToolSpec>,
        registry: ToolRegistry?,
    ): Outcome {
        val input = runCatching { ProxyJson.parseToJsonElement(call.argumentsJson) }
            .getOrNull() as? kotlinx.serialization.json.JsonObject

        // — the meta-tool —
        if (call.name == ToolSearchIndex.TOOL_SEARCH_SPEC.name) {
            val query = input?.str("query").orEmpty()
            val limit = input?.i("max_results") ?: ToolSearchIndex.MAX_RESULTS
            val matched = index.search(query, limit).filterNot { it.name in loaded }

            log.intercept(
                requestId,
                InterceptRecord(
                    InterceptRecord.Kind.TOOL_SEARCH, query,
                    matched.joinToString(", ") { it.name }.ifBlank { "nothing" },
                ),
            )

            if (matched.isEmpty()) {
                // Distinguish "already have it" from "no such thing", because
                // the model's next move differs: one is call the tool, the
                // other is search again with different words.
                val already = index.search(query, limit).filter { it.name in loaded }
                return if (already.isNotEmpty()) {
                    Outcome(
                        "● ToolSearch(\"${query.take(60)}\")\n" +
                            "└  already loaded · call it directly",
                        "These are already loaded and can be called right now: " +
                            already.joinToString(", ") { it.name } + ".",
                    )
                } else {
                    Outcome(
                        "● ToolSearch(\"${query.take(60)}\")\n└  no matches",
                        "Nothing matched \"$query\". The tools that exist but are not loaded " +
                            "are: " + deferred.keys.joinToString(", ") + ".",
                    )
                }
            }

            visible += matched
            loaded += matched.map { it.name }
            return Outcome(
                "● ToolSearch(\"${query.take(60)}\")\n" +
                    "└  ${matched.size} loaded: ${matched.joinToString(", ") { it.name }}",
                ToolSearchIndex.renderSchemas(matched),
            )
        }

        // — a tool this device runs —
        if (call.name in prepared.localToolNames && registry != null) {
            val result = registry.call(call.name, call.argumentsJson)
            log.intercept(
                requestId,
                InterceptRecord(
                    InterceptRecord.Kind.RAN_TOOL, call.name,
                    result.text.take(200),
                ),
            )
            val summary = result.text.lineSequence().firstOrNull()?.take(90).orEmpty()
            return Outcome(
                "● ${call.name}\n└  ${if (result.isError) "failed · " else ""}$summary",
                result.text,
            )
        }

        // — called blind —
        val schema = deferred[call.name]
        if (schema != null) {
            return if (config.autoLoadTools) {
                visible += schema
                loaded += schema.name
                log.intercept(
                    requestId,
                    InterceptRecord(InterceptRecord.Kind.AUTO_LOADED, call.name, "schema injected"),
                )
                Outcome(
                    "● Loaded ${call.name}\n└  schema delivered · awaiting retry",
                    "The schema for `${call.name}` is now loaded:\n\n" +
                        ToolSearchIndex.renderSchemas(listOf(schema)) +
                        "\n\nCall it again using exactly these parameter names.",
                )
            } else {
                log.intercept(
                    requestId,
                    InterceptRecord(InterceptRecord.Kind.BLOCKED, call.name, "not loaded"),
                )
                Outcome(
                    "● Blocked ${call.name}\n└  not loaded · told to ToolSearch first",
                    "`${call.name}` is not loaded. Call " +
                        "ToolSearch(query=\"select:${call.name}\") first, then call it again.",
                )
            }
        }

        // — a name nobody has —
        val suggestions = index.search(call.name, 5)
        log.intercept(
            requestId,
            InterceptRecord(
                InterceptRecord.Kind.UNKNOWN_TOOL, call.name,
                suggestions.joinToString(", ") { it.name }.ifBlank { "no close matches" },
            ),
        )
        return if (suggestions.isEmpty()) {
            Outcome(
                "● Unknown tool ${call.name}\n└  no close matches",
                "There is no tool called `${call.name}`, and nothing similar. " +
                    "Call ToolSearch with keywords describing what you need.",
            )
        } else {
            Outcome(
                "● Unknown tool ${call.name}\n" +
                    "└  did you mean ${suggestions.joinToString(", ") { it.name }}",
                "There is no tool called `${call.name}`. These exist:\n\n" +
                    ToolSearchIndex.renderSchemas(suggestions) +
                    "\n\nCall one of them by its exact name.",
            )
        }
    }

    companion object {
        /**
         * Turn an engine failure into a refusal both protocols can render.
         *
         * The suggestion survives. It is the half that tells somebody what to
         * do about it, and it is produced by the engine precisely so that it
         * can be shown.
         */
        fun refusalFor(failure: Throwable): ProxyRefusal = when (failure) {
            is ProxyRefusal -> failure
            is ModelRunFailure -> ProxyRefusal(
                500, "api_error", failure.message, failure.suggestion,
            )
            else -> ProxyRefusal(
                500, "api_error",
                failure.message ?: "The run failed with no message.",
                null,
            )
        }

        /** Which modality a route needs, so a request naming the wrong model is refused by name. */
        fun requireModality(model: ModelEntity, wanted: Set<Modality>, route: String) {
            if (model.modality !in wanted) {
                throw ProxyRefusal.badRequest(
                    "`${model.id}` is a ${model.modality.label.lowercase()} model, and $route " +
                        "needs ${wanted.joinToString(" or ") { it.label.lowercase() }}.",
                    "Ask GET /v1/models for what is installed.",
                )
            }
        }
    }
}
