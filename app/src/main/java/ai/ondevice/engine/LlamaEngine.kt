package ai.ondevice.engine

import ai.ondevice.core.BackendId
import ai.ondevice.core.SparseParams
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

/**
 * llama.cpp, for real.
 *
 * The whole of this class is translation: Kotlin data classes to JSON, JSON back
 * to [GenerationEvent]. It contains no knowledge of any model — the chat
 * template, the reasoning tags, the tool-call syntax and the stop sequences all
 * come from the GGUF via upstream's own parsers (SPEC §1.3). There is no
 * `when (modelName)` here and there is not supposed to be.
 *
 * Streaming is pull-based: the flow asks the native side for the next token, so
 * cancellation is simply the flow stopping, and the teardown in `onCompletion`
 * is what actually frees the sampler and the KV — which is Appendix A #7's
 * requirement that cancelling frees memory rather than detaching a callback.
 */
class LlamaEngine(
    override val descriptor: RuntimeDescriptor,
) : InferenceEngine {

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }
    private val mutex = Mutex()

    private var handle: Long = 0L
    private var loaded: LoadedModel? = null
    private var appliedParams: SparseParams = SparseParams.EMPTY

    override val isLoaded: Boolean get() = handle != 0L
    override val loadedModelId: String? get() = loaded?.modelId

    override suspend fun load(request: LoadRequest): Result<LoadedModel> = mutex.withLock {
        runCatching {
            check(LlamaBridge.available) {
                LlamaBridge.loadError ?: "The llama.cpp runtime is not installed in this build."
            }
            LlamaBridge.nativeInit()

            // Warm-swap: the old model goes before the new one arrives, so the
            // app never holds two at once (SPEC §3.5).
            if (handle != 0L) freeHandle()

            val started = System.currentTimeMillis()
            val newHandle = withContext(Dispatchers.IO) {
                LlamaBridge.nativeLoad(request.modelPath, request.params.toJsonString())
            }
            check(newHandle != 0L) { "The runtime returned no handle for ${request.modelPath}." }
            handle = newHandle
            appliedParams = request.params

            val info = json.parseToJsonElement(LlamaBridge.nativeInfo(handle)).jsonObject
            val model = LoadedModel(
                modelId = request.modelId,
                backend = request.backend,
                contextLength = info.int("contextLoaded") ?: 0,
                layers = info.int("layers") ?: 0,
                embeddingLength = info.int("embeddingLength") ?: 0,
                embeddingLengthKv = info.int("embeddingLengthKv") ?: 0,
                chatTemplate = info.string("chatTemplate")?.takeIf { it.isNotBlank() },
                // The end-of-generation tokens the *vocabulary* declares. A
                // hand-kept list per model family is exactly the model-locking
                // §1.3 exists to prevent.
                stopSequences = info["eogTokens"]?.jsonArray
                    ?.mapNotNull { it.jsonPrimitive.contentOrNull }
                    ?.filter { it.isNotBlank() }
                    ?: emptyList(),
                loadMillis = System.currentTimeMillis() - started,
            )
            loaded = model
            model
        }
    }

    override suspend fun unload() = mutex.withLock { freeHandle() }

    private fun freeHandle() {
        if (handle != 0L) {
            LlamaBridge.nativeFree(handle)
            handle = 0L
        }
        loaded = null
        appliedParams = SparseParams.EMPTY
    }

    override suspend fun applyParams(params: SparseParams): ParamReport {
        appliedParams = appliedParams.overlaidWith(params)
        if (handle == 0L) return ParamReport(rejected = params.keys.toList())

        val report = json.parseToJsonElement(
            LlamaBridge.nativeApplyParams(handle, params.toJsonString()),
        ).jsonObject
        return ParamReport(
            applied = report.strings("applied"),
            rejected = report.strings("rejected"),
        )
    }

    override fun generate(request: GenerateRequest): Flow<GenerationEvent> = flow {
        if (handle == 0L) {
            emit(GenerationEvent.Failed("No model is loaded.", "Pick a model in chat settings."))
            return@flow
        }

        appliedParams = appliedParams.overlaidWith(request.params)
        LlamaBridge.nativeApplyParams(handle, appliedParams.toJsonString())

        val formatted = formatPrompt(request, addGenerationPrompt = true)
        val start = json.parseToJsonElement(
            LlamaBridge.nativeStartGeneration(handle, formatted.string("prompt").orEmpty(), "[]"),
        ).jsonObject

        start.string("error")?.let { error ->
            emit(GenerationEvent.Failed(error, start.string("suggestion")))
            return@flow
        }

        emit(
            GenerationEvent.PromptProcessed(
                promptTokens = start.int("promptTokens") ?: 0,
                cachedTokens = start.int("cachedTokens") ?: 0,
                promptTokensPerSecond = start.float("promptPerSecond") ?: 0f,
            ),
        )

        val backend = loaded?.backend ?: BackendId.CPU
        var index = 0
        var thinkingTokens = 0
        var thinkingStarted = 0L
        var thinkingClosed = false

        while (true) {
            currentCoroutineContext().ensureActive()
            val step = json.parseToJsonElement(LlamaBridge.nativeNextToken(handle)).jsonObject

            val reasoning = step.string("reasoningDelta").orEmpty()
            if (reasoning.isNotEmpty()) {
                if (thinkingStarted == 0L) thinkingStarted = System.currentTimeMillis()
                thinkingTokens++
                emit(GenerationEvent.ThinkingDelta(reasoning))
            }

            val content = step.string("contentDelta").orEmpty()
            if (content.isNotEmpty()) {
                // The first ordinary token is the end of the reasoning block —
                // upstream's parser tells us that by switching which delta it
                // fills, so the app never looks for a `</think>` itself.
                if (thinkingStarted != 0L && !thinkingClosed) {
                    thinkingClosed = true
                    emit(
                        GenerationEvent.ThinkingDone(
                            thinkingTokens,
                            System.currentTimeMillis() - thinkingStarted,
                        ),
                    )
                }
                emit(GenerationEvent.Token(content, index))
                index++
            }

            if (index > 0 && index % 8 == 0 && content.isNotEmpty()) {
                emit(
                    GenerationEvent.Stats(
                        tokensPerSecond = step.float("tokensPerSecond") ?: 0f,
                        generatedTokens = step.int("generated") ?: index,
                        contextUsed = step.int("contextUsed") ?: 0,
                        backend = backend,
                    ),
                )
            }

            if (step.bool("done") == true) {
                if (thinkingStarted != 0L && !thinkingClosed) {
                    emit(
                        GenerationEvent.ThinkingDone(
                            thinkingTokens,
                            System.currentTimeMillis() - thinkingStarted,
                        ),
                    )
                }
                emit(
                    GenerationEvent.Stats(
                        tokensPerSecond = step.float("tokensPerSecond") ?: 0f,
                        generatedTokens = step.int("generated") ?: index,
                        contextUsed = step.int("contextUsed") ?: 0,
                        backend = backend,
                    ),
                )
                step["toolCalls"]?.jsonArray?.forEach { call ->
                    val obj = call.jsonObject
                    emit(
                        GenerationEvent.ToolCall(
                            name = obj.string("name").orEmpty(),
                            argumentsJson = obj.string("arguments").orEmpty(),
                            id = obj.string("id").orEmpty(),
                        ),
                    )
                }
                emit(
                    GenerationEvent.Done(
                        stopReason = runCatching {
                            StopReason.valueOf(step.string("stopReason") ?: "EOS")
                        }.getOrDefault(StopReason.EOS),
                        generatedTokens = step.int("generated") ?: index,
                        elapsedMillis = step.long("elapsedMillis") ?: 0L,
                    ),
                )
                return@flow
            }
        }
    }.flowOn(Dispatchers.Default).onCompletion {
        // Cancellation has to reach the native loop, not just stop collecting —
        // otherwise the decode keeps running and keeps its buffers.
        if (handle != 0L) LlamaBridge.nativeCancel(handle)
    }

    override suspend fun renderPrompt(request: GenerateRequest): RenderedPrompt =
        withContext(Dispatchers.Default) {
            val model = loaded
            if (handle == 0L || model == null) {
                return@withContext RenderedPrompt(
                    text = "",
                    tokens = emptyList(),
                    template = null,
                    templateSource = "no model loaded",
                    totalTokens = 0,
                    imageTokens = 0,
                    cachedTokens = 0,
                    contextLimit = 0,
                    stopSequences = emptyList(),
                )
            }

            val formatted = formatPrompt(request, addGenerationPrompt = true)
            val text = formatted.string("prompt").orEmpty()
            val tokens = json.parseToJsonElement(LlamaBridge.nativeTokenize(handle, text))
                .jsonArray.map { element ->
                    val obj = element.jsonObject
                    PromptToken(
                        text = obj.string("text").orEmpty(),
                        id = obj.int("id") ?: 0,
                        special = obj.bool("special") ?: false,
                    )
                }

            RenderedPrompt(
                text = text,
                tokens = tokens,
                template = model.chatTemplate,
                templateSource = formatted.string("templateSource")?.takeIf { it.isNotBlank() }
                    ?.let { "gguf.chat_template" } ?: "runtime default",
                totalTokens = tokens.size,
                imageTokens = request.messages.sumOf { it.imagePaths.size } * IMAGE_TOKENS,
                cachedTokens = 0,
                contextLimit = model.contextLength,
                stopSequences = model.stopSequences,
            )
        }

    override suspend fun tokenCount(text: String): Int = withContext(Dispatchers.Default) {
        if (handle == 0L) 0 else LlamaBridge.nativeTokenCount(handle, text)
    }

    /** Drop the KV without unloading — §8.3's first response to memory pressure. */
    fun clearCache() {
        if (handle != 0L) LlamaBridge.nativeClearCache(handle)
    }

    // — translation —

    private fun formatPrompt(request: GenerateRequest, addGenerationPrompt: Boolean): JsonObject {
        val messages = buildJsonArray {
            request.systemPrompt?.takeIf { it.isNotBlank() }?.let { prompt ->
                add(
                    buildJsonObject {
                        put("role", "system")
                        put("content", prompt)
                    },
                )
            }
            request.messages.forEach { message ->
                add(
                    buildJsonObject {
                        put("role", message.role)
                        put("content", message.content)
                        message.toolCallId?.let { put("tool_call_id", it) }
                        message.toolName?.let { put("tool_name", it) }
                        if (message.toolCalls.isNotEmpty()) {
                            put(
                                "tool_calls",
                                buildJsonArray {
                                    message.toolCalls.forEach { call ->
                                        add(
                                            buildJsonObject {
                                                put("name", call.name)
                                                put("arguments", call.argumentsJson)
                                                put("id", call.id)
                                            },
                                        )
                                    }
                                },
                            )
                        }
                    },
                )
            }
        }

        val tools = buildJsonArray {
            request.tools.forEach { tool ->
                add(
                    buildJsonObject {
                        put("name", tool.name)
                        put("description", tool.description)
                        put("parameters", json.parseToJsonElement(tool.parametersJson))
                    },
                )
            }
        }

        return json.parseToJsonElement(
            LlamaBridge.nativeFormatPrompt(handle, messages.toString(), tools.toString(), addGenerationPrompt),
        ).jsonObject
    }

    private companion object {
        /** Only used to label the inspector; the real count comes from mtmd. */
        const val IMAGE_TOKENS = 1456
    }
}

// — small readers, so the translation above stays readable —

private fun JsonObject.string(key: String): String? =
    (this[key] as? JsonPrimitive)?.takeIf { it.isString || it.contentOrNull != null }?.contentOrNull

private fun JsonObject.int(key: String): Int? =
    (this[key] as? JsonPrimitive)?.let { runCatching { it.int }.getOrNull() }

private fun JsonObject.long(key: String): Long? =
    (this[key] as? JsonPrimitive)?.let { runCatching { it.content.toLong() }.getOrNull() }

private fun JsonObject.float(key: String): Float? =
    (this[key] as? JsonPrimitive)?.let { runCatching { it.float }.getOrNull() }

private fun JsonObject.bool(key: String): Boolean? =
    (this[key] as? JsonPrimitive)?.let { runCatching { it.boolean }.getOrNull() }

private fun JsonObject.strings(key: String): List<String> =
    (this[key] as? JsonArray)?.mapNotNull { it.jsonPrimitive.contentOrNull } ?: emptyList()
