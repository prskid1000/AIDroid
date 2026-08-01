package ai.ondevice.engine

import ai.ondevice.core.SparseParams
import kotlinx.coroutines.flow.Flow

/** The boundary between the app and a native runtime. */
interface InferenceEngine {

    val descriptor: RuntimeDescriptor

    val isLoaded: Boolean

    val loadedModelId: String?

    /** Load a model. */
    suspend fun load(request: LoadRequest): Result<LoadedModel>

    suspend fun unload()

    /** Apply live parameters. */
    suspend fun applyParams(params: SparseParams): ParamReport

    /** Generate. */
    fun generate(request: GenerateRequest): Flow<GenerationEvent>

    /** The exact string that will reach the tokenizer, plus its token boundaries. */
    suspend fun renderPrompt(request: GenerateRequest): RenderedPrompt

    suspend fun tokenCount(text: String): Int
}

data class LoadRequest(
    val modelId: String,
    val modelPath: String,
    val companionPaths: Map<String, String> = emptyMap(),
    val params: SparseParams = SparseParams.EMPTY,
    val chatTemplate: String? = null,
)

data class LoadedModel(
    val modelId: String,
    val contextLength: Int,
    val layers: Int,
    val embeddingLength: Int,
    val embeddingLengthKv: Int,
    /** Query heads, as the GGUF declares them. */
    val heads: Int = 0,
    val chatTemplate: String?,
    /** Where [chatTemplate] came from: the GGUF, or a `chat_template` override. */
    val templateSource: String = "gguf.chat_template",
    /** `--chat-template-kwargs` as the runtime currently holds it. */
    val templateKwargsJson: String = "{}",
    val stopSequences: List<String>,
    val loadMillis: Long,
)

/** What the runtime did with the keys it was handed. */
data class ParamReport(
    val applied: List<String> = emptyList(),
    val rejected: List<String> = emptyList(),
    val clamped: Map<String, String> = emptyMap(),
) {
    val hasRejections: Boolean get() = rejected.isNotEmpty()
}

data class GenerateRequest(
    val messages: List<EngineMessage>,
    val params: SparseParams,
    val systemPrompt: String? = null,
    val imagePaths: List<String> = emptyList(),
    val grammar: String? = null,
    val seed: Long = -1,
    /** Tools offered to the model this turn. */
    val tools: List<ToolSpec> = emptyList(),
)

data class EngineMessage(
    val role: String,
    val content: String,
    val imagePaths: List<String> = emptyList(),
    /** Set on an assistant message that asked for tools. */
    val toolCalls: List<ToolCallRequest> = emptyList(),
    /** Set on a `tool` message carrying a result back. */
    val toolCallId: String? = null,
    val toolName: String? = null,
)

/** A tool as the model sees it: a name, a sentence, and a JSON Schema. */
data class ToolSpec(
    val name: String,
    val description: String,
    val parametersJson: String,
)

data class ToolCallRequest(
    val name: String,
    val argumentsJson: String,
    val id: String,
)

/** Streaming output. Everything the generation UI shows comes through here. */
sealed interface GenerationEvent {
    data class PromptProcessed(
        val promptTokens: Int,
        val cachedTokens: Int,
        val promptTokensPerSecond: Float,
    ) : GenerationEvent

    /** A reasoning-block delta, parsed out of the configured tag pair. */
    data class ThinkingDelta(val text: String) : GenerationEvent

    data class ThinkingDone(val totalTokens: Int, val elapsedMillis: Long) : GenerationEvent

    data class Token(
        val text: String,
        val tokenIndex: Int,
        /** Populated when `n_probs` > 0 — powers the token-probability inspector. */
        val topProbabilities: List<TokenProbability> = emptyList(),
    ) : GenerationEvent

    data class Stats(
        val tokensPerSecond: Float,
        val generatedTokens: Int,
        val contextUsed: Int,
    ) : GenerationEvent

    /** The model asked for a tool. */
    data class ToolCall(val name: String, val argumentsJson: String, val id: String) : GenerationEvent

    data class Done(val stopReason: StopReason, val generatedTokens: Int, val elapsedMillis: Long) : GenerationEvent

    /** A native failure surfaced as a real message with numbers, per SPEC §8.4 — an OOM must never present as a bare crash. */
    data class Failed(val message: String, val suggestion: String?) : GenerationEvent
}

data class TokenProbability(val token: String, val probability: Float)

enum class StopReason { EOS, STOP_SEQUENCE, MAX_TOKENS, CONTEXT_FULL, CANCELLED }

/** The prompt inspector's three views: rendered string, tokens, template. */
data class RenderedPrompt(
    val text: String,
    val tokens: List<PromptToken>,
    val template: String?,
    val templateSource: String,
    val totalTokens: Int,
    val imageTokens: Int,
    val cachedTokens: Int,
    val contextLimit: Int,
    val stopSequences: List<String>,
)

data class PromptToken(
    val text: String,
    val id: Int,
    /** Special tokens are painted with the accent, ordinary text with a tint. */
    val special: Boolean = false,
)
