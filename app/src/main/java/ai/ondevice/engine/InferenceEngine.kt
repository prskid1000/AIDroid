package ai.ondevice.engine

import ai.ondevice.core.BackendId
import ai.ondevice.core.SparseParams
import kotlinx.coroutines.flow.Flow

/**
 * The boundary between the app and a native runtime.
 *
 * **The parameter contract is a string-keyed map, and it is that way from the
 * very first call** (SPEC §16.7, Appendix A #10). A typed struct across JNI
 * would mean every upstream parameter addition becomes a Kotlin code change,
 * which defeats §1.5 entirely. The native side maps keys onto `common_params`
 * and the sampler chain through a generated dispatch table, reports the keys it
 * didn't recognise, and never crashes on one.
 *
 * That is also why [applyParams] returns a report rather than Unit: the escape
 * hatch in §16.6 lets a user pass arbitrary JSON straight through, and they are
 * owed an answer about what the runtime actually accepted.
 */
interface InferenceEngine {

    val descriptor: RuntimeDescriptor

    val isLoaded: Boolean

    val loadedModelId: String?

    /**
     * Load a model. [params] carries the §4.1 load-time parameters — the ones
     * that need a reload to change — as raw keys.
     */
    suspend fun load(request: LoadRequest): Result<LoadedModel>

    suspend fun unload()

    /**
     * Apply live parameters. Returns which keys were taken and which were not
     * recognised; unknown keys are never fatal.
     */
    suspend fun applyParams(params: SparseParams): ParamReport

    /**
     * Generate. The Flow is cold and cancellable; **cancellation must free
     * native memory**, not merely detach the callback (Appendix A #7), which is
     * why implementations do their teardown in `onCompletion`/`finally` rather
     * than relying on the consumer.
     */
    fun generate(request: GenerateRequest): Flow<GenerationEvent>

    /**
     * The exact string that will reach the tokenizer, plus its token
     * boundaries. This powers the prompt inspector (SPEC §4.4), which is the
     * single most useful debugging affordance for a local LLM app.
     */
    suspend fun renderPrompt(request: GenerateRequest): RenderedPrompt

    suspend fun tokenCount(text: String): Int
}

data class LoadRequest(
    val modelId: String,
    val modelPath: String,
    val companionPaths: Map<String, String> = emptyMap(),
    val backend: BackendId,
    val params: SparseParams = SparseParams.EMPTY,
    val chatTemplate: String? = null,
)

data class LoadedModel(
    val modelId: String,
    val backend: BackendId,
    val contextLength: Int,
    val layers: Int,
    val embeddingLength: Int,
    val embeddingLengthKv: Int,
    val chatTemplate: String?,
    val stopSequences: List<String>,
    val loadMillis: Long,
)

/**
 * What the runtime did with the keys it was handed. `rejected` is information,
 * not an error — a preset written under a newer engine can legitimately carry
 * keys this build doesn't have, and §11 requires we keep them anyway.
 */
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
)

data class EngineMessage(
    val role: String,
    val content: String,
    val imagePaths: List<String> = emptyList(),
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
        val backend: BackendId,
    ) : GenerationEvent

    data class Done(val stopReason: StopReason, val generatedTokens: Int, val elapsedMillis: Long) : GenerationEvent

    /**
     * A native failure surfaced as a real message with numbers, per SPEC §8.4 —
     * an OOM must never present as a bare crash.
     */
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
