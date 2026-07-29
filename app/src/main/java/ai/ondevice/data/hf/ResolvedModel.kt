package ai.ondevice.data.hf

import ai.ondevice.core.Modality
import ai.ondevice.core.ModelFormat
import ai.ondevice.core.RefusalKind
import ai.ondevice.core.SpeedClass
import ai.ondevice.core.Verdict

/** What the resolver produces: either something loadable, or a reasoned no. */
sealed interface Resolution {

    data class Resolved(val model: ResolvedModel) : Resolution

    /**
     * SPEC §3.2 — each failure case gets its own message *and its own remedy*.
     * A refusal with no way forward is just a dead end, which is the thing §1.2
     * is trying to avoid.
     */
    data class Refused(
        val kind: RefusalKind,
        val title: String,
        val detail: String,
        val subject: String,
        val working: String? = null,
        val remedies: List<Remedy> = emptyList(),
    ) : Resolution
}

/** A remedy is an action, not advice. */
data class Remedy(
    val label: String,
    val action: RemedyAction,
    val primary: Boolean = false,
)

sealed interface RemedyAction {
    data class SearchRepo(val query: String) : RemedyAction
    data class OpenMirror(val owner: String, val repo: String) : RemedyAction
    data class OpenUrl(val url: String) : RemedyAction
    data object EnterToken : RemedyAction
    data object CheckRuntimeUpdate : RemedyAction
    data class ShowSmallerQuants(val repoId: String) : RemedyAction
    data class ContinueAnyway(val repoId: String) : RemedyAction
}

/**
 * A model the app knows how to fetch and load.
 *
 * Everything model-specific here came from GGUF metadata or the HF API — no
 * field is filled from a table keyed by model name. That is what makes SPEC
 * §1.1 true: a model released tomorrow resolves without an app update, provided
 * the bundled runtime already knows its architecture.
 */
data class ResolvedModel(
    val repoId: String,
    val owner: String,
    val repo: String,
    val revision: String,
    val displayName: String,
    val architecture: String?,
    val modality: Modality,
    val format: ModelFormat,
    val contextLength: Int?,
    val chatTemplate: String?,
    val bosToken: String?,
    val eosToken: String?,
    val parameterCount: Long?,
    val layers: Int?,
    val embeddingLength: Int?,
    val embeddingLengthKv: Int?,
    val gated: Boolean,
    val quants: List<QuantVariant>,
    val companions: List<CompanionFile>,
    /** True when metadata came from the Range-request header parser, not the API. */
    val metadataFromHeader: Boolean,
    val securityStatus: String?,
    val hasPickleFiles: Boolean,
) {
    val isVision: Boolean get() = companions.any { it.role == CompanionRole.VISION_PROJECTOR }
}

/**
 * One downloadable variant. A sharded model is one variant with several
 * [files] — SPEC §3.2 step 6: `model-00001-of-00003.gguf` is *one logical
 * model*, and the downloader treats the parts as a single atomic job.
 */
data class QuantVariant(
    val name: String,
    val files: List<RemoteFile>,
    val speedClass: SpeedClass,
    val note: String,
) {
    val totalBytes: Long get() = files.sumOf { it.sizeBytes }
    val isSharded: Boolean get() = files.size > 1
}

data class RemoteFile(
    val filename: String,
    val sizeBytes: Long,
    val sha256: String? = null,
    val commitId: String? = null,
    val securityStatus: String? = null,
)

enum class CompanionRole {
    VISION_PROJECTOR,
    VAE,
    CLIP_L,
    CLIP_G,
    T5XXL,
    TAESD,
    CONTROLNET,
    UPSCALER,
    VOICES,
    VAD,
    ;

    val label: String
        get() = when (this) {
            VISION_PROJECTOR -> "Vision projector"
            VAE -> "VAE"
            CLIP_L -> "CLIP-L text encoder"
            CLIP_G -> "CLIP-G text encoder"
            T5XXL -> "T5-XXL text encoder"
            TAESD -> "TAESD (live preview)"
            CONTROLNET -> "ControlNet"
            UPSCALER -> "Upscaler"
            VOICES -> "Voice style vectors"
            VAD -> "Silero VAD"
        }

    /** Whether the primary model is unusable without it. */
    val required: Boolean
        get() = this == VISION_PROJECTOR || this == VAE || this == CLIP_L || this == T5XXL || this == VOICES
}

/**
 * SPEC §3.2 step 5: companions are detected and auto-paired. "Never make the
 * user manually assemble a multi-file model."
 */
data class CompanionFile(
    val file: RemoteFile,
    val role: CompanionRole,
    val autoSelected: Boolean = true,
)

/** The compatibility verdict attached to a specific quant at a specific context. */
data class VerdictResult(
    val verdict: Verdict,
    val estimate: FitEstimate,
    val availableRamBytes: Long,
    val freeStorageBytes: Long,
) {
    /** "You have 10.40 GB free of 12.00 GB. Headroom 3.20 GB." */
    fun headroomNote(totalRamBytes: Long): String {
        val headroom = estimate.headroomBytes(availableRamBytes)
        return buildString {
            append("You have ${ai.ondevice.core.Fmt.gb(availableRamBytes)} GB free")
            append(" of ${ai.ondevice.core.Fmt.gb(totalRamBytes)} GB.")
            if (headroom > 0) {
                append(" Headroom ${ai.ondevice.core.Fmt.gb(headroom)} GB.")
            } else {
                append(" Short by ${ai.ondevice.core.Fmt.gb(-headroom)} GB.")
            }
        }
    }
}
