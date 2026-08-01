package ai.ondevice.data.hf

import ai.ondevice.core.Modality
import ai.ondevice.core.ModelFormat
import ai.ondevice.core.RefusalKind
import ai.ondevice.core.Verdict

/** What the resolver produces: either something loadable, or a reasoned no. */
sealed interface Resolution {

    data class Resolved(val model: ResolvedModel) : Resolution

    /** SPEC §3.2 — each failure case gets its own message *and its own remedy*. */
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

/** A model the app knows how to fetch and load. */
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
    /** Auxiliary files, grouped by role, each carrying its own default. */
    val companions: List<CompanionGroup>,
    /** True when metadata came from the Range-request header parser, not the API. */
    val metadataFromHeader: Boolean,
    val securityStatus: String?,
    val hasPickleFiles: Boolean,
) {
    val isVision: Boolean get() = companions.any { it.role == CompanionRole.VISION_PROJECTOR }
}

/** One downloadable variant. */
data class QuantVariant(
    val name: String,
    val files: List<RemoteFile>,
    val note: String,
    /** Why this build cannot run this variant, or null when it can. */
    val blockedReason: String? = null,
    /** Why this variant is a bad idea, when it will nonetheless run. */
    val cautionReason: String? = null,
) {
    val totalBytes: Long get() = files.sumOf { it.sizeBytes }
    val isSharded: Boolean get() = files.size > 1
    val runnable: Boolean get() = blockedReason == null
}

data class RemoteFile(
    val filename: String,
    val sizeBytes: Long,
    val sha256: String? = null,
    val commitId: String? = null,
    val securityStatus: String? = null,
)

/** How many files of one role the thing that consumes them can actually take. */
enum class Cardinality {
    /** One path, and only one. */
    ONE,

    /** Every file is part of one thing. */
    ALL,
}

enum class CompanionRole(val cardinality: Cardinality) {
    VISION_PROJECTOR(Cardinality.ONE),
    VAE(Cardinality.ONE),
    CLIP_L(Cardinality.ONE),
    CLIP_G(Cardinality.ONE),
    T5XXL(Cardinality.ONE),
    /** FLUX.2's prompt encoder: a whole language model, not a CLIP. */
    LLM_ENCODER(Cardinality.ONE),
    TAESD(Cardinality.ONE),
    CONTROLNET(Cardinality.ONE),
    UPSCALER(Cardinality.ONE),
    VOICES(Cardinality.ALL),
    VAD(Cardinality.ONE),
    ;

    val label: String
        get() = when (this) {
            VISION_PROJECTOR -> "Vision projector"
            VAE -> "VAE"
            CLIP_L -> "CLIP-L text encoder"
            CLIP_G -> "CLIP-G text encoder"
            T5XXL -> "T5-XXL text encoder"
            LLM_ENCODER -> "Text encoder (language model)"
            TAESD -> "TAESD (live preview)"
            CONTROLNET -> "ControlNet"
            UPSCALER -> "Upscaler"
            VOICES -> "Voice style vectors"
            VAD -> "Silero VAD"
        }

    /**
     * Whether the primary model is unusable without it.
     *
     * Per role, not per architecture, which makes this advice rather than a
     * rule — a T5-XXL is indispensable to SD 3.5 and meaningless to SDXL, and
     * this cannot tell them apart. It reads as a warning and blocks nothing:
     * the file that fills a role can come from anywhere in the library, so
     * skipping the copy in one repo is a legitimate choice.
     *
     * CLIP-G is here because SDXL conditions on *both* encoders — ViT-L and
     * ViT-bigG — and a prompt read through only one of them is half a prompt.
     * It was missing, which is why an SDXL card asked for CLIP-L alone.
     */
    val required: Boolean
        get() = this == VISION_PROJECTOR || this == VAE || this == CLIP_L || this == CLIP_G ||
            this == T5XXL || this == LLM_ENCODER || this == VOICES
}

/** SPEC §3.2 step 5: companions are detected and auto-paired. */
data class CompanionFile(
    val file: RemoteFile,
    val role: CompanionRole,
)

/** Every file found for one role, and which of them to download. */
data class CompanionGroup(
    val role: CompanionRole,
    val candidates: List<CompanionFile>,
    /** Filenames chosen by default. The screen may replace this wholesale. */
    val selected: Set<String>,
    val kind: Kind,
) {
    enum class Kind {
        /** All of them, together, are one thing. Kokoro's voice packs. */
        PARTS,

        /** The same file at different precisions. Pick one; prefer F16. */
        ALTERNATIVES,

        /** Different things that happen to fill the same slot — ControlNet's canny against its depth, an upscaler's ×2 against its ×4. */
        CHOICES,
    }

    val chosen: List<CompanionFile> get() = candidates.filter { it.file.filename in selected }
    val selectedBytes: Long get() = chosen.sumOf { it.file.sizeBytes }

    /** Null when there is nothing to say, so a screen can skip the line. */
    val note: String?
        get() = when {
            candidates.size <= 1 -> null
            kind == Kind.PARTS -> "${candidates.size} files, all needed"
            kind == Kind.ALTERNATIVES ->
                "${candidates.size} precisions available" +
                    (chosen.firstOrNull()?.let { ", ${precisionOf(it.file.filename)} chosen" } ?: "")
            selected.isEmpty() -> "${candidates.size} to choose from, none selected"
            else -> "${candidates.size} to choose from"
        }

    private fun precisionOf(filename: String): String =
        CompanionGrouping.precisionToken(filename) ?: "one"
}

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
