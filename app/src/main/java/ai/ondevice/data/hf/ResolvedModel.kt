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
    /**
     * Auxiliary files, grouped by role, each carrying its own default.
     *
     * Grouped rather than flat because the flat list could not express the
     * difference between "these three files are all needed" and "these three
     * files are the same file three times".
     */
    val companions: List<CompanionGroup>,
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
    /**
     * Why this build cannot run this variant, or null when it can.
     *
     * Set before anything is downloaded, from the graph rather than the
     * filename. It is shown rather than hidden: a variant that silently
     * disappears reads as a repo that does not have it, and the useful thing to
     * say is which one to pick instead.
     */
    val blockedReason: String? = null,
    /**
     * Why this variant is a bad idea, when it will nonetheless run.
     *
     * Distinct from [blockedReason] on purpose: this does not stop the
     * download, it just refuses to let the smallest number on the screen be the
     * only thing the user is told. Below about two bits per weight a small
     * model stops being a worse version of itself and starts producing
     * confident nonsense, and "1.1 GB" reads like the sensible choice right up
     * until it is loaded.
     */
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

/**
 * How many files of one role the thing that consumes them can actually take.
 *
 * A property of the runtimes this app bundles rather than of any model, which
 * is why it is safe to state here: it is read off the native call each role
 * feeds, not guessed from a filename.
 */
enum class Cardinality {
    /**
     * One path, and only one. `sd_ctx_params_t` has a single `jstring` field
     * per role (sd_jni.cpp), llama.cpp takes one projector, and the app's own
     * `companionPathsJson` is a map keyed by role — so a second file of the
     * same role is not merely unused, it is unaddressable.
     */
    ONE,

    /**
     * Every file is part of one thing. Kokoro reads one voice pack per
     * utterance, but which one is the user's choice at speak time, so all of
     * them have to be on disk for the choice to exist.
     */
    ALL,
}

enum class CompanionRole(val cardinality: Cardinality) {
    VISION_PROJECTOR(Cardinality.ONE),
    VAE(Cardinality.ONE),
    CLIP_L(Cardinality.ONE),
    CLIP_G(Cardinality.ONE),
    T5XXL(Cardinality.ONE),
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
)

/**
 * Every file found for one role, and which of them to download.
 *
 * The spec's rule is about *parts*: nobody should have to work out that a
 * projector belongs with the weights. It says nothing about *alternatives*, and
 * treating those as parts is how three vision projectors — 2.68 GB — ended up
 * queued for a model that loads one. Worse, only one survives: the manifest
 * this all ends up in is keyed by role, so the other two are bytes on disk that
 * nothing can ever refer to.
 *
 * So the group states which of the three cases it is and carries a default that
 * the user is free to change.
 */
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

        /**
         * Different things that happen to fill the same slot — ControlNet's
         * canny against its depth, an upscaler's ×2 against its ×4. The
         * runtime still takes one, but which one is a question about the
         * picture the user wants, and no default can answer it.
         */
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
