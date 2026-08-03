package ai.ondevice.core

/** What kind of thing a model is. */
enum class Modality {
    TEXT,
    VISION,
    EMBEDDING,
    DIFFUSION,
    SPEECH_TO_TEXT,
    TEXT_TO_SPEECH,
    UNKNOWN,
    ;

    val label: String
        get() = when (this) {
            TEXT -> "Text"
            VISION -> "Vision"
            EMBEDDING -> "Embedding"
            DIFFUSION -> "Diffusion"
            SPEECH_TO_TEXT -> "Speech"
            TEXT_TO_SPEECH -> "Voice"
            UNKNOWN -> "Unknown"
        }
}

/** The artifact format, which is what decides whether a runtime can load it. */
enum class ModelFormat { GGUF, GGML_BIN, ONNX, SAFETENSORS, PYTORCH_BIN, UNKNOWN }

enum class Capability { TEXT, VISION, TOOLS, EMBEDDING, DIFFUSION, TRANSCRIBE, SYNTHESIZE }

/**
 * SPEC §3.3.
 *
 * There used to be a WORKS_SLOWER between FAST and TIGHT, and it meant one
 * thing only: a quant with no Adreno OpenCL kernel, which would fall back to
 * the CPU. Everything falls back to the CPU now — it is the only device — so
 * the verdict is purely about whether the model fits.
 */
enum class Verdict {
    FAST,
    TIGHT,

    /**
     * Nothing in this build claims the architecture — a caution, not a stop.
     *
     * It used to stop the download, and that made a list of names into the
     * thing deciding what could be installed. Those names come from two
     * vocabularies that were never agreed between: a GGUF says what its
     * *family* is, while a runtime enumerates the specific versions it builds,
     * so the same model is `ltxv` on one side and `ltxav` on the other. Every
     * disagreement of that shape read as "no runtime can load this", which was
     * not true and could not be checked without downloading the file.
     *
     * So the claim is weakened to what is actually known: this build does not
     * recognise the name. The loader is the thing that can answer the real
     * question, and it answers it in seconds against a file on disk.
     */
    UNSUPPORTED_ARCH,
    WONT_FIT,
    NOT_RUNNABLE,
    ;

    /**
     * Whether to let the download start.
     *
     * The two that say no are the two the device settles: it will not fit, or
     * no runtime here reads the format at all. Everything else — including an
     * unrecognised architecture — is advice attached to a download that is
     * allowed to proceed.
     */
    val runnable: Boolean get() = this != WONT_FIT && this != NOT_RUNNABLE

    val tone: VerdictTone
        get() = when (this) {
            FAST -> VerdictTone.AFFIRMATIVE
            TIGHT, UNSUPPORTED_ARCH -> VerdictTone.CAVEAT
            WONT_FIT, NOT_RUNNABLE -> VerdictTone.REFUSAL
        }

    val label: String
        get() = when (this) {
            FAST -> "Fits"
            TIGHT -> "Tight"
            WONT_FIT -> "Won't fit"
            UNSUPPORTED_ARCH -> "Unrecognised architecture"
            NOT_RUNNABLE -> "Not runnable"
        }
}

/** Accent means runnable; an accent outline means caveat; neutral means no. */
enum class VerdictTone { AFFIRMATIVE, CAVEAT, REFUSAL }

/** SPEC §9 — tiering controls default visibility only; nothing is hidden for good. */
enum class Tier {
    BASIC,
    ADVANCED,
    EXPERT,
    ;

    val label: String
        get() = when (this) {
            BASIC -> "Basic"
            ADVANCED -> "Advanced"
            EXPERT -> "Expert"
        }
}

enum class DownloadState { QUEUED, RUNNING, PAUSED, VERIFYING, COMPLETE, FAILED }

/** SPEC §17.3 — a bundle declaring an unsupported contract is refused, never loaded. */

enum class MessageRole { USER, ASSISTANT, SYSTEM, TOOL_CALL, TOOL_RESULT }

/** What a recorded run produced. */
enum class PredictionKind {
    CHAT,
    IMAGE,
    VIDEO,
    SPEECH,
    TRANSCRIBE,
    ;

    val label: String
        get() = when (this) {
            CHAT -> "Chat"
            IMAGE -> "Image"
            VIDEO -> "Video"
            SPEECH -> "Speech"
            TRANSCRIBE -> "Transcript"
        }
}

/** SPEC §3.2 — every refusal gets its own message and its own remedy. */
enum class RefusalKind(val heading: String, val explanation: String) {
    WONT_FIT(
        "Won't fit",
        "The weights, the KV cache at the chosen context, and the compute buffer add up to " +
            "more than this device has. The live refusal shows the arithmetic and offers " +
            "smaller quants.",
    ),
    PYTORCH_ONLY(
        "PyTorch weights only",
        "The repo ships safetensors or .bin with no GGUF or ONNX export. Converting needs a " +
            "desktop, and the app will not pretend otherwise.",
    ),
    GATED(
        "Gated repo",
        "Accept the licence on Hugging Face, then paste a token. The token goes in the " +
            "Android Keystore and is used for nothing else.",
    ),
    NOT_FOUND(
        "No such repo",
        "Hugging Face has no repo by that id. Usually a typo or a private repo the token " +
            "cannot see.",
    ),
    PICKLE_BLOCKED(
        "Pickle files blocked",
        "Pickle executes code on load. There is no safe way to open one, so it is refused " +
            "outright rather than warned about.",
    ),
    UNSCANNED(
        "Unscanned files",
        "A warning, not a block. Hugging Face has not scanned these files, and GGUF has had " +
            "template-injection vulnerabilities.",
    ),
    NO_RUNTIME(
        "No runtime for this format",
        "The format is recognised but the engine that reads it is not installed in this " +
            "build.",
    ),
}
