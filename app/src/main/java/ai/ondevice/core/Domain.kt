package ai.ondevice.core

/**
 * What kind of thing a model is. Classified from architecture and file shape,
 * never from the repo name (SPEC §3.2 step 3).
 */
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

/** Detected at boot. SPEC §8.1 — Vulkan is deliberately out of v1. */
enum class BackendId {
    OPENCL,
    HEXAGON,
    CPU,
    ;

    /**
     * The piece of silicon, which is the question a user is actually asking.
     *
     * These were named for their APIs — "OpenCL", "Hexagon HTP" — and that is
     * the wrong axis to offer a choice on: nobody wants OpenCL, they want the
     * GPU, and OpenCL is one of several ways to get there. The API is still
     * worth showing where the subject really is the build (see [api] and the
     * Runtimes screen); it is not what a settings list should ask.
     */
    val label: String
        get() = when (this) {
            OPENCL -> "GPU"
            HEXAGON -> "NPU"
            CPU -> "CPU"
        }

    /** How this build reaches [label] — the detail, for the screen about builds. */
    val api: String
        get() = when (this) {
            OPENCL -> "OpenCL"
            HEXAGON -> "Hexagon HTP"
            CPU -> "CPU"
        }

    /**
     * Whether a name ggml registered means this backend.
     *
     * ggml names its registries after the API — "OpenCL", "CPU" — this enum
     * names them for a database column, and [label] names them for a person,
     * so all three only coincide by luck.
     * Matching on both spellings, case-insensitively, means a rename upstream
     * degrades to "unrecognised backend" rather than to a wrong one.
     */
    fun matches(registered: String): Boolean =
        registered.equals(name, ignoreCase = true) ||
            registered.equals(api, ignoreCase = true) ||
            registered.equals(label, ignoreCase = true)
}

/**
 * SPEC §3.3. Six verdicts, computed before any download.
 *
 * The design canvas is explicit that these carry no red or green: the palette
 * is mono, so [FAST] is an accent tag, the two warnings are accent outlines,
 * and the three refusals are a neutral disc with a slash. Weight comes from the
 * mark, not the hue — which is why this enum carries a [tone] rather than a
 * colour.
 */
enum class Verdict {
    FAST,
    WORKS_SLOWER,
    TIGHT,
    WONT_FIT,
    UNSUPPORTED_ARCH,
    NOT_RUNNABLE,
    ;

    val runnable: Boolean get() = this == FAST || this == WORKS_SLOWER || this == TIGHT

    val tone: VerdictTone
        get() = when (this) {
            FAST -> VerdictTone.AFFIRMATIVE
            WORKS_SLOWER, TIGHT -> VerdictTone.CAVEAT
            WONT_FIT, UNSUPPORTED_ARCH, NOT_RUNNABLE -> VerdictTone.REFUSAL
        }

    val label: String
        get() = when (this) {
            FAST -> "Fast"
            WORKS_SLOWER -> "Works, slower"
            TIGHT -> "Tight"
            WONT_FIT -> "Won't fit"
            UNSUPPORTED_ARCH -> "Unsupported architecture"
            NOT_RUNNABLE -> "Not runnable"
        }
}

/** Accent means runnable; an accent outline means caveat; neutral means no. */
enum class VerdictTone { AFFIRMATIVE, CAVEAT, REFUSAL }

/**
 * The speed class shown next to each quant variant. Q4_0 is the only quant with
 * an Adreno OpenCL kernel, so everything else falls to CPU — the canvas says so
 * in as many words under the variant list.
 */
enum class SpeedClass {
    OPENCL_FAST,
    CPU_PATH,
    ;

    val label: String get() = if (this == OPENCL_FAST) "fast · OpenCL" else "CPU path"
}

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
enum class RuntimeState { NOT_INSTALLED, INSTALLED, UPDATE_AVAILABLE, ROLLED_BACK }

enum class MessageRole { USER, ASSISTANT, SYSTEM, TOOL_CALL, TOOL_RESULT }

/**
 * What a recorded run produced.
 *
 * Four kinds because there are four things this device makes and four places an
 * artifact row is written. It is deliberately not [Modality]: a modality
 * describes a *model*, and one model can be driven by two different runs —
 * whisper transcribes both a file and a live recording, and the distinction that
 * matters to a stored run is which artifact table its id points into.
 */
enum class PredictionKind {
    CHAT,
    IMAGE,
    SPEECH,
    TRANSCRIBE,
    ;

    val label: String
        get() = when (this) {
            CHAT -> "Chat"
            IMAGE -> "Image"
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
    UNKNOWN_ARCHITECTURE(
        "Unsupported architecture",
        "The installed runtime does not list this architecture. The live refusal names the " +
            "build and how many it knows; a newer one may add it.",
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
