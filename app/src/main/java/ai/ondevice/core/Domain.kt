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

/** Detected at boot. SPEC §8.1 — Vulkan is deliberately out of v1. */
enum class BackendId {
    OPENCL,
    CPU,
    ;

    /** The piece of silicon, which is the question a user is actually asking. */
    val label: String
        get() = when (this) {
            OPENCL -> "GPU"
            CPU -> "CPU"
        }

    /** How this build reaches [label] — the detail, for the screen about builds. */
    val api: String
        get() = when (this) {
            OPENCL -> "OpenCL"
            CPU -> "CPU"
        }

    val registryNames: List<String>
        get() = when (this) {
            OPENCL -> listOf("OpenCL")
            CPU -> listOf("CPU")
        }

    /** Whether a name ggml registered means this backend. */
    fun matches(registered: String): Boolean =
        registryNames.any { it.equals(registered, ignoreCase = true) } ||
            registered.equals(name, ignoreCase = true) ||
            registered.equals(api, ignoreCase = true) ||
            registered.equals(label, ignoreCase = true)
}

/** SPEC §3.3. */
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

/** The speed class shown next to each quant variant. */
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

/** What a recorded run produced. */
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
