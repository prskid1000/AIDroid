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

    val label: String
        get() = when (this) {
            OPENCL -> "OpenCL"
            HEXAGON -> "Hexagon HTP"
            CPU -> "CPU"
        }
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

/** SPEC §8.3 — the configurable policy on THERMAL_STATUS_SEVERE. */
enum class ThermalPolicy {
    CONTINUE,
    REDUCE_THREADS,
    DOWNSHIFT_CPU,
    PAUSE,
    ;

    val label: String
        get() = when (this) {
            CONTINUE -> "Continue regardless"
            REDUCE_THREADS -> "Reduce threads at severe"
            DOWNSHIFT_CPU -> "Downshift to CPU"
            PAUSE -> "Pause generation"
        }
}

enum class MessageRole { USER, ASSISTANT, SYSTEM, TOOL_CALL, TOOL_RESULT }

/** SPEC §3.2 — every refusal gets its own message and its own remedy. */
enum class RefusalKind {
    WONT_FIT,
    PYTORCH_ONLY,
    UNKNOWN_ARCHITECTURE,
    GATED,
    NOT_FOUND,
    PICKLE_BLOCKED,
    UNSCANNED,
    NO_RUNTIME,
}
