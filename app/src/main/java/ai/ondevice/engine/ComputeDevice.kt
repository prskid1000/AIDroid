package ai.ondevice.engine

import ai.ondevice.core.BackendId
import ai.ondevice.data.prefs.AppPrefs
import kotlinx.coroutines.flow.first

/**
 * Which piece of silicon runs a model: the Settings choice, clamped to what
 * ggml actually registered.
 *
 * One class rather than the same six lines in three engines. The setting is
 * global — it says "run on the NPU", not "run chat on the NPU" — so chat,
 * transcribe and image have to answer it identically or the screen is lying
 * about two of them. It lived inside EngineManager and reached llama.cpp only,
 * which is exactly how the Voice and Image tabs came to ignore a setting that
 * looked like it applied to everything.
 *
 * The clamp is the load-bearing part, and it is per runtime because the runtimes
 * are separate shared objects: whisper.cpp answers for whisper.cpp's binary.
 * A device that did not register is not a device, and the floor is CPU — never
 * a refusal, because there is always a CPU.
 *
 * There is deliberately no "auto". It was a benchmark once, and then "the first
 * backend registered", which is an ordering dressed up as a decision.
 */
class ComputeDevice(
    private val prefs: AppPrefs,
    private val registry: RuntimeRegistry,
) {

    /**
     * @param override a per-model backend, which wins: someone who has set one
     *   has said something more specific than the global setting.
     */
    suspend fun chosen(runtimeId: String, override: BackendId? = null): BackendId {
        val available = registry.backendsFor(runtimeId)
        fun clamp(backend: BackendId) = backend.takeIf { it in available } ?: BackendId.CPU

        override?.let { return clamp(it) }
        val setting = runCatching { BackendId.valueOf(prefs.backendMode.first()) }.getOrNull()
        return clamp(setting ?: BackendId.CPU)
    }

    /**
     * The same answer in the spelling the JNI layers pass to ggml.
     *
     * ggml's registry names are its own — the NPU registers as "HTP" — and all
     * three runtimes resolve against them, so this is the one string that
     * crosses the boundary rather than three translations of an enum.
     */
    suspend fun registryName(runtimeId: String, override: BackendId? = null): String =
        chosen(runtimeId, override).registryNames.first()
}
