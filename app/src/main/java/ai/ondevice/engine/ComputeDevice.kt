package ai.ondevice.engine

import ai.ondevice.core.BackendId
import ai.ondevice.data.prefs.AppPrefs
import kotlinx.coroutines.flow.first

/** Which piece of silicon runs a model: the Settings choice, clamped to what ggml actually registered. */
class ComputeDevice(
    private val prefs: AppPrefs,
    private val registry: RuntimeRegistry,
) {

    /** @param override a per-model backend, which wins: someone who has set one has said something more specific than the global setting. */
    suspend fun chosen(runtimeId: String, override: BackendId? = null): BackendId {
        val available = registry.backendsFor(runtimeId)
        fun clamp(backend: BackendId) = backend.takeIf { it in available } ?: BackendId.CPU

        override?.let { return clamp(it) }
        val setting = runCatching { BackendId.valueOf(prefs.backendMode.first()) }.getOrNull()
        return clamp(setting ?: BackendId.CPU)
    }

    /** The same answer in the spelling the JNI layers pass to ggml. */
    suspend fun registryName(runtimeId: String, override: BackendId? = null): String =
        chosen(runtimeId, override).registryNames.first()
}
