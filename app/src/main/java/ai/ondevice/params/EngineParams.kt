package ai.ondevice.params

import ai.ondevice.core.AttachmentRole
import ai.ondevice.engine.LlamaBridge
import ai.ondevice.engine.RuntimeRegistry
import ai.ondevice.engine.SdBridge
import ai.ondevice.engine.WhisperBridge
import ai.ondevice.speech.KokoroEngine
import ai.ondevice.speech.OmniVoiceEngine
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** One parameter key some part of this build actually acts on. */
data class ParamCapability(
    val key: String,
    val requiresReload: Boolean,
    val appliedBy: Applier,
    /** The runtime's own default, when it can report one. */
    val default: JsonElement? = null,
) {
    enum class Applier { RUNTIME, APP }
}

/** What each runtime in *this build* will do something with. */
object EngineParams {

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    private val cache = mutableMapOf<String, Map<String, ParamCapability>?>()

    /** @return the keys [runtimeId] will act on, or null when this build has no way to ask — a runtime that is not compiled in, or an id nothing here knows. */
    @Synchronized
    fun capabilities(runtimeId: String): Map<String, ParamCapability>? =
        cache.getOrPut(runtimeId) { resolve(runtimeId) }

    private fun resolve(runtimeId: String): Map<String, ParamCapability>? = when (runtimeId) {
        RuntimeRegistry.LLAMA ->
            native(LlamaBridge.available) { LlamaBridge.nativeSupportedParams() }

        RuntimeRegistry.WHISPER ->
            native(WhisperBridge.available) { WhisperBridge.nativeSupportedParams() }
                ?.plus(app(CAPTURE_KEYS))

        RuntimeRegistry.STABLE_DIFFUSION ->
            native(SdBridge.available) { SdBridge.nativeSupportedParams() }
                ?.plus(app(diffusionLoadKeys()))

        RuntimeRegistry.KOKORO -> app(KokoroEngine.PARAM_KEYS)
        RuntimeRegistry.OMNIVOICE -> app(OmniVoiceEngine.PARAM_KEYS)

        else -> null
    }

    private inline fun native(
        available: Boolean,
        report: () -> String,
    ): Map<String, ParamCapability>? {
        if (!available) return null
        val text = runCatching(report).getOrNull() ?: return null
        val parsed = runCatching { json.parseToJsonElement(text).jsonObject }.getOrNull() ?: return null
        return parsed.mapValues { (key, value) ->
            val row = value as? JsonObject
            ParamCapability(
                key = key,
                requiresReload = row?.get("reload")?.jsonPrimitive?.booleanOrNull ?: false,
                appliedBy = ParamCapability.Applier.RUNTIME,
                default = row?.get("default"),
            )
        }
    }

    private fun app(keys: Collection<String>): Map<String, ParamCapability> =
        keys.associateWith {
            ParamCapability(it, requiresReload = false, appliedBy = ParamCapability.Applier.APP)
        }

    /** The diffusion keys that reach sd.cpp as `nativeLoad` arguments instead of through the dispatch table. */
    private fun diffusionLoadKeys(): Set<String> =
        AttachmentRole.entries.map { it.paramKey }.toSet() + SD_APP_KEYS

    /** Read by DiffusionEngine.load and by the upscale path, not by the table. */
    private val SD_APP_KEYS = setOf("threads", "upscale_model")

    /** Read by the voice screen's recorder, not by whisper.cpp: how often the capture loop hands a window over. */
    private val CAPTURE_KEYS = setOf("step_ms")
}
