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

/**
 * One parameter key some part of this build actually acts on.
 *
 * @property appliedBy which part. The distinction matters because "the runtime
 *   ignores this key" is not the same as "nothing happens if you set it": half
 *   of the diffusion parameters are constructor arguments to `nativeLoad` and
 *   never go near the dispatch table, and the voice screen's capture settings
 *   belong to the recorder rather than to whisper.cpp.
 */
data class ParamCapability(
    val key: String,
    val requiresReload: Boolean,
    val appliedBy: Applier,
    /**
     * The runtime's own default, when it can report one.
     *
     * llama.cpp's setters already fall back to the live value, so its defaults
     * exist in a default-constructed `common_params` and are read back from
     * there. Null means the engine did not say — the manifest's description
     * stands, which is what it is for.
     */
    val default: JsonElement? = null,
) {
    enum class Applier { RUNTIME, APP }
}

/**
 * What each runtime in *this build* will do something with.
 *
 * SPEC §16.1 says there are no hardcoded parameter widgets, and there are none
 * — the renderer is still generic over the manifest. What the manifest could
 * not do was tell the truth about the binary underneath it. It is a JSON file
 * that has never met the `.so`, and the only guard against the two disagreeing
 * was a hand-written `sinceBuild` string, which is a second thing to keep in
 * sync rather than a way of not having to.
 *
 * They had drifted, and not slightly. Asking the runtimes directly found
 * **nine** llama parameters the screen offered that the binary does not accept
 * — `cache_prompt`, `defrag_thold`, `split_mode`, `lora`, `mmproj`,
 * `mmproj_use_gpu`, `pooling_type`, `penalize_nl`, `json_schema` — plus
 * whisper's `flash_attn` and stable-diffusion's `type`. Every one of them
 * rendered a control that moved, saved a value that persisted, and changed
 * nothing whatsoever. That is worse than the parameter being absent, because
 * the user has no way to tell.
 *
 * It also found one in the other direction: whisper's `single_segment` is
 * accepted by the binary and was described nowhere, so it could not be reached.
 *
 * So the question is asked of the thing that answers it. The native runtimes
 * enumerate their own dispatch tables over JNI; the app-side keys are declared
 * beside the code that reads them. The manifest keeps the job it is good at —
 * labels, help text, ranges, grouping, tiers — and loses the one it was bad at,
 * which was claiming a capability on someone else's behalf.
 *
 * A key the runtime reports and the manifest does not describe is still shown,
 * as a plain text field in its own group. Undescribed beats unreachable, and it
 * means a parameter upstream adds is usable the day the runtime is rebuilt
 * rather than the day someone remembers to write a manifest row for it.
 */
object EngineParams {

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    private val cache = mutableMapOf<String, Map<String, ParamCapability>?>()

    /**
     * @return the keys [runtimeId] will act on, or null when this build has no
     *   way to ask — a runtime that is not compiled in, or an id nothing here
     *   knows. Null is deliberately different from an empty map: "cannot ask"
     *   must not read as "accepts nothing", or a missing `.so` would silently
     *   empty the parameter screen instead of saying the runtime is absent.
     */
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

        // Both voice engines are Kotlin all the way down — ONNX Runtime has no
        // dispatch table of ours to enumerate — so they declare their keys next
        // to the code that reads them and this only collects the declaration.
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
            // An app-applied key is read on the next run rather than pushed into
            // a live context, so none of them need the reload batching that
            // exists for llama's load-time parameters.
            ParamCapability(it, requiresReload = false, appliedBy = ParamCapability.Applier.APP)
        }

    /**
     * The diffusion keys that reach sd.cpp as `nativeLoad` arguments instead of
     * through the dispatch table.
     *
     * Derived from [AttachmentRole] rather than listed, because that enum
     * already owns the key each role's path travels under and DiffusionEngine
     * builds its load call by walking it. Writing the same ten strings again
     * here would be the exact duplication this file exists to remove.
     */
    private fun diffusionLoadKeys(): Set<String> =
        AttachmentRole.entries.map { it.paramKey }.toSet() + SD_APP_KEYS

    /** Read by DiffusionEngine.load and by the upscale path, not by the table. */
    private val SD_APP_KEYS = setOf("threads", "upscale_model")

    /**
     * Read by the voice screen's recorder, not by whisper.cpp: how often the
     * capture loop hands a window over.
     *
     * `vad` used to be listed here too, on the belief that the capture loop
     * dropped silent windows. It never did — nothing read the key on either
     * side. whisper's own parameter table now owns it, so listing it here as an
     * app key would take it back off the runtime that finally implements it.
     */
    private val CAPTURE_KEYS = setOf("step_ms")
}
