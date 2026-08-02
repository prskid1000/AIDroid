package ai.ondevice.params

import ai.ondevice.engine.RuntimeRegistry
import ai.ondevice.speech.KokoroEngine
import ai.ondevice.speech.OmniVoiceEngine
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/**
 * That nothing a runtime accepts reaches the screen as an untyped text box.
 *
 * `ParamRepository.specsFor` renders any reported-but-undescribed key under
 * "not described yet" with a generic help line. That is the right fallback —
 * better than hiding a parameter the runtime will act on — but it is a
 * fallback, and it had quietly become the resting state for four of them:
 * `enable_thinking` and `chat_template_kwargs` on the chat screen,
 * `single_segment` and `step_ms` on the voice one. `enable_thinking` in
 * particular is a switch most people want and it was an empty box.
 *
 * The native keys are read out of the C++ dispatch tables rather than listed
 * here, because that table *is* the contract — a unit test cannot call JNI, and
 * a hand-copied list would drift the same way the manifest did.
 */
class DescribedParamsTest {

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    private val manifest: ParamManifest by lazy {
        json.decodeFromString(
            ParamManifest.serializer(),
            File("src/main/assets/params-manifest.json").readText(),
        )
    }

    private fun described(runtimeId: String): Set<String> =
        manifest.paramsFor(runtimeId).map { it.key }.toSet()

    /**
     * The keys a native runtime says it will act on.
     *
     * Each apply table is written as `{ "key", { [](…) } }`, with llama's
     * carrying a reload flag before the lambda — which is why that part is
     * optional here. This is the same table `nativeSupportedParams` enumerates
     * over JNI at runtime, read from source because a unit test has no JNI.
     */
    private fun dispatchTable(source: String): Set<String> {
        val text = File("src/main/cpp/$source").readText()
        return Regex("""\{\s*"([a-z_0-9]+)",\s*\{\s*(?:(?:true|false)\s*,\s*)?\[""")
            .findAll(text)
            .map { it.groupValues[1] }
            .toSet()
    }

    private fun assertAllDescribed(runtimeId: String, reported: Set<String>) {
        val missing = (reported - described(runtimeId)).sorted()
        assertEquals(
            "$runtimeId reports these and the manifest describes none of them, so each " +
                "renders as an empty text box: $missing",
            emptyList<String>(),
            missing,
        )
    }

    @Test
    fun `every llama key is described`() {
        val table = dispatchTable("llama_jni.cpp")
        assertEquals("the dispatch table could not be read", true, table.size > 20)
        assertAllDescribed(RuntimeRegistry.LLAMA, table)
    }

    @Test
    fun `every whisper key is described`() {
        val table = dispatchTable("whisper_jni.cpp")
        assertEquals("the dispatch table could not be read", true, table.size > 10)
        // `step_ms` is read by the recorder rather than by whisper.cpp, and is
        // added to the reported set the same way EngineParams adds it.
        assertAllDescribed(RuntimeRegistry.WHISPER, table + "step_ms")
    }

    /**
     * The diffusion table only. Its load-time settings and component paths do
     * not appear in it — they reach `sd_ctx_params_t` rather than the dispatch
     * table, and [LoadContractTest] covers those.
     */
    @Test
    fun `every diffusion key is described`() {
        val table = dispatchTable("sd_jni.cpp")
        assertEquals("the dispatch table could not be read", true, table.size > 10)
        assertAllDescribed(RuntimeRegistry.STABLE_DIFFUSION, table)
    }

    @Test
    fun `every voice key is described`() {
        assertAllDescribed(RuntimeRegistry.KOKORO, KokoroEngine.PARAM_KEYS)
        assertAllDescribed(RuntimeRegistry.OMNIVOICE, OmniVoiceEngine.PARAM_KEYS)
    }

    /**
     * A described key that names a JSON value has to say so, or the screen
     * gives it a plain string field and the runtime is handed a quoted blob.
     */
    @Test
    fun `the map-valued keys are typed as maps`() {
        listOf("logit_bias", "chat_template_kwargs").forEach { key ->
            assertEquals(
                "$key takes a JSON object and must be typed `map` to be editable as one",
                ParamType.MAP,
                manifest.spec(RuntimeRegistry.LLAMA, key)?.type,
            )
        }
    }
}
