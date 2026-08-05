package ai.ondevice.params

import ai.ondevice.engine.RuntimeRegistry
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/**
 * That a dial nobody has touched still means "inherit".
 *
 * Wan 2.2 is two denoisers either side of a noise boundary and upstream gives
 * each a full sampler. This app copied the low-noise one wholesale and changed
 * only the step count, so the rest were shared whether or not that was wanted.
 * Opening them up is only safe while an untouched one keeps following the other
 * expert — and the trap is that most of these have no harmless zero. A CFG of 0
 * is not a gentler CFG, it is an unconditioned pass, and a default of 0 here
 * would quietly wreck every clip made by a model with one denoiser.
 *
 * So the defaults are sentinels, and this asserts they stay sentinels. It is
 * the same rule the loaded-model settings follow: the app does not get to
 * invent a value the runtime is perfectly able to choose.
 */
class InheritSentinelTest {

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    private val manifest: ParamManifest by lazy {
        json.decodeFromString(
            ParamManifest.serializer(),
            File("src/main/assets/params-manifest.json").readText(),
        )
    }

    private fun spec(runtime: String, key: String): ParamSpec =
        manifest.spec(runtime, key) ?: throw AssertionError("$key is not in the manifest")

    @Test
    fun `the high-noise expert's numeric dials default to following the other one`() {
        listOf("high_noise_cfg_scale", "high_noise_guidance", "high_noise_eta").forEach { key ->
            val s = spec(RuntimeRegistry.STABLE_DIFFUSION, key)
            assertEquals(
                "$key must default to -1 (inherit). A real number here is applied to the " +
                    "high-noise pass of every model, including the ones that have no such pass",
                -1.0,
                s.default?.jsonPrimitive?.doubleOrNull,
            )
            assertEquals(
                "$key's minimum has to reach the sentinel or it cannot be set back to inherit",
                -1.0,
                s.min,
            )
        }
    }

    @Test
    fun `the high-noise sampler defaults to auto rather than to a named method`() {
        val s = spec(RuntimeRegistry.STABLE_DIFFUSION, "high_noise_sampling_method")
        assertEquals(ParamType.ENUM, s.type)
        assertEquals("auto", s.default?.jsonPrimitive?.content)
        assertEquals(
            "\"auto\" is the inherit value and must be offered first",
            "auto",
            s.values.firstOrNull(),
        )
    }

    @Test
    fun `every high-noise dial is video-only`() {
        listOf(
            "high_noise_steps", "high_noise_cfg_scale", "high_noise_guidance",
            "high_noise_eta", "high_noise_sampling_method",
        ).forEach { key ->
            assertEquals(
                "$key describes a second denoiser, which only a video architecture has",
                listOf("video"),
                spec(RuntimeRegistry.STABLE_DIFFUSION, key).appliesTo?.output,
            )
        }
    }

    /**
     * -1 is upstream's own "unrestricted", so the default changes nothing for
     * anyone who does not set it. It is here to be *lowered*: a reasoning model
     * left to think as long as it likes is the most reliable way to cook a
     * phone, and until this existed there was no dial for it short of cutting
     * the reply short with `n_predict`, which truncates the answer instead.
     */
    @Test
    fun `the thinking budget defaults to unrestricted and can reach zero`() {
        val s = spec(RuntimeRegistry.LLAMA, "reasoning_budget")
        assertEquals(ParamType.INT, s.type)
        assertEquals(-1, s.default?.jsonPrimitive?.intOrNull)
        assertEquals(
            "0 must be reachable — it is the setting that ends the thinking block at once",
            -1.0,
            s.min,
        )
    }
}
