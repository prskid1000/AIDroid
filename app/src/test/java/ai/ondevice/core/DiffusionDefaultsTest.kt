package ai.ondevice.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** That a model's family reaches the settings it needs, from the name the runtime prints. */
class DiffusionDefaultsTest {

    @Test
    fun `a guidance-distilled model is not asked to guide twice`() {
        // CFG above 1 on Flux is what turned the lynx into coloured noise.
        assertEquals(1.0f, DiffusionDefaults.forName("Flux")!!.cfgScale, 0f)
        assertEquals(1.0f, DiffusionDefaults.forName("Flux.2 klein")!!.cfgScale, 0f)
    }

    @Test
    fun `klein is matched before the family it belongs to`() {
        assertEquals(4, DiffusionDefaults.forName("Flux.2 klein")!!.steps)
        assertEquals(20, DiffusionDefaults.forName("Flux.2")!!.steps)
    }

    @Test
    fun `the runtime's own spelling and a gguf architecture both land`() {
        assertEquals(
            DiffusionDefaults.forName("Flux.2 klein"),
            DiffusionDefaults.forName("flux2_klein"),
        )
        assertEquals(DiffusionDefaults.forName("SDXL"), DiffusionDefaults.forName("sdxl"))
    }

    @Test
    fun `a name nothing recognises changes nothing`() {
        assertNull(DiffusionDefaults.forName(null))
        assertNull(DiffusionDefaults.forName(""))
        assertNull(DiffusionDefaults.forName("   "))
        assertNull(DiffusionDefaults.forName("some model published next year"))
    }

    @Test
    fun `a UNet keeps the settings it was tuned for`() {
        val sd1 = DiffusionDefaults.forName("SD 1.x")!!
        assertEquals(7.0f, sd1.cfgScale, 0f)
        assertEquals("dpm++2m", sd1.samplingMethod)
    }

    @Test
    fun `every sampler and schedule named here is one the manifest offers`() {
        val samplers = setOf(
            "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
            "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd",
        )
        val schedules = setOf("discrete", "karras", "exponential", "ays", "gits")
        val names = listOf(
            "Flux", "Flux.2 klein", "SDXL", "SD 1.x", "SD3.x", "Qwen Image", "Chroma Radiance",
        )
        names.forEach { name ->
            val d = DiffusionDefaults.forName(name) ?: error("$name should be recognised")
            assertEquals("$name sampler", true, d.samplingMethod in samplers)
            assertEquals("$name schedule", true, d.schedule in schedules)
        }
    }
}
