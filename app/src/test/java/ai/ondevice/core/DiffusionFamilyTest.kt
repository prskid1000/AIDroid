package ai.ondevice.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** That a model's family reaches the component list it needs, from the name the runtime prints. */
class DiffusionFamilyTest {

    @Test
    fun `SDXL conditions on two encoders, not one`() {
        assertEquals(setOf("clip_l", "clip_g"), DiffusionFamily.forName("SDXL")!!.encoders)
    }

    @Test
    fun `FLUX 2 reads its prompt with a language model`() {
        assertEquals(setOf("llm"), DiffusionFamily.forName("Flux.2 klein")!!.encoders)
        assertEquals(setOf("clip_l", "t5xxl"), DiffusionFamily.forName("Flux")!!.encoders)
    }

    @Test
    fun `the runtime's own spelling and a gguf architecture both land`() {
        assertEquals(
            DiffusionFamily.forName("Flux.2 klein"),
            DiffusionFamily.forName("flux2_klein"),
        )
        assertEquals(DiffusionFamily.forName("SDXL"), DiffusionFamily.forName("sdxl"))
    }

    @Test
    fun `a name nothing recognises claims nothing`() {
        assertNull(DiffusionFamily.forName(null))
        assertNull(DiffusionFamily.forName(""))
        assertNull(DiffusionFamily.forName("   "))
        assertNull(DiffusionFamily.forName("some model published next year"))
    }

    @Test
    fun `only the UNet families keep their decoder in the checkpoint`() {
        assertEquals(false, DiffusionFamily.forName("SD 1.x")!!.vaeSeparate)
        assertEquals(false, DiffusionFamily.forName("SDXL")!!.vaeSeparate)
        assertTrue(DiffusionFamily.forName("SD3.x")!!.vaeSeparate)
        assertTrue(DiffusionFamily.forName("Flux.2 klein")!!.vaeSeparate)
    }

    @Test
    fun `every encoder named here is a role this app can fill`() {
        val keys = AttachmentRole.entries.map { it.paramKey }.toSet()
        listOf("Flux", "Flux.2 klein", "SDXL", "SD 1.x", "SD3.x", "Qwen Image", "Chroma Radiance")
            .forEach { name ->
                val family = DiffusionFamily.forName(name) ?: error("$name should be recognised")
                (family.encoders + family.optionalEncoders).forEach {
                    assertEquals("$name names $it", true, it in keys)
                }
            }
    }
}
