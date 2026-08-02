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
        SD_CPP_VERSIONS.forEach { name ->
            val family = DiffusionFamily.forName(name) ?: error("$name should be recognised")
            (family.encoders + family.optionalEncoders).forEach {
                assertEquals("$name names $it", true, it in keys)
            }
        }
    }

    /**
     * Every string `model_version_to_str` can print reaches an answer.
     *
     * The gap this closes was silent by construction: an unrecognised name
     * returns null, null means "claim nothing is missing", and a screen that
     * claims nothing looks exactly like a screen with nothing to say. Half of
     * upstream's versions fell through — Wan, Anima, Z-Image, HiDream, LTX-AV
     * and a dozen others — so a model from any of them ran with no idea which
     * encoders it needed and no warning that the app did not know.
     */
    @Test
    fun `every version stable-diffusion cpp can name is recognised`() {
        SD_CPP_VERSIONS.forEach {
            assertTrue("$it should be recognised", DiffusionFamily.forName(it) != null)
        }
    }

    @Test
    fun `the two the old table got wrong`() {
        // Z-Image builds an LLMEmbedder, not the SD 3.x CLIP pair.
        assertEquals(setOf("llm"), DiffusionFamily.forName("Z-Image")!!.encoders)
        // Chroma Radiance decodes to pixels, so there is no VAE to ask for.
        assertEquals(false, DiffusionFamily.forName("Chroma Radiance")!!.vaeSeparate)
        assertEquals(setOf("t5xxl"), DiffusionFamily.forName("Chroma Radiance")!!.encoders)
    }

    @Test
    fun `an SDXL pix2pix is SDXL, not the SD 1 x one`() {
        assertEquals(
            DiffusionFamily.forName("SDXL"),
            DiffusionFamily.forName("SDXL Instruct-Pix2Pix"),
        )
        assertEquals(
            DiffusionFamily.forName("SD 1.x"),
            DiffusionFamily.forName("Instruct-Pix2Pix"),
        )
    }

    private companion object {
        /** Copied from `model_version_to_str` in stable-diffusion.cpp. */
        val SD_CPP_VERSIONS = listOf(
            "SD 1.x", "SD 1.x Inpaint", "Instruct-Pix2Pix", "SD 1.x Tiny UNet",
            "SD 2.x", "SD 2.x Inpaint", "SD 2.x Tiny UNet",
            "SDXS (512-DS)", "SDXS (09)",
            "SDXL", "SDXL Inpaint", "SDXL Instruct-Pix2Pix", "SDXL (Vega)", "SDXL (SSD1B)",
            "SVD", "SD3.x",
            "Flux", "Flux Fill", "Flux Control", "Flex.2", "Chroma Radiance",
            "Wan 2.x", "Wan 2.2 I2V", "Wan 2.2 TI2V", "LingBot Video",
            "Qwen Image", "Qwen Image Layered", "Hunyuan Video", "Anima",
            "Flux.2", "Flux.2 klein", "LTXAV", "HiDream O1", "Z-Image",
            "Boogu Image", "Ovis Image", "Ernie Image", "Lens", "MiniT2I",
            "Longcat-Image", "PiD", "Ideogram 4", "SeFi-Image", "Krea2",
            "Mage Flow", "ESRGAN",
        )
    }
}
