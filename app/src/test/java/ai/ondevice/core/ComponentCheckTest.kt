package ai.ondevice.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** What each diffusion family needs beside itself, and what it does not. */
class ComponentCheckTest {

    private fun attachment(role: AttachmentRole, enabled: Boolean = true) = ModelAttachment(
        modelId = role.name,
        role = role,
        path = "/models/${role.paramKey}.safetensors",
        displayName = role.label,
        weight = 1.0f,
        enabled = enabled,
    )

    private fun missingRoles(
        architecture: String,
        attached: List<AttachmentRole> = emptyList(),
        installed: Set<AttachmentRole> = emptySet(),
    ): List<String> = ComponentCheck
        .forDiffusion(attached.map { attachment(it) }, architecture, installed)
        .map { it.what }

    @Test
    fun `SDXL is told about both of its text encoders, not one`() {
        val missing = missingRoles("SDXL")
        assertTrue("$missing", missing.any { it.contains("CLIP-L") })
        assertTrue("$missing", missing.any { it.contains("CLIP-G") })
    }

    @Test
    fun `a UNet carries its own decoder, so no VAE is demanded`() {
        assertTrue(missingRoles("SD 1.x").none { it.contains("VAE") })
        assertTrue(missingRoles("SDXL").none { it.contains("VAE") })
    }

    @Test
    fun `a bare transformer is asked for the decoder it does not carry`() {
        assertTrue(missingRoles("Flux.2 klein").any { it.contains("VAE") })
        assertTrue(missingRoles("SD3_x").any { it.contains("VAE") })
    }

    @Test
    fun `FLUX 2 wants a language model and neither CLIP nor T5`() {
        val missing = missingRoles("Flux.2 klein")
        assertTrue("$missing", missing.any { it.contains("LLM") })
        assertTrue("$missing", missing.none { it.contains("T5") || it.contains("CLIP") })
    }

    @Test
    fun `FLUX 1 wants CLIP-L and T5, which is not what FLUX 2 wants`() {
        val missing = missingRoles("Flux")
        assertTrue("$missing", missing.any { it.contains("CLIP-L") })
        assertTrue("$missing", missing.any { it.contains("T5") })
        assertTrue("$missing", missing.none { it.contains("LLM") })
    }

    @Test
    fun `SD 3 does not insist on T5, which it can run without`() {
        assertTrue(missingRoles("SD3.x").none { it.contains("T5") })
    }

    @Test
    fun `supplying what a family asks for silences it`() {
        val missing = missingRoles("SDXL", attached = listOf(AttachmentRole.CLIP_L, AttachmentRole.CLIP_G))
        assertEquals(emptyList<String>(), missing)
    }

    @Test
    fun `installed but unattached is a different problem from absent`() {
        val absent = ComponentCheck.forDiffusion(emptyList(), "SDXL", emptySet())
        assertTrue(absent.all { it.state == MissingComponent.State.NOT_INSTALLED })

        val onTheShelf = ComponentCheck.forDiffusion(
            emptyList(),
            "SDXL",
            setOf(AttachmentRole.CLIP_L, AttachmentRole.CLIP_G),
        )
        assertTrue(onTheShelf.all { it.state == MissingComponent.State.INSTALLED_NOT_ATTACHED })
    }

    @Test
    fun `an unrecognised architecture is not lectured about parts nobody can name`() {
        assertEquals(emptyList<String>(), missingRoles("some model published next year"))
    }

    // Whether the parts are in the file is a property of the file, not of the
    // family. SDXL ships both ways, and the quantised one is what runs here.

    @Test
    fun `a full checkpoint is asked for nothing, whatever its family`() {
        assertEquals(
            emptyList<String>(),
            ComponentCheck.forDiffusion(emptyList(), "SDXL", bareDenoiser = false).map { it.what },
        )
        assertEquals(
            emptyList<String>(),
            ComponentCheck.forDiffusion(emptyList(), "Flux", bareDenoiser = false).map { it.what },
        )
    }

    @Test
    fun `a quantised SDXL is asked for the decoder a full one would carry`() {
        val bare = ComponentCheck.forDiffusion(emptyList(), "SDXL", bareDenoiser = true).map { it.what }
        assertTrue("$bare", bare.any { it.contains("VAE") })
        assertTrue("$bare", bare.any { it.contains("CLIP-G") })
    }

    /**
     * The UNet question, asked in the runtime's own spelling.
     *
     * These went through a private list of exact strings — `sd1`, `sd2`, `sdxl`
     * — and sd.cpp does not print any of them but the third. `SD1.x` missed,
     * and SD 1.5 is the architecture with more ControlNets written for it than
     * every other one here put together.
     */
    @Test
    fun `SD 1_x takes a ControlNet and an IP-Adapter, whatever the version suffix`() {
        listOf("SD1.x", "SD 1.x", "SD2.x", "SDXL", "SVD").forEach { architecture ->
            val complaints = ComponentCheck.forDiffusion(
                available = listOf(
                    attachment(AttachmentRole.CONTROLNET),
                    attachment(AttachmentRole.IP_ADAPTER),
                    attachment(AttachmentRole.CLIP_VISION),
                ),
                architecture = architecture,
                bareDenoiser = true,
            ).filter { it.state == MissingComponent.State.WONT_ATTACH }
            assertEquals("$architecture is a UNet", emptyList<MissingComponent>(), complaints)
        }
    }

    @Test
    fun `a transformer is still told that neither one will attach`() {
        listOf("SD3.x", "Flux", "Flux.2 klein", "Z-Image", "Chroma Radiance").forEach { architecture ->
            val complaints = ComponentCheck.forDiffusion(
                available = listOf(attachment(AttachmentRole.CONTROLNET)),
                architecture = architecture,
                bareDenoiser = true,
            ).filter { it.state == MissingComponent.State.WONT_ATTACH }
            assertTrue("$architecture has no UNet to attach to", complaints.isNotEmpty())
        }
    }
}
