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
}
