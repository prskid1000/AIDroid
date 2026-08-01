package ai.ondevice.core

import org.junit.Assert.assertEquals
import org.junit.Test

/** Labels that have to be unique, because the control hands back the label. */
class LabelsTest {

    @Test
    fun `filenames that already differ are left alone`() {
        assertEquals(
            mapOf(
                "models/clip_l.safetensors" to "clip_l.safetensors",
                "models/clip_g.safetensors" to "clip_g.safetensors",
            ),
            FileLabels.distinguish(
                listOf("models/clip_l.safetensors", "models/clip_g.safetensors"),
            ),
        )
    }

    @Test
    fun `the diffusers layout grows one folder, and only one`() {
        val labels = FileLabels.distinguish(
            listOf(
                "vae/diffusion_pytorch_model.safetensors",
                "controlnet/diffusion_pytorch_model.safetensors",
            ),
        )
        assertEquals("vae/diffusion_pytorch_model.safetensors", labels.values.first())
        assertEquals(2, labels.values.toSet().size)
    }

    @Test
    fun `it keeps growing while a collision survives`() {
        val labels = FileLabels.distinguish(
            listOf("a/enc/model.safetensors", "b/enc/model.safetensors"),
        )
        assertEquals(2, labels.values.toSet().size)
        assertEquals("a/enc/model.safetensors", labels.getValue("a/enc/model.safetensors"))
    }

    @Test
    fun `two files that are genuinely the same path stay the same`() {
        val labels = FileLabels.distinguish(listOf("vae/model.safetensors", "vae/model.safetensors"))
        assertEquals(1, labels.size)
    }

    @Test
    fun `a name that stands alone takes no qualifier`() {
        assertEquals(
            listOf("whisper.cpp", "Kokoro"),
            Labels.unique(
                listOf(
                    Labels.Item("whisper.cpp", listOf("large-v3")),
                    Labels.Item("Kokoro", listOf("int8")),
                ),
            ),
        )
    }

    @Test
    fun `namesakes take the first qualifier that separates them`() {
        assertEquals(
            listOf(
                "stable-diffusion-3.5-fp8 · CLIP-L",
                "stable-diffusion-3.5-fp8 · CLIP-G",
                "stable-diffusion-3.5-fp8 · T5-XXL",
            ),
            Labels.unique(
                listOf(
                    Labels.Item("stable-diffusion-3.5-fp8", listOf("CLIP-L", "fp8")),
                    Labels.Item("stable-diffusion-3.5-fp8", listOf("CLIP-G", "fp8")),
                    Labels.Item("stable-diffusion-3.5-fp8", listOf("T5-XXL", "fp8")),
                ),
            ),
        )
    }

    @Test
    fun `it takes a second qualifier when the first still collides`() {
        val labels = Labels.unique(
            listOf(
                Labels.Item("repo", listOf("VAE", "fp16")),
                Labels.Item("repo", listOf("VAE", "fp32")),
            ),
        )
        assertEquals(2, labels.toSet().size)
        assertEquals("repo · VAE · fp16", labels.first())
    }

    @Test
    fun `nulls and blanks are not offered as qualifiers`() {
        val labels = Labels.unique(
            listOf(
                Labels.Item("repo", listOf(null, "", "Q4")),
                Labels.Item("repo", listOf(null, "", "Q8")),
            ),
        )
        assertEquals(listOf("repo · Q4", "repo · Q8"), labels)
    }
}
