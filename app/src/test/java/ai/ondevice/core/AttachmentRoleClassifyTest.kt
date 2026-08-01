package ai.ondevice.core

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The real filenames of the repos listed twice or three times on Add model.
 *
 * These are the cards that looked identical: one repo, several roles, and the
 * screen showing every file under each. Narrowing a card to its role is only
 * as good as this, so the repos that made the problem visible are the ones
 * pinned here.
 */
class AttachmentRoleClassifyTest {

    /** h94/IP-Adapter — the adapters and the two encoders they read through. */
    private val ipAdapterRepo = listOf(
        "models/ip-adapter_sd15.safetensors",
        "models/ip-adapter_sd15_light.safetensors",
        "models/ip-adapter_sd15_vit-G.safetensors",
        "models/ip-adapter-plus_sd15.safetensors",
        "models/ip-adapter-plus-face_sd15.safetensors",
        "models/ip-adapter-full-face_sd15.safetensors",
        "models/image_encoder/model.safetensors",
        "sdxl_models/ip-adapter_sdxl.safetensors",
        "sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
        "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
        "sdxl_models/image_encoder/model.safetensors",
    )

    /** Comfy-Org/stable-diffusion-3.5-fp8 — three encoders in one repo. */
    private val sd35Encoders = listOf(
        "split_files/text_encoders/clip_l.safetensors",
        "split_files/text_encoders/clip_g.safetensors",
        "split_files/text_encoders/t5xxl_fp8_e4m3fn.safetensors",
    )

    private fun forRole(files: List<String>, role: AttachmentRole) =
        files.filter { AttachmentRole.classify(it) == role }

    @Test
    fun `the IP-Adapter card and the CLIP-Vision card do not offer the same files`() {
        val adapters = forRole(ipAdapterRepo, AttachmentRole.IP_ADAPTER)
        val encoders = forRole(ipAdapterRepo, AttachmentRole.CLIP_VISION)

        assertEquals(10, adapters.size)
        assertEquals(2, encoders.size)
        assertEquals(emptyList<String>(), adapters.intersect(encoders.toSet()).toList())
    }

    @Test
    fun `an image encoder is read from its directory, since the file says nothing`() {
        assertEquals(
            AttachmentRole.CLIP_VISION,
            AttachmentRole.classify("sdxl_models/image_encoder/model.safetensors"),
        )
    }

    @Test
    fun `SD 3-5's three encoders land in three different slots`() {
        assertEquals(1, forRole(sd35Encoders, AttachmentRole.CLIP_L).size)
        assertEquals(1, forRole(sd35Encoders, AttachmentRole.CLIP_G).size)
        assertEquals(1, forRole(sd35Encoders, AttachmentRole.T5XXL).size)
    }

    @Test
    fun `CLIP-G is not swallowed by CLIP-L, which is a prefix problem waiting to happen`() {
        assertEquals(
            AttachmentRole.CLIP_G,
            AttachmentRole.classify("split_files/text_encoders/clip_g.safetensors"),
        )
        assertEquals(
            AttachmentRole.CLIP_L,
            AttachmentRole.classify("split_files/text_encoders/clip_l.safetensors"),
        )
    }

    @Test
    fun `every file in both repos is claimed by exactly one role`() {
        (ipAdapterRepo + sd35Encoders).forEach { file ->
            val role = AttachmentRole.classify(file)
            assertEquals("$file should classify", true, role != null)
        }
    }
}
