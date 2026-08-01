package ai.ondevice.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * A role belongs to weights, and the diffusers layout hides it in the directory.
 *
 * Both mistakes here were found on one repo. `stabilityai/stable-diffusion-3.5-medium`
 * keeps its VAE at `vae/diffusion_pytorch_model.safetensors` — the filename says
 * nothing — and ships a ComfyUI example workflow whose name contains "upscal".
 */
class RoleFromPathTest {

    @Test
    fun `a diffusers VAE is recognised from its directory`() {
        assertEquals(
            AttachmentRole.VAE,
            AttachmentRole.classify("vae/diffusion_pytorch_model.safetensors"),
        )
    }

    @Test
    fun `so are the encoders that use the same layout`() {
        assertEquals(
            AttachmentRole.CLIP_L,
            AttachmentRole.classify("text_encoder/clip_l/model.safetensors"),
        )
        assertEquals(
            AttachmentRole.T5XXL,
            AttachmentRole.classify("split_files/text_encoders/t5xxl_fp16.safetensors"),
        )
    }

    @Test
    fun `an example workflow is not an upscaler`() {
        assertNull(
            AttachmentRole.classify("SD3.5L_plus_SD3.5M_upscaling_example_workflow.json"),
        )
    }

    @Test
    fun `nor is the config sitting beside the weights a VAE`() {
        assertNull(AttachmentRole.classify("vae/config.json"))
    }

    @Test
    fun `notes shipped alongside a model claim no role at all`() {
        listOf(
            "README.md",
            "model_index.json",
            "vae/config.json",
            "controlnet/config.yaml",
            "lora_readme.txt",
        ).forEach { assertNull(it, AttachmentRole.classify(it)) }
    }

    @Test
    fun `real weights still classify, whatever they are packed in`() {
        assertEquals(AttachmentRole.VAE, AttachmentRole.classify("sdxl.vae.safetensors"))
        assertEquals(AttachmentRole.UPSCALER, AttachmentRole.classify("RealESRGAN_x4.pth"))
        assertEquals(
            AttachmentRole.IP_ADAPTER,
            AttachmentRole.classify("models/ip-adapter_sd15.safetensors"),
        )
        assertEquals(
            AttachmentRole.CLIP_VISION,
            AttachmentRole.classify("sdxl_models/image_encoder/model.safetensors"),
        )
    }
}
