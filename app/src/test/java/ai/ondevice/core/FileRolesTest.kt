package ai.ondevice.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * One case per naming convention the wild actually uses, and one per
 * disagreement the two old classifiers had with each other.
 */
class FileRolesTest {

    private fun role(path: String, vararg tags: String) = FileRoles.of(path, tags.toList())

    // — the diffusers layout, where the folder is the only thing that speaks —

    @Test
    fun `a diffusers component is named by its directory`() {
        val file = "diffusion_pytorch_model.safetensors"
        assertEquals(AttachmentRole.VAE, role("vae/$file"))
        assertEquals(AttachmentRole.CONTROLNET, role("controlnet/$file"))
        assertEquals(AttachmentRole.CLIP_L, role("text_encoder/model.safetensors"))
        assertEquals(AttachmentRole.CLIP_G, role("text_encoder_2/model.safetensors"))
        assertEquals(AttachmentRole.T5XXL, role("text_encoder_3/model.safetensors"))
        assertEquals(AttachmentRole.CLIP_VISION, role("models/image_encoder/model.safetensors"))
    }

    @Test
    fun `the denoiser is the model, not something attached to it`() {
        assertNull(role("unet/diffusion_pytorch_model.safetensors"))
        assertNull(role("transformer/diffusion_pytorch_model.safetensors"))
    }

    @Test
    fun `a repo's own name is not a claim about the file inside it`() {
        // The whole reason the folder is read as segments: this file is a UNet
        // in a repo that happens to be called sd-vae-ft-mse.
        assertNull(role("sd-vae-ft-mse/unet/model.safetensors"))
        // …and the split halves of one VAE are still a VAE.
        assertEquals(AttachmentRole.VAE, role("vae_decoder/model.safetensors"))
        assertEquals(AttachmentRole.VAE, role("vae_encoder/model.safetensors"))
    }

    // — filenames —

    @Test
    fun `the comfy split-files layout names its encoders in the filename`() {
        assertEquals(AttachmentRole.CLIP_L, role("split_files/text_encoders/clip_l.safetensors"))
        assertEquals(AttachmentRole.CLIP_G, role("split_files/text_encoders/clip_g.safetensors"))
        assertEquals(
            AttachmentRole.T5XXL,
            role("split_files/text_encoders/t5xxl_fp8_e4m3fn.safetensors"),
        )
        assertEquals(AttachmentRole.VAE, role("split_files/vae/flux2-vae.safetensors"))
    }

    @Test
    fun `the roles only the resolver used to know`() {
        assertEquals(AttachmentRole.VISION_PROJECTOR, role("mmproj-model-f16.gguf"))
        assertEquals(AttachmentRole.VOICES, role("voices-v1.0.bin"))
        assertEquals(AttachmentRole.VAD, role("silero_vad.onnx"))
    }

    @Test
    fun `the roles only the library used to know`() {
        assertEquals(AttachmentRole.IP_ADAPTER, role("ip-adapter_sdxl_vit-h.safetensors"))
        assertEquals(AttachmentRole.LORA, role("pytorch_lora_weights.safetensors"))
        assertEquals(AttachmentRole.UPSCALER, role("RealESRGAN_x4.pth"))
    }

    // — the word "control", which means three different things —

    @Test
    fun `control means ControlNet where nothing says otherwise`() {
        assertEquals(
            AttachmentRole.CONTROLNET,
            role("control_v11p_sd15_canny_fp16.safetensors"),
        )
        assertEquals(AttachmentRole.CONTROLNET, role("controlnet-union-sdxl-1.0.safetensors"))
    }

    @Test
    fun `a ControlNet shipped as a LoRA is still a ControlNet`() {
        assertEquals(
            AttachmentRole.CONTROLNET,
            role("control-lora-openposeXL2-rank256.safetensors"),
        )
    }

    @Test
    fun `a LoRA that steers from a reference is a LoRA, and its repo says so`() {
        // Both old classifiers called this a ControlNet, because the name
        // contains "control". The repo is tagged `lora` and is not tagged
        // `controlnet`, which is the only evidence there is.
        assertEquals(
            AttachmentRole.LORA,
            role("flux2_klein_4b_refcontrol_depth.safetensors", "lora", "diffusers"),
        )
        assertEquals(
            AttachmentRole.CONTROLNET,
            role("some_control_model.safetensors", "controlnet"),
        )
    }

    @Test
    fun `an unnamed file falls back to what the repo says it is`() {
        assertEquals(AttachmentRole.LORA, role("checkpoint-1000/model.safetensors", "lora"))
        assertEquals(AttachmentRole.EMBEDDING, role("learned_embeds.bin", "textual_inversion"))
        assertNull(role("model.safetensors"))
    }

    // — the extension guard —

    @Test
    fun `notes shipped beside the weights are not weights`() {
        assertNull(role("SD3.5L_plus_SD3.5M_upscaling_example_workflow.json"))
        assertNull(role("vae/config.json"))
        assertNull(role("README.md"))
        assertNull(role("lora_readme.txt", "lora"))
    }

    @Test
    fun `every container a runtime here can load is accepted`() {
        listOf("safetensors", "gguf", "ckpt", "pth", "pt", "bin", "onnx")
            .forEach { assertEquals(it, AttachmentRole.VAE, role("sdxl.vae.$it")) }
    }
}
