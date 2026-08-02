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

    // — the bug class: a rule matched against the wrong half of the path —
    //
    // A role nothing classifies is a component the resolver never offers and
    // the downloader never fetches, and the model installs half-complete. That
    // is not a warning anywhere: the file simply is not there afterwards.

    /**
     * Kokoro's voice packs, which is how this class of bug was found.
     *
     * The rule was `listOf("voices")` matched against the *basename*, and the
     * word appears only in the folder. Every pack classified as nothing, so a
     * Kokoro install arrived as a graph with no speakers — and the screen said
     * "Kokoro is not installed" about a folder containing Kokoro.
     */
    @Test
    fun `Kokoro's speaker vectors are found by their folder`() {
        assertEquals(AttachmentRole.VOICES, role("voices/af_alloy.bin"))
        assertEquals(AttachmentRole.VOICES, role("voices/bm_george.bin"))
        assertEquals(AttachmentRole.VOICES, role("onnx/../voices/jf_alpha.bin"))
    }

    /**
     * `text_encoders/` — plural — is what every Comfy-Org repackage uses, and
     * the folder rule was written for the singular diffusers spelling. The
     * encoders Krea 2 and Ovis cannot run without were both invisible.
     */
    @Test
    fun `a language-model encoder in the plural encoder folder is found`() {
        assertEquals(
            AttachmentRole.LLM_ENCODER,
            role("split_files/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"),
        )
        assertEquals(
            AttachmentRole.LLM_ENCODER,
            role("split_files/text_encoders/ovis_2.5.safetensors"),
        )
    }

    /** The named encoders still win inside that folder — it narrows, it does not decide. */
    @Test
    fun `a named encoder beats the folder it sits in`() {
        assertEquals(AttachmentRole.CLIP_L, role("text_encoders/clip_l.safetensors"))
        assertEquals(AttachmentRole.T5XXL, role("text_encoders/t5xxl_fp16.safetensors"))
    }

    /**
     * The same file kind in two runtimes, told apart by where it sits.
     *
     * llama's projector and a diffusion encoder's vision tower are both an
     * `mmproj`, and they go to different fields of different structs.
     */
    @Test
    fun `an mmproj belongs to whichever runtime its folder names`() {
        assertEquals(AttachmentRole.VISION_PROJECTOR, role("mmproj-F16.gguf"))
        assertEquals(
            AttachmentRole.LLM_VISION,
            role("split_files/text_encoders/mmproj-qwen3vl.gguf"),
        )
    }

    @Test
    fun `the flat clip-vision spelling is found without its folder`() {
        assertEquals(AttachmentRole.CLIP_VISION, role("clip_vision_h.safetensors"))
        assertEquals(AttachmentRole.CLIP_VISION, role("clip_vision_g.safetensors"))
    }

    @Test
    fun `LTX-AV's audio decoder is not handed to the video decoder`() {
        assertEquals(AttachmentRole.AUDIO_VAE, role("ltxav_audio_vae.safetensors"))
        assertEquals(AttachmentRole.VAE, role("ltxav_vae.safetensors"))
    }

    @Test
    fun `the identity adapters are named`() {
        assertEquals(AttachmentRole.PHOTO_MAKER, role("photomaker-v1.bin"))
        assertEquals(AttachmentRole.PULID, role("pulid_flux_v0.9.1.safetensors"))
    }

    @Test
    fun `the companion denoisers are told from the checkpoint beside them`() {
        assertEquals(AttachmentRole.UNCOND_DIFFUSION, role("ideogram4_uncond-Q4_0.gguf"))
        assertNull(role("ideogram4-Q4_0.gguf"))
        assertEquals(
            AttachmentRole.HIGH_NOISE_DIFFUSION,
            role("wan2.2_i2v_high_noise_14B_Q4_0.gguf"),
        )
        assertEquals(AttachmentRole.MOTION_MODULE, role("mm_sd_v15_v2.ckpt"))
        assertEquals(AttachmentRole.MOTION_MODULE, role("v3_sd15_mm.ckpt"))
    }

    /**
     * The audit this class of bug asks for: every role a diffusion context can
     * be handed must have at least one path that reaches it. A role with no
     * rule is a slot the picker renders empty forever, because nothing on disk
     * can ever be classified into it.
     */
    @Test
    fun `every attachable role is reachable from some real filename`() {
        val reachable = listOf(
            "voices/af_alloy.bin",
            "split_files/text_encoders/qwen3vl_4b.safetensors",
            "text_encoders/clip_l.safetensors",
            "text_encoders/clip_g.safetensors",
            "text_encoders/t5xxl_fp16.safetensors",
            "clip_vision_h.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "ltxav_audio_vae.safetensors",
            "controlnet/diffusion_pytorch_model.safetensors",
            "ip-adapter_sdxl_vit-h.safetensors",
            "photomaker-v1.bin",
            "pulid_flux_v0.9.1.safetensors",
            "mm_sd_v15_v2.ckpt",
            "ideogram4_uncond-Q4_0.gguf",
            "wan2.2_high_noise_14B_Q4_0.gguf",
            "style-lora.safetensors",
            "embeddings/easynegative.safetensors",
            "split_files/text_encoders/mmproj-qwen3vl.gguf",
            "RealESRGAN_x4plus.pth",
            "mmproj-F16.gguf",
            "silero_vad.onnx",
        ).mapNotNull { FileRoles.of(it) }.toSet()

        val unreachable = AttachmentRole.entries.filterNot { it in reachable }
        assertEquals(
            "no filename in this test reaches these roles, so nothing on disk " +
                "can be classified into them: $unreachable",
            emptyList<AttachmentRole>(),
            unreachable,
        )
    }
}
