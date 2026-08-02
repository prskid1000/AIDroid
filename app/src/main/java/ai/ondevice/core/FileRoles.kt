package ai.ondevice.core

/**
 * What a file *is*, decided once.
 *
 * There were two of these. `AttachmentRole.classify` read the basename and a
 * couple of folders; `ModelResolver.companionRole` read the whole path with
 * `contains` and knew a different, overlapping set of roles. They disagreed
 * often enough to be worth listing, because every entry below is a bug someone
 * met:
 *
 *  - `sd-vae-ft-mse/unet/model.safetensors` — the resolver said VAE, because
 *    the repo's *name* contains "vae". The folder plainly says UNet.
 *  - `flux2_klein_4b_refcontrol_depth.safetensors` — both said ControlNet,
 *    because the name contains "control". It is a LoRA, and its repo says so.
 *  - `text_encoder/model.safetensors` — neither said anything, so the encoders
 *    of every diffusers-layout repo were invisible.
 *  - `image_encoder/model.safetensors` — only one of the two recognised it, so
 *    an IP-Adapter's encoder was never auto-paired.
 *  - LoRAs, IP-Adapters, CLIP-Vision and embeddings had no rule at all on the
 *    resolver side, so none of them could ever be found as a companion.
 *
 * The evidence is graded, and the grade is the whole design:
 *
 *  1. **The folder**, when it names a component. The diffusers layout calls
 *     every file `diffusion_pytorch_model.safetensors` and puts the answer in
 *     the directory, so where a directory speaks it is the most reliable thing
 *     there is — a repo's own name is not a claim about a file inside it.
 *  2. **A strong token in the filename** — `clip_l`, `t5xxl`, `mmproj`,
 *     `ip-adapter`, `esrgan`. These name one thing and nothing else.
 *  3. **A weak token**, of which `control` is the only one: it appears in
 *     `control_v11p_canny` (a ControlNet), in `control-lora-openpose` (a
 *     ControlNet shipped as a LoRA) and in `refcontrol_depth` (a LoRA). Where
 *     it is all we have, the repo's own tags break the tie.
 *
 * Nothing here is a list of known model names. Every rule is about how the
 * file is *labelled*, which is a property of the file rather than of a curated
 * table that goes stale.
 */
object FileRoles {

    /**
     * The containers weights come in.
     *
     * A role describes a file the runtime will load, and repos ship notes
     * beside them — `config.json`, a README, a ComfyUI workflow. Matching on
     * the name alone read `SD3.5L_plus_SD3.5M_upscaling_example_workflow.json`
     * as an upscaler, and offered a 21 KB diagram as a model.
     */
    private val WEIGHT_CONTAINERS = setOf(
        "safetensors", "gguf", "ckpt", "pth", "pt", "bin", "onnx", "npz", "npy",
    )

    /**
     * A folder that names the component inside it.
     *
     * `vae_decoder` and `vae_encoder` are one component split in two, so a
     * prefix match is right; `vaeless` is not, so the match is on a whole
     * segment or a segment followed by `_`.
     */
    private val BY_FOLDER: List<Pair<String, AttachmentRole?>> = listOf(
        "image_encoder" to AttachmentRole.CLIP_VISION,
        "vae" to AttachmentRole.VAE,
        // SD 3.x numbers its three: CLIP-L, CLIP-G, T5. FLUX.1 uses the same
        // two slots for CLIP-L and T5, which is why `t5xxl` in the filename
        // beats the folder below — and it usually is in the filename.
        "text_encoder_3" to AttachmentRole.T5XXL,
        "text_encoder_2" to AttachmentRole.CLIP_G,
        "text_encoder" to AttachmentRole.CLIP_L,
        "controlnet" to AttachmentRole.CONTROLNET,
        "control_net" to AttachmentRole.CONTROLNET,
        "lora" to AttachmentRole.LORA,
        "loras" to AttachmentRole.LORA,
        "embeddings" to AttachmentRole.EMBEDDING,
        // Kokoro's speaker vectors, which live in `voices/` and are named for
        // the speaker — `af_alloy.bin`, `bm_george.bin`. The rule for these was
        // in BY_NAME, matched against the basename, and the word "voices"
        // appears in no basename: every pack classified as nothing, so the
        // resolver never offered them and the downloader never fetched them.
        // Kokoro then installed as a graph with no voices, which
        // `looksInstalled` correctly reports as not installed — the model was
        // on the device and unusable, and the screen said "not installed" about
        // a folder plainly containing it.
        "voices" to AttachmentRole.VOICES,
        // The denoiser is the model itself, not something attached to it, and
        // saying so stops the folder rules above matching a repo *named* for a
        // component that also ships the model.
        "unet" to null,
        "transformer" to null,
        "diffusion_model" to null,
    )

    /** Tokens that name one component and nothing else. */
    private val BY_NAME: List<Pair<List<String>, AttachmentRole>> = listOf(
        // The companion denoisers, first because their files are otherwise
        // named exactly like the checkpoint they sit beside —
        // `ideogram4_uncond-Q4_0.gguf` differs from `ideogram4-Q4_0.gguf` by one
        // token, and the two are the same size.
        listOf("uncond") to AttachmentRole.UNCOND_DIFFUSION,
        listOf("high_noise", "high-noise", "highnoise") to AttachmentRole.HIGH_NOISE_DIFFUSION,
        // AnimateDiff names its modules `mm_sd_v15_v2` and `v3_sd15_mm`, so the
        // token sits at either end. A bare `mm` anywhere would catch far too
        // much, which is why these are anchored rather than contained.
        listOf("motion_module", "motion-module", "mm_sd", "_mm.", "animatediff") to
            AttachmentRole.MOTION_MODULE,
        listOf("mmproj") to AttachmentRole.VISION_PROJECTOR,
        listOf("photomaker", "photo_maker", "photo-maker") to AttachmentRole.PHOTO_MAKER,
        listOf("pulid") to AttachmentRole.PULID,
        listOf("ip-adapter", "ip_adapter", "ipadapter") to AttachmentRole.IP_ADAPTER,
        // Before CLIP-L, because `clip_vision_l` names both and is the vision
        // tower. The `image_encoder` folder catches the diffusers layout; these
        // catch the flat one, where h94's own file is `clip_vision_h`.
        listOf("clip_vision", "clip-vision", "clip_vit_h", "clip_vit_bigg_vision") to
            AttachmentRole.CLIP_VISION,
        listOf("clip_l", "clip-l", "clip_vit_l") to AttachmentRole.CLIP_L,
        listOf("clip_g", "clip-g", "clip_vit_bigg") to AttachmentRole.CLIP_G,
        listOf("t5xxl", "t5-xxl", "t5_xxl") to AttachmentRole.T5XXL,
        listOf("esrgan", "upscal") to AttachmentRole.UPSCALER,
        listOf("silero", "_vad", "vad_") to AttachmentRole.VAD,
        listOf("voices") to AttachmentRole.VOICES,
        // Before the bare `vae`, which would otherwise take it: LTX-AV's audio
        // decoder is a second VAE, and handing it to the video one decodes
        // latents it has never seen.
        listOf("audio_vae", "audio-vae", "vae_audio") to AttachmentRole.AUDIO_VAE,
        listOf("vae") to AttachmentRole.VAE,
    )

    /**
     * A folder that narrows the answer without giving it.
     *
     * `text_encoders/` — plural, and the layout every Comfy-Org repackage uses
     * — says only that what is inside conditions the prompt. Which encoder it
     * is comes from the filename, and the strong rules above already answer for
     * CLIP-L, CLIP-G and T5. Whatever is left in that folder is the language
     * model: `qwen3vl_4b_fp8_scaled`, `ovis_2.5`, `umt5_xxl`.
     *
     * It is consulted after the filename rather than with the other folders
     * because it is the weaker signal of the two — and because the singular
     * `text_encoder/` above means CLIP-L specifically, in the diffusers layout,
     * which is a different claim.
     */
    private const val GENERIC_ENCODER_FOLDER = "text_encoders"

    /**
     * @param path the repo-relative path, folders and all.
     * @param tags the repo's own tags, which settle the ambiguous cases.
     */
    fun of(path: String, tags: List<String> = emptyList()): AttachmentRole? {
        if (path.substringAfterLast('.', "").lowercase() !in WEIGHT_CONTAINERS) return null

        val lower = path.lowercase()
        val name = lower.substringAfterLast('/')
        val folders = lower.split('/', '\\').dropLast(1)
        val tagSet = tags.map { it.lowercase() }.toSet()

        // 1. The folder, where one names a component. `null` is an answer too:
        //    it means "this is the model", and stops a weaker rule guessing.
        BY_FOLDER.firstOrNull { (segment, _) ->
            folders.any { it == segment || it.startsWith(segment + "_") }
        }?.let { (_, role) -> return role }

        // 2. A strong token in the filename.
        //    `controlnet` and `control-lora` are strong; a bare `control` is not.
        if (name.contains("controlnet") ||
            name.contains("control-lora") ||
            name.contains("control_lora")
        ) {
            return AttachmentRole.CONTROLNET
        }
        if (name.contains("lora") || name.contains("lycoris") || name.contains("locon")) {
            return AttachmentRole.LORA
        }
        // 2a. A vision tower inside `text_encoders/` belongs to the diffusion
        //     model's language encoder, not to a chat model. It is the same
        //     kind of file as llama's `mmproj` — which is exactly the problem,
        //     and the folder is the only thing that tells them apart.
        if (folders.any { it == GENERIC_ENCODER_FOLDER } &&
            (name.contains("mmproj") || name.contains("vision"))
        ) {
            return AttachmentRole.LLM_VISION
        }
        if (name.contains("llm_vision") || name.contains("llm-vision")) {
            return AttachmentRole.LLM_VISION
        }

        BY_NAME.firstOrNull { (tokens, _) -> tokens.any { name.contains(it) } }
            ?.let { (_, role) -> return role }

        // 2b. Inside `text_encoders/`, anything the tokens above did not name is
        //     the language model. Krea 2's `qwen3vl_4b_fp8_scaled` and Ovis'
        //     `ovis_2.5` are both this, and both classified as nothing at all —
        //     so the encoder each of them cannot run without was invisible to
        //     the resolver and never offered as a companion.
        if (folders.any { it == GENERIC_ENCODER_FOLDER }) {
            return AttachmentRole.LLM_ENCODER
        }

        // 3. The weak token, and the repo's own word on it. `refcontrol_depth`
        //    is a LoRA in a repo tagged `lora`; `control_v11p_canny` is a
        //    ControlNet in a repo tagged `controlnet`; with neither tag, the
        //    word means what it usually means.
        if (name.contains("control") && !name.contains("uncond")) {
            return when {
                "lora" in tagSet && "controlnet" !in tagSet -> AttachmentRole.LORA
                else -> AttachmentRole.CONTROLNET
            }
        }

        // 4. Nothing in the file says what it is, so the repo's declaration is
        //    all there is. Only for the roles a whole repo is plausibly *for*.
        return when {
            "controlnet" in tagSet -> AttachmentRole.CONTROLNET
            "lora" in tagSet -> AttachmentRole.LORA
            "textual_inversion" in tagSet -> AttachmentRole.EMBEDDING
            else -> null
        }
    }
}
