package ai.ondevice.core

/**
 * Auxiliary models, as *data with a role*.
 *
 * A diffusion run is a base model plus an ordered list of attachments. That is
 * the whole model, and it is deliberately the same shape whether the base is
 * SD 1.5, SDXL, SD3, Flux or something released next month: the app pairs a
 * role with a file path and hands both to the runtime. It does not know which
 * ControlNets are compatible with which architectures, and it must not pretend
 * to — that knowledge lives in stable-diffusion.cpp, which is the thing that
 * loads the tensors and is therefore the thing entitled to refuse them.
 *
 * SPEC §1.5 applied one level up from parameters: adding support for a new
 * kind of auxiliary is a manifest key and a row here, never a branch on the
 * base model's family.
 */
enum class AttachmentRole(
    val label: String,
    /** The manifest key its path is passed under (SPEC §16.7). */
    val paramKey: String,
    /** Whether more than one can be attached to a single run. */
    val multiple: Boolean = false,
    /** Whether the runtime takes a per-attachment weight. */
    val weighted: Boolean = false,
) {
    /**
     * LoRAs are the only auxiliary that is genuinely a *list* with weights, and
     * sd.cpp takes them as one — `sd_lora_t[]` — rather than as a directory
     * scan, so the app can offer several at once with individual strengths.
     */
    LORA("LoRA", "loras", multiple = true, weighted = true),

    /** Structural guidance from a pose, depth or edge map. */
    CONTROLNET("ControlNet", "control_net"),

    /** Style transfer from a reference image rather than from text. */
    IP_ADAPTER("IP-Adapter", "ip_adapter"),

    /**
     * The image encoder an IP-Adapter reads its reference picture through.
     *
     * Not optional, and not part of the adapter file. sd.cpp builds a
     * `FrozenCLIPVisionEmbedder` as soon as `ip_adapter_path` is set, populating
     * it from the same tensor map every other module loads from — so with no
     * `clip_vision_path` supplied there is nothing in that map for it to bind,
     * because SD 1.5's own CLIP is a *text* encoder. The adapter had no way to
     * work without this, which is why it needed its own role rather than being
     * treated as a detail of the adapter.
     */
    CLIP_VISION("CLIP vision encoder", "clip_vision"),

    /** A replacement decoder — usually to fix washed-out colour. */
    VAE("VAE", "vae"),

    /** The tiny decoder that makes a live preview cheap enough to show. */
    TAESD("TAESD preview decoder", "taesd"),

    /** Flux and SD3 carry their text encoders separately. */
    CLIP_L("CLIP-L", "clip_l"),
    CLIP_G("CLIP-G", "clip_g"),
    T5XXL("T5-XXL", "t5xxl"),

    /** Textual-inversion embeddings, loaded from a directory. */
    EMBEDDING("Embedding", "embd_dir", multiple = true),

    UPSCALER("Upscaler", "upscale_model"),
    ;

    val isDiffusionAuxiliary: Boolean get() = true

    companion object {
        /**
         * Classify a repo or file from its *metadata*, never from a curated list
         * of known model names.
         *
         * The signals used are the ones the ecosystem actually publishes: the
         * HF `pipeline_tag`, the repo's declared tags, and the filename
         * conventions the training tools themselves emit. A repo that adopts a
         * new convention is unrecognised rather than mis-filed, and the user
         * can still set the role by hand — which is the honest failure mode.
         */
        fun classify(filename: String, tags: List<String> = emptyList()): AttachmentRole? {
            val name = filename.substringAfterLast('/').lowercase()
            val path = filename.lowercase()
            val tagSet = tags.map { it.lowercase() }.toSet()

            return when {
                // Read from the *directory*, because the file is called
                // `model.safetensors` and says nothing about itself. That is a
                // diffusers layout convention rather than a curated name — every
                // repo publishing an image encoder puts it under
                // `image_encoder/` — and without it h94/IP-Adapter's encoder was
                // skipped entirely, leaving the adapter unusable.
                path.contains("image_encoder") -> CLIP_VISION
                // ControlNet is tested *before* LoRA on purpose. The most widely
                // used ControlNet pack ships its weights as
                // `control_lora_rank128_v11p_sd15_canny_fp16.safetensors` — a
                // ControlNet distilled into LoRA form — and matching "lora"
                // first files the whole pack under the wrong role, so sd.cpp is
                // handed it as `loras` and the structural guidance silently
                // does nothing.
                "controlnet" in tagSet || name.contains("control") && !name.contains("uncond") -> CONTROLNET
                "lora" in tagSet || name.contains("lora") -> LORA
                name.contains("ip-adapter") || name.contains("ip_adapter") -> IP_ADAPTER
                name.startsWith("taesd") || name.contains("taesd") -> TAESD
                name.contains("vae") -> VAE
                name.contains("clip_l") || name.contains("clip-l") -> CLIP_L
                name.contains("clip_g") || name.contains("clip-g") -> CLIP_G
                name.contains("t5xxl") || name.contains("t5-xxl") -> T5XXL
                name.contains("esrgan") || name.contains("upscal") -> UPSCALER
                "textual_inversion" in tagSet -> EMBEDDING
                else -> null
            }
        }
    }
}

/**
 * One attachment as selected for a run.
 *
 * [weight] is only meaningful for roles that declare themselves weighted; the
 * others ignore it rather than silently applying it.
 */
data class ModelAttachment(
    val modelId: String,
    val role: AttachmentRole,
    val path: String,
    val displayName: String,
    val weight: Float = 1.0f,
    val enabled: Boolean = true,
)
