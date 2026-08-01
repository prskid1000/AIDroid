package ai.ondevice.core

/**
 * What kind of job a role does, so four spellings of one job read as four
 * spellings of one job.
 *
 * CLIP-L, CLIP-G, T5-XXL and the LLM all turn a prompt into conditioning; which
 * one a model wants is a property of its architecture, not a choice between
 * unrelated files. Listed flat, as they were, they read as four separate
 * things to go and find.
 *
 * The order is the order they appear in.
 */
enum class RoleFamily(val label: String) {
    PROMPT_ENCODER("Prompt encoder"),
    DECODER("Decoder"),
    STYLE_AND_CONTROL("Style & control"),
    POST("Post-processing"),
}

/** Auxiliary models, as *data with a role*. */
enum class AttachmentRole(
    val label: String,
    /** The manifest key its path is passed under (SPEC §16.7). */
    val paramKey: String,
    val family: RoleFamily,
    /** Whether more than one can be attached to a single run. */
    val multiple: Boolean = false,
    /** Whether the runtime takes a per-attachment weight. */
    val weighted: Boolean = false,
    /** A role this one cannot work without, and why in one clause. */
    val requires: Requirement? = null,
) {
    LORA("LoRA", "loras", RoleFamily.STYLE_AND_CONTROL, multiple = true, weighted = true),

    /** Structural guidance from a pose, depth or edge map. */
    CONTROLNET("ControlNet", "control_net", RoleFamily.STYLE_AND_CONTROL),

    /** Style transfer from a reference image rather than from text. */
    IP_ADAPTER(
        "IP-Adapter", "ip_adapter", RoleFamily.STYLE_AND_CONTROL,
        requires = Requirement(
            roleName = "CLIP_VISION",
            because = "sd.cpp builds the vision embedder the moment an IP-Adapter path is set, " +
                "and with no encoder supplied there is nothing for it to bind to",
        ),
    ),

    /** The image encoder an IP-Adapter reads its reference picture through. */
    CLIP_VISION("CLIP vision encoder", "clip_vision", RoleFamily.STYLE_AND_CONTROL),

    /** A replacement decoder — usually to fix washed-out colour. */
    VAE("VAE", "vae", RoleFamily.DECODER),

    /** The tiny decoder that makes a live preview cheap enough to show. */
    TAESD("TAESD preview decoder", "taesd", RoleFamily.DECODER),

    /** Flux and SD3 carry their text encoders separately. */
    CLIP_L("CLIP-L", "clip_l", RoleFamily.PROMPT_ENCODER),
    CLIP_G("CLIP-G", "clip_g", RoleFamily.PROMPT_ENCODER),
    T5XXL("T5-XXL", "t5xxl", RoleFamily.PROMPT_ENCODER),

    /**
     * FLUX.2's text encoder, which is a language model rather than CLIP or T5
     * — Qwen3 for Klein, Mistral Small for dev. It is the size of a chat model
     * and is the reason a 4B diffusion model costs more than 4B to run.
     */
    LLM_ENCODER("LLM", "llm", RoleFamily.PROMPT_ENCODER),

    /** Textual-inversion embeddings, loaded from a directory. */
    EMBEDDING("Embedding", "embd_dir", RoleFamily.STYLE_AND_CONTROL, multiple = true),

    UPSCALER("Upscaler", "upscale_model", RoleFamily.POST),
    ;

    val isDiffusionAuxiliary: Boolean get() = true

    /** Named by string, not by [AttachmentRole], because an enum entry cannot name a sibling in its own constructor call. */
    data class Requirement(val roleName: String, val because: String)

    /** The role this one cannot work without, or null. */
    val required: AttachmentRole? get() = requires?.let { valueOf(it.roleName) }

    companion object {
        /** Classify a repo or file from its *metadata*, never from a curated list of known model names. */
        /**
         * The containers weights actually come in.
         *
         * A role describes a file the runtime will load, and the repos that
         * ship one also ship notes about it — `config.json`, a README, a
         * ComfyUI workflow. Matching on the name alone read
         * `SD3.5L_plus_SD3.5M_upscaling_example_workflow.json` as an upscaler,
         * because it does contain "upscal", and offered a 21 KB diagram as a
         * model.
         */
        private val WEIGHT_CONTAINERS = setOf(
            "safetensors", "gguf", "ckpt", "pth", "pt", "bin", "onnx", "npz", "npy",
        )

        private fun isWeightFile(filename: String) =
            filename.substringAfterLast('.', "").lowercase() in WEIGHT_CONTAINERS

        fun classify(filename: String, tags: List<String> = emptyList()): AttachmentRole? {
            if (!isWeightFile(filename)) return null
            val name = filename.substringAfterLast('/').lowercase()
            val path = filename.lowercase()
            val tagSet = tags.map { it.lowercase() }.toSet()

            return when {
                // Read from the *directory*, because the file is called `model.safetensors` and says nothing about itself.
                path.contains("image_encoder") -> CLIP_VISION
                // ControlNet is tested *before* LoRA on purpose.
                "controlnet" in tagSet || name.contains("control") && !name.contains("uncond") -> CONTROLNET
                "lora" in tagSet || name.contains("lora") -> LORA
                name.contains("ip-adapter") || name.contains("ip_adapter") -> IP_ADAPTER
                name.startsWith("taesd") || name.contains("taesd") -> TAESD
                // Read the whole path, not just the filename. The diffusers
                // layout names the role in the *directory* and calls every file
                // `diffusion_pytorch_model.safetensors`, so a VAE fetched that
                // way came back unclassified while the resolver, which reads the
                // path, had already filed it correctly. Two classifiers, two
                // answers, for the same file.
                path.contains("vae") -> VAE
                path.contains("clip_l") || path.contains("clip-l") -> CLIP_L
                path.contains("clip_g") || path.contains("clip-g") -> CLIP_G
                path.contains("t5xxl") || path.contains("t5-xxl") -> T5XXL
                name.contains("esrgan") || name.contains("upscal") -> UPSCALER
                "textual_inversion" in tagSet -> EMBEDDING
                else -> null
            }
        }
    }
}

/** One attachment as selected for a run. */
data class ModelAttachment(
    val modelId: String,
    val role: AttachmentRole,
    val path: String,
    val displayName: String,
    val weight: Float = 1.0f,
    val enabled: Boolean = true,
)
