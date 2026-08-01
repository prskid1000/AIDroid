package ai.ondevice.core

/** Auxiliary models, as *data with a role*. */
enum class AttachmentRole(
    val label: String,
    /** The manifest key its path is passed under (SPEC §16.7). */
    val paramKey: String,
    /** Whether more than one can be attached to a single run. */
    val multiple: Boolean = false,
    /** Whether the runtime takes a per-attachment weight. */
    val weighted: Boolean = false,
    /** A role this one cannot work without, and why in one clause. */
    val requires: Requirement? = null,
) {
    LORA("LoRA", "loras", multiple = true, weighted = true),

    /** Structural guidance from a pose, depth or edge map. */
    CONTROLNET("ControlNet", "control_net"),

    /** Style transfer from a reference image rather than from text. */
    IP_ADAPTER(
        "IP-Adapter", "ip_adapter",
        requires = Requirement(
            roleName = "CLIP_VISION",
            because = "sd.cpp builds the vision embedder the moment an IP-Adapter path is set, " +
                "and with no encoder supplied there is nothing for it to bind to",
        ),
    ),

    /** The image encoder an IP-Adapter reads its reference picture through. */
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

    /** Named by string, not by [AttachmentRole], because an enum entry cannot name a sibling in its own constructor call. */
    data class Requirement(val roleName: String, val because: String)

    /** The role this one cannot work without, or null. */
    val required: AttachmentRole? get() = requires?.let { valueOf(it.roleName) }

    companion object {
        /** Classify a repo or file from its *metadata*, never from a curated list of known model names. */
        fun classify(filename: String, tags: List<String> = emptyList()): AttachmentRole? {
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

/** One attachment as selected for a run. */
data class ModelAttachment(
    val modelId: String,
    val role: AttachmentRole,
    val path: String,
    val displayName: String,
    val weight: Float = 1.0f,
    val enabled: Boolean = true,
)
