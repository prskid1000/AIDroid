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

    // The three the resolver used to know about and this enum did not. They
    // belong to the text and voice runtimes rather than to diffusion, which is
    // why they were in the other enum — and being in the other enum is exactly
    // what made a file mean two things at once.

    /** The projector that lets a text model be shown a picture. */
    VISION_PROJECTOR("Vision projector", "mmproj", RoleFamily.PROMPT_ENCODER),

    /** Kokoro's speaker vectors, all of which are one component. */
    VOICES("Voice style vectors", "voices", RoleFamily.STYLE_AND_CONTROL, multiple = true),

    /** Silero's voice-activity detector, which whisper takes as a file. */
    VAD("Silero VAD", "vad_model", RoleFamily.STYLE_AND_CONTROL),
    ;

    val isDiffusionAuxiliary: Boolean get() = true

    /** Named by string, not by [AttachmentRole], because an enum entry cannot name a sibling in its own constructor call. */
    data class Requirement(val roleName: String, val because: String)

    /** The role this one cannot work without, or null. */
    val required: AttachmentRole? get() = requires?.let { valueOf(it.roleName) }

    /**
     * How many files of this role the thing consuming them can take.
     *
     * Kokoro's voice packs are the odd one: every file is part of one
     * component, so "all of them" is the answer rather than "pick one".
     */
    val cardinality: Cardinality
        get() = if (this == VOICES) Cardinality.ALL else Cardinality.ONE

    companion object {
        /**
         * What this file is, from how it is labelled — never from a curated
         * list of known model names.
         *
         * The rules live in [FileRoles], which is the only place that answers
         * this question. It used to be answered here *and* in the resolver, by
         * two rule sets that disagreed.
         */
        fun classify(filename: String, tags: List<String> = emptyList()): AttachmentRole? =
            FileRoles.of(filename, tags)
    }
}

/** How many files of one role the thing that consumes them can actually take. */
enum class Cardinality {
    /** One path, and only one. */
    ONE,

    /** Every file is part of one thing. */
    ALL,
}

/** One attachment as selected for a run. */
data class ModelAttachment(
    val modelId: String,
    val role: AttachmentRole,
    val path: String,
    val displayName: String,
    val weight: Float = 1.0f,
    val enabled: Boolean = true,
    /**
     * The strength stored with the choice, as against [weight], which is what
     * this run is using.
     *
     * The two are separate so that a dial moved on the Image sheet survives a
     * refresh — it is a per-run thought — while a strength changed on the All
     * Parameters screen still lands, because the stored value it was compared
     * against has changed too.
     */
    val chosenWeight: Float = weight,
    /**
     * Whether the loaded model has any use for this role.
     *
     * False keeps the row visible and unusable rather than removing it, and the
     * distinction matters more than it looks: the loader falls back to the
     * stored path for any role the caller does not mention, so a role dropped
     * from this list is a role the loader quietly loads anyway. An LLM encoder
     * left over from FLUX was costing SDXL two and a half gigabytes that way,
     * with nothing on screen to say so.
     */
    val applicable: Boolean = true,
)
