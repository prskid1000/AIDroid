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
    /**
     * A second set of denoiser weights, beside the checkpoint's own.
     *
     * Its own family because it is neither an encoder nor a decoder nor a
     * style: these files are the model, published in more than one piece, and
     * the run is wrong rather than merely plainer without them. That also keeps
     * them out of `ADOPTABLE_FAMILIES` — a denoiser is never plumbing to be
     * filled in on the user's behalf.
     */
    COMPANION_DENOISER("Companion denoiser"),
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

    /**
     * The denoiser a model runs for its unconditional pass.
     *
     * Ideogram 4 ships two files of equal size and uses the second wherever
     * classifier-free guidance needs the un-prompted branch. Upstream takes it
     * on `--uncond-diffusion-model`; without it the architecture cannot be run
     * at all, which is a different thing from running plainly.
     */
    UNCOND_DIFFUSION("Unconditional denoiser", "uncond_diffusion_model", RoleFamily.COMPANION_DENOISER),

    /**
     * The denoiser used for the noisy end of the schedule.
     *
     * Wan 2.2's I2V and TI2V split the model in two by timestep — high-noise
     * steps run one set of weights and low-noise steps the other. Both are
     * required, and each is the full size of a denoiser.
     */
    HIGH_NOISE_DIFFUSION("High-noise denoiser", "high_noise_diffusion_model", RoleFamily.COMPANION_DENOISER),

    /**
     * AnimateDiff's temporal layers, which turn a still model into a video one.
     *
     * Not a LoRA: it is a separate module sd.cpp inserts between an SD 1.5
     * UNet's existing blocks, and it is what makes frames relate to each other
     * rather than being N unrelated pictures of the same prompt.
     */
    MOTION_MODULE("Motion module", "motion_module", RoleFamily.COMPANION_DENOISER),

    /** The decoder for LTX-AV's audio track, which is a separate latent space from its video. */
    AUDIO_VAE("Audio decoder", "audio_vae", RoleFamily.DECODER),

    /**
     * The vision tower of an LLM text encoder.
     *
     * Distinct from CLIP_VISION, which an IP-Adapter reads through: this one
     * belongs to the language model, and is what lets an edit model be shown
     * the picture it is being asked about.
     */
    LLM_VISION("LLM vision tower", "llm_vision", RoleFamily.PROMPT_ENCODER),

    /** Keep one person's face across generations, from a few photographs. */
    PHOTO_MAKER("PhotoMaker", "photo_maker", RoleFamily.STYLE_AND_CONTROL),

    /** The same job as PhotoMaker by a different method, and the two do not combine. */
    PULID("PuLID", "pulid", RoleFamily.STYLE_AND_CONTROL),

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

    /**
     * Whether stable-diffusion.cpp has anywhere to put this.
     *
     * This returned `true` for everything, which made it a property that
     * answered no question. Three roles here belong to the other runtimes —
     * llama's `mmproj`, whisper's Silero VAD, Kokoro's voice packs — and the
     * diffusion parameter screen declared all three as keys it accepts. The
     * manifest has no diffusion row for any of them, so they rendered under
     * "not described yet" as empty text boxes on the image screen: three
     * settings offered to a runtime that has no field for them, which is the
     * same silent no-op this codebase keeps finding, dressed as a feature.
     *
     * They stay in this enum because a file has one role whichever runtime
     * loads it — that was the point of merging the two role enums. What they
     * are not is something the diffusion loader can be told about.
     */
    val isDiffusionAuxiliary: Boolean
        get() = this !in setOf(VISION_PROJECTOR, VOICES, VAD)

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
