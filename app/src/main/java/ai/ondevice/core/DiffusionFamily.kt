package ai.ondevice.core

/**
 * What a diffusion family needs supplied *beside* the checkpoint.
 *
 * This is a question about files, not about settings. Steps, CFG, the sampler
 * and the schedule used to be answered here too, and the app wrote them onto
 * the model on the user's behalf whenever it thought it had recognised
 * something. That is gone: those four are the user's, and a table guessing at
 * them from a name is a table that overrides a deliberate choice the moment it
 * guesses wrong.
 *
 * What is left cannot be guessed by the user and does not change what the
 * picture looks like — SDXL conditions on two encoders and SD 3.x on three,
 * whoever is asking. Missing one is not a preference, it is a run that cannot
 * start.
 *
 * Nothing here is inferred from a filename or a repo name. The key is the
 * version stable-diffusion.cpp works out from the tensors themselves and
 * announces on load, or the GGUF's own `general.architecture` where it has one.
 */
data class DiffusionFamily(
    /**
     * What this family reads its prompt through, as `AttachmentRole.paramKey`
     * values — the one vocabulary both role enums in this app agree on.
     *
     * A checkpoint that carries its own encoders needs none of these supplied;
     * a bare denoiser, which is what most quantised releases are, needs all of
     * them or it has no way to read the prompt at all.
     */
    val encoders: Set<String> = emptySet(),
    /** Read when present and workable without — SD 3.x's T5 is the only one. */
    val optionalEncoders: Set<String> = emptySet(),
    /**
     * Whether this family's releases normally ship the decoder separately.
     *
     * True for the diffusion transformers, whose quantised weights are the
     * denoiser alone. False for the UNet families, where a full checkpoint
     * carrying denoiser, encoders and VAE is still the usual shape — an
     * external VAE there is a fix for a known fault, not a missing part.
     */
    val vaeSeparate: Boolean = true,
    /**
     * Whether the denoiser is a UNet, which decides what can be bolted onto it.
     *
     * sd.cpp builds a ControlNet only in the SD1/SD2/SDXL/SVD shape, and an
     * IP-Adapter injects at `input_blocks.N.1` names that exist in a UNet and
     * nowhere else. On a diffusion transformer either file loads, costs its
     * memory and changes nothing.
     *
     * `vaeSeparate` is not a stand-in for this even though it agrees on four of
     * the five: Chroma Radiance keeps its decoder inside and is still a
     * transformer.
     */
    val unet: Boolean = false,
) {
    companion object {
        /** SD 1.x and SD 2.x: one CLIP, and the decoder in the file. */
        val UNET = DiffusionFamily(
            encoders = setOf(CLIP_L),
            vaeSeparate = false,
            unet = true,
        )

        /** Two text encoders, not one: SDXL conditions on ViT-L *and* ViT-bigG. */
        private val SDXL = UNET.copy(encoders = setOf(CLIP_L, CLIP_G))

        /** SVD conditions on a picture, not on words — there is no text encoder to supply. */
        private val PICTURE_CONDITIONED = UNET.copy(encoders = emptySet())

        private val SD3 = DiffusionFamily(
            // SD 3.x runs on the two CLIPs alone; T5 is what it reads long,
            // written-out prompts with, and dropping it costs prompt fidelity
            // rather than the ability to run.
            encoders = setOf(CLIP_L, CLIP_G),
            optionalEncoders = setOf(T5XXL),
        )

        private val FLUX = DiffusionFamily(encoders = setOf(CLIP_L, T5XXL))

        /**
         * Everything that reads its prompt with a language model rather than
         * with CLIP or T5 — FLUX.2 (Qwen3 for Klein, Mistral Small for dev),
         * Qwen-Image (Qwen2.5-VL), Z-Image (Qwen3), Anima, LTX-AV (Gemma 3) and
         * the rest of the recent transformers. It is why a 4B image model costs
         * far more than 4B to run.
         */
        private val LLM_CONDITIONED = DiffusionFamily(encoders = setOf(LLM))

        /** T5 alone: Chroma (a de-distilled Flux with CLIP-L dropped), MiniT2I, and Wan's UMT5. */
        private val T5_ONLY = DiffusionFamily(encoders = setOf(T5XXL))

        /**
         * Chroma Radiance decodes straight to pixels, so it has no VAE to
         * supply and asking for one is asking for a part that does not exist.
         */
        private val CHROMA_RADIANCE = T5_ONLY.copy(vaeSeparate = false)

        /**
         * Wan reads a picture through a CLIP-vision encoder as well as the
         * prompt through UMT5, and runs without one.
         */
        private val WAN = T5_ONLY.copy(optionalEncoders = setOf(CLIP_VISION))

        /** Conditioners that live entirely inside the checkpoint. */
        private val SELF_CONDITIONED = DiffusionFamily(encoders = emptySet())

        /**
         * Every version stable-diffusion.cpp can name, matched against the
         * string it prints or against a GGUF `general.architecture`.
         *
         * The list is the one in `model_version_to_str`, and each entry's
         * encoders are the ones the matching branch of `sd_ctx_t`'s conditioner
         * dispatch actually constructs — not a reading of what the family is
         * known for elsewhere. Two of those disagreed: Z-Image was filed under
         * SD 3.x and builds an `LLMEmbedder`, and Chroma Radiance was asked for
         * a VAE it has no use for.
         *
         * Order is longest-match-first within each family, so "Flux.2 klein" is
         * not caught by "flux" and "SDXL Instruct-Pix2Pix" is not caught by the
         * SD 1.x "instruct-pix2pix".
         */
        private val BY_NAME: List<Pair<String, DiffusionFamily>> = listOf(
            // Language-model conditioners, most specific first.
            "flux.2 klein" to LLM_CONDITIONED,
            "flux2_klein" to LLM_CONDITIONED,
            "flux2klein" to LLM_CONDITIONED,
            "flux.2" to LLM_CONDITIONED,
            "flux2" to LLM_CONDITIONED,
            "qwen image" to LLM_CONDITIONED,
            "qwen_image" to LLM_CONDITIONED,
            "z-image" to LLM_CONDITIONED,
            "z_image" to LLM_CONDITIONED,
            "ovis image" to LLM_CONDITIONED,
            "sefi-image" to LLM_CONDITIONED,
            "boogu image" to LLM_CONDITIONED,
            "ernie image" to LLM_CONDITIONED,
            "longcat" to LLM_CONDITIONED,
            "ideogram" to LLM_CONDITIONED,
            "hunyuan video" to LLM_CONDITIONED,
            "lingbot video" to LLM_CONDITIONED,
            "ltxav" to LLM_CONDITIONED,
            "mage flow" to LLM_CONDITIONED,
            "krea2" to LLM_CONDITIONED,
            "anima" to LLM_CONDITIONED,
            "lens" to LLM_CONDITIONED,
            "pid" to LLM_CONDITIONED,

            // T5, with or without a CLIP beside it.
            "chroma radiance" to CHROMA_RADIANCE,
            "chroma" to T5_ONLY,
            "minit2i" to T5_ONLY,
            "wan 2" to WAN,
            "wan2" to WAN,

            // Its conditioner reads the checkpoint's own visual tower.
            "hidream" to SELF_CONDITIONED,
            // An upscaler, which conditions on nothing at all.
            "esrgan" to SELF_CONDITIONED,

            // CLIP-L and T5.
            "flex" to FLUX,
            "flux" to FLUX,

            // The UNet families. SDXL before the SD 1.x pix2pix spelling.
            "sd3" to SD3,
            "sdxl" to SDXL,
            "sdxs" to UNET,
            // SVD stays here and is *not* offered as a supported architecture in
            // `runtimes.json`. Upstream lists it under
            // `sd_version_supports_video_generation` and then has no branch for
            // it in `prepare_video_generation_latents`: no image conditioning,
            // and no `motion_bucket`, `cond_aug` or `fps_id` anywhere in the
            // source. It would load, sample, and hand back noise.
            //
            // Keeping the name mapped means a file that turns out to be SVD is
            // still described correctly rather than falling through as unknown;
            // dropping it from the runtime's advertised list means the resolver
            // stops telling anyone it will work.
            "svd" to PICTURE_CONDITIONED,
            "sd 2" to UNET,
            "sd2" to UNET,
            "sd 1" to UNET,
            "sd1" to UNET,
            "instruct-pix2pix" to UNET,
        )

        // The role param keys this table speaks in. AttachmentRole owns them;
        // they are repeated here rather than imported so that core has no
        // ordering dependency between two enums that reference each other.
        private const val CLIP_L = "clip_l"
        private const val CLIP_G = "clip_g"
        private const val T5XXL = "t5xxl"
        private const val LLM = "llm"
        private const val CLIP_VISION = "clip_vision"

        /**
         * The architectures that make frames rather than stills.
         *
         * Upstream's `sd_version_supports_video_generation`, name for name.
         * It cannot live on [DiffusionFamily] itself because the families cut
         * across it — LTX-AV and Z-Image are both `LLM_CONDITIONED`, and only
         * one of them makes video — so it is its own question about the name.
         *
         * SVD is listed because upstream lists it, though nothing can usefully
         * run it. What this answers is "does the video branch claim this",
         * which is what a parameter needs to know before deciding whether it
         * applies.
         */
        private val VIDEO_NAMES = listOf(
            "svd", "wan 2", "wan2", "hunyuan video", "lingbot video", "ltxav",
        )

        /**
         * Whether this architecture generates video, or null for a name this
         * table does not know.
         *
         * Null rather than false, so a parameter gated on it shows itself for
         * an unrecognised model: hiding a setting because of a name nobody
         * recognises is hiding it for no reason anyone could act on.
         */
        fun isVideo(name: String?): Boolean? {
            val key = name?.lowercase()?.trim()?.takeIf { it.isNotEmpty() } ?: return null
            if (VIDEO_NAMES.any { key.contains(it) }) return true
            return if (forName(key) != null) false else null
        }

        /**
         * What this model needs beside itself, or null when the name means
         * nothing here — in which case nothing is claimed to be missing,
         * because a wrong warning is worse than no warning.
         */
        fun forName(name: String?): DiffusionFamily? {
            val key = name?.lowercase()?.trim()?.takeIf { it.isNotEmpty() } ?: return null
            return BY_NAME.firstOrNull { (needle, _) -> key.contains(needle) }?.second
        }
    }
}
