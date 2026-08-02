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
) {
    companion object {
        /** SD 1.x, and what everything used to inherit. */
        val UNET = DiffusionFamily(
            encoders = setOf(CLIP_L),
            vaeSeparate = false,
        )

        /** Two text encoders, not one: SDXL conditions on ViT-L *and* ViT-bigG. */
        private val SDXL = UNET.copy(encoders = setOf(CLIP_L, CLIP_G))

        private val SD3 = DiffusionFamily(
            // SD 3.x runs on the two CLIPs alone; T5 is what it reads long,
            // written-out prompts with, and dropping it costs prompt fidelity
            // rather than the ability to run.
            encoders = setOf(CLIP_L, CLIP_G),
            optionalEncoders = setOf(T5XXL),
        )

        private val FLUX = DiffusionFamily(encoders = setOf(CLIP_L, T5XXL))

        /**
         * FLUX.2 reads its prompt with a language model instead of CLIP and T5
         * — Qwen3 for Klein, Mistral Small for dev. That is why a 4B image
         * model costs more than 4B to run.
         */
        private val FLUX2 = FLUX.copy(encoders = setOf(LLM))

        /** Qwen-Image reads its prompt with a Qwen2.5-VL, so it is an LLM encoder too. */
        private val QWEN_IMAGE = FLUX.copy(encoders = setOf(LLM))

        /** Chroma is a de-distilled Flux with CLIP-L dropped: T5 alone. */
        private val CHROMA = FLUX.copy(encoders = setOf(T5XXL))

        /**
         * Matched against the version string stable-diffusion.cpp prints, or a
         * GGUF `general.architecture`, whichever the caller has. Longest match
         * first, so "Flux.2 klein" is not caught by "flux".
         */
        private val BY_NAME: List<Pair<String, DiffusionFamily>> = listOf(
            "flux.2 klein" to FLUX2,
            "flux2_klein" to FLUX2,
            "flux2klein" to FLUX2,
            "flux.2" to FLUX2,
            "flux2" to FLUX2,
            "chroma" to CHROMA,
            "flex" to FLUX,
            "flux" to FLUX,
            "qwen image" to QWEN_IMAGE,
            "qwen_image" to QWEN_IMAGE,
            "sd3" to SD3,
            "z-image" to SD3,
            "sdxl" to SDXL,
            "sdxs" to UNET,
            "sd 2" to UNET,
            "sd2" to UNET,
            "sd 1" to UNET,
            "sd1" to UNET,
        )

        // The role param keys this table speaks in. AttachmentRole owns them;
        // they are repeated here rather than imported so that core has no
        // ordering dependency between two enums that reference each other.
        private const val CLIP_L = "clip_l"
        private const val CLIP_G = "clip_g"
        private const val T5XXL = "t5xxl"
        private const val LLM = "llm"

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
