package ai.ondevice.core

/**
 * The settings a diffusion family expects, where one set of numbers cannot
 * serve all of them.
 *
 * Steps, CFG and the sampler are not neutral knobs. A guidance-distilled flow
 * model has classifier-free guidance *baked in*: give it CFG 7 and it is asked
 * to push twice, which burns the picture to coloured noise. A four-step
 * distilled model given 28 steps costs seven times the wall clock for no gain.
 * The old single default — 28 steps, CFG 7, DPM++(2M)/Karras — is SD 1.5's, and
 * it is wrong for everything published since.
 *
 * Nothing here is inferred from a filename or a repo name. The key is the
 * version stable-diffusion.cpp works out from the tensors themselves and
 * announces on load, or the GGUF's own `general.architecture` where it has one.
 */
data class DiffusionDefaults(
    val steps: Int,
    val cfgScale: Float,
    val samplingMethod: String,
    val schedule: String,
    /** Why these numbers and not the others, in one clause for the screen. */
    val because: String,
    /**
     * What this family reads its prompt through, as `AttachmentRole.paramKey`
     * values — the one vocabulary both role enums in this app agree on.
     *
     * A checkpoint that carries its own encoders needs none of these supplied;
     * a bare denoiser, which is what every quantised release is, needs all of
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
        val UNET = DiffusionDefaults(
            steps = 24,
            cfgScale = 7.0f,
            samplingMethod = "dpm++2m",
            schedule = "karras",
            because = "a UNet trained at 512 with classifier-free guidance",
            encoders = setOf(CLIP_L),
            vaeSeparate = false,
        )

        /**
         * Size is deliberately not one of these. SDXL and Flux were trained at
         * 1024, and 1024² on this class of device is four times the compute of
         * 512² for a picture nobody waits for. Which resolution to spend is a
         * device decision and stays the user's.
         */
        /** Two text encoders, not one: SDXL conditions on ViT-L *and* ViT-bigG. */
        private val SDXL = UNET.copy(
            steps = 24,
            because = "a UNet with classifier-free guidance",
            encoders = setOf(CLIP_L, CLIP_G),
        )

        private val SD3 = DiffusionDefaults(
            steps = 28,
            cfgScale = 4.5f,
            samplingMethod = "euler",
            schedule = "discrete",
            because = "a rectified-flow transformer, which wants Euler and gentler guidance",
            // SD 3.x runs on the two CLIPs alone; T5 is what it reads long,
            // written-out prompts with, and dropping it costs prompt fidelity
            // rather than the ability to run.
            encoders = setOf(CLIP_L, CLIP_G),
            optionalEncoders = setOf(T5XXL),
        )

        /**
         * Flux and its relatives are guidance-distilled: the model already
         * carries the effect of CFG, so the sampler must not apply it again.
         * sd.cpp's own README says the same — `--cfg-scale 1.0`.
         */
        private val FLUX = DiffusionDefaults(
            steps = 20,
            cfgScale = 1.0f,
            samplingMethod = "euler",
            schedule = "discrete",
            because = "guidance is distilled into the weights, so a second push burns the image",
            encoders = setOf(CLIP_L, T5XXL),
        )

        /**
         * Klein is the four-step distillation, and FLUX.2 reads its prompt with
         * a language model instead of CLIP and T5 — Qwen3 here, Mistral Small
         * for dev. That is why a 4B image model costs more than 4B to run.
         */
        private val FLUX2 = FLUX.copy(encoders = setOf(LLM))

        private val FLUX_KLEIN = FLUX2.copy(
            steps = 4,
            because = "distilled to four steps, with guidance already in the weights",
        )

        /** Qwen-Image reads its prompt with a Qwen2.5-VL, so it is an LLM encoder too. */
        private val QWEN_IMAGE = FLUX.copy(
            steps = 20,
            cfgScale = 2.5f,
            because = "a flow transformer that takes only light guidance",
            encoders = setOf(LLM),
        )

        /** Chroma is a de-distilled Flux with CLIP-L dropped: T5 alone. */
        private val CHROMA = FLUX.copy(
            cfgScale = 4.0f,
            because = "a de-distilled Flux, which takes guidance again",
            encoders = setOf(T5XXL),
        )

        /**
         * Matched against the version string stable-diffusion.cpp prints, or a
         * GGUF `general.architecture`, whichever the caller has. Longest match
         * first, so "Flux.2 klein" is not caught by "flux".
         */
        private val BY_NAME: List<Pair<String, DiffusionDefaults>> = listOf(
            "flux.2 klein" to FLUX_KLEIN,
            "flux2_klein" to FLUX_KLEIN,
            "flux2klein" to FLUX_KLEIN,
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

        /**
         * What this model wants, or null when the name means nothing here —
         * in which case whatever is on screen stays, because a wrong guess is
         * worse than an old one.
         */
        // The role param keys this table speaks in. AttachmentRole owns them;
        // they are repeated here rather than imported so that core has no
        // ordering dependency between two enums that reference each other.
        private const val CLIP_L = "clip_l"
        private const val CLIP_G = "clip_g"
        private const val T5XXL = "t5xxl"
        private const val LLM = "llm"

        fun forName(name: String?): DiffusionDefaults? {
            val key = name?.lowercase()?.trim()?.takeIf { it.isNotEmpty() } ?: return null
            return BY_NAME.firstOrNull { (needle, _) -> key.contains(needle) }?.second
        }
    }
}
