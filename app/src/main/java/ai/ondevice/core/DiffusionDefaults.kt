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
) {
    companion object {
        /** SD 1.x, and what everything used to inherit. */
        val UNET = DiffusionDefaults(
            steps = 24,
            cfgScale = 7.0f,
            samplingMethod = "dpm++2m",
            schedule = "karras",
            because = "a UNet trained at 512 with classifier-free guidance",
        )

        /**
         * Size is deliberately not one of these. SDXL and Flux were trained at
         * 1024, and 1024² on this class of device is four times the compute of
         * 512² for a picture nobody waits for. Which resolution to spend is a
         * device decision and stays the user's.
         */
        private val SDXL = UNET.copy(steps = 24, because = "a UNet with classifier-free guidance")

        private val SD3 = DiffusionDefaults(
            steps = 28,
            cfgScale = 4.5f,
            samplingMethod = "euler",
            schedule = "discrete",
            because = "a rectified-flow transformer, which wants Euler and gentler guidance",
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
        )

        /** Klein is the four-step distillation. Twenty-eight steps buys nothing. */
        private val FLUX_KLEIN = FLUX.copy(
            steps = 4,
            because = "distilled to four steps, with guidance already in the weights",
        )

        private val QWEN_IMAGE = FLUX.copy(
            steps = 20,
            cfgScale = 2.5f,
            because = "a flow transformer that takes only light guidance",
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
            "chroma" to FLUX.copy(cfgScale = 4.0f, because = "a de-distilled Flux, which takes guidance again"),
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
        fun forName(name: String?): DiffusionDefaults? {
            val key = name?.lowercase()?.trim()?.takeIf { it.isNotEmpty() } ?: return null
            return BY_NAME.firstOrNull { (needle, _) -> key.contains(needle) }?.second
        }
    }
}
