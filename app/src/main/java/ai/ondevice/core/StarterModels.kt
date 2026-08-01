package ai.ondevice.core

/** A short list of repositories known to resolve, as a starting point. */
data class StarterModel(
    /** What goes in the paste field, verbatim. */
    val repoId: String,
    val modality: Modality,
    /** What it is, in the fewest words that distinguish it from its neighbours. */
    val summary: String,
    /** Roughly what the download costs, so the choice is informed before it starts. */
    val sizeHint: String,
    /** Set for things that hang off a diffusion model rather than being one. */
    val role: AttachmentRole? = null,
)

object StarterModels {

    /** OmniVoice's repo id, named once. */
    /** Not `onnx-community/OmniVoice-Onnx`, and the reason is measured. */
    const val OMNIVOICE_REPO = "prskid1000/OmniVoice-Onnx-bidirectional"

    /** "`<repo id>` — `<size>`", for a screen nudging the user toward a download. */
    fun installHint(repoId: String): String? =
        (ALL + ADDONS).firstOrNull { it.repoId == repoId }?.let { "${it.repoId} — ${it.sizeHint}" }

    val ALL: List<StarterModel> = listOf(
        // — chat — One family at three sizes, rather than four families at one size each.
        StarterModel(
            repoId = "unsloth/Qwen3.5-2B-GGUF",
            modality = Modality.TEXT,
            summary = "Small and quick. The one to try first on a modest phone.",
            sizeHint = "1.28 GB at Q4_K_M",
        ),
        StarterModel(
            repoId = "unsloth/Qwen3.5-4B-GGUF",
            modality = Modality.TEXT,
            summary = "Noticeably better answers, noticeably slower.",
            sizeHint = "2.74 GB at Q4_K_M",
        ),
        StarterModel(
            repoId = "unsloth/Qwen3.5-9B-GGUF",
            modality = Modality.TEXT,
            summary = "The best of the three, and it wants a phone with the RAM to hold it.",
            sizeHint = "5.68 GB at Q4_K_M",
        ),

        // — speech to text —
        StarterModel(
            repoId = "ggerganov/whisper.cpp",
            modality = Modality.SPEECH_TO_TEXT,
            summary = "Every whisper size in one repo — base or small suits a phone.",
            sizeHint = "75 MB base, 466 MB small",
        ),

        // — text to speech —
        StarterModel(
            repoId = "onnx-community/Kokoro-82M-v1.0-ONNX",
            modality = Modality.TEXT_TO_SPEECH,
            summary = "The neural voice. Its 55 voice packs come with it.",
            sizeHint = "~116 MB with voices",
        ),
        StarterModel(
            repoId = OMNIVOICE_REPO,
            modality = Modality.TEXT_TO_SPEECH,
            summary = "Any language, [laughter] and [sigh], a voice you describe in words. " +
                "Much slower than Kokoro.",
            sizeHint = "~750 MB",
        ),

        // — images —
        //
        // FLUX.2 Klein is what the app leads with now. It does two things SD
        // cannot: it follows a written instruction, and it edits a picture you
        // hand it rather than only making one from nothing. The 4B is the one
        // a phone can hold — the 9B needs an 8B text encoder beside it and
        // comes to about 11 GB before the image is even started.
        //
        // It costs more than its own size, and the reason is [FLUX_ENCODER]:
        // FLUX.2 reads the prompt with a language model, not with CLIP.
        StarterModel(
            repoId = "leejet/FLUX.2-klein-4B-GGUF",
            modality = Modality.DIFFUSION,
            summary = "Follows instructions and edits a picture you give it, in four steps. " +
                "Needs the Qwen3 4B encoder and the FLUX.2 VAE below — 5.4 GB in all.",
            sizeHint = "~2.5 GB at Q4",
        ),
    )

    /** The text encoder FLUX.2 Klein 4B reads its prompt with. */
    const val FLUX_ENCODER = "unsloth/Qwen3-4B-GGUF"

    /** The things that attach to a diffusion model. */
    val ADDONS: List<StarterModel> = listOf(
        // — the two FLUX.2 Klein cannot run without —
        StarterModel(
            repoId = FLUX_ENCODER,
            modality = Modality.DIFFUSION,
            role = AttachmentRole.LLM_ENCODER,
            summary = "FLUX.2 Klein 4B reads its prompt with this. Required, and the " +
                "reason a 4B image model costs 5 GB. Pick Q4_K_M.",
            sizeHint = "~2.5 GB at Q4",
        ),
        StarterModel(
            repoId = "Comfy-Org/flux2-klein-4B",
            modality = Modality.DIFFUSION,
            role = AttachmentRole.VAE,
            summary = "FLUX.2's decoder, at split_files/vae/flux2-vae.safetensors. " +
                "Required: Klein produces latents and nothing else can read them.",
            sizeHint = "~336 MB",
        ),

        // Nothing for SD 1.5 is listed any more, and neither is SD 1.5: a
        // ControlNet, an IP-Adapter and a LoRA that only fit a model the app no
        // longer offers are five downloads that cannot be used. The upscaler
        // below is the exception — it enlarges a finished PNG and does not care
        // which model made it.
        StarterModel(
            repoId = "ai-forever/Real-ESRGAN",
            modality = Modality.DIFFUSION,
            role = AttachmentRole.UPSCALER,
            // The repo holds ×2, ×4 and ×8, all named RealESRGAN_* so the role is read off the filename.
            summary = "Enlarge a finished picture ×4. Pick RealESRGAN_x4 — the ×2 and ×8 " +
                "networks need upsample blocks this build's upscaler does not have.",
            sizeHint = "~67 MB",
        ),
    )

    /** Grouped for display, in the order a new install would want them. */
    val BY_MODALITY: List<Pair<Modality, List<StarterModel>>> = listOf(
        Modality.TEXT,
        Modality.SPEECH_TO_TEXT,
        Modality.TEXT_TO_SPEECH,
        Modality.DIFFUSION,
    ).map { modality -> modality to ALL.filter { it.modality == modality } }
}
