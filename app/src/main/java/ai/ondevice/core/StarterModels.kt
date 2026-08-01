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
        StarterModel(
            repoId = "second-state/stable-diffusion-v1-5-GGUF",
            modality = Modality.DIFFUSION,
            summary = "SD 1.5. The smallest thing that makes a real picture.",
            sizeHint = "~2.0 GB at Q4",
        ),
        StarterModel(
            repoId = "city96/stable-diffusion-3.5-medium-gguf",
            modality = Modality.DIFFUSION,
            summary = "SD 3.5 medium. Better prompt following, wants a big phone.",
            sizeHint = "~3.5 GB at Q4",
        ),
    )

    /** The things that attach to a diffusion model. */
    val ADDONS: List<StarterModel> = listOf(
        StarterModel(
            repoId = "comfyanonymous/ControlNet-v1-1_fp16_safetensors",
            modality = Modality.DIFFUSION,
            role = AttachmentRole.CONTROLNET,
            summary = "Canny, depth, openpose and eleven more, for SD 1.5. Pick one file.",
            sizeHint = "~723 MB each",
        ),
        StarterModel(
            repoId = "h94/IP-Adapter",
            modality = Modality.DIFFUSION,
            role = AttachmentRole.IP_ADAPTER,
            summary = "Style from a reference picture instead of from words.",
            sizeHint = "~44 MB for sd15",
        ),
        StarterModel(
            repoId = "latent-consistency/lcm-lora-sdv1-5",
            modality = Modality.DIFFUSION,
            role = AttachmentRole.LORA,
            summary = "The useful first LoRA: usable pictures in 4–8 steps instead of 28.",
            sizeHint = "~135 MB",
        ),
        StarterModel(
            repoId = "madebyollin/taesd",
            modality = Modality.DIFFUSION,
            role = AttachmentRole.TAESD,
            summary = "The tiny decoder behind the live preview. Cheap, and worth it.",
            sizeHint = "~5 MB",
        ),
        StarterModel(
            repoId = "stabilityai/sd-vae-ft-mse-original",
            modality = Modality.DIFFUSION,
            role = AttachmentRole.VAE,
            summary = "A better decoder for SD 1.5 — fixes washed-out colour.",
            sizeHint = "~335 MB",
        ),
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
