package ai.ondevice.core

/**
 * A short list of repositories known to resolve, as a starting point.
 *
 * Appendix A #8 is explicit that a curated list must never be the only path,
 * which is why the paste field stays first on the screen and this sits below
 * it. But "paste a Hugging Face ID" assumes you already know one, and for
 * Kokoro and whisper that is a genuinely hard thing to guess: whisper's weights
 * live in a repo named after the *runtime* rather than the model, and Kokoro's
 * ONNX export is a different owner from the PyTorch original.
 *
 * The rule for adding an entry: it must have been checked against the live
 * Hugging Face API, and this build must have a runtime that can execute it. A
 * suggestion that 404s wastes the user's time; a suggestion that downloads 400
 * MB and then cannot speak wastes their storage as well.
 *
 * OmniVoice is listed now that there is an engine for it, with its cost stated
 * rather than buried: it is genuinely six to seven times slower than Kokoro, so
 * the summary says so before the download starts rather than after.
 */
data class StarterModel(
    /** What goes in the paste field, verbatim. */
    val repoId: String,
    val modality: Modality,
    /** What it is, in the fewest words that distinguish it from its neighbours. */
    val summary: String,
    /** Roughly what the download costs, so the choice is informed before it starts. */
    val sizeHint: String,
    /**
     * Set for things that hang off a diffusion model rather than being one.
     * They are listed separately because installing one without a base model
     * gives you nothing to attach it to.
     */
    val role: AttachmentRole? = null,
)

object StarterModels {

    val ALL: List<StarterModel> = listOf(
        // — chat —
        StarterModel(
            repoId = "unsloth/Qwen3-1.7B-GGUF",
            modality = Modality.TEXT,
            summary = "Small and quick. The one to try first on a modest phone.",
            sizeHint = "~1.1 GB at Q4",
        ),
        StarterModel(
            repoId = "unsloth/Qwen3-4B-Instruct-2507-GGUF",
            modality = Modality.TEXT,
            summary = "Noticeably better answers, noticeably slower.",
            sizeHint = "~2.4 GB at Q4",
        ),
        StarterModel(
            repoId = "bartowski/Llama-3.2-3B-Instruct-GGUF",
            modality = Modality.TEXT,
            summary = "A different family, for when Qwen's style does not suit.",
            sizeHint = "~2.0 GB at Q4",
        ),
        StarterModel(
            repoId = "unsloth/gemma-3-4b-it-GGUF",
            modality = Modality.TEXT,
            summary = "Gemma 3, and it takes images as well as text.",
            sizeHint = "~2.5 GB at Q4",
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
            repoId = "onnx-community/OmniVoice-Onnx",
            modality = Modality.TEXT_TO_SPEECH,
            summary = "Any language, [laughter] and [sigh], several speakers — but " +
                "six to seven times slower than Kokoro.",
            // Not the int4 figure. The backbone quantises to int4 fine, but the
            // audio embedding table and codebook heads must stay fp16 or the
            // output is noise rather than a worse voice, so the honest number
            // includes them.
            sizeHint = "~683 MB",
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

    /**
     * The things that attach to a diffusion model.
     *
     * The Image screen has had LoRA, ControlNet and IP-Adapter support since it
     * was written — it classifies anything installed by role and offers a
     * weight dial for the roles sd.cpp weights. What it had no answer for was
     * "where do I get one", so the section was permanently empty and looked
     * unimplemented.
     *
     * Filenames matter here in a way they do not elsewhere, because the role is
     * read from the filename. Each entry below was checked to make sure its
     * files classify as the role claimed.
     */
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
            // The repo holds ×2, ×4 and ×8, all named RealESRGAN_* so the role is
            // read off the filename. Only the ×4 file runs here, and that is a
            // property of the bundled upscaler rather than of the weights:
            // sd.cpp's RRDBNet reads its scale back from the tensors it finds
            // (`conv_up2` present → ×4) and builds no `conv_up3`, so the ×8
            // network's third upsample block has nowhere to load. The ×2 file is
            // worse than unsupported — BasicSR trains scale-2 with a
            // pixel-unshuffled input, giving `conv_first` twelve input channels
            // where this build assumes three, so it cannot even bind. Saying so
            // beats a 67 MB download that fails at load.
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
