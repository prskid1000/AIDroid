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
 * rather than buried: it is more than an order of magnitude slower than Kokoro,
 * so the summary says so before the download starts rather than after.
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

    /**
     * OmniVoice's repo id, named once.
     *
     * Screens that find it missing tell the user what to paste, and they used to
     * each carry their own copy of the id *and* their own size — "about 683 MB"
     * in two places against "~1.0 GB" here, for the same download. At least one
     * had to be wrong, and it was the 683: nothing in the repo adds up to it.
     * [installHint] is how a screen asks instead of remembering.
     */
    /**
     * Not `onnx-community/OmniVoice-Onnx`, and the reason is measured.
     *
     * That repo's `llm_decoder` was built by onnxruntime-genai's ModelBuilder,
     * which emits autoregressive decoders: 28 fused `GroupQueryAttention` nodes,
     * causal by design and taking no arbitrary mask. OmniVoice is a masked
     * diffusion LM — a frame must attend to frames committed after it — so it
     * produced a buzz rather than speech. Probe: change the last of twelve
     * tokens and every earlier hidden state is bit-identical, max|diff| 0.0.
     *
     * This repo is that export redone with the attention mask as a real 4-D
     * input, quantised back to 4 bits (284 MB against the original's 296 MB),
     * measured at 54.6 dB dynamic range against the PyTorch reference's 54.7.
     *
     * It also carries the Higgs vocoder converted to fp32. onnx-community's is
     * float16 throughout with no Cast node in it, which is fine on x86 — ONNX
     * Runtime has no fp16 CPU kernels there and wraps the graph in fp32 casts —
     * and fatal on arm64, where the CPU has real ARMv8.2 fp16 arithmetic and
     * float16 stops at 65504. Measured on a phone: every earlier graph finite
     * and healthy, then 48000 samples of NaN out of the vocoder. The embeddings
     * encoder, heads decoder and tokenizer are copied unchanged.
     */
    const val OMNIVOICE_REPO = "prskid1000/OmniVoice-Onnx-bidirectional"

    /**
     * "`<repo id>` — `<size>`", for a screen nudging the user toward a download.
     *
     * Null when the repo is not one we suggest, which is the honest answer: this
     * list is a starting point, not a catalogue, and the resolver gives the real
     * byte count once an id is actually pasted.
     */
    fun installHint(repoId: String): String? =
        (ALL + ADDONS).firstOrNull { it.repoId == repoId }?.let { "${it.repoId} — ${it.sizeHint}" }

    val ALL: List<StarterModel> = listOf(
        // — chat —
        //
        // One family at three sizes, rather than four families at one size each.
        // The choice a phone actually forces is how much RAM to spend, and a list
        // that answers it with four different model families answers a question
        // nobody asked while leaving that one implicit. Sizes are the Q4_K_M
        // files as the Hugging Face API reports them, not estimates.
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
            // The int4 variant, as the resolver counted it on a device: the
            // 284 MB backbone from `int4/`, 506 MB of shared graphs from
            // `components/`, and the 11 MB tokenizer at the root — 800 MB.
            // `fp32/` swaps the backbone for a 1.76 GB one and comes to 2.28 GB.
            // The components grew by 43 MB when the vocoder went to fp32, which
            // is what it costs for it to work on the platform this app runs on.
            //
            // Worth knowing before the download: 328 MB of those components are
            // the acoustic, semantic and quantizer encoders, which only matter
            // for cloning a voice from a reference clip. That is not implemented
            // yet, so today they are 44% of the download doing nothing. They are
            // bundled anyway because the resolver adds every component directory
            // to whichever variant is chosen, and splitting them out would not
            // change that — only removing them from the repo would, and then
            // cloning would need a second download.
            //
            // ONNX Runtime 1.23 is the floor, for two unrelated schema reasons:
            // int4's `bits` attribute on GatherBlockQuantized, and the twelve
            // inputs to GroupQueryAttention. This build ships 1.28.
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
