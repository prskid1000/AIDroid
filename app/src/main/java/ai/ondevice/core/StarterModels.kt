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
        // One entry per architecture, each the smallest quant still worth
        // running. What each needs *beside* itself lives in [BUNDLES] below,
        // keyed to the same architecture, so a component can no longer be read
        // as belonging to whichever base it happens to sit next to.
        StarterModel(
            repoId = Z_IMAGE_TURBO,
            modality = Modality.DIFFUSION,
            summary = "The one to try first. Turbo, and the whole bundle is about 5 GB — " +
                "less than SDXL once its encoder is counted. Published by the author of the " +
                "runtime this app uses, so its tensor names are the ones it expects.",
            sizeHint = "2.59 GB at Q2_K",
        ),
        StarterModel(
            repoId = SD35_TURBO,
            modality = Modality.DIFFUSION,
            summary = "The smallest base here, and the only one under 2 GB. Takes CLIP-L, " +
                "CLIP-G and a decoder; its own repo carries the decoder.",
            sizeHint = "1.79 GB at Q4_K_M",
        ),
        StarterModel(
            repoId = SDXL_TURBO,
            modality = Modality.DIFFUSION,
            summary = "The only one here that takes a ControlNet or an IP-Adapter — both are " +
                "built for a UNet and this is the only UNet. Carries its own encoders and " +
                "decoder, so it needs nothing else to run.",
            sizeHint = "3.94 GB at Q4_0",
        ),
        StarterModel(
            repoId = FLUX2_KLEIN,
            modality = Modality.DIFFUSION,
            summary = "Follows a written instruction and edits a picture you hand it, in " +
                "four steps. Reads its prompt with a language model, so the encoder beside " +
                "it is most of what it costs.",
            sizeHint = "2.46 GB at Q4_0",
        ),
    )

    // Repo ids used by more than one card, or long enough to be worth a name.
    private const val Z_IMAGE_TURBO = "leejet/Z-Image-Turbo-GGUF"
    private const val SD35_TURBO = "tensorart/stable-diffusion-3.5-medium-turbo"
    private const val SD35_ENCODERS = "Comfy-Org/stable-diffusion-3.5-fp8"
    private const val SDXL_TURBO = "gpustack/stable-diffusion-xl-1.0-turbo-GGUF"
    private const val IP_ADAPTER_REPO = "h94/IP-Adapter"
    private const val FLUX2_KLEIN = "leejet/FLUX.2-klein-4B-GGUF"

    /**
     * What one architecture needs beside its base model, and what it can take.
     *
     * Grouped because the flat list could not say which component went with
     * which base. That was recorded in comment banners and in English inside
     * each summary — "SDXL only", "Built for Klein 4B" — which the screen could
     * not read, so it rendered every add-on in one undifferentiated block. With
     * four architectures and their LoRAs that block is unreadable, and picking
     * wrong is not obviously wrong: a LoRA from another family loads, costs its
     * time and changes nothing.
     *
     * Every repo and filename here was checked against the Hugging Face API
     * rather than recalled. Where nothing suitable exists the row is absent —
     * a card that 404s is worse than no card, and a component from the wrong
     * family is worse than both.
     */
    data class StarterBundle(
        /** As stable-diffusion.cpp names it, so [DiffusionFamily] agrees. */
        val architecture: String,
        val label: String,
        val base: StarterModel,
        val parts: List<StarterModel>,
    )

    val BUNDLES: List<StarterBundle> = listOf(
        StarterBundle(
            architecture = "Z-Image",
            label = "Z-Image Turbo",
            base = ALL.first { it.repoId == Z_IMAGE_TURBO },
            parts = listOf(
                StarterModel(
                    repoId = "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.LLM_ENCODER,
                    summary = "Required. Z-Image reads its prompt with Qwen3, and this " +
                        "conversion was made for it rather than adapted from a chat model.",
                    sizeHint = "2.08 GB at Q3_K_M",
                ),
                StarterModel(
                    repoId = "Comfy-Org/z_image_turbo",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.VAE,
                    summary = "Required, at split_files/vae/ae.safetensors. The same " +
                        "16-channel autoencoder FLUX uses — this copy is not gated, and the " +
                        "one the runtime's own docs point at is.",
                    sizeHint = "~335 MB",
                ),
                StarterModel(
                    repoId = "alfredplpl/z-image-modern-anime-lora",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.LORA,
                    summary = "Anime. Trained on this architecture rather than adapted to it.",
                    sizeHint = "~184 MB",
                ),
                StarterModel(
                    repoId = "suayptalha/Z-Image-Turbo-Realism-LoRA",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.LORA,
                    summary = "Photoreal. Ships in diffusers key naming, which the runtime " +
                        "may not match — the Image screen says so if none of it applied.",
                    sizeHint = "~85 MB",
                ),
            ),
        ),
        StarterBundle(
            architecture = "SD3.x",
            label = "SD 3.5 medium turbo",
            base = ALL.first { it.repoId == SD35_TURBO },
            parts = listOf(
                StarterModel(
                    repoId = SD35_ENCODERS,
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.CLIP_L,
                    summary = "Required, at text_encoders/clip_l.",
                    sizeHint = "~246 MB",
                ),
                StarterModel(
                    repoId = SD35_ENCODERS,
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.CLIP_G,
                    summary = "Required, at text_encoders/clip_g.",
                    sizeHint = "~1.39 GB",
                ),
                StarterModel(
                    repoId = SD35_TURBO,
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.VAE,
                    summary = "Required, at vae/. The base model's own repo carries it.",
                    sizeHint = "~168 MB",
                ),
                StarterModel(
                    repoId = "city96/t5-v1_1-xxl-encoder-gguf",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.T5XXL,
                    summary = "Optional. Long written-out prompts read better with it. " +
                        "Quantised to under half the fp8 release, which is 4.89 GB.",
                    sizeHint = "2.10 GB at Q3_K_S",
                ),
            ),
        ),
        StarterBundle(
            architecture = "SDXL",
            label = "SDXL turbo",
            base = ALL.first { it.repoId == SDXL_TURBO },
            parts = listOf(
                StarterModel(
                    repoId = "Linaqruf/anime-detailer-xl-lora",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.LORA,
                    summary = "Anime, and small enough to be worth trying on anything.",
                    sizeHint = "~43 MB",
                ),
                StarterModel(
                    repoId = "ostris/photorealistic-slider-sdxl-lora",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.LORA,
                    summary = "Photoreal, at a strength you set rather than a look you take.",
                    sizeHint = "~24 MB",
                ),
                StarterModel(
                    repoId = "diffusers/controlnet-canny-sdxl-1.0-small",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.CONTROLNET,
                    summary = "Hold an outline while the prompt changes everything else. " +
                        "Distilled — an eighth the size of the full union model.",
                    sizeHint = "~320 MB",
                ),
                StarterModel(
                    repoId = IP_ADAPTER_REPO,
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.IP_ADAPTER,
                    summary = "Take the style of a picture you supply. Use the vit-h file at " +
                        "sdxl_models/ip-adapter_sdxl_vit-h — the other one needs an encoder " +
                        "a gigabyte larger for no gain.",
                    sizeHint = "~698 MB",
                ),
                StarterModel(
                    repoId = IP_ADAPTER_REPO,
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.CLIP_VISION,
                    summary = "Required by the IP-Adapter above and useless without it, at " +
                        "models/image_encoder. This is the vit-h encoder that file expects.",
                    sizeHint = "~2.53 GB",
                ),
                StarterModel(
                    repoId = "madebyollin/sdxl-vae-fp16-fix",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.VAE,
                    summary = "Optional, and a substitution rather than a gap: this " +
                        "checkpoint has its own decoder. Fixes the fp16 artefacts SDXL's " +
                        "original is known for.",
                    sizeHint = "~335 MB",
                ),
            ),
        ),
        StarterBundle(
            architecture = "Flux.2 klein",
            label = "FLUX.2 Klein 4B",
            base = ALL.first { it.repoId == FLUX2_KLEIN },
            parts = listOf(
                StarterModel(
                    repoId = "unsloth/Qwen3-4B-GGUF",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.LLM_ENCODER,
                    summary = "Required. Klein reads its prompt with Qwen3 rather than with " +
                        "CLIP — take the UD-IQ2_M file, a gigabyte under Q4_K_M and still an " +
                        "imatrix quant.",
                    sizeHint = "1.53 GB at UD-IQ2_M",
                ),
                StarterModel(
                    repoId = "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.VAE,
                    summary = "Required, at split_files/vae/flux2-vae.safetensors. Klein " +
                        "produces latents and nothing else can read them.",
                    sizeHint = "~336 MB",
                ),
                StarterModel(
                    repoId = "nomadoor/flux-2-klein-4B-360-erp-outpaint-lora",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.LORA,
                    summary = "Extends a picture outward into a 360 panorama. The only LoRA " +
                        "found for this architecture — it is new enough that the style ones " +
                        "do not exist yet.",
                    sizeHint = "~46 MB",
                ),
            ),
        ),
    )

    /** Every add-on, flattened, for whatever still wants one list. */
    val ADDONS: List<StarterModel> = BUNDLES.flatMap { it.parts }

    /** Grouped for display, in the order a new install would want them. */
    val BY_MODALITY: List<Pair<Modality, List<StarterModel>>> = listOf(
        Modality.TEXT,
        Modality.SPEECH_TO_TEXT,
        Modality.TEXT_TO_SPEECH,
        Modality.DIFFUSION,
    ).map { modality -> modality to ALL.filter { it.modality == modality } }
}
