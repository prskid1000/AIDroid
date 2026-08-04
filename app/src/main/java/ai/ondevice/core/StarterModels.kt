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
        StarterModel(
            repoId = SD15,
            modality = Modality.DIFFUSION,
            summary = "The oldest and the smallest, and the one a modest phone finishes " +
                "fastest. Carries its own encoder and decoder, and takes every kind of " +
                "add-on this app supports — it is the only base here that does.",
            sizeHint = "1.75 GB at Q4_0",
        ),
        StarterModel(
            repoId = FLUX1_SCHNELL,
            modality = Modality.DIFFUSION,
            summary = "Four steps, and the strongest prompt-following here. Needs CLIP-L, a " +
                "T5 and a decoder supplied, which together cost more than half again what " +
                "the denoiser does — budget about 6.5 GB before it will load.",
            sizeHint = "4.01 GB at Q2_K",
        ),
        StarterModel(
            repoId = ERNIE_TURBO,
            modality = Modality.DIFFUSION,
            summary = "Eight steps, and the best here at putting readable words inside a " +
                "picture — posters, labels, signage. Its text encoder is 3B rather than the " +
                "7–8B the others want, so the whole bundle is about 5.5 GB.",
            sizeHint = "3.18 GB at Q2_K",
        ),
        StarterModel(
            repoId = KREA2_TURBO,
            modality = Modality.DIFFUSION,
            summary = "Photographic by default rather than by prompt — the look most models " +
                "need a LoRA to reach. About 6.7 GB with its encoder and decoder.",
            sizeHint = "4.89 GB at Q2_K",
        ),

        // — video —
        //
        // A vid_gen model needs more beside it than an image model does, so the
        // size below is the whole bundle rather than the checkpoint — the
        // checkpoint is the part that fits.
        StarterModel(
            repoId = WAN22_TI2V_5B_TURBO,
            modality = Modality.DIFFUSION,
            summary = "Video, and the one that fits. Text or a picture in, a few seconds of " +
                "clip out, silently — sound needs an architecture no phone here has the " +
                "memory for. Step-distilled: four steps at CFG 1 rather than twenty at CFG 6, " +
                "which on a phone is the difference between a clip and an afternoon.",
            sizeHint = "2.55 GB at Q3_K_M, ~6.7 GB the bundle",
        ),
    )

    // Repo ids used by more than one card, or long enough to be worth a name.
    /**
     * The step-distilled TI2V 5B, which is the one worth recommending.
     *
     * Same architecture, same encoder, same decoder as the base model — only
     * the denoiser differs, and it differs by taking four steps at CFG 1 where
     * the original takes twenty at CFG 6. CFG 1 also drops the unconditional
     * pass, so each of those four steps costs half of what one used to.
     * Measured on the device this was written against, a 384² clip went from
     * three quarters of an hour to a few minutes.
     */
    private const val WAN22_TI2V_5B_TURBO = "hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF"

    /** Still the source of the 2.2 decoder, which Turbo does not ship. */
    private const val WAN22_TI2V_5B = "QuantStack/Wan2.2-TI2V-5B-GGUF"
    private const val Z_IMAGE_TURBO = "leejet/Z-Image-Turbo-GGUF"
    private const val SD35_TURBO = "tensorart/stable-diffusion-3.5-medium-turbo"
    private const val SD35_ENCODERS = "Comfy-Org/stable-diffusion-3.5-fp8"
    private const val SDXL_TURBO = "gpustack/stable-diffusion-xl-1.0-turbo-GGUF"
    private const val IP_ADAPTER_REPO = "h94/IP-Adapter"
    private const val FLUX2_KLEIN = "leejet/FLUX.2-klein-4B-GGUF"
    private const val SD15 = "gpustack/stable-diffusion-v1-5-GGUF"
    private const val FLUX1_SCHNELL = "city96/FLUX.1-schnell-gguf"

    private const val ERNIE_TURBO = "unsloth/ERNIE-Image-Turbo-GGUF"
    private const val KREA2_TURBO = "vantagewithai/Krea-2-Turbo-GGUF"

    /** SD 3.5 and FLUX.1 both read through it, at different quants. */
    private const val T5_ENCODER_GGUF = "city96/t5-v1_1-xxl-encoder-gguf"
    private const val FLUX_TEXT_ENCODERS = "comfyanonymous/flux_text_encoders"

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
                    repoId = T5_ENCODER_GGUF,
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
        StarterBundle(
            architecture = "SD1.x",
            label = "SD 1.5",
            base = ALL.first { it.repoId == SD15 },
            parts = listOf(
                StarterModel(
                    repoId = "latent-consistency/lcm-lora-sdv1-5",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.LORA,
                    summary = "Optional, and the one to try first: it cuts a run to four " +
                        "steps at a CFG scale near 1. Not a style — every style LoRA below " +
                        "stacks on top of it.",
                    sizeHint = "~135 MB",
                ),
                StarterModel(
                    repoId = "comfyanonymous/ControlNet-v1-1_fp16_safetensors",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.CONTROLNET,
                    summary = "Hold an outline while the prompt changes everything else. Take " +
                        "control_v11p_sd15_canny_fp16 — the control_lora_rank128 files beside " +
                        "it are a fifth the size and are LoRA-compressed, which is not the " +
                        "layout the runtime's ControlNet loader reads.",
                    sizeHint = "~723 MB",
                ),
                StarterModel(
                    repoId = IP_ADAPTER_REPO,
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.IP_ADAPTER,
                    summary = "Take the style of a picture you supply, at " +
                        "models/ip-adapter_sd15. A sixteenth the size of the SDXL one, and " +
                        "the same vision encoder reads for both.",
                    sizeHint = "~45 MB",
                ),
                StarterModel(
                    repoId = IP_ADAPTER_REPO,
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.CLIP_VISION,
                    summary = "Required by the IP-Adapter above and useless without it, at " +
                        "models/image_encoder. Fifty times the adapter's size, which is the " +
                        "real cost of this feature.",
                    sizeHint = "~2.53 GB",
                ),
                StarterModel(
                    repoId = "stabilityai/sd-vae-ft-mse",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.VAE,
                    summary = "Optional, and a substitution rather than a gap: this " +
                        "checkpoint has its own decoder. The one SD 1.5's own publisher " +
                        "retrained afterwards, and it is kinder to faces.",
                    sizeHint = "~335 MB",
                ),
            ),
        ),
        StarterBundle(
            architecture = "Flux",
            label = "FLUX.1 schnell",
            base = ALL.first { it.repoId == FLUX1_SCHNELL },
            parts = listOf(
                StarterModel(
                    repoId = FLUX_TEXT_ENCODERS,
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.CLIP_L,
                    summary = "Required, at clip_l.safetensors. FLUX reads its prompt through " +
                        "this and the T5 together, and will not load without both.",
                    sizeHint = "~246 MB",
                ),
                StarterModel(
                    repoId = T5_ENCODER_GGUF,
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.T5XXL,
                    summary = "Required. Take Q3_K_S — the fp16 release in the repo beside " +
                        "clip_l is 9.79 GB, which is more than the denoiser.",
                    sizeHint = "2.10 GB at Q3_K_S",
                ),
                StarterModel(
                    repoId = "diffusers/FLUX.1-vae",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.VAE,
                    summary = "Required. FLUX produces 16-channel latents and nothing else " +
                        "reads them. This is the fp16 copy — black-forest-labs' own repo is " +
                        "gated and this app cannot sign in to one.",
                    sizeHint = "~168 MB",
                ),
            ),
        ),
        StarterBundle(
            architecture = "Ernie Image",
            label = "ERNIE Image turbo",
            base = ALL.first { it.repoId == ERNIE_TURBO },
            parts = listOf(
                StarterModel(
                    repoId = "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.LLM_ENCODER,
                    summary = "Required, and take Q4_K_M. sd.cpp builds a Ministral 3.3B for " +
                        "this architecture specifically — not a Qwen, and not the prompt " +
                        "enhancer of the same name published beside the weights.",
                    sizeHint = "2.15 GB at Q4_K_M",
                ),
                StarterModel(
                    repoId = "baidu/ERNIE-Image-Turbo",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.VAE,
                    summary = "Required, at vae/. The publisher's own repo carries it, and it " +
                        "is the smallest decoder of any bundle here.",
                    sizeHint = "~168 MB",
                ),
            ),
        ),
        StarterBundle(
            architecture = "Krea2",
            label = "Krea 2 turbo",
            base = ALL.first { it.repoId == KREA2_TURBO },
            parts = listOf(
                StarterModel(
                    repoId = "unsloth/Qwen3-VL-4B-Instruct-GGUF",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.LLM_ENCODER,
                    summary = "Required. Four billion, not the eight the file name elsewhere " +
                        "suggests — Comfy-Org's repackage of this model ships a qwen3vl_4b, so " +
                        "4B is the size the checkpoint was trained against. UD-IQ2_M keeps it " +
                        "to a gigabyte and a half.",
                    sizeHint = "1.53 GB at UD-IQ2_M",
                ),
                StarterModel(
                    repoId = "Comfy-Org/Krea-2",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.VAE,
                    summary = "Required, at vae/qwen_image_vae.safetensors. Krea 2 borrows " +
                        "Qwen-Image's decoder rather than shipping one of its own.",
                    sizeHint = "~254 MB",
                ),
            ),
        ),

        StarterBundle(
            architecture = "wan2_2_ti2v",
            label = "Wan 2.2 TI2V 5B Turbo",
            base = ALL.first { it.repoId == WAN22_TI2V_5B_TURBO },
            parts = listOf(
                StarterModel(
                    repoId = "city96/umt5-xxl-encoder-gguf",
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.T5XXL,
                    summary = "Required. Wan reads its prompt with UMT5-XXL, and the encoder " +
                        "is larger than the model it feeds — Q3_K_M is where it stops being " +
                        "the biggest thing in the bundle. Distilling the denoiser did not " +
                        "change the encoder, and this is the same file the undistilled model " +
                        "reads.",
                    sizeHint = "2.85 GB at Q3_K_M",
                ),
                StarterModel(
                    repoId = WAN22_TI2V_5B,
                    modality = Modality.DIFFUSION,
                    role = AttachmentRole.VAE,
                    summary = "Required, at VAE/Wan2.2_VAE.safetensors — from the undistilled " +
                        "repo, because Turbo ships a denoiser and nothing else. The 2.2 " +
                        "decoder, not the 2.1 one every other Wan uses: TI2V 5B is the " +
                        "exception, and the wrong one decodes to noise rather than failing.",
                    sizeHint = "1.31 GB",
                ),
            ),
        ),

        // LTX-2.3 had a card and a bundle here and no longer does.
        //
        // It is the only architecture that makes sound, which is the whole
        // reason it was worth trying, and it wants about 14 GB of weights
        // before a single buffer is allocated — against roughly 9.9 GB free on
        // the phone this was measured on. A starter card is a recommendation,
        // and recommending a download that ends in "won't fit" after seven
        // gigabytes is not one. Nothing stops anyone pasting the repo id; what
        // is gone is the app suggesting it.
    )

    /** Every add-on, flattened, for whatever still wants one list. */
    val ADDONS: List<StarterModel> = BUNDLES.flatMap { it.parts }

    /**
     * The bases that make video, taken from their bundle's architecture.
     *
     * A [StarterModel] carries no architecture of its own — the bundle does —
     * so this is derived rather than declared, and stays right when a bundle is
     * added. `isVideo` is the same answer the screens use, which is upstream's
     * `sd_version_supports_video_generation` by name.
     */
    private val VIDEO_BASES: Set<String> = BUNDLES
        .filter { DiffusionFamily.isVideo(it.architecture) == true }
        .map { it.base.repoId }
        .toSet()

    /**
     * Grouped for display, in the order a new install would want them.
     *
     * Stills and clips are one modality to the runtime and two different
     * decisions to a person: they share DIFFUSION, the same loader and the
     * same parameters, but a 6.5 GB video bundle listed between two 2 GB image
     * models reads as an image model that is inexplicably enormous. Split by
     * what they make, not by what loads them.
     */
    val BY_SECTION: List<Pair<String, List<StarterModel>>> = listOf(
        Modality.TEXT.label to ALL.filter { it.modality == Modality.TEXT },
        Modality.SPEECH_TO_TEXT.label to ALL.filter { it.modality == Modality.SPEECH_TO_TEXT },
        Modality.TEXT_TO_SPEECH.label to ALL.filter { it.modality == Modality.TEXT_TO_SPEECH },
        "Image" to ALL.filter { it.modality == Modality.DIFFUSION && it.repoId !in VIDEO_BASES },
        "Video" to ALL.filter { it.modality == Modality.DIFFUSION && it.repoId in VIDEO_BASES },
    )
}
