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

        // Stable Diffusion's current generation. Like Klein it keeps its text
        // encoders outside the checkpoint, so it also costs more than its own
        // size — three of them, listed below as required.
        StarterModel(
            repoId = "city96/stable-diffusion-3.5-medium-gguf",
            modality = Modality.DIFFUSION,
            summary = "SD 3.5 medium. Reads a prompt closely and writes legible text. " +
                "Needs all three encoders below — about 5 GB together.",
            sizeHint = "~1.8 GB at Q4",
        ),

        // The only model here a ControlNet or an IP-Adapter fits.
        //
        // sd.cpp builds both for a UNet and nothing else — control.hpp branches
        // on SD1/SD2/SDXL/SVD, and the IP-Adapter's injection map is a list of
        // UNet block names — so on Klein or SD 3.5, which are both diffusion
        // transformers, either would load, cost its memory and change no
        // pixels. SDXL is the last model built that way, which is the whole
        // reason it is here: nothing newer can be steered by a pose or a depth
        // map at all.
        //
        // It is slow. A 1024-pixel picture is minutes on a phone CPU, against
        // seconds for Klein at four steps. That is the trade, stated up front.
        StarterModel(
            repoId = "HyperX-Sentience/SDXL-GGUF",
            modality = Modality.DIFFUSION,
            summary = "The last model a ControlNet or IP-Adapter fits — steer it with a pose, " +
                "depth or edge map. UNet only, so it needs CLIP-L, CLIP-G and the SDXL VAE " +
                "below. Minutes per picture here.",
            sizeHint = "1.42 GB at Q4_K_S",
        ),
    )

    /** The text encoder FLUX.2 Klein 4B reads its prompt with. */
    const val FLUX_ENCODER = "unsloth/Qwen3-4B-GGUF"

    /** SD 3.5's three, which live together in one repo. */
    const val SD35_ENCODERS = "Comfy-Org/stable-diffusion-3.5-fp8"

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

        // — the three SD 3.5 cannot run without, all from one repo —
        StarterModel(
            repoId = SD35_ENCODERS,
            modality = Modality.DIFFUSION,
            role = AttachmentRole.CLIP_L,
            summary = "SD 3.5's first text encoder — split_files/text_encoders/clip_l.",
            sizeHint = "~246 MB",
        ),
        StarterModel(
            repoId = SD35_ENCODERS,
            modality = Modality.DIFFUSION,
            role = AttachmentRole.CLIP_G,
            summary = "SD 3.5's second — split_files/text_encoders/clip_g.",
            sizeHint = "~1.4 GB",
        ),
        StarterModel(
            repoId = SD35_ENCODERS,
            modality = Modality.DIFFUSION,
            role = AttachmentRole.T5XXL,
            summary = "SD 3.5's third, and the one that reads a long prompt as a sentence " +
                "rather than as keywords — split_files/text_encoders/t5xxl.",
            sizeHint = "~4.9 GB fp8",
        ),

        // — the two SDXL needs, and the two only SDXL can use —
        //
        // SDXL reads its prompt through the same CLIP-L and CLIP-G that SD 3.5
        // does, so the two encoders above serve both and are not listed twice.
        StarterModel(
            repoId = "madebyollin/sdxl-vae-fp16-fix",
            modality = Modality.DIFFUSION,
            role = AttachmentRole.VAE,
            summary = "SDXL's decoder, rebuilt so it does not produce a black image in fp16. " +
                "Required with the SDXL card above.",
            sizeHint = "319 MB",
        ),
        StarterModel(
            repoId = "r3gm/controlnet-union-sdxl-1.0-fp16",
            modality = Modality.DIFFUSION,
            role = AttachmentRole.CONTROLNET,
            // xinsir's original is fp32 at the same tensor count; this is the
            // fp16 mirror. There is no GGUF of it anywhere — ComfyUI's GGUF
            // loader does not handle ControlNets, so nobody has quantised one.
            summary = "Canny, depth, pose, scribble and tile in one file. Attach a control " +
                "image on the Image screen and it steers the composition. SDXL only.",
            sizeHint = "2.39 GB fp16",
        ),
        StarterModel(
            repoId = "h94/IP-Adapter",
            modality = Modality.DIFFUSION,
            role = AttachmentRole.IP_ADAPTER,
            summary = "Style from a reference picture instead of from words. Pick " +
                "ip-adapter_sdxl_vit-h, and take the CLIP-Vision encoder below with it — " +
                "without one it binds to nothing.",
            sizeHint = "666 MB",
        ),
        StarterModel(
            repoId = "h94/IP-Adapter",
            modality = Modality.DIFFUSION,
            role = AttachmentRole.CLIP_VISION,
            // models/image_encoder is the ViT-H, which is what the _vit-h
            // adapters were trained against; sdxl_models/image_encoder is the
            // bigG and 1.1 GB larger for no gain unless you picked that adapter.
            summary = "The eye the IP-Adapter looks through — models/image_encoder. " +
                "Nothing else uses it, and it is the expensive half of the pair.",
            sizeHint = "2.4 GB",
        ),

        // The upscaler is model-agnostic: it enlarges a finished PNG and does
        // not care what made it.
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
