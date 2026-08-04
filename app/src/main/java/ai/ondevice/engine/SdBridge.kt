package ai.ondevice.engine

/** The stable-diffusion.cpp JNI surface. */
object SdBridge {

    val available: Boolean = runCatching {
        System.loadLibrary("ondevice_sd")
        true
    }.getOrElse {
        loadError = it.message
        false
    }

    @Volatile
    var loadError: String? = null
        private set

    external fun nativeSystemInfo(): String

    external fun nativeSupportedParams(): String

    /** What the loader worked out the last checkpoint to be, e.g. "Flux.2 klein". */
    external fun nativeDetectedVersion(): String

    /** True when the last load found a denoiser alone rather than a full checkpoint. */
    external fun nativeIsBareDiffusion(): Boolean

    /**
     * The last thing the loader said it was doing, in its own words.
     *
     * A load is one JNI call that takes minutes, and sd.cpp narrates it to the
     * log callback throughout. This is that narration, so the screen can say
     * where it has got to instead of showing a spinner for four gigabytes.
     */
    /**
     * The components the loader actually took, as `[{"role","path"}]`.
     *
     * What is resident, not what was asked for. A checkpoint carrying its own
     * encoders, a file the loader declined, one it never reached — all three
     * make the two lists differ, and only this one is a fact about memory.
     */
    external fun nativeLoadedComponents(): String

    external fun nativeLoadStage(): String

    /**
     * The runtime's own working-memory reservations, as
     * `[{"what","computeMb","cacheMb"}]`.
     *
     * ggml prints these once each, at reserve time, and the app only ever kept
     * the most recent line — so one module's graph buffer would sit on screen
     * while the cache buffer that dwarfed it was overwritten and lost. Kept
     * per module now, both figures, because a decode observed taking three
     * gigabytes described by a lone "851.60 MB" is the app hiding the answer
     * rather than reporting it.
     *
     * Not a total, and never to be added to the weight sizes: these are
     * reserved and released at different points in a run.
     */
    external fun nativeBuffers(): String

    /**
     * How many of each LoRA's tensors the last run actually applied, as
     * `[{"file","applied","total"}]`.
     *
     * A LoRA for the wrong architecture is not refused. sd.cpp matches tensors
     * by name, finds none that fit, and generates a picture the LoRA had no
     * part in — same duration, same everything, no sign on screen. This is the
     * count it keeps while doing that.
     */
    external fun nativeLoraReport(): String

    /**
     * Build a context.
     *
     * @param componentsJson every auxiliary file, keyed by
     *   `AttachmentRole.paramKey` — `{"vae":"/p","clip_l":"/p"}`. A key left
     *   out is a component not supplied, which is not the same as one supplied
     *   empty. Textual inversions are the one array: `"embeddings":
     *   [{"name":..,"path":..}]`.
     * @param settingsJson the rest of `sd_ctx_params_t`, keyed by the struct's
     *   own field names — `wtype`, `max_vram`, `stream_layers`, `flash_attn`,
     *   `tensor_type_rules`, `prediction`, and so on.
     *
     * Two JSON objects rather than the eleven positional strings this used to
     * take. The old shape did not merely make the call long — it made the
     * fields it *omitted* invisible, and two whole architectures were
     * unrunnable because nothing here could name a file they needed. A field
     * added upstream now costs one line on each side.
     */
    external fun nativeLoad(
        modelPath: String,
        componentsJson: String,
        settingsJson: String,
        threads: Int,
    ): Long

    /**
     * What the loaded context can make, as the runtime answered at load.
     *
     * Exclusive for every checkpoint but one: an SD 1.x with a motion module
     * attached says yes to both, and is the only thing here that can.
     */
    external fun nativeSupportsImage(handle: Long): Boolean

    external fun nativeSupportsVideo(handle: Long): Boolean

    /**
     * Generate a clip and leave it on disk, returning a manifest rather than
     * pixels — `{"dir","frames":[…],"width","height","fps","audio"}`.
     *
     * A five-second 480p clip is ~147 MB of raw RGB and the runtime returns
     * every frame at once. Passing that back as a `ByteArray` would hold it
     * three times over: the runtime's buffer, the Java copy, and whatever the
     * screen decodes it into.
     *
     * @param end the last frame, when the request is to travel between two
     *   stills. It has no counterpart in image generation.
     * @param outputDir must exist and be writable.
     */
    external fun nativeGenerateVideo(
        handle: Long,
        init: ByteArray?, initWidth: Int, initHeight: Int,
        end: ByteArray?, endWidth: Int, endHeight: Int,
        control: ByteArray?, controlWidth: Int, controlHeight: Int,
        attachmentsJson: String,
        outputDir: String,
    ): String?

    /** ESRGAN upscaling. */
    external fun nativeUpscale(
        esrganPath: String,
        rgb: ByteArray,
        width: Int,
        height: Int,
        factor: Int,
        threads: Int,
        tileSize: Int,
    ): ByteArray?

    external fun nativeFree(handle: Long)

    external fun nativeApplyParams(handle: Long, paramsJson: String): String

    external fun nativeProgress(handle: Long): String

    /** Packed RGB with an 8-byte header, or null before the first preview. */
    external fun nativePreview(handle: Long): ByteArray?

    external fun nativeCancel(handle: Long)

    external fun nativeGenerate(
        handle: Long,
        init: ByteArray?, initWidth: Int, initHeight: Int,
        mask: ByteArray?, maskWidth: Int, maskHeight: Int,
        control: ByteArray?, controlWidth: Int, controlHeight: Int,
        /** The picture an edit model is shown — `-r` upstream, not img2img. */
        reference: ByteArray?, referenceWidth: Int, referenceHeight: Int,
        /** The picture an IP-Adapter takes its style from, read through CLIP-Vision. */
        style: ByteArray?, styleWidth: Int, styleHeight: Int,
        /** The face PhotoMaker and PuLID are asked to keep. */
        identity: ByteArray?, identityWidth: Int, identityHeight: Int,
        attachmentsJson: String,
    ): ByteArray?
}
