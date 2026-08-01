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

    external fun nativeLoad(
        modelPath: String,
        vaePath: String,
        taesdPath: String,
        controlNetPath: String,
        /** The remaining sd_ctx_params_t auxiliary paths. */
        clipLPath: String,
        clipGPath: String,
        t5xxlPath: String,
        ipAdapterPath: String,
        embeddingsPath: String,
        /** Required whenever [ipAdapterPath] is set — see AttachmentRole.CLIP_VISION. */
        clipVisionPath: String,
        /** FLUX.2's text encoder, which is a language model rather than CLIP or T5. */
        llmPath: String,
        threads: Int,
    ): Long

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
        attachmentsJson: String,
    ): ByteArray?
}
