package ai.ondevice.engine

/**
 * The stable-diffusion.cpp JNI surface.
 *
 * Two things differ from the other bridges, and both follow from
 * `generate_image` blocking for the whole run:
 *
 *  - Progress and the live preview are **polled**, not pushed. sd.cpp publishes
 *    them from its own callbacks into native state; Kotlin reads them from a
 *    second coroutine. No native thread ever calls into the JVM.
 *  - Images cross as packed RGB with an 8-byte width/height header, so a caller
 *    never has to guess the dimensions of what it just received.
 *
 * Attachments — LoRAs, ControlNet, IP-Adapter — arrive as one role-tagged JSON
 * list rather than as named arguments, which is what lets a new auxiliary kind
 * be added without changing this signature.
 */
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

    /** `{"<key>":{"reload":bool}, …}` — see [LlamaBridge.nativeSupportedParams]. */
    external fun nativeSupportedParams(): String

    external fun nativeLoad(
        modelPath: String,
        vaePath: String,
        taesdPath: String,
        controlNetPath: String,
        /**
         * The remaining sd_ctx_params_t auxiliary paths. Load-time in sd.cpp,
         * which is why routing them through the per-run attachment list silently
         * dropped them: that loop only understands LORA and CONTROLNET.
         */
        clipLPath: String,
        clipGPath: String,
        t5xxlPath: String,
        ipAdapterPath: String,
        embeddingsPath: String,
        /** Required whenever [ipAdapterPath] is set — see AttachmentRole.CLIP_VISION. */
        clipVisionPath: String,
        threads: Int,
        /**
         * The ggml registry name of the chosen compute device — "OpenCL",
         * "HTP", "CPU" — or empty to let sd.cpp choose as it always did.
         * See [ai.ondevice.core.BackendId.registryNames].
         */
        backend: String,
    ): Long

    /**
     * ESRGAN upscaling. Takes no handle: sd.cpp gives the upscaler its own
     * context, separate from the diffusion one, so this does not need a model
     * loaded and does not disturb one that is.
     *
     * Returns the same shape as [nativeGenerate] — 8 bytes of width and height,
     * then packed RGB — because the output size depends on the model's own
     * factor when `factor` is 0.
     */
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
        attachmentsJson: String,
    ): ByteArray?
}
