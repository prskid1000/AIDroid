package ai.ondevice.engine

/**
 * The whisper.cpp JNI surface.
 *
 * Same discipline as [LlamaBridge]: JSON in, JSON out, no typed parameter
 * struct across the boundary (SPEC §16.7). The one non-string argument is the
 * audio itself, because turning a few hundred thousand floats into JSON to
 * satisfy a rule about *parameters* would be following the letter of the thing
 * and missing its point.
 */
object WhisperBridge {

    val available: Boolean = runCatching {
        System.loadLibrary("ondevice_whisper")
        true
    }.getOrElse {
        loadError = it.message
        false
    }

    @Volatile
    var loadError: String? = null
        private set

    /** `{"<key>":{"reload":bool}, …}` — see [LlamaBridge.nativeSupportedParams]. */
    external fun nativeSupportedParams(): String

    /** `{"backends":[…]}` — what ggml registered in this binary. */
    external fun nativeSystemInfo(): String

    /**
     * @param backend the ggml registry name of the chosen compute device —
     *   "OpenCL", "HTP", "CPU" — or empty for CPU. See
     *   [ai.ondevice.core.BackendId.registryNames].
     */
    external fun nativeLoad(path: String, backend: String): Long

    external fun nativeFree(handle: Long)

    external fun nativeInfo(handle: Long): String

    external fun nativeApplyParams(handle: Long, paramsJson: String): String

    /** @param samples mono PCM, 16 kHz, in [-1, 1]. */
    external fun nativeTranscribe(handle: Long, samples: FloatArray): String
}
