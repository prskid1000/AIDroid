package ai.ondevice.engine

/** The whisper.cpp JNI surface. */
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

    external fun nativeSupportedParams(): String

    external fun nativeSystemInfo(): String

    external fun nativeLoad(path: String): Long

    external fun nativeFree(handle: Long)

    external fun nativeInfo(handle: Long): String

    external fun nativeApplyParams(handle: Long, paramsJson: String): String

    /** @param samples mono PCM, 16 kHz, in [-1, 1]. */
    external fun nativeTranscribe(handle: Long, samples: FloatArray): String
}
