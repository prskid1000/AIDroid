package ai.ondevice.engine

/** The JNI surface, and nothing else. */
object LlamaBridge {

    /** Whether `libondevice_llama.so` is present and loadable in this build. */
    val available: Boolean = runCatching {
        System.loadLibrary("ondevice_llama")
        true
    }.getOrElse {
        loadError = it.message
        false
    }

    @Volatile
    var loadError: String? = null
        private set

    external fun nativeInit()


    external fun nativeSystemInfo(): String

    external fun nativeSupportedParams(): String

    /** @return an opaque handle, or throws with a message that names the file. */
    external fun nativeLoad(path: String, paramsJson: String): Long

    external fun nativeFree(handle: Long)

    external fun nativeInfo(handle: Long): String

    external fun nativeApplyParams(handle: Long, paramsJson: String): String

    external fun nativeFormatPrompt(
        handle: Long,
        messagesJson: String,
        toolsJson: String,
        addGenerationPrompt: Boolean,
    ): String

    external fun nativeTokenize(handle: Long, text: String): String

    external fun nativeTokenCount(handle: Long, text: String): Int

    /** [imagePathsJson] must hold one path per media marker in [prompt], in order. */
    external fun nativeStartGeneration(
        handle: Long,
        prompt: String,
        stopsJson: String,
        imagePathsJson: String,
    ): String

    external fun nativeNextToken(handle: Long): String

    external fun nativeCancel(handle: Long)

    external fun nativeClearCache(handle: Long)
}
