package ai.ondevice.engine

/**
 * The JNI surface, and nothing else.
 *
 * Every method here takes and returns strings — JSON in, JSON out — which is
 * SPEC §16.7 taken literally: *the parameter contract is a string-keyed map, and
 * it is that way from the very first call*. A typed struct across this boundary
 * would mean every upstream parameter addition becomes a signature change, a
 * recompile of both sides, and an app update, which defeats §1.5 entirely.
 *
 * The cost is a JSON parse per token. On a mid-range phone that is tens of
 * microseconds against tens of milliseconds of decode, and it buys a boundary
 * that never has to move.
 */
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

    /**
     * Where the NPU's DSP loader should look for its skels (ADSP_LIBRARY_PATH).
     *
     * Must be called before anything else here, including [nativeSystemInfo] —
     * see [HexagonSkels].
     */
    external fun nativeSetDspSearchPath(path: String)

    external fun nativeSystemInfo(): String

    /**
     * `{"<key>":{"reload":bool}, …}` — every parameter this binary acts on.
     *
     * Static, so it can be called before any model is loaded, which is when the
     * parameter screen needs it.
     */
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

    external fun nativeStartGeneration(handle: Long, prompt: String, stopsJson: String): String

    external fun nativeNextToken(handle: Long): String

    external fun nativeCancel(handle: Long)

    external fun nativeClearCache(handle: Long)
}
