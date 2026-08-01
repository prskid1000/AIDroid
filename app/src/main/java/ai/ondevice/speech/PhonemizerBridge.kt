package ai.ondevice.speech

/** The espeak-ng JNI surface. */
object PhonemizerBridge {

    val available: Boolean = runCatching {
        System.loadLibrary("ondevice_phonemizer")
        true
    }.getOrElse {
        loadError = it.message
        false
    }

    @Volatile
    var loadError: String? = null
        private set

    /** [dataParent] is the directory *containing* `espeak-ng-data`. */
    external fun nativeInit(dataParent: String, voice: String)

    external fun nativePhonemize(text: String): String

    external fun nativeVersion(): String

    external fun nativeFree()
}
