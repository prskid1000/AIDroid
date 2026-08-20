package ai.ondevice.engine

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * What the engines say about themselves, kept where the app can show it.
 *
 * The engines are already talkative — llama.cpp reports prompt tokens, cache
 * hits, template origin and a token rate on every turn; sd.cpp names its phase
 * and its tile; whisper reports its decode; the ONNX voices report graph
 * shapes when an export is refused. All of it went to logcat and nowhere else,
 * which means it was available exactly when a laptop was plugged in, and the
 * problems worth reading it for are the ones that happen on a phone in
 * somebody's pocket forty minutes into a clip.
 *
 * A ring in memory rather than a file: this is diagnosis, not an audit trail,
 * and the prompts that pass through it are not worth keeping past the process
 * that produced them. `proxy.debug` is the separate, deliberate switch for
 * writing anything down.
 *
 * Mirrored to `android.util.Log` as well, so `adb logcat` keeps working exactly
 * as it did — nothing here replaces it, and a crash still leaves its trace in
 * the place a crash is looked for.
 */
object EngineLog {

    /** Which part of the app said it, so the tail can be filtered by engine. */
    enum class Source(val label: String) {
        TEXT("Text"),
        DIFFUSION("Image & video"),
        SPEECH("Speech"),
        VOICE("Voice"),
        PROXY("Proxy"),
        /** ggml and the JNI wrappers, which belong to no single engine. */
        NATIVE("Native"),
        APP("App"),
        ;

        companion object {
            /**
             * Which engine a tag belongs to.
             *
             * Derived from the tag rather than passed at every call site: the
             * tags already exist, they are already accurate, and threading a
             * second argument through two hundred call sites is how half of
             * them come to be wrong.
             */
            fun of(tag: String): Source = when {
                // The native tags first: `ondevice.llama` is llama.cpp's own
                // callback and `LlamaEngine` is this app's wrapper, and they
                // belong under the same heading because a failing load says
                // half of it in each.
                tag == "ondevice.llama" || tag.startsWith("Llama") -> TEXT
                tag == "ondevice.sd" || tag.startsWith("Sd") ||
                    tag.startsWith("Diffusion") -> DIFFUSION
                tag == "ondevice.whisper" || tag.startsWith("Whisper") ||
                    tag.startsWith("Transcriber") -> SPEECH
                tag.startsWith("onnxruntime", ignoreCase = true) ||
                    tag.startsWith("Kokoro") || tag.startsWith("OmniVoice") ||
                    tag.startsWith("Speech") || tag.startsWith("Phonemizer") -> VOICE
                tag.startsWith("Proxy") || tag.startsWith("Ktor") -> PROXY
                // ggml picks the backend for every engine, so it has no single
                // home; it is worth seeing whichever one is loading.
                tag.startsWith("ggml") || tag.startsWith("ondevice.") -> NATIVE
                else -> APP
            }
        }
    }

    data class Entry(
        val at: Long,
        val level: Char,
        val tag: String,
        val message: String,
        /** True for a line read back out of logcat rather than written here. */
        val native: Boolean = false,
    ) {
        val source: Source get() = Source.of(tag)
    }

    private val _entries = MutableStateFlow<List<Entry>>(emptyList())
    val entries: StateFlow<List<Entry>> = _entries.asStateFlow()

    fun d(tag: String, message: String) = write('D', tag, message) { android.util.Log.d(tag, message) }

    fun i(tag: String, message: String) = write('I', tag, message) { android.util.Log.i(tag, message) }

    fun w(tag: String, message: String) = write('W', tag, message) { android.util.Log.w(tag, message) }

    fun w(tag: String, message: String, error: Throwable) =
        write('W', tag, message) { android.util.Log.w(tag, message, error) }

    fun e(tag: String, message: String) = write('E', tag, message) { android.util.Log.e(tag, message) }

    fun e(tag: String, message: String, error: Throwable) =
        write('E', tag, "$message: ${error.message}") { android.util.Log.e(tag, message, error) }

    /**
     * A line the C++ side wrote, arriving by way of logcat.
     *
     * Not mirrored back to `android.util.Log` — it is already there, and
     * writing it again would have the tail read its own output.
     */
    fun native(level: Char, tag: String, message: String) {
        append(Entry(System.currentTimeMillis(), level, tag, trim(message), native = true))
    }

    fun clear() {
        _entries.value = emptyList()
    }

    private inline fun write(level: Char, tag: String, message: String, mirror: () -> Unit) {
        mirror()
        append(Entry(System.currentTimeMillis(), level, tag, trim(message)))
    }

    /**
     * Long lines are the useful ones — a signal summary, a refused ONNX graph —
     * but not at unbounded length, and one of them must not be able to push the
     * rest of the ring out on its own.
     */
    private fun trim(message: String): String =
        if (message.length > MAX_LINE) message.take(MAX_LINE) + "…" else message

    private fun append(entry: Entry) {
        // Synchronised because these arrive from several threads at once: the
        // diffusion callback is a native one, llama's is a dispatcher's, and the
        // logcat tail has a thread of its own.
        synchronized(this) {
            val next = _entries.value + entry
            _entries.value = if (next.size > CAPACITY) next.takeLast(CAPACITY) else next
        }
    }

    private const val CAPACITY = 600
    private const val MAX_LINE = 2_000
}
