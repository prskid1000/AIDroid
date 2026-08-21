package ai.ondevice.engine

import ai.ondevice.data.log.LogFile
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

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
 * The ring is backed by a file once [persistTo] has been called, and that is a
 * change of mind worth recording. It was memory-only on the argument that this
 * is diagnosis rather than an audit trail — true, and beside the point: the
 * thing worth diagnosing is usually why the process is no longer there, and the
 * process going away was what emptied the ring. The file holds exactly what the
 * ring holds, [CAPACITY] lines, so nothing is kept that the screen would not
 * have shown anyway, and Clear empties both.
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

    @Serializable
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
        synchronized(this) {
            _entries.value = emptyList()
            pending.clear()
            store?.clear()
        }
    }

    // ── the file behind the ring ────────────────────────────────────────

    /**
     * Read the tail back, and write new lines to it from here on.
     *
     * Called once, from the application, with a scope that outlives every
     * screen. Before it is called this object still works — the lines go into
     * the ring and into [pending], and the first flush after install writes the
     * ones from startup that would otherwise have been the only ones lost.
     *
     * Flushed on a timer rather than per line. A line costs a `write` syscall
     * and the engines produce them in bursts — a diffusion step, a token rate,
     * a whole ggml backend announcing itself — so per-line writing would put a
     * file append inside the sampling loop. One second is short enough that a
     * kill takes at most a second of context with it, which is the thing this
     * file exists to keep.
     */
    fun persistTo(file: File, scope: CoroutineScope) {
        scope.launch(Dispatchers.IO) {
            val log = LogFile(file, CAPACITY)
            val restored = log.read().mapNotNull { line ->
                runCatching { JSON.decodeFromString(Entry.serializer(), line) }.getOrNull()
            }
            // The read finishes before [store] is set, and that order is the
            // whole of the interlock: a flush that ran first would append this
            // session's buffered lines to the file, the read would find them,
            // and the merge below would put them into the ring a second time.
            synchronized(EngineLog) {
                store = log
                // Restored first — they are older than anything this process
                // has said, and the ring is oldest-first.
                if (restored.isNotEmpty()) {
                    _entries.value = (restored + _entries.value).takeLast(CAPACITY)
                }
            }
            while (isActive) {
                delay(FLUSH_MILLIS)
                flush()
            }
        }
    }

    private fun flush() {
        val batch = synchronized(this) {
            if (pending.isEmpty()) return
            val copy = pending.toList()
            pending.clear()
            copy
        }
        store?.append(batch)
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
            // Buffered even before there is a file, so the lines written while
            // the app is still starting are not the ones that go missing. Bounded
            // by the same number as the ring, because a flush that never came
            // must not be able to grow without one.
            runCatching { JSON.encodeToString(Entry.serializer(), entry) }
                .getOrNull()
                ?.let { pending += it }
            if (pending.size > CAPACITY) repeat(pending.size - CAPACITY) { pending.removeFirst() }
        }
    }

    /** Written on the flush after they are added. Guarded by `this`. */
    private val pending = ArrayDeque<String>()

    @Volatile
    private var store: LogFile? = null

    private val JSON = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    private const val FLUSH_MILLIS = 1_000L

    private const val CAPACITY = 600
    private const val MAX_LINE = 2_000
}
