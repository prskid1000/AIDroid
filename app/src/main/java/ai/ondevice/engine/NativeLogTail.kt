package ai.ondevice.engine

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * The native engines' own log, brought into the app's.
 *
 * llama.cpp, sd.cpp and whisper.cpp all say a great deal about themselves and
 * none of it goes through Kotlin: each installs a C callback —
 * `llama_log_set`, `sd_set_log_callback`, `whisper_log_set` — that forwards to
 * `__android_log_print` under `ondevice.llama`, `ondevice.sd` and
 * `ondevice.whisper`. ONNX Runtime logs from its own native layer too, under
 * tags nobody here chose. All of it lands in logcat and stops there, which is
 * to say it is available when a cable is attached and nowhere else.
 *
 * Read back out of logcat rather than routed through JNI, and the reason is
 * proportion. Calling up into Java from those callbacks means a cached
 * `JavaVM`, a method id, attaching threads that ggml created and this app does
 * not own, and getting all of it right in four `.cpp` files behind a
 * twenty-minute native build — to move text that is already being written to a
 * buffer this process is allowed to read. Android restricts an app's logcat to
 * its own entries, so `--pid` is both the filter and the whole of the
 * permission story.
 *
 * The dump comes first and the follow after, so opening the screen shows the
 * load that already happened rather than only what happens next.
 */
object NativeLogTail {

    private var job: Job? = null
    private var readers = 0

    /**
     * Start following, if nobody already is.
     *
     * Reference-counted, because the log screen can be opened, left and opened
     * again faster than a process spawns — and two tails would double every
     * line.
     */
    @Synchronized
    fun start(scope: CoroutineScope) {
        readers++
        if (job != null) return
        job = scope.launch(Dispatchers.IO) { follow() }
    }

    @Synchronized
    fun stop() {
        readers = (readers - 1).coerceAtLeast(0)
        if (readers > 0) return
        job?.cancel()
        job = null
    }

    private suspend fun follow() {
        val pid = android.os.Process.myPid()
        val process = runCatching {
            ProcessBuilder(
                "logcat",
                "--pid=$pid",
                // `time` carries the clock and the priority and nothing else —
                // the pid and tid are ours by construction and would be noise.
                "-v", "time",
                // An allowlist, ending in `*:S`.
                //
                // Filtering by pid alone is not enough: everything the platform
                // logs on this process's behalf comes with it, and on this
                // device that is the Adreno driver announcing its build
                // configuration a hundred lines at a time. Measured on the
                // first run of this screen: fifty-four lines captured, every
                // one of them from `AdrenoVK-0`, and not one from an engine.
                //
                // The cost is that an unexpected native tag is missed. That is
                // the right way round — this is the *engine* log, the tags below
                // are the ones the four native runtimes actually write under,
                // and a list that has to be added to is better than a screen
                // nobody can read.
                *ALLOWED.flatMap { listOf("$it:V") }.toTypedArray(),
                "*:S",
            ).redirectErrorStream(true).start()
        }.getOrElse {
            EngineLog.w(
                "NativeLogTail",
                "cannot read this app's own logcat, so native engine output is not shown here: " +
                    "${it.message}",
            )
            return
        }

        try {
            process.inputStream.bufferedReader().use { reader ->
                while (kotlinx.coroutines.currentCoroutineContext().isActive) {
                    val line = reader.readLine() ?: break
                    parse(line)?.let { (level, tag, message) ->
                        EngineLog.native(level, tag, message)
                    }
                }
            }
        } catch (_: Throwable) {
            // A cancelled read throws on the way out; there is nothing to say
            // about it that the absence of further lines does not already say.
        } finally {
            runCatching { process.destroy() }
        }
    }

    /**
     * `08-20 11:37:16.614 I/ondevice.llama( 1234): text`
     *
     * Hand-parsed rather than regex-per-line: this runs on every line a busy
     * engine emits, and sd.cpp emits one per tile.
     */
    private fun parse(line: String): Triple<Char, String, String>? {
        // Continuation lines from a multi-line native message have no header.
        if (line.length < MIN_LINE || line.startsWith("-")) return null
        val slash = line.indexOf('/')
        if (slash < 2 || line[slash - 2] != ' ') return null
        val level = line[slash - 1]
        if (level !in "VDIWEF") return null
        val paren = line.indexOf('(', slash)
        val colon = line.indexOf("): ", paren.coerceAtLeast(0))
        if (paren < 0 || colon < 0) return null
        val tag = line.substring(slash + 1, paren).trim()
        val message = line.substring(colon + 3)
        if (message.isBlank()) return null
        return Triple(level, tag, message)
    }

    /**
     * The tags the native side writes under.
     *
     * The first four are this app's own JNI wrappers, spelled as
     * `app/src/main/cpp` spells them. `ggml` and its backends are shared by
     * three of the runtimes and are where a load says which accelerator it
     * chose. `onnxruntime` is the two voices, and is the one that explains a
     * refused graph rather than merely refusing it.
     *
     * Deliberately not this app's Kotlin tags: those reach [EngineLog]
     * directly, and listening for them here would show every line twice — once
     * when it is written and again when logcat catches up.
     */
    private val ALLOWED = listOf(
        "ondevice.llama", "ondevice.sd", "ondevice.whisper", "ondevice.quickjs",
        "ggml", "ggml-opencl", "ggml-cpu", "ggml-backend",
        "onnxruntime", "onnxruntime-genai",
    )

    private const val MIN_LINE = 20
}
