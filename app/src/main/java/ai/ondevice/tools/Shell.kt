package ai.ondevice.tools

import ai.ondevice.engine.ToolSpec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.util.concurrent.TimeUnit

/**
 * One command the model ran, kept so the Tools screen can show what happened.
 */
data class ShellRun(
    val command: String,
    val exitCode: Int,
    val millis: Long,
    val at: Long,
    /** The first line or so of what came back, for the list. */
    val summary: String,
)

/**
 * What the model has run, most recent first.
 *
 * An object rather than a field on the provider because the provider is rebuilt
 * for every turn — a log that lived on it would be empty by the time anyone
 * looked. Bounded, because this is a phone and nobody scrolls back 500 commands.
 */
object ShellLog {
    private val _runs = MutableStateFlow<List<ShellRun>>(emptyList())
    val runs: StateFlow<List<ShellRun>> = _runs.asStateFlow()

    fun record(run: ShellRun) {
        _runs.value = (listOf(run) + _runs.value).take(LIMIT)
    }

    fun clear() {
        _runs.value = emptyList()
    }

    private const val LIMIT = 200
}

/**
 * A shell, for the model.
 *
 * Android has no bash. What it has is mksh at `/system/bin/sh` and toybox,
 * which between them cover roughly 210 applets — sed, grep, find, cut, sort,
 * tr, xargs, tar, diff, patch, stat — and notably do not include awk, python or
 * node. The tool description below says so, because a model that assumes a GNU
 * userland writes commands that fail for reasons it cannot see.
 *
 * The other platform rule worth knowing is W^X: since API 29 an app may not
 * execute a file it wrote itself, so `chmod +x script.sh && ./script.sh` is
 * refused by the kernel. Running the interpreter over the script —
 * `sh script.sh` — is fine, because the executable is the system's and the
 * script is only ever read. That is the documented way to run something you
 * just wrote, and the description says that too.
 */
class ShellToolProvider(
    private val workspace: Workspace,
    private val settings: ToolSettings = ToolSettings.EMPTY,
) : ToolProvider {

    override val id: String = ID

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    override fun settings(): List<ai.ondevice.params.ParamSpec> = listOf(
        ToolSettings.int(
            "run_shell", "default_timeout", DEFAULT_TIMEOUT_SECONDS, 1, MAX_TIMEOUT_SECONDS,
            label = "Default timeout",
            help = "Seconds a command gets when it does not ask for a number itself.",
        ),
        ToolSettings.int(
            "run_shell", "max_timeout", MAX_TIMEOUT_SECONDS, 5, 600,
            label = "Longest timeout",
            help = "The ceiling the model cannot ask past. A command still running is killed.",
        ),
        ToolSettings.int(
            "run_shell", "max_output_kb", MAX_OUTPUT_CHARS / 1024, 1, 256,
            label = "Output kept",
            help = "Output beyond this is cut, with a note saying so. Raising it lets one " +
                "chatty command crowd out the conversation.",
        ),
    )

    private val defaultTimeout get() = settings.int("run_shell.default_timeout", DEFAULT_TIMEOUT_SECONDS)
    private val maxTimeout get() = settings.int("run_shell.max_timeout", MAX_TIMEOUT_SECONDS)
    private val maxOutputChars get() = settings.int("run_shell.max_output_kb", MAX_OUTPUT_CHARS / 1024) * 1024

    override suspend fun specs(): List<ToolSpec> = listOf(
        ToolSpec(
            name = "run_shell",
            description = "Run a shell command in the workspace and return its output. " +
                "This is Android: the shell is mksh and the tools are toybox — sed, grep, find, " +
                "cut, sort, tr, xargs, tar, diff, wc, stat and about 200 more. There is no awk, " +
                "no python and no node, and no package manager to add them. " +
                "To run a script you wrote, use \"sh script.sh\" — the system forbids executing " +
                "a file this app created, so chmod +x will not work. " +
                "stdout and stderr come back together, with the exit code.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "command": {
                      "type": "string",
                      "description": "e.g. ls -la, or grep -rn TODO . | head -20"
                    },
                    "timeout_seconds": {
                      "type": "integer",
                      "description": "How long to allow, 1 to $MAX_TIMEOUT_SECONDS. Default $DEFAULT_TIMEOUT_SECONDS."
                    }
                  },
                  "required": ["command"]
                }
            """.trimIndent(),
        ),
    )

    override suspend fun call(name: String, argumentsJson: String): ToolResult =
        withContext(Dispatchers.IO) {
            if (name != "run_shell") return@withContext fail("No shell tool named \"$name\".")

            val args = runCatching { json.parseToJsonElement(argumentsJson).jsonObject }.getOrNull()
            val command = args?.get("command")?.jsonPrimitive?.content?.trim()
            if (command.isNullOrEmpty()) return@withContext fail("run_shell needs a \"command\".")

            val timeout = (args["timeout_seconds"]?.jsonPrimitive?.content?.toIntOrNull()
                ?: defaultTimeout).coerceIn(1, maxTimeout)

            run(command, timeout)
        }

    private fun run(command: String, timeoutSeconds: Int): ToolResult {
        val started = System.currentTimeMillis()
        val process = runCatching {
            ProcessBuilder(SHELL, "-c", command)
                .directory(workspace.root)
                .redirectErrorStream(true)
                .apply {
                    // A shell with no HOME writes its history to / and fails;
                    // TMPDIR keeps mktemp inside the sandbox rather than in a
                    // /tmp this app cannot write to.
                    environment()["HOME"] = workspace.root.absolutePath
                    environment()["TMPDIR"] = workspace.root.absolutePath
                    environment()["PATH"] = "/system/bin:/system/xbin"
                }
                .start()
        }.getOrElse {
            return fail("The shell could not start: ${it.message}")
        }

        val output = StringBuilder()
        // Drained on this thread while the process runs. A pipe that fills up
        // blocks the child forever, so waiting first and reading after is a
        // deadlock for any command that prints more than the buffer holds.
        val reader = Thread {
            runCatching {
                process.inputStream.bufferedReader().forEachLine { line ->
                    synchronized(output) {
                        if (output.length < maxOutputChars) output.append(line).append('\n')
                    }
                }
            }
        }.apply { isDaemon = true; start() }

        val finished = runCatching {
            process.waitFor(timeoutSeconds.toLong(), TimeUnit.SECONDS)
        }.getOrDefault(false)

        if (!finished) {
            process.destroyForcibly()
            reader.join(READER_GRACE_MILLIS)
            val elapsed = System.currentTimeMillis() - started
            ShellLog.record(ShellRun(command, TIMED_OUT, elapsed, started, "timed out"))
            return fail(
                "The command was still running after ${timeoutSeconds}s and was stopped.\n" +
                    trimmed(output) +
                    "\nAsk for a longer timeout_seconds, or narrow what it does.",
            )
        }

        reader.join(READER_GRACE_MILLIS)
        val exit = process.exitValue()
        val elapsed = System.currentTimeMillis() - started
        val text = trimmed(output)

        ShellLog.record(
            ShellRun(
                command = command,
                exitCode = exit,
                millis = elapsed,
                at = started,
                summary = text.lineSequence().firstOrNull { it.isNotBlank() }?.take(120).orEmpty(),
            ),
        )
        android.util.Log.i(TAG, "$ $command → exit $exit in ${elapsed}ms")

        // A non-zero exit is reported as an error so the model stops and reads
        // it, but the output still goes back — the message on stderr is almost
        // always the thing that explains the code.
        val body = buildString {
            if (text.isBlank()) appendLine("(no output)") else appendLine(text)
            if (exit != 0) append("exit code $exit")
        }.trim()
        return if (exit == 0) ok(body) else fail(body)
    }

    private fun trimmed(output: StringBuilder): String {
        val text = synchronized(output) { output.toString() }.trimEnd()
        return if (text.length >= MAX_OUTPUT_CHARS) {
            text.take(MAX_OUTPUT_CHARS) + "\n… output cut at ${MAX_OUTPUT_CHARS / 1024} KB. " +
                "Pipe it through head, tail or grep."
        } else {
            text
        }
    }

    private fun ok(text: String) = ToolResult(text, providerId = ID)
    private fun fail(text: String) = ToolResult(text, isError = true, providerId = ID)

    companion object {
        const val ID = "shell"

        /** mksh on every Android since 10, and the only shell that is always there. */
        private const val SHELL = "/system/bin/sh"
        private const val TAG = "ShellTool"

        private const val DEFAULT_TIMEOUT_SECONDS = 20
        private const val MAX_TIMEOUT_SECONDS = 120
        private const val MAX_OUTPUT_CHARS = 24 * 1024
        private const val READER_GRACE_MILLIS = 500L

        /** Not a real exit code — no process returns it — so the log can say "timed out". */
        const val TIMED_OUT = -1

        fun toolNames(): List<String> = listOf("run_shell")
    }
}
