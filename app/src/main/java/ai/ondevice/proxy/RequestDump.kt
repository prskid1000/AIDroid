package ai.ondevice.proxy

import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * The request and its answer, written down, when `proxy.debug` says so.
 *
 * The in-memory log already keeps both bodies and is the right place to read
 * them from — this exists for the one thing a ring in RAM cannot do, which is
 * survive the process. A generation that ends with the app being killed takes
 * its own evidence with it, and that is exactly the run somebody wants to look
 * at afterwards.
 *
 * Off by default, and the row that turns it on says plainly that it writes
 * prompts to disk. Telecode does the same thing under `data/logs/`, clears the
 * previous run's dumps at startup for the same reason, and keeps the switch
 * off for the same one.
 */
class RequestDump(private val directory: () -> File) {

    /**
     * Clear what the last run left.
     *
     * At startup rather than on a timer: these are worth exactly as long as the
     * session that produced them, and a folder that only grows is a folder
     * somebody eventually finds full of prompts they had forgotten about.
     */
    fun clearPrevious(): Int = runCatching {
        val folder = directory()
        val files = folder.listFiles { file -> file.name.startsWith(PREFIX) }.orEmpty()
        files.count { it.delete() }
    }.getOrDefault(0)

    /**
     * One request, once it has finished.
     *
     * Written whole rather than appended to as it goes: a half-written dump of
     * a run that was killed is worse than none, because it reads as a complete
     * record of a shorter run.
     */
    fun write(record: RequestRecord) = runCatching {
        val folder = directory().apply { mkdirs() }
        sweep(folder)

        val stamp = STAMP.format(Date(record.startedAt))
        val file = File(folder, "$PREFIX$stamp-${record.id.take(8)}.json")

        file.writeText(
            buildString {
                appendLine("{")
                appendLine("""  "at": "$stamp",""")
                appendLine("""  "method": ${quote(record.method)},""")
                appendLine("""  "path": ${quote(record.path)},""")
                appendLine("""  "protocol": ${quote(record.protocol.name.lowercase())},""")
                appendLine("""  "client": ${quote(record.client)},""")
                appendLine("""  "phase": ${quote(record.phase)},""")
                appendLine("""  "requested_model": ${quote(record.requestedModel)},""")
                appendLine("""  "resolved_model": ${quote(record.resolvedModel)},""")
                appendLine("""  "status": ${record.status},""")
                appendLine("""  "duration_ms": ${record.durationMillis},""")
                appendLine("""  "streaming": ${record.streaming},""")
                appendLine("""  "frames": ${record.frames},""")
                appendLine("""  "rounds": ${record.rounds},""")
                appendLine("""  "prompt_tokens": ${record.promptTokens},""")
                appendLine("""  "generated_tokens": ${record.generatedTokens},""")
                appendLine("""  "tokens_per_second": ${record.tokensPerSecond},""")
                record.error?.let { appendLine("""  "error": ${quote(it)},""") }
                appendLine("""  "intercepts": [""")
                record.intercepts.forEachIndexed { index, intercept ->
                    val comma = if (index == record.intercepts.lastIndex) "" else ","
                    appendLine(
                        """    {"kind": ${quote(intercept.kind.name)}, """ +
                            """"name": ${quote(intercept.name)}, """ +
                            """"detail": ${quote(intercept.detail)}}$comma""",
                    )
                }
                appendLine("  ],")
                // The bodies are already redacted and capped by the log — see
                // RequestLog.forDisplay. Base64 is not written to disk here for
                // the same reason it is not kept in memory: a screenshot is four
                // megabytes of unreadable text, and twenty of them is a folder
                // nobody wanted.
                appendLine("""  "request": ${quote(record.requestBody)},""")
                appendLine("""  "response": ${quote(record.responseBody)}""")
                append("}")
            },
        )
        file
    }.getOrNull()

    /** Newest kept, oldest dropped. A session should not fill the device. */
    private fun sweep(folder: File) {
        runCatching {
            val files = folder.listFiles { file -> file.name.startsWith(PREFIX) }.orEmpty()
            if (files.size < MAX_FILES) return
            files.sortedBy { it.lastModified() }
                .take(files.size - MAX_FILES + 1)
                .forEach { it.delete() }
        }
    }

    /** Enough for the JSON to be valid; the bodies are the only risky part. */
    private fun quote(value: String): String = buildString {
        append('"')
        value.forEach { c ->
            when (c) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (c < ' ') append("\\u%04x".format(c.code)) else append(c)
            }
        }
        append('"')
    }

    private companion object {
        const val PREFIX = "proxy_full_"
        const val MAX_FILES = 200
        val STAMP = SimpleDateFormat("yyyyMMdd-HHmmss.SSS", Locale.UK)
    }
}
