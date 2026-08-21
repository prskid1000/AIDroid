package ai.ondevice.data.log

import java.io.File

/**
 * A log kept on disk as one line per record, of which only the tail survives.
 *
 * Both logs in this app were rings in memory and nowhere else, which was the
 * right call for what they were for — diagnosis inside one session — and the
 * wrong one for how they are actually used. The interesting question is almost
 * always "what did it do before it stopped", and the process stopping is what
 * emptied the ring: a proxy killed under memory pressure took the only record
 * of the requests leading up to it with it.
 *
 * One line per record rather than one document, because the write that matters
 * is the append: re-encoding two hundred records with their bodies on every
 * token would cost more than serving the request. Every record is JSON, so a
 * body with newlines in it is still exactly one line, and a line that will not
 * parse — a half-written tail after a kill — is dropped rather than taking the
 * file with it.
 *
 * [keep] is the cap and it is the *same* number as the ring in front of it, so
 * there is one answer to "how much is kept" rather than two that disagree. The
 * file is allowed to grow to twice that before it is rewritten: trimming on
 * every append would mean reading and rewriting the whole file per line, and
 * the slack is what makes the common path an append and nothing else.
 */
class LogFile(private val file: File, private val keep: Int) {

    private val lock = Any()

    /** Lines currently in the file. Negative until the file has been looked at. */
    private var count = -1

    /** The tail, oldest first. At most [keep] of them. */
    fun read(): List<String> = synchronized(lock) {
        if (!file.isFile) {
            count = 0
            return emptyList()
        }
        val all = runCatching { file.readLines() }
            .getOrDefault(emptyList())
            .filter { it.isNotBlank() }
        count = all.size
        if (all.size > keep) all.takeLast(keep) else all
    }

    fun append(lines: List<String>) {
        if (lines.isEmpty()) return
        synchronized(lock) {
            runCatching {
                file.parentFile?.mkdirs()
                // Counted before the write, not after. It was after, and
                // `countLines()` then included the lines just appended — which
                // were then added a second time, so the file was believed to be
                // twice its real size and got trimmed at half the threshold.
                if (count < 0) count = countLines()
                file.appendText(lines.joinToString(separator = "\n", postfix = "\n"))
                count += lines.size
                if (count > keep * 2) trim()
            }
        }
    }

    fun clear() {
        synchronized(lock) {
            runCatching { file.delete() }
            count = 0
        }
    }

    /** Called with [lock] held. Keeps the newest [keep] and drops the rest. */
    private fun trim() {
        val all = runCatching { file.readLines() }.getOrNull() ?: return
        val tail = all.filter { it.isNotBlank() }.takeLast(keep)
        // Written beside and moved into place, because a rewrite interrupted
        // half way is the one failure that would lose everything rather than
        // the oldest half — and this runs while the thing being logged is
        // still going on.
        val staging = File(file.parentFile, file.name + ".trim")
        runCatching {
            staging.writeText(tail.joinToString(separator = "\n", postfix = "\n"))
            // `Files.move` rather than `File.renameTo`, because renameTo onto a
            // file that already exists is allowed to fail and does — it returns
            // false rather than throwing, so the trim silently did nothing and
            // the file grew for as long as the app ran.
            java.nio.file.Files.move(
                staging.toPath(),
                file.toPath(),
                java.nio.file.StandardCopyOption.REPLACE_EXISTING,
            )
            count = tail.size
        }.onFailure { staging.delete() }
    }

    private fun countLines(): Int =
        runCatching { file.readLines().count { it.isNotBlank() } }.getOrDefault(0)
}
