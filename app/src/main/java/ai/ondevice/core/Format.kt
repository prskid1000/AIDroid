package ai.ondevice.core

import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Number and byte formatting, in the shapes the design canvas uses.
 *
 * The canvas is consistent about this: sizes read "2.50 GB", context reads
 * "32K" with the exact token count beside it, throughput reads "14.1 t/s", and
 * the fit arithmetic is shown to two decimals so the addition visibly works out.
 */
object Fmt {

    private const val GB = 1_000_000_000.0
    private const val MB = 1_000_000.0
    private const val KB = 1_000.0

    /** "2.50 GB" / "742 MB" / "1.8 MB" — the canvas' own thresholds. */
    fun bytes(b: Long): String = when {
        b >= GB -> String.format("%.2f GB", b / GB)
        b >= 100 * MB -> String.format("%.0f MB", b / MB)
        b >= MB -> String.format("%.1f MB", b / MB)
        b >= KB -> String.format("%.0f KB", b / KB)
        else -> "$b B"
    }

    /** Always in GB, two decimals — used everywhere the fit sum is shown. */
    fun gb(bytes: Long): String = String.format("%.2f", bytes / GB)

    fun gbFromGb(value: Double): String = String.format("%.2f", value)

    /** "8K" / "32K" / "262K" / "2 048". */
    fun contextLabel(tokens: Int): String = when {
        tokens >= 1024 && tokens % 1024 == 0 -> "${tokens / 1024}K"
        tokens >= 1000 -> "${tokens / 1000}K"
        else -> tokens.toString()
    }

    /** "2 140" — the canvas uses a thin space as the thousands separator. */
    fun grouped(n: Int): String {
        val s = abs(n).toString()
        val out = StringBuilder()
        for ((i, c) in s.withIndex()) {
            if (i > 0 && (s.length - i) % 3 == 0) out.append(' ')
            out.append(c)
        }
        return (if (n < 0) "-" else "") + out
    }

    /** "14.1 t/s". */
    fun tokensPerSecond(v: Float): String = String.format("%.1f t/s", v)

    /** "18.4 MB/s". */
    fun transferRate(bytesPerSecond: Long): String = when {
        bytesPerSecond >= MB -> String.format("%.1f MB/s", bytesPerSecond / MB)
        else -> String.format("%.0f KB/s", bytesPerSecond / KB)
    }

    fun percent(fraction: Float): String = "${(fraction.coerceIn(0f, 1f) * 100).roundToInt()}%"

    /** "18:42" / "00:12.4" — durations, with the canvas' tenths on segments. */
    fun duration(ms: Long, tenths: Boolean = false): String {
        val totalSeconds = ms / 1000.0
        val minutes = (totalSeconds / 60).toInt()
        val seconds = totalSeconds - minutes * 60
        return if (tenths) {
            String.format("%02d:%04.1f", minutes, seconds)
        } else {
            String.format("%02d:%02d", minutes, seconds.toInt())
        }
    }

    /** "4m ago" / "yesterday" / "2h ago" — the canvas' relative-time voice. */
    fun relative(epochMillis: Long, now: Long = System.currentTimeMillis()): String {
        val delta = now - epochMillis
        val minutes = delta / 60_000
        val hours = delta / 3_600_000
        val days = delta / 86_400_000
        return when {
            minutes < 1 -> "just now"
            minutes < 60 -> "${minutes}m ago"
            hours < 24 -> "${hours}h ago"
            days == 1L -> "yesterday"
            days < 30 -> "${days}d ago"
            else -> "${days / 30}mo ago"
        }
    }

    /** "9f2c…41ab" — the truncated hash the checksum-mismatch card shows. */
    fun shortHash(hex: String): String =
        if (hex.length <= 9) hex else "${hex.take(4)}…${hex.takeLast(4)}"

    /** "~52 s left" / "~3 min left". */
    fun eta(seconds: Long): String = when {
        seconds < 90 -> "~$seconds s left"
        seconds < 3600 -> "~${seconds / 60} min left"
        else -> "~${seconds / 3600} h left"
    }
}
