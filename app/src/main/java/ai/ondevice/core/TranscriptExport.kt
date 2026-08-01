package ai.ondevice.core

/** SPEC §6.5 — a transcript leaves the app in the formats the rest of the world already reads. */
@kotlinx.serialization.Serializable
data class TranscriptSegment(
    val startMillis: Long,
    val endMillis: Long,
    val text: String,
    val confidence: Float = 1f,
)

/** How a transcript is stored, and how to read one back. */
object TranscriptSegments {

    private val json = kotlinx.serialization.json.Json { ignoreUnknownKeys = true; isLenient = true }
    private val listSerializer =
        kotlinx.serialization.builtins.ListSerializer(TranscriptSegment.serializer())

    fun encode(segments: List<TranscriptSegment>): String =
        json.encodeToString(listSerializer, segments)

    fun parse(raw: String?): List<TranscriptSegment> {
        if (raw.isNullOrBlank()) return emptyList()
        // The current shape first.
        runCatching { return json.decodeFromString(listSerializer, raw) }
        // Then the old one: a sparse object holding an array of plain strings.
        return SparseParams.parse(raw).stringList("segments").orEmpty()
            .map { TranscriptSegment(startMillis = 0, endMillis = 0, text = it) }
    }
}

enum class TranscriptFormat(val label: String, val extension: String, val mime: String) {
    TXT("TXT", "txt", "text/plain"),
    SRT("SRT", "srt", "application/x-subrip"),
    VTT("VTT", "vtt", "text/vtt"),
    JSON("JSON", "json", "application/json"),
}

object TranscriptExport {

    fun render(
        format: TranscriptFormat,
        segments: List<TranscriptSegment>,
        title: String,
        modelId: String? = null,
    ): String = when (format) {
        TranscriptFormat.TXT -> segments.joinToString("\n") { it.text }

        // SRT indexes from 1 and separates the clock fields with a comma.
        TranscriptFormat.SRT -> segments.mapIndexed { index, s ->
            "${index + 1}\n${clock(s.startMillis, ',')} --> ${clock(s.endMillis, ',')}\n${s.text}"
        }.joinToString("\n\n", postfix = "\n")

        // WebVTT is the same shape with a header and a full stop.
        TranscriptFormat.VTT -> buildString {
            append("WEBVTT\n\n")
            segments.forEach { s ->
                append("${clock(s.startMillis, '.')} --> ${clock(s.endMillis, '.')}\n${s.text}\n\n")
            }
        }

        TranscriptFormat.JSON -> buildString {
            append("{\n")
            append("  \"title\": ${quote(title)},\n")
            append("  \"model\": ${modelId?.let { quote(it) } ?: "null"},\n")
            append("  \"segments\": [\n")
            segments.forEachIndexed { index, s ->
                append("    { \"start\": ${s.startMillis / 1000.0}, \"end\": ${s.endMillis / 1000.0}, ")
                append("\"confidence\": ${s.confidence}, \"text\": ${quote(s.text)} }")
                append(if (index == segments.lastIndex) "\n" else ",\n")
            }
            append("  ]\n}\n")
        }
    }

    /** `HH:MM:SS,mmm` for SRT, `HH:MM:SS.mmm` for VTT. */
    private fun clock(millis: Long, decimal: Char): String {
        val hours = millis / 3_600_000
        val minutes = (millis % 3_600_000) / 60_000
        val seconds = (millis % 60_000) / 1000
        val remainder = millis % 1000
        return String.format("%02d:%02d:%02d%c%03d", hours, minutes, seconds, decimal, remainder)
    }

    private fun quote(value: String): String = buildString {
        append('"')
        value.forEach { char ->
            when (char) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (char < ' ') append(String.format("\\u%04x", char.code)) else append(char)
            }
        }
        append('"')
    }
}
