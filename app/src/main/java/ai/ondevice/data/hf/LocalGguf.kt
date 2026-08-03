package ai.ondevice.data.hf

import java.io.File

/**
 * What a GGUF already on the device says about itself.
 *
 * The resolver reads a header over the network before a download; this reads
 * one off the disk afterwards, for the questions that only come up once a file
 * is installed — which T5 is this, and so is it the one this checkpoint reads.
 *
 * Off the file rather than out of a column, because a column has to be filled
 * in by whatever wrote the row, and the rows that predate the column stay null
 * for ever. The file is the same file either way and it has always known.
 *
 * Cached by path and modification time: an attachment list is rebuilt on every
 * refresh, and 1 MB per candidate per refresh is a lot of reading to answer a
 * question whose answer cannot change unless the file does.
 */
object LocalGguf {

    private data class Stamp(val length: Long, val modified: Long)

    private val cache = HashMap<String, Pair<Stamp, GgufMetadata?>>()

    /** The header, parsed, or null when the file is absent or is not a GGUF. */
    fun metadata(path: String): GgufMetadata? {
        val file = File(path)
        if (!file.isFile) return null
        val stamp = Stamp(file.length(), file.lastModified())
        synchronized(cache) {
            cache[path]?.let { (seen, value) -> if (seen == stamp) return value }
        }
        val parsed = runCatching {
            file.inputStream().use { GgufHeaderReader.parse(it).getOrNull() }
        }.getOrNull()
        synchronized(cache) { cache[path] = stamp to parsed }
        return parsed
    }

    /** How many tokens this file's tokenizer holds, or null. */
    fun vocabSize(path: String): Int? = metadata(path)?.vocabSize
}
