package ai.ondevice.data.hf

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject

/** Tensor names, read from the front of a safetensors file. */
object SafetensorsHeaderReader {

    /** Enough for the header of any auxiliary; a full SD VAE's is ~50 KB. */
    const val HEADER_BYTES: Int = 256 * 1024

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    data class Header(val tensorNames: Set<String>) {
        fun hasPrefix(prefix: String): Boolean = tensorNames.any { it.startsWith(prefix) }
    }

    fun parse(bytes: ByteArray): Header? {
        if (bytes.size < 8) return null
        var length = 0L
        for (i in 7 downTo 0) length = (length shl 8) or (bytes[i].toLong() and 0xFF)
        if (length <= 0 || length > Int.MAX_VALUE) return null
        val end = 8 + length.toInt()
        if (end > bytes.size) return null

        val text = String(bytes, 8, length.toInt(), Charsets.UTF_8)
        val names = runCatching {
            json.parseToJsonElement(text).jsonObject.keys.filterNot { it == "__metadata__" }
        }.getOrNull() ?: return null
        return Header(names.toSet()).takeIf { it.tensorNames.isNotEmpty() }
    }
}
