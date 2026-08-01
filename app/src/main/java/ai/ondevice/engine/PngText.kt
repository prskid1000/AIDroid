package ai.ondevice.engine

import java.io.ByteArrayOutputStream
import java.util.zip.CRC32

/** Writes a `tEXt` chunk into an already-encoded PNG. */
internal object PngText {

    fun withTextChunk(png: ByteArray, keyword: String, value: String): ByteArray {
        val iendIndex = findIendStart(png) ?: return png

        val out = ByteArrayOutputStream(png.size + value.length + 64)
        out.write(png, 0, iendIndex)
        out.write(textChunk(keyword, value))
        out.write(png, iendIndex, png.size - iendIndex)
        return out.toByteArray()
    }

    /** `tEXt` is Latin-1 by specification. */
    private fun textChunk(keyword: String, value: String): ByteArray {
        val payload = ByteArrayOutputStream()
        payload.write(keyword.toByteArray(Charsets.ISO_8859_1))
        payload.write(0)
        payload.write(
            value.map { if (it.code in 32..255) it else '?' }
                .joinToString("")
                .toByteArray(Charsets.ISO_8859_1),
        )
        val data = payload.toByteArray()

        val chunk = ByteArrayOutputStream()
        chunk.write(intBytes(data.size))
        val typed = "tEXt".toByteArray(Charsets.US_ASCII) + data
        chunk.write(typed)
        val crc = CRC32().apply { update(typed) }.value
        chunk.write(intBytes(crc.toInt()))
        return chunk.toByteArray()
    }

    /** The offset of the `IEND` chunk's length field. */
    private fun findIendStart(png: ByteArray): Int? {
        var index = 8 // skip the signature
        while (index + 8 <= png.size) {
            val length = readInt(png, index)
            val type = String(png, index + 4, 4, Charsets.US_ASCII)
            if (type == "IEND") return index
            index += 12 + length // length + type + data + crc
        }
        return null
    }

    private fun readInt(bytes: ByteArray, offset: Int): Int =
        ((bytes[offset].toInt() and 0xFF) shl 24) or
            ((bytes[offset + 1].toInt() and 0xFF) shl 16) or
            ((bytes[offset + 2].toInt() and 0xFF) shl 8) or
            (bytes[offset + 3].toInt() and 0xFF)

    private fun intBytes(value: Int): ByteArray = byteArrayOf(
        (value ushr 24).toByte(),
        (value ushr 16).toByte(),
        (value ushr 8).toByte(),
        value.toByte(),
    )
}
