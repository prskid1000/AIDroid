package ai.ondevice.speech

import java.io.File
import java.io.OutputStream

/**
 * Mono 16-bit PCM WAV.
 *
 * Kokoro returns floats; every player expects a container. Writing the header
 * by hand rather than pulling in a media library keeps the exported file
 * exactly what it claims to be — 44 bytes of RIFF and then samples — which is
 * also what makes it checkable from a shell.
 */
object WavFile {

    private const val HEADER_BYTES = 44

    fun write(destination: File, samples: FloatArray, sampleRate: Int): File {
        destination.parentFile?.mkdirs()
        destination.outputStream().buffered().use { out ->
            writeHeader(out, samples.size, sampleRate)
            val frame = ByteArray(2)
            samples.forEach { sample ->
                // Clamp before scaling: Kokoro occasionally overshoots ±1 on
                // plosives, and letting that wrap turns a loud consonant into a
                // click of the opposite sign.
                val clamped = sample.coerceIn(-1f, 1f)
                val value = (clamped * Short.MAX_VALUE).toInt()
                frame[0] = (value and 0xFF).toByte()
                frame[1] = ((value shr 8) and 0xFF).toByte()
                out.write(frame)
            }
        }
        return destination
    }

    /** What is actually in a WAV, for a file this app may not have written. */
    data class Info(val sampleRate: Int, val frames: Long, val millis: Long)

    /**
     * Read the rate and length back off disk.
     *
     * The alternative was to assume 24 kHz because that is what the two neural
     * engines produce — but the system engine writes its own file at whatever
     * rate it likes, and a duration derived from the wrong rate is a number that
     * looks right and is not. Walks the chunk list rather than trusting the
     * 44-byte layout, because only files this object wrote are guaranteed to
     * have it.
     */
    fun describe(file: File): Info? = runCatching {
        file.inputStream().buffered().use { input ->
            val header = ByteArray(12)
            if (input.read(header) != 12) return null
            if (String(header, 0, 4, Charsets.US_ASCII) != "RIFF" ||
                String(header, 8, 4, Charsets.US_ASCII) != "WAVE"
            ) {
                return null
            }

            var sampleRate = 0
            var bitsPerSample = 16
            var channels = 1
            val chunk = ByteArray(8)
            while (input.read(chunk) == 8) {
                val id = String(chunk, 0, 4, Charsets.US_ASCII)
                val size = le32(chunk, 4)
                if (size < 0) return null
                when (id) {
                    "fmt " -> {
                        val fmt = ByteArray(size)
                        if (input.read(fmt) != size) return null
                        channels = le16(fmt, 2).coerceAtLeast(1)
                        sampleRate = le32(fmt, 4)
                        bitsPerSample = le16(fmt, 14).coerceAtLeast(8)
                    }
                    "data" -> {
                        if (sampleRate <= 0) return null
                        val bytesPerFrame = (bitsPerSample / 8) * channels
                        val frames = size.toLong() / bytesPerFrame.coerceAtLeast(1)
                        return Info(sampleRate, frames, frames * 1000L / sampleRate)
                    }
                    else -> if (input.skip(size.toLong()) != size.toLong()) return null
                }
                // Chunks are word-aligned; an odd size carries a pad byte.
                if (size % 2 == 1) input.read()
            }
            null
        }
    }.getOrNull()

    private fun le16(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xFF) or ((bytes[offset + 1].toInt() and 0xFF) shl 8)

    private fun le32(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xFF) or
            ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
            ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
            ((bytes[offset + 3].toInt() and 0xFF) shl 24)

    /** Exposed so [WavWriter] can lay down a placeholder and patch it later. */
    internal fun writeHeaderFor(out: OutputStream, sampleCount: Int, sampleRate: Int) =
        writeHeader(out, sampleCount, sampleRate)

    private fun writeHeader(out: OutputStream, sampleCount: Int, sampleRate: Int) {
        val dataBytes = sampleCount * 2
        val byteRate = sampleRate * 2

        fun ascii(value: String) = out.write(value.toByteArray(Charsets.US_ASCII))
        fun int32(value: Int) = out.write(
            byteArrayOf(
                (value and 0xFF).toByte(),
                ((value shr 8) and 0xFF).toByte(),
                ((value shr 16) and 0xFF).toByte(),
                ((value shr 24) and 0xFF).toByte(),
            ),
        )

        fun int16(value: Int) = out.write(
            byteArrayOf((value and 0xFF).toByte(), ((value shr 8) and 0xFF).toByte()),
        )

        ascii("RIFF")
        int32(HEADER_BYTES - 8 + dataBytes)
        ascii("WAVE")
        ascii("fmt ")
        int32(16)      // PCM header size
        int16(1)       // format: PCM
        int16(1)       // channels: mono
        int32(sampleRate)
        int32(byteRate)
        int16(2)       // block align
        int16(16)      // bits per sample
        ascii("data")
        int32(dataBytes)
    }
}

/**
 * A WAV written as the audio arrives, rather than from a finished array.
 *
 * A recording used to exist only as a ten-second rolling window inside the
 * decoder — the samples were read, fed to whisper and dropped, so there was
 * nothing to replay, re-run at a different setting, or export. Keeping them
 * meant either holding the whole take in memory (a half-hour recording is
 * upwards of a hundred megabytes of Float) or streaming it to disk. This
 * streams.
 *
 * The header is written first with a zero length and patched on [close], which
 * is the ordinary way to write a RIFF file whose length nobody knows yet. A
 * take that is never closed — the process dies mid-recording — leaves a file
 * whose header says zero samples, and every player will read that as an empty
 * clip rather than as garbage.
 */
class WavWriter(
    private val destination: java.io.File,
    private val sampleRate: Int,
) : java.io.Closeable {

    private val out = java.io.BufferedOutputStream(java.io.FileOutputStream(destination))
    private var samples = 0
    private var closed = false

    init {
        destination.parentFile?.mkdirs()
        WavFile.writeHeaderFor(out, sampleCount = 0, sampleRate = sampleRate)
    }

    /** Append [count] 16-bit samples, little-endian, as they came off the mic. */
    fun append(buffer: ShortArray, count: Int) {
        if (closed) return
        val bytes = ByteArray(count * 2)
        for (i in 0 until count) {
            val value = buffer[i].toInt()
            bytes[i * 2] = (value and 0xFF).toByte()
            bytes[i * 2 + 1] = ((value shr 8) and 0xFF).toByte()
        }
        out.write(bytes)
        samples += count
    }

    val sampleCount: Int get() = samples

    override fun close() {
        if (closed) return
        closed = true
        runCatching { out.flush(); out.close() }
        // Patch the two lengths the header could not know when it was written.
        runCatching {
            java.io.RandomAccessFile(destination, "rw").use { file ->
                val dataBytes = samples * 2
                file.seek(4)
                file.write(littleEndian(36 + dataBytes))
                file.seek(40)
                file.write(littleEndian(dataBytes))
            }
        }
    }

    private fun littleEndian(value: Int) = byteArrayOf(
        (value and 0xFF).toByte(),
        ((value shr 8) and 0xFF).toByte(),
        ((value shr 16) and 0xFF).toByte(),
        ((value shr 24) and 0xFF).toByte(),
    )
}
