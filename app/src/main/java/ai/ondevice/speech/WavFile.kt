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
