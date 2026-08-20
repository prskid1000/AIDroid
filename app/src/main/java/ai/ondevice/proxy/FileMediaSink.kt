package ai.ondevice.proxy

import java.io.File
import java.security.MessageDigest

/**
 * Where an inbound picture or recording lands.
 *
 * On disk, because the engines take paths and nothing else —
 * `GenerateRequest.imagePaths`, `Transcriber.transcribeFile`, and everything
 * downstream of them. The same discipline `PortType` states for workflow edges:
 * values are paths, never pixels or samples.
 *
 * Content-addressed, and that is the whole reason this is a class rather than a
 * line. A chat client re-sends its entire history every turn, so the same
 * screenshot arrives on turn two, turn three and turn twenty; writing it under
 * a fresh name each time would fill the device and, worse, hand llama.cpp a
 * different path for identical bytes on every turn — which invalidates the
 * prompt cache and re-encodes the image from scratch each time.
 */
class FileMediaSink(private val directory: () -> File) : MediaSink {

    override fun writeBase64(data: String, mediaType: String): String? = runCatching {
        val bytes = android.util.Base64.decode(data, android.util.Base64.DEFAULT)
        writeBytes(bytes, extensionFor(mediaType))
    }.getOrNull()

    override fun writeBytes(bytes: ByteArray, extension: String): String? = runCatching {
        val folder = directory().apply { mkdirs() }
        sweep(folder)
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") { "%02x".format(it) }
            .take(32)
        val file = File(folder, "$digest.$extension")
        if (!file.exists()) file.writeBytes(bytes)
        file.absolutePath
    }.getOrNull()

    /**
     * Sweep on write rather than on a timer.
     *
     * There is no scheduler here and adding one for a scratch folder would be a
     * second thing to keep alive. Writes are the only moment the folder grows,
     * so they are the only moment it needs pruning.
     */
    private fun sweep(folder: File) {
        val cutoff = System.currentTimeMillis() - RETENTION_MILLIS
        runCatching {
            folder.listFiles()?.forEach { file ->
                if (file.isFile && file.lastModified() < cutoff) file.delete()
            }
        }
    }

    private fun extensionFor(mediaType: String): String = when {
        mediaType.endsWith("jpeg") || mediaType.endsWith("jpg") -> "jpg"
        mediaType.endsWith("webp") -> "webp"
        mediaType.endsWith("gif") -> "gif"
        mediaType.endsWith("wav") -> "wav"
        mediaType.endsWith("mpeg") || mediaType.endsWith("mp3") -> "mp3"
        mediaType.endsWith("mp4") -> "mp4"
        mediaType.endsWith("ogg") -> "ogg"
        mediaType.endsWith("flac") -> "flac"
        mediaType.endsWith("webm") -> "webm"
        mediaType.endsWith("m4a") -> "m4a"
        else -> "png"
    }

    private companion object {
        /**
         * A day.
         *
         * Long enough that a conversation running all afternoon keeps re-using
         * the same file rather than re-decoding it, short enough that a week of
         * screenshots does not accumulate on a phone.
         */
        const val RETENTION_MILLIS = 24 * 60 * 60 * 1000L
    }
}
