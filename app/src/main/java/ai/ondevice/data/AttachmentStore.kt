package ai.ondevice.data

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

/**
 * Files the user attaches to a conversation.
 *
 * Everything is **copied** into the app's own directory, never referenced by
 * content URI. Two reasons, and both are the kind of thing that only bites
 * later: a picker's read grant dies with the process, so a conversation would
 * lose its images on the next launch; and an export (see [ConversationArchive])
 * cannot bundle a file it has no durable handle to.
 *
 * The directory is under `getExternalFilesDir`, so SPEC §13 still holds — these
 * are ordinary files the user can open in any file manager, not a private store.
 */
class AttachmentStore(
    private val context: Context,
    private val storage: ModelStorage,
) {

    suspend fun copyIn(uri: Uri): Attachment? = withContext(Dispatchers.IO) {
        val resolver = context.contentResolver
        val displayName = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
            ?: uri.lastPathSegment?.substringAfterLast('/')
            ?: "attachment"

        val mime = resolver.getType(uri).orEmpty()
        // The id prefix keeps two files called "photo.jpg" apart without
        // renaming them past recognition.
        val target = File(storage.attachmentsDir(), "${UUID.randomUUID().toString().take(8)}-${sanitize(displayName)}")

        runCatching {
            resolver.openInputStream(uri)?.use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            } ?: return@withContext null
        }.getOrElse { return@withContext null }

        Attachment(
            path = target.absolutePath,
            displayName = displayName,
            mimeType = mime,
            sizeBytes = target.length(),
            kind = AttachmentKind.of(mime, displayName),
        )
    }

    /**
     * Pull readable text out of a document so it can go into the prompt.
     *
     * Plain text and the text-ish formats are read directly. A PDF is *not*
     * parsed here: doing it properly needs a real layout-aware extractor, and a
     * naive one silently produces scrambled column order that the model then
     * confidently answers questions about. Saying so is better than that.
     */
    suspend fun extractText(attachment: Attachment): TextExtraction = withContext(Dispatchers.IO) {
        val file = File(attachment.path)
        if (!file.exists()) {
            return@withContext TextExtraction(error = "That file is no longer there.")
        }
        if (file.length() > MAX_TEXT_BYTES) {
            return@withContext TextExtraction(
                error = "${attachment.displayName} is ${file.length() / 1024} kB. " +
                    "Only the first ${MAX_TEXT_BYTES / 1024} kB was read.",
                text = file.inputStream().use { input ->
                    String(input.readNBytes(MAX_TEXT_BYTES.toInt()), Charsets.UTF_8)
                },
            )
        }

        when {
            attachment.isTextual -> TextExtraction(text = file.readText())
            attachment.mimeType == "application/pdf" -> TextExtraction(
                error = "PDF text extraction is not implemented. Attach the text, or a screenshot " +
                    "of the page if the model can see images.",
            )
            else -> TextExtraction(
                error = "${attachment.displayName} is ${attachment.mimeType.ifBlank { "an unknown type" }}, " +
                    "which this build cannot read as text.",
            )
        }
    }

    private fun sanitize(name: String): String =
        name.map { if (it.isLetterOrDigit() || it == '-' || it == '.' || it == '_') it else '_' }
            .joinToString("")
            .take(64)

    private companion object {
        const val MAX_TEXT_BYTES = 512L * 1024
    }
}

data class Attachment(
    val path: String,
    val displayName: String,
    val mimeType: String,
    val sizeBytes: Long,
    val kind: AttachmentKind,
) {
    val isTextual: Boolean
        get() = mimeType.startsWith("text/") ||
            mimeType in setOf(
                "application/json", "application/xml", "application/x-yaml",
                "application/javascript", "application/x-sh",
            ) ||
            displayName.substringAfterLast('.', "").lowercase() in TEXT_EXTENSIONS

    private companion object {
        val TEXT_EXTENSIONS = setOf(
            "txt", "md", "markdown", "json", "yaml", "yml", "xml", "csv", "tsv", "log",
            "kt", "java", "py", "js", "ts", "tsx", "c", "cpp", "h", "hpp", "rs", "go",
            "sh", "toml", "ini", "cfg", "gradle", "properties", "sql", "html", "css",
        )
    }
}

data class TextExtraction(val text: String = "", val error: String? = null)

enum class AttachmentKind {
    IMAGE, AUDIO, DOCUMENT;

    companion object {
        fun of(mime: String, name: String): AttachmentKind = when {
            mime.startsWith("image/") -> IMAGE
            mime.startsWith("audio/") || mime.startsWith("video/") -> AUDIO
            name.substringAfterLast('.', "").lowercase() in setOf("jpg", "jpeg", "png", "webp", "gif") -> IMAGE
            name.substringAfterLast('.', "").lowercase() in setOf("wav", "mp3", "m4a", "ogg", "flac") -> AUDIO
            else -> DOCUMENT
        }
    }
}
