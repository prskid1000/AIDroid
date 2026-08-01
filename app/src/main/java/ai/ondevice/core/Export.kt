package ai.ondevice.core

import java.io.File

/** A finished artifact on its way out of the app. */
data class Export(
    val staged: File,
    /** What to call it in the user's folder; SAF makes it unique if it collides. */
    val suggestedName: String,
    val mime: String,
) {
    constructor(staged: File, mime: String) : this(staged, staged.name, mime)

    companion object {
        const val MIME_MARKDOWN = "text/markdown"
        const val MIME_ZIP = "application/zip"
        const val MIME_PNG = "image/png"
        const val MIME_WAV = "audio/wav"
        const val MIME_TEXT = "text/plain"
        const val MIME_JSON = "application/json"

        /** The MIME an extension implies, for the formats this app produces. */
        fun mimeFor(name: String): String = when (name.substringAfterLast('.', "").lowercase()) {
            "md" -> MIME_MARKDOWN
            "zip" -> MIME_ZIP
            "png" -> MIME_PNG
            "wav" -> MIME_WAV
            "json" -> MIME_JSON
            // srt and vtt have registered types, but no Android app matches on
            // them and a share sheet with no targets is worse than a plain one.
            else -> MIME_TEXT
        }
    }
}

/** Where a save went, in words a person can check. */
data class SavedTo(val displayPath: String, val count: Int = 1)
