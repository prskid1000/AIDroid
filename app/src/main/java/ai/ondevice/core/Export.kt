package ai.ondevice.core

import java.io.File

/**
 * A finished artifact on its way out of the app.
 *
 * Every producer — a conversation, a picture, a rendered WAV, a transcript —
 * returns one of these, and nothing else in the app builds an `Intent` or picks
 * a destination for itself. Before this there were four near-identical
 * `share*` helpers and four export methods, each with its own idea of where a
 * file should go, and all four went to the same place: an app-private
 * `exports/` directory that a phone's file manager cannot browse. A button
 * labelled "Save as WAV" that saves somewhere you cannot reach is not a save.
 *
 * [staged] is the app's own copy, which is what a share sheet reads through
 * FileProvider. Saving copies it to wherever the user chose; the staged file
 * stays, because the library still lists it.
 */
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

/**
 * Where a save went, in words a person can check.
 *
 * A save that reports nothing but success is the failure mode this whole change
 * exists to end: the file has to be findable, and the only way the user knows
 * where to look is if the app says so.
 */
data class SavedTo(val displayPath: String, val count: Int = 1)
