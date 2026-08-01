package ai.ondevice.data

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import ai.ondevice.core.Export
import ai.ondevice.core.SavedTo
import ai.ondevice.data.prefs.AppPrefs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext

/** Writing artifacts to a folder the user picked, and remembering which one. */
class ExportStore(
    private val context: Context,
    private val prefs: AppPrefs,
) {

    /** The remembered folder, or null if there is none this app may still write to. */
    suspend fun folder(): Uri? {
        val stored = prefs.exportFolder.first()?.let(Uri::parse) ?: return null
        val held = context.contentResolver.persistedUriPermissions
            .any { it.uri == stored && it.isWritePermission }
        return stored.takeIf { held }
    }

    /** Remember a tree the user just granted, and hold onto it across reboots. */
    suspend fun remember(tree: Uri) {
        runCatching {
            context.contentResolver.takePersistableUriPermission(
                tree,
                android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION or
                    android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
            )
        }
        prefs.setExportFolder(tree.toString())
    }

    /** A name for the chosen folder that means something on screen. */
    suspend fun folderLabel(): String? = folder()?.let { readableName(it) }

    /** Copy [exports] into [tree]. */
    suspend fun save(exports: List<Export>, tree: Uri): Result<SavedTo> =
        withContext(Dispatchers.IO) {
            runCatching {
                val parent = DocumentsContract.buildDocumentUriUsingTree(
                    tree,
                    DocumentsContract.getTreeDocumentId(tree),
                )
                var written = 0
                var lastError: Throwable? = null
                exports.forEach { export ->
                    runCatching { writeOne(export, parent) }
                        .onSuccess { written++ }
                        .onFailure { lastError = it }
                }
                if (written == 0) {
                    throw lastError ?: IllegalStateException("There was nothing to export.")
                }
                SavedTo(displayPath = readableName(tree), count = written)
            }
        }

    /** Write one export to a document URI the user named themselves. */
    suspend fun saveAs(export: Export, document: Uri): Result<SavedTo> =
        withContext(Dispatchers.IO) {
            runCatching {
                copyInto(export, document)
                SavedTo(displayPath = readableName(document))
            }
        }

    private fun writeOne(export: Export, parent: Uri) {
        check(export.staged.isFile) {
            "${export.suggestedName} is listed but its file is gone from disk."
        }
        val target = DocumentsContract.createDocument(
            context.contentResolver,
            parent,
            export.mime,
            export.suggestedName,
        ) ?: error("${export.suggestedName} could not be created in that folder.")
        copyInto(export, target)
    }

    private fun copyInto(export: Export, target: Uri) {
        context.contentResolver.openOutputStream(target)?.use { out ->
            export.staged.inputStream().use { it.copyTo(out) }
        } ?: error("That folder would not accept ${export.suggestedName}.")
    }

    /** A tree or document URI as something a person can look for. */
    private fun readableName(uri: Uri): String {
        val decoded = Uri.decode(uri.toString())
        val tail = decoded.substringAfterLast(':', "").trim('/')
        return when {
            tail.isBlank() -> decoded.substringAfterLast('/')
            tail.equals("primary", ignoreCase = true) -> "Internal storage"
            else -> tail
        }
    }
}
