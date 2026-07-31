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

/**
 * Writing artifacts to a folder the user picked, and remembering which one.
 *
 * On Android 11 and later an app cannot write outside its own directories by
 * path at all, whatever it asks for in the manifest. The Storage Access
 * Framework is not one option among several here — it is the only mechanism
 * that exists, which is why every "save" in this app until now quietly landed
 * in app-private storage instead.
 *
 * The folder is asked for once and then persisted. `takePersistableUriPermission`
 * is what makes that survive a reboot; without it the grant dies with the
 * process and the second save of the day would ask again, which is how a
 * perfectly correct SAF integration still manages to feel broken.
 */
class ExportStore(
    private val context: Context,
    private val prefs: AppPrefs,
) {

    /**
     * The remembered folder, or null if there is none this app may still write
     * to.
     *
     * Checked against the live permission list rather than trusted: a user can
     * revoke access, or delete the folder, long after we stored its URI, and a
     * remembered destination that no longer works must read as "ask me again"
     * rather than as a failed save.
     */
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

    /**
     * Copy [exports] into [tree]. Returns where they went, or the first failure.
     *
     * All-or-nothing is deliberately *not* the contract: a batch export of
     * forty artifacts should not lose thirty-nine because one file is gone from
     * disk. The count that comes back is what actually landed.
     */
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

    /**
     * A tree or document URI as something a person can look for.
     *
     * `content://com.android.externalstorage.documents/tree/primary%3ADownload`
     * is accurate and useless. The last path segment after the colon is the
     * part that matches what a file manager shows.
     */
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
