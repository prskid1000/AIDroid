package ai.ondevice.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.core.content.FileProvider
import ai.ondevice.core.Export

/**
 * The two ways an artifact leaves this app, in one place.
 *
 * There used to be four copies of the share code — `shareFile`, `shareImage`,
 * `shareAudio`, `shareTranscript` — differing only in the chooser title, and
 * every screen that produced anything carried its own. Four copies of a thing
 * is four places for it to drift, and it had: the titles disagreed about what
 * was being sent and one of them said "Send conversation" for a WAV.
 */

/** Hand [export] to another app. Not a save: nothing is written where you can find it. */
fun shareExport(context: Context, export: Export) {
    val uri = FileProvider.getUriForFile(
        context,
        "${context.packageName}.fileprovider",
        export.staged,
    )
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = export.mime
        putExtra(Intent.EXTRA_STREAM, uri)
        putExtra(Intent.EXTRA_TITLE, export.suggestedName)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(intent, "Send ${export.suggestedName}"))
}

/**
 * Ask for a folder to export into, once.
 *
 * `OpenDocumentTree` rather than `CreateDocument` because the answer is reused:
 * a per-file picker on every save is how an export feature becomes one nobody
 * uses. The returned lambda opens the picker; the grant is persisted by
 * [ai.ondevice.data.ExportStore].
 */
@Composable
fun rememberFolderPicker(onPicked: (Uri) -> Unit): () -> Unit {
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocumentTree(),
    ) { uri -> uri?.let(onPicked) }
    // Null start location: let the picker open wherever the user last was,
    // rather than sending them to a directory this app chose for them.
    return { launcher.launch(null) }
}
