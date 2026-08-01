package ai.ondevice.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.core.content.FileProvider
import ai.ondevice.core.Export

/** The two ways an artifact leaves this app, in one place. */

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

/** Ask for a folder to export into, once. */
@Composable
fun rememberFolderPicker(onPicked: (Uri) -> Unit): () -> Unit {
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocumentTree(),
    ) { uri -> uri?.let(onPicked) }
    // Null start location: let the picker open wherever the user last was,
    // rather than sending them to a directory this app chose for them.
    return { launcher.launch(null) }
}
