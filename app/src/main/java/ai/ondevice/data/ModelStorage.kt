package ai.ondevice.data

import android.content.Context
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.download.Downloader
import ai.ondevice.data.download.toJob
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Where files live on disk.
 *
 * SPEC §13: "All model files user-accessible; nothing in a private opaque
 * store." These are external app directories, browsable in any file manager,
 * which is also why the gallery screen can say the images sit in a normal
 * folder.
 */
class ModelStorage(private val context: Context, private val db: OnDeviceDatabase) {

    fun root(): File = context.getExternalFilesDir(null) ?: context.filesDir

    fun modelsDir(): File = File(root(), "models").apply { mkdirs() }

    fun galleryDir(): File = File(root(), "gallery").apply { mkdirs() }

    fun transcriptsDir(): File = File(root(), "transcripts").apply { mkdirs() }

    /**
     * Rendered speech. Its own folder rather than sharing `transcripts/`, which
     * held the opposite direction of the same feature — audio in, text out —
     * and made the exported WAVs read as inputs.
     */
    fun speechDir(): File = File(root(), "speech").apply { mkdirs() }

    /**
     * Images, audio and documents attached to a conversation.
     *
     * They are *copied* here rather than referenced by content URI: a picker
     * grant dies with the process, and a conversation that renders its
     * attachments only until the next reboot is not a conversation you can
     * keep. It also makes the export in [ConversationArchive] possible at all.
     */
    fun attachmentsDir(): File = File(root(), "attachments").apply { mkdirs() }

    fun exportsDir(): File = File(root(), "exports").apply { mkdirs() }

    /**
     * A model id carries both a repo path and a quant (`owner/repo:Q4_0`), and
     * neither `/` nor `:` survives every filesystem the user might point us at —
     * SPEC §3.4 allows an SD card over SAF, which can be FAT32. So the whole id
     * is reduced to a portable directory name.
     */
    fun modelDir(modelId: String): File = File(modelsDir(), sanitize(modelId)).apply { mkdirs() }

    private fun sanitize(id: String): String =
        id.map { if (it.isLetterOrDigit() || it == '-' || it == '.' || it == '_') it else '_' }
            .joinToString("")

    fun pathFor(modelId: String, filename: String): String =
        File(modelDir(modelId), filename.substringAfterLast('/')).absolutePath

    /**
     * Orphan cleanup, the other direction from the boot sweep: files on disk
     * with no library record, and records whose file has gone.
     *
     * A model is a *directory*, not a file. Matching only against each record's
     * `localPath` treats every companion as a stray, and companions are not
     * exotic: a Kokoro install is one graph plus 55 voice packs, so the screen
     * offered to "clean up" 55 orphans and one tap would have left Kokoro
     * installed, listed, selectable — and unable to speak in any voice, failing
     * at synthesis time with a missing pack rather than at the moment the files
     * were destroyed. Anything inside a known model's own directory belongs to
     * that model whether or not it is the file the record happens to name.
     */
    suspend fun findOrphans(): OrphanReport = withContext(Dispatchers.IO) {
        val records = db.models().getAll()
        val knownPaths = records.map { it.localPath }.toSet()
        // A download in flight owns everything it has written so far, including
        // the record that has no file yet.
        val downloading = db.models().pendingModelIds().toSet()
        val livePartFiles = db.downloads().getUnfinished()
            .flatMap { job -> job.toJob().files.map { it.destPath + Downloader.PART_SUFFIX } }
            .toSet()
        // Canonical, because the walk yields canonical paths and a record could
        // have been written with a symlinked or differently-cased parent.
        val ownedDirs = records
            .mapNotNull { runCatching { modelDir(it.id).canonicalPath }.getOrNull() }
            .toSet()

        fun isOwned(file: File): Boolean {
            if (file.absolutePath in knownPaths) return true
            var parent = runCatching { file.canonicalFile.parentFile }.getOrNull()
            while (parent != null) {
                if (parent.path in ownedDirs) return true
                parent = parent.parentFile
            }
            return false
        }

        // Every format, not an allowlist of three.
        //
        // The old filter was `gguf`, `bin` or `onnx`, which silently excluded
        // most of what this app installs: a diffusion auxiliary is
        // `.safetensors`, an upscaler is `.pth`, and — the expensive one — an
        // ONNX model's weights live in `.onnx.data` sidecars, so an interrupted
        // OmniVoice install left ~700 MB that the report could not see and the
        // storage meter counted anyway. Anything unowned under the models
        // directory is a stray whatever it is named; `isOwned` already keeps a
        // real model's config and tokenizer files out of this.
        val strayFiles = modelsDir().walkTopDown()
            .filter { it.isFile && it.absolutePath !in livePartFiles }
            .filterNot(::isOwned)
            .toList()

        val missingFiles = records.filter { it.id !in downloading && !File(it.localPath).exists() }

        OrphanReport(strayFiles = strayFiles, recordsWithoutFiles = missingFiles)
    }

    suspend fun deleteModel(model: ModelEntity) = withContext(Dispatchers.IO) {
        runCatching { File(model.localPath).delete() }
        runCatching { modelDir(model.id).deleteRecursively() }
        db.models().deleteById(model.id)
    }

    fun usedBytes(): Long = runCatching {
        modelsDir().walkTopDown().filter { it.isFile }.sumOf { it.length() }
    }.getOrDefault(0L)
}

data class OrphanReport(
    val strayFiles: List<File>,
    val recordsWithoutFiles: List<ModelEntity>,
) {
    val hasAny: Boolean get() = strayFiles.isNotEmpty() || recordsWithoutFiles.isNotEmpty()
    val strayBytes: Long get() = strayFiles.sumOf { it.length() }
}
