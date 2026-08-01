package ai.ondevice.data

import android.content.Context
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.download.Downloader
import ai.ondevice.data.download.toJob
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/** Where files live on disk. */
class ModelStorage(private val context: Context, private val db: OnDeviceDatabase) {

    fun root(): File = context.getExternalFilesDir(null) ?: context.filesDir

    fun modelsDir(): File = File(root(), "models").apply { mkdirs() }

    fun galleryDir(): File = File(root(), "gallery").apply { mkdirs() }

    fun transcriptsDir(): File = File(root(), "transcripts").apply { mkdirs() }

    /** Rendered speech. */
    fun speechDir(): File = File(root(), "speech").apply { mkdirs() }

    /** Images, audio and documents attached to a conversation. */
    fun attachmentsDir(): File = File(root(), "attachments").apply { mkdirs() }

    fun exportsDir(): File = File(root(), "exports").apply { mkdirs() }

    fun modelDir(modelId: String): File = File(modelsDir(), sanitize(modelId)).apply { mkdirs() }

    private fun sanitize(id: String): String =
        id.map { if (it.isLetterOrDigit() || it == '-' || it == '.' || it == '_') it else '_' }
            .joinToString("")

    fun pathFor(modelId: String, filename: String): String =
        File(modelDir(modelId), filename.substringAfterLast('/')).absolutePath

    /** Orphan cleanup, the other direction from the boot sweep: files on disk with no library record, and records whose file has gone. */
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
