package ai.ondevice.data

import android.content.Context
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
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
     */
    suspend fun findOrphans(): OrphanReport = withContext(Dispatchers.IO) {
        val records = db.models().getAll()
        val knownPaths = records.map { it.localPath }.toSet()

        val strayFiles = modelsDir().walkTopDown()
            .filter { it.isFile && (it.extension == "gguf" || it.extension == "bin" || it.extension == "onnx") }
            .filter { it.absolutePath !in knownPaths }
            .toList()

        val missingFiles = records.filter { !File(it.localPath).exists() }

        OrphanReport(strayFiles = strayFiles, recordsWithoutFiles = missingFiles)
    }

    suspend fun deleteModel(model: ModelEntity) = withContext(Dispatchers.IO) {
        runCatching { File(model.localPath).delete() }
        runCatching { modelDir(model.id).deleteRecursively() }
        db.models().deleteById(model.id)
        db.benchmarks().clearFor(model.id)
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
