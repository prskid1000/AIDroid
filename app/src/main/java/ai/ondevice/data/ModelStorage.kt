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

    /**
     * A gallery file no earlier picture is already using.
     *
     * Images were named after their seed alone, and a seed is the one thing
     * people reuse on purpose — rerunning 812934177 to compare two settings
     * wrote the second result over the first. Losing the file was the smaller
     * half of it: the database keeps a row per render, so the older row went on
     * pointing at a path that now held the newer picture, and a gallery meant
     * to show a comparison showed the same image twice.
     *
     * The seed stays in the name because it is what makes a file recognisable
     * on disk; a counter is appended only when it has to be.
     */
    fun galleryFile(seed: Long, suffix: String = ""): File =
        uniqueFile(galleryDir(), "$seed$suffix", "png")

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

        // "The file has gone" is a claim about a file that was once there.
        //
        // A row is written when a download *starts*, so between that write and
        // the job row landing beside it there is a moment with a record, no
        // file and nothing in `downloading` to vouch for it — and the Models
        // screen, which is where you are taken after starting a download, runs
        // this on open. The record was reported as an orphan and Clean up
        // deleted it, stranding a part-written multi-gigabyte download with no
        // library row to finish into.
        //
        // `completedAt` closes it for good: it is stamped once, by the
        // downloader, when the last file verifies. A row that has never had it
        // is a download in progress or a download that died, and neither is a
        // file that went missing. What it does leave behind — a row for a job
        // that was cancelled out from under it — is reachable by deleting the
        // model, which is a deliberate act rather than a sweep.
        val missingFiles = records.filter {
            it.id !in downloading && it.completedAt != null && !File(it.localPath).exists()
        }

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

    companion object {
        /**
         * `dir/base.ext`, or the first `base-2`, `base-3`… that is free.
         *
         * Separate from [galleryFile] only so it can be tested against a real
         * temporary directory without an Android Context — the behaviour it
         * guards is "an artifact never lands on a path another artifact is
         * already recorded at", and that is worth an assertion rather than a
         * reading of the code.
         */
        fun uniqueFile(dir: File, base: String, ext: String): File {
            File(dir, "$base.$ext").takeIf { !it.exists() }?.let { return it }
            var n = 2
            while (true) {
                val candidate = File(dir, "$base-$n.$ext")
                if (!candidate.exists()) return candidate
                n++
            }
        }
    }
}

data class OrphanReport(
    val strayFiles: List<File>,
    val recordsWithoutFiles: List<ModelEntity>,
) {
    val hasAny: Boolean get() = strayFiles.isNotEmpty() || recordsWithoutFiles.isNotEmpty()
    val strayBytes: Long get() = strayFiles.sumOf { it.length() }
}
