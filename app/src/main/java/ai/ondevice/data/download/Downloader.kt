package ai.ondevice.data.download

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import ai.ondevice.core.DownloadState
import ai.ondevice.data.db.DownloadJobEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.data.secure.TokenStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.RandomAccessFile
import java.security.MessageDigest
import kotlin.math.min
import kotlin.math.pow

/** SPEC §3.4 — the downloader. */
class Downloader(
    private val context: Context,
    private val client: OkHttpClient,
    private val db: OnDeviceDatabase,
    private val prefs: AppPrefs,
    private val tokens: TokenStore,
    private val scope: CoroutineScope,
) {
    private val activeJobs = mutableMapOf<String, Job>()
    private val rates = MutableStateFlow<Map<String, Long>>(emptyMap())

    fun observeJobs(): Flow<List<DownloadJob>> = db.downloads().observeAll().map { rows ->
        val currentRates = rates.value
        rows.map { it.toJob(currentRates[it.id] ?: 0) }
    }

    fun observeActiveCount(): Flow<Int> = db.downloads().observeActiveCount()

    suspend fun enqueue(job: DownloadJob) {
        // Adding the same model twice is one download, not two.
        //
        // A job's key is its own id and a fresh one is minted per enqueue, so
        // nothing stopped a second job for the same model existing beside the
        // first — with the same destination paths, the same `.part` files, and
        // two coroutines writing into them at once. Whichever finished second
        // wrote over the first's offsets, and the result was a file that
        // downloaded twice and verified once, if at all.
        //
        // An unfinished job for this model is the download the person is
        // asking for, so this resumes it. A FAILED or COMPLETE one is not
        // unfinished, and re-adding after either still starts a fresh job —
        // which is what makes "add it again" a working repair.
        val existing = db.downloads().forModel(job.modelId)
            .firstOrNull { it.state in UNFINISHED }
        if (existing != null) {
            android.util.Log.i(TAG, "already downloading ${job.modelId}; resuming instead")
            start(existing.id)
            return
        }
        db.downloads().upsert(job.toEntity())
        start(job.id)
    }

    /**
     * Stop and forget every job for a model, and drop what they had written.
     *
     * Deleting a model deleted its row, its file and its folder and left the
     * download running — so the bytes kept arriving into a directory nobody
     * owned, for a library entry that no longer existed. The row came back off
     * the list, the traffic did not, and the only sign was the queue.
     */
    suspend fun cancelForModel(modelId: String) {
        db.downloads().forModel(modelId).forEach { entity ->
            activeJobs.remove(entity.id)?.cancel()
            entity.toJob().files.forEach { file ->
                runCatching { File(file.destPath).delete() }
                runCatching { File(file.destPath + PART_SUFFIX).delete() }
            }
            db.downloads().deleteById(entity.id)
        }
    }

    fun start(jobId: String) {
        if (activeJobs[jobId]?.isActive == true) return
        // Raise the foreground service before the transfer begins.
        //
        // There has been a DownloadService, declared in the manifest with a
        // dataSync type and a notification that names the file and its
        // progress, since the downloads were written — and nothing ever
        // started it. Every caller went to `enqueue` or here directly, so the
        // transfer ran on this class's application scope with no service
        // behind it: alive while the process was, and the process is a ten-
        // gigabyte one that Android reclaims within moments of the app leaving
        // the screen. A download survived switching apps only for as long as
        // the system had no use for the memory, and there was no notification
        // to say it had stopped.
        //
        // Started here rather than at each call site, because this is the one
        // place every download passes through — resume, retry and a fresh
        // enqueue all end up on this line.
        DownloadService.ensureRunning(context)
        activeJobs[jobId] = scope.launch(Dispatchers.IO) { runJob(jobId) }
    }

    /** Pick up jobs whose process died mid-flight. */
    suspend fun resumeInterrupted() {
        val stalled = db.downloads().getActive()
        android.util.Log.i(TAG, "resumeInterrupted: ${stalled.size} stalled job(s)")
        stalled.forEach { entity ->
            if (activeJobs[entity.id]?.isActive == true) return@forEach
            android.util.Log.i(TAG, "resuming ${entity.displayName} at ${entity.bytesDone}/${entity.bytesTotal}")
            start(entity.id)
        }
    }

    fun pause(jobId: String) {
        activeJobs.remove(jobId)?.cancel()
        scope.launch {
            db.downloads().get(jobId)?.let {
                db.downloads().upsert(it.copy(state = DownloadState.PAUSED, updatedAt = now()))
            }
        }
    }

    /** Cancelling removes the partial files — no orphans left behind. */
    fun cancel(jobId: String) {
        activeJobs.remove(jobId)?.cancel()
        scope.launch(Dispatchers.IO) {
            val entity = db.downloads().get(jobId) ?: return@launch
            val job = entity.toJob()
            job.files.forEach { file ->
                runCatching { File(file.destPath).delete() }
                runCatching { File(file.destPath + PART_SUFFIX).delete() }
            }
            db.downloads().deleteById(jobId)

            // The library row goes too, unless the model was already installed.
            val model = db.models().get(job.modelId)
            if (model != null && !File(model.localPath).exists()) {
                db.models().deleteById(job.modelId)
            }
        }
    }

    private suspend fun runJob(jobId: String) {
        var entity = db.downloads().get(jobId) ?: return
        var job = entity.toJob()
        android.util.Log.i(TAG, "runJob ${job.displayName} wifiOnly=${job.wifiOnly} unmetered=${isUnmetered()}")

        if (job.wifiOnly && !isUnmetered()) {
            update(entity.copy(state = DownloadState.PAUSED, error = "Waiting for Wi-Fi", updatedAt = now()))
            return
        }

        update(entity.copy(state = DownloadState.RUNNING, error = null, errorDetailJson = null, updatedAt = now()))

        val maxRetries = runCatching { prefs.maxRetries.let { 5 } }.getOrDefault(5)
        val files = job.files.toMutableList()

        for ((index, file) in files.withIndex()) {
            if (file.complete) continue
            currentCoroutineContext().ensureActive()

            var attempt = 0
            var succeeded = false
            var lastError: DownloadError? = null

            while (attempt <= maxRetries && !succeeded) {
                currentCoroutineContext().ensureActive()
                val result = downloadFile(jobId, file) { done, rate ->
                    files[index] = files[index].copy(bytesDone = done)
                    rates.value = rates.value + (jobId to rate)
                    persistProgress(jobId, files)
                }
                result.onSuccess { finished ->
                    files[index] = finished
                    succeeded = true
                }.onFailure { throwable ->
                    lastError = classify(throwable)
                    attempt++
                    if (attempt == 1 && lastError?.kind == DownloadErrorKind.CHECKSUM_MISMATCH) {
                        android.util.Log.w(TAG, "${file.filename}: checksum failed, retrying from zero")
                        files[index] = files[index].copy(bytesDone = 0)
                        persistProgress(jobId, files)
                    } else if (attempt <= maxRetries && lastError?.kind == DownloadErrorKind.NETWORK_LOST) {
                        // Exponential backoff, capped so a long outage doesn't
                        // turn into a busy loop.
                        kotlinx.coroutines.delay(min(30_000L, (2.0.pow(attempt) * 1000).toLong()))
                    } else {
                        attempt = maxRetries + 1
                    }
                }
            }

            if (!succeeded) {
                val current = db.downloads().get(jobId) ?: return
                update(
                    current.copy(
                        state = DownloadState.FAILED,
                        filesJson = DownloadJson.encodeToString(
                            kotlinx.serialization.builtins.ListSerializer(DownloadFile.serializer()),
                            files,
                        ),
                        error = lastError?.kind?.title,
                        errorDetailJson = lastError?.let {
                            DownloadJson.encodeToString(DownloadError.serializer(), it)
                        },
                        attempts = attempt,
                        updatedAt = now(),
                    ),
                )
                return
            }
        }

        // A sharded model completes atomically: only once every part has
        // verified does the job report complete.
        val current = db.downloads().get(jobId) ?: return
        // Nothing to finish into.
        //
        // `markCompleted` is an UPDATE, so a job whose library row has gone
        // updated nothing and still reported COMPLETE: gigabytes verified onto
        // the disk, no row anywhere, and the files then read as strays by the
        // orphan sweep that had deleted the row in the first place. Saying so
        // keeps the bytes — the part files stay where they are — and leaves a
        // sentence rather than a silence.
        if (db.models().get(current.modelId) == null) {
            android.util.Log.w(TAG, "no library row for ${current.modelId}; leaving files in place")
            update(
                current.copy(
                    state = DownloadState.FAILED,
                    error = "The library entry for this model is gone. Add it again — the " +
                        "downloaded bytes are still here and it will carry on from them.",
                    updatedAt = now(),
                ),
            )
            return
        }
        // Stamped before the job flips to COMPLETE, so there is no instant in
        // which the queue says finished and the library still says not.
        db.models().markCompleted(current.modelId, now())
        update(
            current.copy(
                state = DownloadState.COMPLETE,
                filesJson = DownloadJson.encodeToString(
                    kotlinx.serialization.builtins.ListSerializer(DownloadFile.serializer()),
                    files,
                ),
                bytesDone = files.sumOf { it.bytesDone },
                updatedAt = now(),
            ),
        )
        activeJobs.remove(jobId)
    }

    /** One file, resuming from its persisted offset via an HTTP Range request, then verified against the expected sha256 before the `.part` is promoted. */
    private suspend fun downloadFile(
        jobId: String,
        file: DownloadFile,
        onProgress: suspend (bytesDone: Long, bytesPerSecond: Long) -> Unit,
    ): Result<DownloadFile> = withContext(Dispatchers.IO) {
        runCatching {
            val partFile = File(file.destPath + PART_SUFFIX)
            partFile.parentFile?.mkdirs()

            // A finished file already at the destination is not re-fetched.
            val dest = File(file.destPath)
            if (dest.isFile && file.sizeBytes > 0 && dest.length() == file.sizeBytes) {
                val trustworthy = file.expectedSha256?.let {
                    sha256Of(dest).equals(it, ignoreCase = true)
                } ?: true
                if (trustworthy) {
                    onProgress(dest.length(), 0)
                    return@runCatching file.copy(bytesDone = dest.length(), complete = true)
                }
            }

            val existing = if (partFile.exists()) partFile.length() else 0L
            val startAt = if (existing in 1 until file.sizeBytes.coerceAtLeast(1)) existing else 0L
            if (startAt == 0L && partFile.exists()) partFile.delete()

            val request = Request.Builder()
                .url(file.url)
                .apply {
                    if (startAt > 0) header("Range", "bytes=$startAt-")
                    tokens.hfToken?.let { header("Authorization", "Bearer $it") }
                }
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    throw java.io.IOException("HTTP ${response.code}")
                }
                val body = response.body ?: throw java.io.IOException("Empty body")

                // A Range request that comes back 200 instead of 206 is the whole file, not the tail.
                val resumed = startAt > 0 && response.code == 206
                val writeFrom = if (resumed) startAt else 0L
                if (startAt > 0 && !resumed) {
                    android.util.Log.w(
                        TAG,
                        "${file.filename}: server ignored Range (HTTP ${response.code}), restarting from 0",
                    )
                    partFile.delete()
                }

                val total = if (resumed) startAt + body.contentLength() else body.contentLength()

                RandomAccessFile(partFile, "rw").use { out ->
                    out.setLength(writeFrom)
                    out.seek(writeFrom)
                    val buffer = ByteArray(BUFFER_BYTES)
                    var done = writeFrom
                    var lastPersist = System.currentTimeMillis()
                    var windowBytes = 0L
                    var windowStart = lastPersist

                    body.byteStream().use { input ->
                        while (true) {
                            currentCoroutineContext().ensureActive()
                            val read = input.read(buffer)
                            if (read <= 0) break
                            out.write(buffer, 0, read)
                            done += read
                            windowBytes += read

                            val nowMs = System.currentTimeMillis()
                            if (nowMs - lastPersist >= PERSIST_INTERVAL_MS) {
                                val elapsed = (nowMs - windowStart).coerceAtLeast(1)
                                onProgress(done, windowBytes * 1000 / elapsed)
                                lastPersist = nowMs
                                windowStart = nowMs
                                windowBytes = 0
                            }
                        }
                    }
                    onProgress(done, 0)
                }

                // Integrity. A mismatch rejects the file outright — nothing is
                // installed, and the user is offered a re-download.
                if (file.expectedSha256 != null) {
                    val actual = sha256Of(partFile)
                    if (!actual.equals(file.expectedSha256, ignoreCase = true)) {
                        partFile.delete()
                        throw ChecksumMismatch(file.expectedSha256, actual)
                    }
                }

                if (dest.exists()) dest.delete()
                if (!partFile.renameTo(dest)) throw java.io.IOException("Could not finalise ${file.filename}")

                file.copy(bytesDone = dest.length(), complete = true, sizeBytes = if (total > 0) total else dest.length())
            }
        }
    }

    private suspend fun persistProgress(jobId: String, files: List<DownloadFile>) {
        val entity = db.downloads().get(jobId) ?: return
        db.downloads().upsert(
            entity.copy(
                filesJson = DownloadJson.encodeToString(
                    kotlinx.serialization.builtins.ListSerializer(DownloadFile.serializer()),
                    files,
                ),
                bytesDone = files.sumOf { it.bytesDone },
                updatedAt = now(),
            ),
        )
    }

    private suspend fun update(entity: DownloadJobEntity) = db.downloads().upsert(entity)

    private fun classify(t: Throwable): DownloadError = when {
        t is ChecksumMismatch -> DownloadError(
            DownloadErrorKind.CHECKSUM_MISMATCH,
            "The downloaded file's sha256 doesn't match the repo's lfs.oid. Nothing was installed.",
            expected = t.expected,
            actual = t.actual,
        )
        t.message?.contains("ENOSPC") == true -> DownloadError(
            DownloadErrorKind.STORAGE_FULL,
            "Ran out of storage part-way through.",
        )
        t.message?.contains("401") == true || t.message?.contains("403") == true -> DownloadError(
            DownloadErrorKind.AUTH_REQUIRED,
            "This file needs a Hugging Face token with access to the repo.",
        )
        t.message?.contains("404") == true -> DownloadError(
            DownloadErrorKind.NOT_FOUND,
            "The file is no longer at the pinned revision.",
        )
        t is java.io.IOException -> DownloadError(
            DownloadErrorKind.NETWORK_LOST,
            t.message ?: "Network error. Offsets persisted; this resumes across reboot.",
        )
        else -> DownloadError(DownloadErrorKind.UNKNOWN, t.message ?: "Download failed.")
    }

    private fun isUnmetered(): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED)
    }

    /** SPEC §3.4 — orphan sweep on boot. */
    suspend fun sweepOrphans(modelsDir: File): List<File> = withContext(Dispatchers.IO) {
        val known = db.downloads().getUnfinished()
            .flatMap { it.toJob().files.map { f -> f.destPath + PART_SUFFIX } }
            .toSet()

        modelsDir.walkTopDown()
            .filter { it.isFile && it.name.endsWith(PART_SUFFIX) && it.absolutePath !in known }
            .onEach { it.delete() }
            .toList()
    }

    private fun sha256Of(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(BUFFER_BYTES)
            while (true) {
                val read = input.read(buffer)
                if (read <= 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun now() = System.currentTimeMillis()

    private class ChecksumMismatch(val expected: String, val actual: String) :
        java.io.IOException("sha256 mismatch")

    companion object {
        private const val TAG = "ondevice.download"
        const val PART_SUFFIX = ".part"

        /** States that mean "this download is still someone's business". */
        private val UNFINISHED = setOf(
            DownloadState.QUEUED,
            DownloadState.RUNNING,
            DownloadState.PAUSED,
            DownloadState.VERIFYING,
        )
        private const val BUFFER_BYTES = 64 * 1024
        private const val PERSIST_INTERVAL_MS = 750L
    }
}

// — mapping —

internal fun DownloadJobEntity.toJob(bytesPerSecond: Long = 0): DownloadJob = DownloadJob(
    id = id,
    modelId = modelId,
    displayName = displayName,
    hfRepo = hfRepo,
    revision = revision,
    files = runCatching {
        DownloadJson.decodeFromString(
            kotlinx.serialization.builtins.ListSerializer(DownloadFile.serializer()),
            filesJson,
        )
    }.getOrDefault(emptyList()),
    state = state,
    error = errorDetailJson?.let {
        runCatching { DownloadJson.decodeFromString(DownloadError.serializer(), it) }.getOrNull()
    },
    connections = connections,
    wifiOnly = wifiOnly,
    attempts = attempts,
    bytesPerSecond = bytesPerSecond,
)

internal fun DownloadJob.toEntity(): DownloadJobEntity = DownloadJobEntity(
    id = id,
    modelId = modelId,
    displayName = displayName,
    hfRepo = hfRepo,
    revision = revision,
    filesJson = DownloadJson.encodeToString(
        kotlinx.serialization.builtins.ListSerializer(DownloadFile.serializer()),
        files,
    ),
    bytesDone = bytesDone,
    bytesTotal = bytesTotal,
    state = state,
    error = error?.kind?.title,
    errorDetailJson = error?.let { DownloadJson.encodeToString(DownloadError.serializer(), it) },
    connections = connections,
    wifiOnly = wifiOnly,
    attempts = attempts,
    createdAt = System.currentTimeMillis(),
    updatedAt = System.currentTimeMillis(),
)
