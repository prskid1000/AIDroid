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

/**
 * SPEC §3.4 — the downloader.
 *
 * The rules that shape it:
 *  - **Resume across app kill and reboot.** Byte offsets are persisted in Room
 *    after every flush, not held in memory.
 *  - **Integrity is checked, not assumed.** Every file is hashed against the
 *    repo's `lfs.oid` and rejected on mismatch — the S4 failure card shows both
 *    hashes rather than saying "something went wrong".
 *  - **Sharded models are one job.** N parts complete atomically or not at all.
 *  - **Companions are auto-queued** beside the primary file.
 *  - **Revision pinning**: files are fetched at the resolved commit, never at a
 *    moving `main`, so an upstream change is detected rather than silently
 *    swapped in.
 *  - Partial files are removed on cancel and swept on boot.
 */
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
        db.downloads().upsert(job.toEntity())
        start(job.id)
    }

    fun start(jobId: String) {
        if (activeJobs[jobId]?.isActive == true) return
        activeJobs[jobId] = scope.launch(Dispatchers.IO) { runJob(jobId) }
    }

    /**
     * Pick up jobs whose process died mid-flight.
     *
     * SPEC §3.4 promises a download survives the app being killed, and the
     * foreground service covers the ordinary cases — but a crash, a force-stop
     * or a reinstall leaves a row saying RUNNING with nothing running. Without
     * this the job sits at 87% forever and the promise is false. Restarting is
     * cheap because the `.part` is still there and the Range request resumes
     * from its length; if the file went too, it starts over rather than
     * trusting a byte count nothing backs up.
     */
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
            entity.toJob().files.forEach { file ->
                runCatching { File(file.destPath).delete() }
                runCatching { File(file.destPath + PART_SUFFIX).delete() }
            }
            db.downloads().deleteById(jobId)
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
                    // A checksum failure after a *resume* is far more often a
                    // bad resume than a bad file on the server, and the partial
                    // has already been deleted — so one clean attempt from zero
                    // is worth making before telling the user the repo is
                    // wrong. A second failure is reported as-is.
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

    /**
     * One file, resuming from its persisted offset via an HTTP Range request,
     * then verified against the expected sha256 before the `.part` is promoted.
     */
    private suspend fun downloadFile(
        jobId: String,
        file: DownloadFile,
        onProgress: suspend (bytesDone: Long, bytesPerSecond: Long) -> Unit,
    ): Result<DownloadFile> = withContext(Dispatchers.IO) {
        runCatching {
            val partFile = File(file.destPath + PART_SUFFIX)
            partFile.parentFile?.mkdirs()

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

                // A Range request that comes back 200 instead of 206 is the
                // whole file, not the tail. Writing it at the resume offset
                // produces a file of the right *length* whose middle is
                // garbage — which then fails the sha256 check with no clue why.
                // A redirect or a proxy can drop the header, so this is checked
                // rather than assumed.
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

                val dest = File(file.destPath)
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

    /**
     * SPEC §3.4 — orphan sweep on boot. Two directions: `.part` files with no
     * job row, and job rows whose files have vanished.
     */
    suspend fun sweepOrphans(modelsDir: File): List<File> = withContext(Dispatchers.IO) {
        // Same literal-query reason as the resume above: a bound enum list
        // matched nothing, which would have made this sweep delete the partial
        // file of every *live* download it was supposed to protect.
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
