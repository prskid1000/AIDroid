package ai.ondevice.data.download

import ai.ondevice.core.DownloadState
import ai.ondevice.core.Fmt
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** A file within a download job. */
@Serializable
data class DownloadFile(
    val filename: String,
    val url: String,
    val destPath: String,
    val sizeBytes: Long,
    /** The repo's `lfs.oid`. A mismatch after download rejects the file. */
    val expectedSha256: String? = null,
    val bytesDone: Long = 0,
    val complete: Boolean = false,
    /** mmproj / VAE / encoders queued alongside the primary weights. */
    val companionRole: String? = null,
    val shardIndex: Int? = null,
    val shardCount: Int? = null,
) {
    val fraction: Float get() = if (sizeBytes > 0) (bytesDone.toFloat() / sizeBytes).coerceIn(0f, 1f) else 0f

    val progressLabel: String get() = when {
        complete -> Fmt.bytes(sizeBytes)
        bytesDone > 0 -> "${Fmt.bytes(bytesDone)} / ${Fmt.bytes(sizeBytes)}"
        else -> "queued"
    }
}

/** Structured failure detail, so a refusal can show the numbers rather than a toast. */
@Serializable
data class DownloadError(
    val kind: DownloadErrorKind,
    val message: String,
    val expected: String? = null,
    val actual: String? = null,
)

@Serializable
enum class DownloadErrorKind {
    CHECKSUM_MISMATCH,
    NETWORK_LOST,
    STORAGE_FULL,
    AUTH_REQUIRED,
    NOT_FOUND,
    CANCELLED,
    UNKNOWN,
    ;

    val title: String get() = when (this) {
        CHECKSUM_MISMATCH -> "Checksum mismatch"
        NETWORK_LOST -> "Network lost"
        STORAGE_FULL -> "Storage full"
        AUTH_REQUIRED -> "Token required"
        NOT_FOUND -> "File not found"
        CANCELLED -> "Cancelled"
        UNKNOWN -> "Download failed"
    }
}

/** The in-memory view of a job, rehydrated from its row. */
data class DownloadJob(
    val id: String,
    val modelId: String,
    val displayName: String,
    val hfRepo: String?,
    val revision: String?,
    val files: List<DownloadFile>,
    val state: DownloadState,
    val error: DownloadError?,
    val connections: Int,
    val wifiOnly: Boolean,
    val attempts: Int,
    val bytesPerSecond: Long = 0,
) {
    val bytesTotal: Long get() = files.sumOf { it.sizeBytes }
    val bytesDone: Long get() = files.sumOf { it.bytesDone }
    val fraction: Float get() = if (bytesTotal > 0) (bytesDone.toFloat() / bytesTotal).coerceIn(0f, 1f) else 0f

    val shardCount: Int get() = files.count { it.shardCount != null && it.shardCount > 1 }
    val companionCount: Int get() = files.count { it.companionRole != null }

    /** "3 shards + mmproj · 4 connections" — the S4 subtitle. */
    val subtitle: String get() = buildString {
        val shards = files.filter { it.shardCount != null && it.shardCount > 1 }
        if (shards.isNotEmpty()) append("${shards.size} shards") else append("${files.size} file${if (files.size == 1) "" else "s"}")
        val companions = files.mapNotNull { it.companionRole }.distinct()
        if (companions.isNotEmpty()) append(" + ${companions.joinToString(", ")}")
        append(" · $connections connections")
    }

    val etaSeconds: Long
        get() = if (bytesPerSecond > 0) (bytesTotal - bytesDone) / bytesPerSecond else 0
}

internal val DownloadJson = Json {
    ignoreUnknownKeys = true
    encodeDefaults = true
}
