package ai.ondevice.data.db

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.DownloadState
import ai.ondevice.core.MessageRole
import ai.ondevice.core.Modality
import ai.ondevice.core.ModelFormat
import ai.ondevice.core.PredictionKind
import ai.ondevice.core.RuntimeState

/** The data model of SPEC §11, one entity per line of that block. */

/**
 * `models.backendOverride`, `messages.backend` and `prediction_runs.backend`
 * are left over from when there was a GPU to choose. Nothing writes them now.
 *
 * They are kept rather than dropped because Room validates the schema against
 * these classes on open and a column it does not expect is a refusal to open,
 * so removing the field means recreating three tables — two of which hold the
 * user's model library and every conversation they have had. That is a real
 * risk to take for three unused TEXT columns, and one worth taking only
 * alongside the next migration that has to touch those tables anyway.
 */

@Entity(tableName = "models", indices = [Index("hfRepo"), Index("modality")])
data class ModelEntity(
    @PrimaryKey val id: String,
    val hfRepo: String?,
    val revision: String?,
    val localPath: String,
    val format: ModelFormat,
    val architecture: String?,
    val quant: String?,
    val sizeBytes: Long,
    val sha256: String?,
    val modality: Modality,
    val contextLength: Int?,
    /** Read from `gguf.chat_template`. Never hardcoded — SPEC §1.3, Appendix A #2. */
    val chatTemplate: String?,
    val bosToken: String?,
    val eosToken: String?,
    /** mmproj / VAE / text encoders / voices, resolved and paired automatically. */
    val companionPathsJson: String,
    /** When the library row was written — which is when the download *started*. */
    val installedAt: Long,
    /** When every byte arrived and verified, or null while it has not. */
    val completedAt: Long? = null,
    val lastUsedAt: Long?,
    /** "Keep loaded" — survives screen-off, not process death (SPEC §3.5). */
    val pinned: Boolean,
    val favourite: Boolean,
    val notes: String?,
    val backendOverride: String? = null,
    /** Sparse per-model parameter overrides. Unknown keys preserved inert. */
    val paramOverridesJson: String,
    val defaultPresetId: String?,
    val displayName: String,
    /**
     * A name given by hand, which outranks everything derived.
     *
     * `displayName` is the repo, and the app can only ever qualify it with what
     * it happens to know — the role, the quant, a folder. That is enough to
     * tell two rows apart and not enough to say which one you meant. A person
     * naming a file "SDXL decoder" has settled the question outright.
     */
    val customLabel: String? = null,
    /** Which add-on slot this model fills, or null for a base model. */
    val attachmentRole: AttachmentRole? = null,
) {
    /** What to call this model wherever it is shown or picked. */
    val label: String get() = customLabel?.takeIf { it.isNotBlank() } ?: displayName
}

/**
 * A model row whose bytes have not all landed yet.
 *
 * The library row is written when a download *starts*, so "there is no model"
 * and "the model is on its way" look identical to a tab that only counts
 * installed rows. Each tab reads this so it can say which of the two it is.
 */
data class InstallingModel(
    val modelId: String,
    val displayName: String,
    val modality: Modality,
    val attachmentRole: AttachmentRole?,
    val bytesDone: Long,
    val bytesTotal: Long,
    val state: DownloadState,
) {
    val fraction: Float
        get() = if (bytesTotal > 0) (bytesDone.toFloat() / bytesTotal).coerceIn(0f, 1f) else 0f

    val paused: Boolean get() = state == DownloadState.PAUSED

    /** "Qwen3.5 4B · 62%", or "· paused" when it is not moving. */
    val label: String
        get() = if (paused) {
            "$displayName · paused at ${(fraction * 100).toInt()}%"
        } else {
            "$displayName · ${(fraction * 100).toInt()}%"
        }
}

@Entity(tableName = "conversations", indices = [Index("updatedAt")])
data class ConversationEntity(
    @PrimaryKey val id: String,
    val title: String,
    val modelId: String?,
    val personaId: String?,
    val systemPrompt: String?,
    val presetId: String?,
    val createdAt: Long,
    val updatedAt: Long,
)

@Entity(tableName = "messages", indices = [Index("conversationId")])
data class MessageEntity(
    @PrimaryKey val id: String,
    val conversationId: String,
    val role: MessageRole,
    val content: String,
    /** Reasoning block, parsed out of the configured tag pair and shown collapsed. */
    val thinking: String?,
    val thinkingMillis: Long?,
    val thinkingTokens: Int?,
    val imagePathsJson: String,
    val toolCallsJson: String?,
    val tokenCount: Int?,
    val imageTokenCount: Int?,
    /** The full parameter set this message was generated under. */
    val generationParamsJson: String,
    val tokensPerSecond: Float?,
    val backend: String? = null,
    val createdAt: Long,
    /** Branching: a regenerate keeps the old message and points the new one here. */
    val parentMessageId: String?,
)

@Entity(tableName = "generated_images", indices = [Index("createdAt")])
data class GeneratedImageEntity(
    @PrimaryKey val id: String,
    val path: String,
    val prompt: String,
    val negativePrompt: String?,
    /** Also written into the PNG's tEXt chunk so the file alone is reproducible. */
    val paramsJson: String,
    val modelId: String?,
    val seed: Long,
    val width: Int,
    val height: Int,
    val createdAt: Long,
)

@Entity(tableName = "transcripts", indices = [Index("createdAt")])
data class TranscriptEntity(
    @PrimaryKey val id: String,
    val sourcePath: String?,
    val title: String,
    val segmentsJson: String,
    val modelId: String?,
    val paramsJson: String,
    val durationMillis: Long,
    val createdAt: Long,
)

/** A saved synthesis. */
@Entity(tableName = "syntheses", indices = [Index("createdAt")])
data class SynthesisEntity(
    @PrimaryKey val id: String,
    val path: String,
    val text: String,
    val engineId: String,
    val modelId: String?,
    val voice: String?,
    val paramsJson: String,
    val durationMillis: Long,
    val sampleRate: Int,
    val createdAt: Long,
)

/** What one prediction cost the device. */
@Entity(tableName = "prediction_runs", indices = [Index("artifactId"), Index("startedAt")])
data class PredictionRunEntity(
    @PrimaryKey val id: String,
    val kind: PredictionKind,
    /** The message, image, synthesis or transcript this run produced. */
    val artifactId: String,
    val modelId: String?,
    val backend: String? = null,
    val startedAt: Long,
    val elapsedMillis: Long,
    /** Summary columns, duplicated out of [traceJson] on purpose. */
    val peakCpuPercent: Int,
    val meanCpuPercent: Int,
    val peakRssBytes: Long,
    /** A serialised [ai.ondevice.engine.ResourceTrace]. */
    val traceJson: String,
    val statsJson: String,
)

@Entity(tableName = "download_jobs", indices = [Index("state")])
data class DownloadJobEntity(
    @PrimaryKey val id: String,
    val modelId: String,
    val displayName: String,
    val hfRepo: String?,
    val revision: String?,
    /** One logical job may hold N shards plus companions — SPEC §3.4. */
    val filesJson: String,
    val bytesDone: Long,
    val bytesTotal: Long,
    val state: DownloadState,
    val error: String?,
    val errorDetailJson: String?,
    val connections: Int,
    val wifiOnly: Boolean,
    val attempts: Int,
    val createdAt: Long,
    val updatedAt: Long,
)

/**
 * What is compiled into this build, as a row per engine.
 *
 * It once described something that could be installed, updated and rolled back
 * on its own. It never could: the `.so` files are inside the APK, Android's W^X
 * enforcement refuses to load a native library from writable storage, and so an
 * engine update is an app update. Nothing ever wrote the six columns that
 * carried the other story, and no code path could reach the buttons that read
 * them. Seeded from `runtimes.json` at first launch.
 */
@Entity(tableName = "runtime_bundles")
data class RuntimeBundleEntity(
    @PrimaryKey val engine: String,
    val buildTag: String?,
    val upstreamCommit: String?,
    val jniContract: Int,
    val installedAt: Long?,
    val sizeBytes: Long,
    val state: RuntimeState,
    val architectureCount: Int,
    val backendsJson: String,
)

/** An MCP server the user added by hand. */
@Entity(tableName = "mcp_servers")
data class McpServerEntity(
    @PrimaryKey val id: String,
    val name: String,
    val url: String,
    /** Sent verbatim as `Authorization`. Stored here, not in the URL. */
    val authHeader: String?,
    /** Paused or not, and the only switch there is. */
    val enabled: Boolean,
    /** What the server said it offers, as a JSON array of `{name, description}`. */
    val lastToolsJson: String?,
    /** Tool names the user switched off on this server, as a JSON array. */
    val disabledToolsJson: String,
    val lastCheckedAt: Long?,
    val lastError: String?,
    val createdAt: Long,
)

@Entity(tableName = "param_manifests")
data class ParamManifestEntity(
    @PrimaryKey val version: Int,
    /** "bundled" or "ota" — the app uses max(bundled, downloaded) by version. */
    val source: String,
    val fetchedAt: Long,
    val signatureOk: Boolean,
    val json: String,
)
