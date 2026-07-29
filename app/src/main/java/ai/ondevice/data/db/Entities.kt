package ai.ondevice.data.db

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import ai.ondevice.core.BackendId
import ai.ondevice.core.DownloadState
import ai.ondevice.core.MessageRole
import ai.ondevice.core.Modality
import ai.ondevice.core.ModelFormat
import ai.ondevice.core.RuntimeState

/**
 * The data model of SPEC §11, one entity per line of that block.
 *
 * Two rules run through all of it:
 *  - The **full parameter set** is stored alongside every generated artifact —
 *    message, image, transcript. Reproducibility costs almost nothing at write
 *    time and is impossible to reconstruct later.
 *  - Parameter blobs are sparse JSON strings, never typed columns, so an
 *    unknown key survives an engine downgrade (SPEC §11, Appendix A #12).
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
    val installedAt: Long,
    val lastUsedAt: Long?,
    /** "Keep loaded" — survives screen-off, not process death (SPEC §3.5). */
    val pinned: Boolean,
    val favourite: Boolean,
    val notes: String?,
    /** Per-model backend override, persisted (SPEC §8.1). */
    val backendOverride: BackendId?,
    /** Sparse per-model parameter overrides. Unknown keys preserved inert. */
    val paramOverridesJson: String,
    val defaultPresetId: String?,
    val displayName: String,
)

@Entity(
    tableName = "benchmarks",
    primaryKeys = ["modelId", "backend"],
)
data class BenchmarkEntity(
    val modelId: String,
    val backend: BackendId,
    val promptTokPerSec: Float,
    val genTokPerSec: Float,
    val measuredAt: Long,
)

@Entity(tableName = "presets", indices = [Index("modality")])
data class PresetEntity(
    @PrimaryKey val id: String,
    val modality: Modality,
    val name: String,
    /** Sparse JSON. A built-in preset is editable and deletable like any other. */
    val paramsJson: String,
    val isBuiltIn: Boolean,
)

@Entity(tableName = "personas")
data class PersonaEntity(
    @PrimaryKey val id: String,
    val name: String,
    val avatarPath: String?,
    val systemPrompt: String,
    val defaultModelId: String?,
    val defaultPresetId: String?,
    val defaultVoice: String?,
    val memoryNotes: String?,
)

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
    val backend: BackendId?,
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

@Entity(tableName = "runtime_bundles")
data class RuntimeBundleEntity(
    @PrimaryKey val engine: String,
    val buildTag: String?,
    val upstreamCommit: String?,
    val jniContract: Int,
    val installedAt: Long?,
    val sizeBytes: Long,
    val state: RuntimeState,
    /** Kept so a runtime that fails to init twice can auto-revert (SPEC §17.8). */
    val previousBuildTag: String?,
    val availableBuildTag: String?,
    val availableSizeBytes: Long?,
    val availableNotes: String?,
    val architectureCount: Int,
    val backendsJson: String,
    val initFailureCount: Int,
    val rolledBackFrom: String?,
)

/**
 * An MCP server the user added by hand.
 *
 * There is no discovery and no default list on purpose: every entry here is a
 * third party the user typed in, and one they can see and delete. `enabled`
 * exists separately from deletion so a server can be switched off for a
 * conversation without losing its URL.
 */
@Entity(tableName = "mcp_servers")
data class McpServerEntity(
    @PrimaryKey val id: String,
    val name: String,
    val url: String,
    /** Sent verbatim as `Authorization`. Stored here, not in the URL. */
    val authHeader: String?,
    val enabled: Boolean,
    val lastToolsJson: String?,
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
