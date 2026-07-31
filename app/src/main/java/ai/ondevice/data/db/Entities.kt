package ai.ondevice.data.db

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.BackendId
import ai.ondevice.core.DownloadState
import ai.ondevice.core.MessageRole
import ai.ondevice.core.Modality
import ai.ondevice.core.ModelFormat
import ai.ondevice.core.PredictionKind
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
    /** When the library row was written — which is when the download *started*. */
    val installedAt: Long,
    /**
     * When every byte arrived and verified, or null while it has not.
     *
     * This is the record that a model is usable. It used to be derived instead,
     * as "no download_jobs row for this model is still active" — the absence of
     * evidence standing in for evidence. That reads correctly right up until a
     * job row goes missing for any reason at all (a pruned history, a failed
     * write, a user clearing the download list), at which point a model that
     * never finished is silently promoted to installed and handed to llama.cpp,
     * which reports it as a corrupt header.
     *
     * Written once, by the downloader, at the moment the last file verifies.
     */
    val completedAt: Long? = null,
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
    /**
     * Which add-on slot this model fills, or null for a base model.
     *
     * Stored because the user said so on the Add screen, not derived from the
     * filename. It used to be re-derived on every read by
     * AttachmentRole.classify(localPath), and a path is a weak thing to hang it
     * on: `model.safetensors` in an `image_encoder/` directory is a CLIP vision
     * encoder, the same basename elsewhere is a base checkpoint, and a
     * ControlNet that classified as nothing became a *diffusion model* and got
     * loaded as one. Now the answer is recorded once, by someone who knows.
     */
    val attachmentRole: AttachmentRole? = null,
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

/**
 * A saved synthesis.
 *
 * Speak wrote its WAV to disk and then forgot it — the path lived in memory
 * until the process died, which meant every rendered take was unreachable the
 * moment the user left the screen. Recording it alongside the script and the
 * parameters makes a synthesis as reproducible as a generated image already is,
 * and gives the library something to list.
 */
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

/**
 * What one prediction cost the device.
 *
 * One table keyed by the artifact rather than a `traceJson` column on each of
 * `messages`, `generated_images`, `syntheses` and `transcripts`. Four columns
 * would have meant four migrations to keep in step and four places to remember,
 * and — the part that decided it — the four kinds would no longer be
 * comparable. A single table answers "what did anything on this device cost"
 * with one query.
 *
 * [artifactId] is a plain column, not a foreign key: the four tables it can
 * point into have no common parent, and a run is worth keeping even if it
 * describes something that has since been deleted by hand. Deletes clean up
 * after themselves in [ai.ondevice.ui.vm.LibraryViewModel] instead.
 */
@Entity(tableName = "prediction_runs", indices = [Index("artifactId"), Index("startedAt")])
data class PredictionRunEntity(
    @PrimaryKey val id: String,
    val kind: PredictionKind,
    /** The message, image, synthesis or transcript this run produced. */
    val artifactId: String,
    val modelId: String?,
    val backend: BackendId?,
    val startedAt: Long,
    val elapsedMillis: Long,
    /**
     * Summary columns, duplicated out of [traceJson] on purpose. A list screen
     * that wants "how hard did this work" must not parse and walk a 180-point
     * array per row to find out.
     */
    val peakCpuPercent: Int,
    val meanCpuPercent: Int,
    val peakRssBytes: Long,
    /** A serialised [ai.ondevice.engine.ResourceTrace]. */
    val traceJson: String,
    /**
     * Whatever throughput figure this kind of run has, sparse — `tokens_per_second`
     * for chat, `seconds_per_step` for image, `realtime_factor` for a transcript.
     * Sparse rather than three nullable columns for the same reason every other
     * parameter blob here is: a kind added later needs no migration.
     */
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
    /**
     * Paused or not, and the only switch there is.
     *
     * This used to be one of two: the column was written `true` at creation and
     * never changed, while the screen's toggle edited a set in the preferences.
     * The column was the one the provider list filtered on, so the dead switch
     * was the one that mattered — a server "paused" in the UI was still filtered
     * in by a flag nobody could see.
     */
    val enabled: Boolean,
    /**
     * What the server said it offers, as a JSON array of `{name, description}`.
     * Kept so the tool list can be read — and switched off one by one — without
     * the server being reachable at that moment.
     */
    val lastToolsJson: String?,
    /**
     * Tool names the user switched off on this server, as a JSON array.
     *
     * Per-server rather than global: two servers may legitimately offer a tool
     * of the same name, and turning one off must not silence the other.
     */
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
