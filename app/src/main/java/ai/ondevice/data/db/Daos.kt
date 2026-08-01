package ai.ondevice.data.db

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Upsert
import ai.ondevice.core.DownloadState
import ai.ondevice.core.Modality
import kotlinx.coroutines.flow.Flow

@Dao
interface ModelDao {
    @Query("SELECT * FROM models ORDER BY lastUsedAt DESC, installedAt DESC")
    fun observeAll(): Flow<List<ModelEntity>>

    @Query("SELECT * FROM models WHERE modality = :modality ORDER BY lastUsedAt DESC")
    fun observeByModality(modality: Modality): Flow<List<ModelEntity>>

    /** Only models whose bytes have actually arrived. */
    @Query(
        """
        SELECT * FROM models
        WHERE completedAt IS NOT NULL
        ORDER BY lastUsedAt DESC, installedAt DESC
        """,
    )
    fun observeInstalled(): Flow<List<ModelEntity>>

    @Query(
        """
        SELECT * FROM models
        WHERE modality = :modality AND completedAt IS NOT NULL
        ORDER BY lastUsedAt DESC
        """,
    )
    fun observeInstalledByModality(modality: Modality): Flow<List<ModelEntity>>

    @Query(
        """
        SELECT * FROM models
        WHERE modality = :modality AND completedAt IS NOT NULL
        ORDER BY lastUsedAt DESC
        """,
    )
    suspend fun getInstalledByModality(modality: Modality): List<ModelEntity>

    @Query(
        """
        SELECT * FROM models
        WHERE completedAt IS NOT NULL
        ORDER BY lastUsedAt DESC, installedAt DESC
        """,
    )
    suspend fun getInstalled(): List<ModelEntity>

    /** Set once, by the downloader, when the last file of a model verifies. */
    @Query("UPDATE models SET completedAt = :at WHERE id = :modelId")
    suspend fun markCompleted(modelId: String, at: Long)

    /** Models with a download still in flight, so the library can say so. */
    @Query(
        """
        SELECT modelId FROM download_jobs
        WHERE state IN ('QUEUED','RUNNING','PAUSED','VERIFYING')
        """,
    )
    fun observePendingModelIds(): Flow<List<String>>

    /** The same set, read once. */
    @Query(
        """
        SELECT modelId FROM download_jobs
        WHERE state IN ('QUEUED','RUNNING','PAUSED','VERIFYING')
        """,
    )
    suspend fun pendingModelIds(): List<String>

    @Query("SELECT * FROM models WHERE id = :id")
    fun observe(id: String): Flow<ModelEntity?>

    @Query("SELECT * FROM models WHERE id = :id")
    suspend fun get(id: String): ModelEntity?

    @Query("SELECT * FROM models")
    suspend fun getAll(): List<ModelEntity>

    @Query("SELECT * FROM models WHERE pinned = 1 LIMIT 1")
    suspend fun getPinned(): ModelEntity?

    @Upsert
    suspend fun upsert(model: ModelEntity)

    @Delete
    suspend fun delete(model: ModelEntity)

    @Query("DELETE FROM models WHERE id = :id")
    suspend fun deleteById(id: String)

    @Query("UPDATE models SET lastUsedAt = :at WHERE id = :id")
    suspend fun touch(id: String, at: Long)

    @Query("UPDATE models SET pinned = 0")
    suspend fun clearPins()

    @Query("UPDATE models SET pinned = :pinned WHERE id = :id")
    suspend fun setPinned(id: String, pinned: Boolean)

    @Query("UPDATE models SET paramOverridesJson = :json WHERE id = :id")
    suspend fun setParamOverrides(id: String, json: String)

    @Query("SELECT SUM(sizeBytes) FROM models")
    fun observeTotalBytes(): Flow<Long?>

    @Query("SELECT SUM(sizeBytes) FROM models WHERE modality IN (:modalities)")
    fun observeBytesFor(modalities: List<Modality>): Flow<Long?>
}

@Dao
interface PresetDao {
    /** Seed order, not alphabetical. */
    @Query("SELECT * FROM presets WHERE modality = :modality ORDER BY isBuiltIn DESC, rowid")
    fun observeFor(modality: Modality): Flow<List<PresetEntity>>

    @Query("SELECT * FROM presets WHERE id = :id")
    suspend fun get(id: String): PresetEntity?

    @Query("SELECT * FROM presets ORDER BY rowid")
    suspend fun getAll(): List<PresetEntity>

    @Query("SELECT COUNT(*) FROM presets")
    suspend fun count(): Int

    @Upsert
    suspend fun upsert(preset: PresetEntity)

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertAll(presets: List<PresetEntity>)

    @Query("DELETE FROM presets WHERE id = :id")
    suspend fun deleteById(id: String)
}

@Dao
interface PersonaDao {
    @Query("SELECT * FROM personas ORDER BY rowid")
    fun observeAll(): Flow<List<PersonaEntity>>

    @Query("SELECT * FROM personas WHERE id = :id")
    suspend fun get(id: String): PersonaEntity?

    @Query("SELECT * FROM personas ORDER BY rowid")
    suspend fun getAll(): List<PersonaEntity>

    @Query("SELECT COUNT(*) FROM personas")
    suspend fun count(): Int

    @Upsert
    suspend fun upsert(persona: PersonaEntity)

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertAll(personas: List<PersonaEntity>)

    @Query("DELETE FROM personas WHERE id = :id")
    suspend fun deleteById(id: String)
}

@Dao
interface McpServerDao {
    @Query("SELECT * FROM mcp_servers ORDER BY createdAt")
    fun observeAll(): Flow<List<McpServerEntity>>

    @Query("SELECT * FROM mcp_servers ORDER BY createdAt")
    suspend fun getAll(): List<McpServerEntity>

    @Query("SELECT * FROM mcp_servers WHERE id = :id")
    suspend fun get(id: String): McpServerEntity?

    @Upsert
    suspend fun upsert(server: McpServerEntity)

    @Query("DELETE FROM mcp_servers WHERE id = :id")
    suspend fun deleteById(id: String)
}

@Dao
interface ConversationDao {
    @Query("SELECT * FROM conversations ORDER BY updatedAt DESC")
    fun observeAll(): Flow<List<ConversationEntity>>

    @Query("SELECT * FROM conversations WHERE id = :id")
    fun observe(id: String): Flow<ConversationEntity?>

    @Query("SELECT * FROM conversations WHERE id = :id")
    suspend fun get(id: String): ConversationEntity?

    @Query("SELECT * FROM conversations ORDER BY updatedAt DESC LIMIT 1")
    suspend fun mostRecent(): ConversationEntity?

    @Query("SELECT * FROM conversations ORDER BY updatedAt DESC")
    suspend fun getAll(): List<ConversationEntity>

    @Upsert
    suspend fun upsert(conversation: ConversationEntity)

    @Query("DELETE FROM conversations WHERE id = :id")
    suspend fun deleteById(id: String)

    @Query("UPDATE conversations SET updatedAt = :at WHERE id = :id")
    suspend fun touch(id: String, at: Long)
}

@Dao
interface MessageDao {
    @Query("SELECT * FROM messages WHERE conversationId = :conversationId ORDER BY createdAt ASC")
    fun observeFor(conversationId: String): Flow<List<MessageEntity>>

    @Query("SELECT * FROM messages WHERE conversationId = :conversationId ORDER BY createdAt ASC")
    suspend fun getFor(conversationId: String): List<MessageEntity>

    /** Every message, oldest first, for the library's per-thread counts and previews. */
    @Query("SELECT * FROM messages ORDER BY createdAt ASC")
    fun observeAllOrdered(): Flow<List<MessageEntity>>

    @Query("SELECT * FROM messages WHERE id = :id")
    suspend fun get(id: String): MessageEntity?

    @Upsert
    suspend fun upsert(message: MessageEntity)

    @Query("DELETE FROM messages WHERE id = :id")
    suspend fun deleteById(id: String)

    @Query("DELETE FROM messages WHERE conversationId = :conversationId")
    suspend fun clearFor(conversationId: String)
}

@Dao
interface GeneratedImageDao {
    @Query("SELECT * FROM generated_images ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<GeneratedImageEntity>>

    @Query("SELECT * FROM generated_images WHERE id = :id")
    suspend fun get(id: String): GeneratedImageEntity?

    @Query("SELECT COUNT(*) FROM generated_images")
    fun observeCount(): Flow<Int>

    @Upsert
    suspend fun upsert(image: GeneratedImageEntity)

    @Query("DELETE FROM generated_images WHERE id = :id")
    suspend fun deleteById(id: String)
}

@Dao
interface TranscriptDao {
    @Query("SELECT * FROM transcripts ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<TranscriptEntity>>

    @Query("SELECT * FROM transcripts WHERE id = :id")
    suspend fun get(id: String): TranscriptEntity?

    @Upsert
    suspend fun upsert(transcript: TranscriptEntity)

    @Query("DELETE FROM transcripts WHERE id = :id")
    suspend fun deleteById(id: String)
}

@Dao
interface SynthesisDao {
    @Query("SELECT * FROM syntheses ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<SynthesisEntity>>

    @Query("SELECT * FROM syntheses WHERE id = :id")
    suspend fun get(id: String): SynthesisEntity?

    @Upsert
    suspend fun upsert(synthesis: SynthesisEntity)

    @Query("DELETE FROM syntheses WHERE id = :id")
    suspend fun deleteById(id: String)
}

@Dao
interface PredictionRunDao {
    @Query("SELECT * FROM prediction_runs ORDER BY startedAt DESC")
    fun observeAll(): Flow<List<PredictionRunEntity>>

    /** Every run for one artifact, oldest first. */
    @Query("SELECT * FROM prediction_runs WHERE artifactId = :artifactId ORDER BY startedAt")
    fun observeFor(artifactId: String): Flow<List<PredictionRunEntity>>

    @Query("SELECT * FROM prediction_runs WHERE artifactId IN (:artifactIds) ORDER BY startedAt")
    fun observeForAny(artifactIds: List<String>): Flow<List<PredictionRunEntity>>

    @Query("SELECT * FROM prediction_runs WHERE artifactId = :artifactId ORDER BY startedAt")
    suspend fun getFor(artifactId: String): List<PredictionRunEntity>

    @Upsert
    suspend fun upsert(run: PredictionRunEntity)

    @Query("DELETE FROM prediction_runs WHERE artifactId = :artifactId")
    suspend fun deleteForArtifact(artifactId: String)

    @Query("DELETE FROM prediction_runs WHERE artifactId IN (:artifactIds)")
    suspend fun deleteForArtifacts(artifactIds: List<String>)
}

@Dao
interface DownloadDao {
    @Query("SELECT * FROM download_jobs ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<DownloadJobEntity>>

    @Query("SELECT * FROM download_jobs WHERE state IN (:states) ORDER BY createdAt")
    fun observeByState(states: List<DownloadState>): Flow<List<DownloadJobEntity>>

    @Query("SELECT * FROM download_jobs WHERE state IN (:states) ORDER BY createdAt")
    suspend fun getByState(states: List<DownloadState>): List<DownloadJobEntity>

    @Query("SELECT * FROM download_jobs WHERE id = :id")
    suspend fun get(id: String): DownloadJobEntity?

    @Upsert
    suspend fun upsert(job: DownloadJobEntity)

    @Query("DELETE FROM download_jobs WHERE id = :id")
    suspend fun deleteById(id: String)

    @Query("SELECT COUNT(*) FROM download_jobs WHERE state IN ('QUEUED','RUNNING','VERIFYING')")
    fun observeActiveCount(): Flow<Int>

    /** Jobs that believe they are in flight. */
    @Query("SELECT * FROM download_jobs WHERE state IN ('QUEUED','RUNNING','VERIFYING') ORDER BY createdAt")
    suspend fun getActive(): List<DownloadJobEntity>

    @Query("SELECT * FROM download_jobs WHERE state IN ('QUEUED','RUNNING','PAUSED','VERIFYING') ORDER BY createdAt")
    suspend fun getUnfinished(): List<DownloadJobEntity>

    /** Clear finished history. */
    @Query("DELETE FROM download_jobs WHERE state IN ('COMPLETE','FAILED')")
    suspend fun clearFinished()
}

@Dao
interface RuntimeDao {
    @Query("SELECT * FROM runtime_bundles")
    fun observeAll(): Flow<List<RuntimeBundleEntity>>

    @Query("SELECT * FROM runtime_bundles WHERE engine = :engine")
    suspend fun get(engine: String): RuntimeBundleEntity?

    @Query("SELECT COUNT(*) FROM runtime_bundles")
    suspend fun count(): Int

    @Upsert
    suspend fun upsert(bundle: RuntimeBundleEntity)

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertAll(bundles: List<RuntimeBundleEntity>)
}

@Dao
interface ParamManifestDao {
    @Query("SELECT * FROM param_manifests ORDER BY version DESC LIMIT 1")
    suspend fun newest(): ParamManifestEntity?

    @Query("SELECT * FROM param_manifests ORDER BY version DESC LIMIT 1")
    fun observeNewest(): Flow<ParamManifestEntity?>

    @Upsert
    suspend fun upsert(manifest: ParamManifestEntity)
}
