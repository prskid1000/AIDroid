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

    /** A name given by hand. Blank is stored as null, so "cleared" is one state. */
    @Query("UPDATE models SET customLabel = :label WHERE id = :modelId")
    suspend fun setCustomLabel(modelId: String, label: String?)

    /** Set once, by the downloader, when the last file of a model verifies. */
    @Query("UPDATE models SET completedAt = :at WHERE id = :modelId")
    suspend fun markCompleted(modelId: String, at: Long)

    /**
     * Models with a download still in flight, so a tab can say "on its way"
     * rather than "none installed" — two states that look the same from
     * [observeInstalledByModality], which sees neither.
     */
    @Query(
        """
        SELECT m.id AS modelId,
               COALESCE(NULLIF(m.customLabel, ''), j.displayName) AS displayName,
               m.modality AS modality,
               m.attachmentRole AS attachmentRole, j.bytesDone AS bytesDone,
               j.bytesTotal AS bytesTotal, j.state AS state
        FROM download_jobs j
        JOIN models m ON m.id = j.modelId
        WHERE j.state IN ('QUEUED','RUNNING','PAUSED','VERIFYING')
          AND m.completedAt IS NULL
        ORDER BY j.createdAt
        """,
    )
    fun observeInstalling(): Flow<List<InstallingModel>>

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
interface GeneratedClipDao {
    @Query("SELECT * FROM generated_clips ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<GeneratedClipEntity>>

    @Query("SELECT * FROM generated_clips WHERE id = :id")
    suspend fun get(id: String): GeneratedClipEntity?

    @Query("SELECT COUNT(*) FROM generated_clips")
    fun observeCount(): Flow<Int>

    @Upsert
    suspend fun upsert(clip: GeneratedClipEntity)

    /**
     * Removes the row only. The frames are a directory the caller has to delete
     * itself — a DAO that reached into the filesystem would be a DAO that could
     * delete a folder a failed transaction still refers to.
     */
    @Query("DELETE FROM generated_clips WHERE id = :id")
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
interface ParamManifestDao {
    @Query("SELECT * FROM param_manifests ORDER BY version DESC LIMIT 1")
    suspend fun newest(): ParamManifestEntity?

    @Query("SELECT * FROM param_manifests ORDER BY version DESC LIMIT 1")
    fun observeNewest(): Flow<ParamManifestEntity?>

    @Upsert
    suspend fun upsert(manifest: ParamManifestEntity)
}
