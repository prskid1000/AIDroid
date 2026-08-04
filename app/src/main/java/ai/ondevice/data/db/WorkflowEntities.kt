package ai.ondevice.data.db

import androidx.room.Dao
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow

/**
 * A workflow the user has built.
 *
 * The graph is one JSON column — see WorkflowGraph for why it is not a pair of
 * node and edge tables.
 */
@Entity(tableName = "workflows", indices = [Index("updatedAt")])
data class WorkflowEntity(
    @PrimaryKey val id: String,
    val name: String,
    val notes: String = "",
    val graphJson: String,
    val createdAt: Long,
    val updatedAt: Long,
    val lastRunAt: Long? = null,
)

/**
 * One run of one workflow.
 *
 * [graphJson] is a *copy* rather than a reference to the workflow's own. What
 * a run did must not change because the workflow was edited afterwards — the
 * same reasoning that puts the generation parameters on every message and a
 * parameters chunk in every PNG this app writes.
 */
@Entity(
    tableName = "workflow_runs",
    indices = [Index("workflowId"), Index("startedAt")],
)
data class WorkflowRunEntity(
    @PrimaryKey val id: String,
    val workflowId: String,
    val graphJson: String,
    /** RUNNING · DONE · FAILED · CANCELLED */
    val state: String,
    val startedAt: Long,
    val finishedAt: Long? = null,
    val error: String? = null,
    val errorHint: String? = null,
    /**
     * Per-node status and where each output landed, written as the run goes.
     *
     * Written incrementally rather than at the end, so a run the system kills
     * part-way through can be picked up from the last step that finished. At
     * these run lengths that is not a nicety.
     */
    val nodeStatesJson: String = "{}",
)

@Dao
interface WorkflowDao {

    @Query("SELECT * FROM workflows ORDER BY updatedAt DESC")
    fun observeAll(): Flow<List<WorkflowEntity>>

    @Query("SELECT * FROM workflows WHERE id = :id")
    suspend fun get(id: String): WorkflowEntity?

    @Upsert
    suspend fun upsert(workflow: WorkflowEntity)

    @Query("DELETE FROM workflows WHERE id = :id")
    suspend fun delete(id: String)

    @Query("UPDATE workflows SET lastRunAt = :at WHERE id = :id")
    suspend fun touch(id: String, at: Long)

    @Query("SELECT * FROM workflow_runs ORDER BY startedAt DESC LIMIT :limit")
    fun observeRuns(limit: Int = 40): Flow<List<WorkflowRunEntity>>

    @Query("SELECT * FROM workflow_runs WHERE id = :id")
    suspend fun getRun(id: String): WorkflowRunEntity?

    /** The run to offer to resume: the last one that never finished. */
    @Query("SELECT * FROM workflow_runs WHERE state = 'RUNNING' ORDER BY startedAt DESC LIMIT 1")
    suspend fun unfinishedRun(): WorkflowRunEntity?

    @Upsert
    suspend fun upsertRun(run: WorkflowRunEntity)

    @Query("DELETE FROM workflow_runs WHERE workflowId = :workflowId")
    suspend fun deleteRunsFor(workflowId: String)
}
