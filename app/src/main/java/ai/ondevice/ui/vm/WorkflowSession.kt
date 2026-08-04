package ai.ondevice.ui.vm

import ai.ondevice.core.workflow.WorkflowGraph
import ai.ondevice.data.db.WorkflowEntity
import ai.ondevice.engine.workflow.NodeProgress
import ai.ondevice.engine.workflow.ResidencyPlan
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * A workflow being edited, and one being run — outliving any screen.
 *
 * The same shape as the other four sessions, for the same reason, and more
 * sharply here than anywhere: a graph is several generations end to end, so a
 * run is not minutes but the better part of an hour. A run that ended because
 * somebody looked at another app would be worse than useless.
 */
@Singleton
class WorkflowSession @Inject constructor() {

    val state = MutableStateFlow(WorkflowState())

    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    var runJob: Job? = null

    /**
     * A choice a step is waiting on.
     *
     * The Pick step suspends the run until somebody answers, which is the
     * whole point of it: on a device where the next step is twenty minutes,
     * choosing before spending them is worth more than any automation. Held
     * here rather than in the runner so the answer can come from a screen that
     * did not exist when the question was asked.
     */
    var pending: CompletableDeferred<String?>? = null

    private var observing = false

    fun claimObservers(): Boolean {
        if (observing) return false
        observing = true
        return true
    }
}

/** S15 — workflows: the list, the one being edited, and the run in flight. */
data class WorkflowState(
    val workflows: List<WorkflowEntity> = emptyList(),
    /** The graph on the editor, which is not saved until it is. */
    val editing: WorkflowEntity? = null,
    val graph: WorkflowGraph = WorkflowGraph(),

    // — the run —
    val running: Boolean = false,
    val cancelling: Boolean = false,
    val runId: String? = null,
    val nodeStates: Map<String, NodeProgress> = emptyMap(),
    /** Which step is being worked on now, for the big view. */
    val activeNodeId: String? = null,
    val loadingWhat: List<String> = emptyList(),
    val unloadReason: String? = null,
    val plan: ResidencyPlan? = null,
    /** Set when a step is waiting for somebody to choose. */
    val choosingNodeId: String? = null,
    val choices: List<String> = emptyList(),
    val error: String? = null,
    val errorHint: String? = null,
    val finishedAt: Long? = null,
    val liveTrace: ai.ondevice.engine.ResourceTrace? = null,
    val runtimeBuffers: List<ai.ondevice.engine.RuntimeBuffer> = emptyList(),
)
