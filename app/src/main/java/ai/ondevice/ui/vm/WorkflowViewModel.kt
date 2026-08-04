package ai.ondevice.ui.vm

import androidx.lifecycle.ViewModel
import ai.ondevice.core.workflow.NodeKind
import ai.ondevice.core.workflow.NodeRecord
import ai.ondevice.core.workflow.PortType
import ai.ondevice.core.workflow.WorkflowGraph
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.WorkflowEntity
import ai.ondevice.data.db.WorkflowRunEntity
import ai.ondevice.engine.InferenceService
import ai.ondevice.engine.record
import ai.ondevice.engine.workflow.NodeProgress
import ai.ondevice.engine.workflow.NodeRunState
import ai.ondevice.engine.workflow.ResidencyPlanner
import ai.ondevice.engine.workflow.RunReporter
import ai.ondevice.engine.workflow.WorkflowRunner
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.util.UUID
import javax.inject.Inject

@HiltViewModel
class WorkflowViewModel @Inject constructor(
    @dagger.hilt.android.qualifiers.ApplicationContext
    private val context: android.content.Context,
    private val session: WorkflowSession,
    private val db: OnDeviceDatabase,
    private val runner: WorkflowRunner,
    private val recorder: ai.ondevice.engine.ResourceRecorder,
) : ViewModel() {

    private val _state get() = session.state
    val state: StateFlow<WorkflowState> = session.state.asStateFlow()
    private val runScope get() = session.scope

    private var runJob: Job?
        get() = session.runJob
        set(value) { session.runJob = value }

    init {
        if (session.claimObservers()) attachObservers()
    }

    private fun attachObservers() {
        runScope.launch {
            db.workflows().observeAll().collect { list ->
                _state.value = _state.value.copy(workflows = list)
            }
        }
        runScope.launch {
            db.models().observeInstalled().collect { models ->
                _state.value = _state.value.copy(
                    models = models.filter { it.attachmentRole == null },
                )
            }
        }
    }

    /** Every installed model, so a step can be pointed at one. */
    suspend fun installedModels(): List<ModelEntity> =
        db.models().getInstalled().filter { it.attachmentRole == null }

    // ── editing ──────────────────────────────────────────────────────────

    fun newWorkflow() {
        runScope.launch {
            val now = System.currentTimeMillis()
            val entity = WorkflowEntity(
                id = UUID.randomUUID().toString(),
                name = "New workflow",
                graphJson = WorkflowGraph().encode(),
                createdAt = now,
                updatedAt = now,
            )
            db.workflows().upsert(entity)
            open(entity.id)
        }
    }

    fun open(workflowId: String) {
        runScope.launch {
            val entity = db.workflows().get(workflowId) ?: return@launch
            _state.value = _state.value.copy(
                editing = entity,
                graph = WorkflowGraph.decode(entity.graphJson),
            )
        }
    }

    fun rename(name: String) = edit { it.copy(name = name) }

    fun addNode(kind: NodeKind) {
        val node = NodeRecord(
            id = UUID.randomUUID().toString().take(8),
            type = kind.type,
            params = defaultsFor(kind),
        )
        mutate { it.copy(nodes = it.nodes + node) }
    }

    fun removeNode(nodeId: String) =
        mutate { graph -> graph.copy(nodes = graph.nodes.filterNot { it.id == nodeId }) }

    fun moveNode(from: Int, to: Int) = mutate { graph ->
        val nodes = graph.nodes.toMutableList()
        if (from !in nodes.indices || to !in nodes.indices) return@mutate graph
        nodes.add(to, nodes.removeAt(from))
        graph.copy(nodes = nodes)
    }

    fun setEnabled(nodeId: String, enabled: Boolean) =
        updateNode(nodeId) { it.copy(enabled = enabled) }

    fun setLabel(nodeId: String, label: String) =
        updateNode(nodeId) { it.copy(label = label) }

    /** Bind a slot to an earlier step's output, or clear it. */
    fun bind(nodeId: String, slot: String, reference: String?) = updateNode(nodeId) { node ->
        val slots = node.slots.toMutableMap()
        if (reference == null) slots.remove(slot) else slots[slot] = reference
        node.copy(slots = slots)
    }

    /**
     * Point a step at a model, and record what that makes it.
     *
     * The shape is stored beside the id rather than looked up when drawing,
     * because the editor asks a node for its slots while composing and a
     * database read there would be a suspend call in the middle of a frame.
     */
    fun chooseModel(nodeId: String, model: ai.ondevice.data.db.ModelEntity) {
        val makesVideo = ai.ondevice.core.DiffusionFamily.isVideo(
            model.architecture ?: model.label,
        ) == true
        val shape = ai.ondevice.core.workflow.ProcessorShape.of(
            model.modality,
            makesVideo,
            isUpscaler = model.attachmentRole == ai.ondevice.core.AttachmentRole.UPSCALER,
        )
        setParam(nodeId, "model", model.id)
        setParam(nodeId, "shape", shape.name)
        updateNode(nodeId) { node ->
            if (node.label.isBlank()) node.copy(label = model.label) else node
        }
    }

    /**
     * Store a setting as the *type* it is, not as the text it was typed as.
     *
     * Everything here came from a text field, and writing it all back as a
     * JSON string is wrong in a way that is invisible until a run: a step
     * asking for sixty tokens sent `"60"` where the runtime wanted `60`,
     * parsed it as nothing, and generated nothing — reporting success, because
     * an empty answer is not an error anywhere along that path.
     *
     * The free-text keys are named rather than guessed at, because a prompt
     * that happens to read "60" is a prompt and not a number.
     */
    fun setParam(nodeId: String, key: String, value: String?) = updateNode(nodeId) { node ->
        val params = node.params.toMutableMap()
        when {
            value == null -> params.remove(key)
            key in TEXT_KEYS -> params[key] = JsonPrimitive(value)
            value.equals("true", true) || value.equals("false", true) ->
                params[key] = JsonPrimitive(value.toBoolean())
            value.toLongOrNull() != null -> params[key] = JsonPrimitive(value.toLong())
            value.toDoubleOrNull() != null -> params[key] = JsonPrimitive(value.toDouble())
            else -> params[key] = JsonPrimitive(value)
        }
        node.copy(params = JsonObject(params))
    }

    private fun updateNode(nodeId: String, block: (NodeRecord) -> NodeRecord) = mutate { graph ->
        graph.copy(nodes = graph.nodes.map { if (it.id == nodeId) block(it) else it })
    }

    private fun mutate(block: (WorkflowGraph) -> WorkflowGraph) {
        val graph = block(_state.value.graph)
        _state.value = _state.value.copy(graph = graph)
        edit { it.copy(graphJson = graph.encode()) }
    }

    private fun edit(block: (WorkflowEntity) -> WorkflowEntity) {
        val current = _state.value.editing ?: return
        val updated = block(current).copy(updatedAt = System.currentTimeMillis())
        _state.value = _state.value.copy(editing = updated)
        runScope.launch { db.workflows().upsert(updated) }
    }

    fun delete(workflowId: String) {
        runScope.launch {
            db.workflows().deleteRunsFor(workflowId)
            db.workflows().delete(workflowId)
            if (_state.value.editing?.id == workflowId) {
                _state.value = _state.value.copy(editing = null, graph = WorkflowGraph())
            }
        }
    }

    private companion object {
        /** Settings that are prose, whatever they happen to look like. */
        val TEXT_KEYS = setOf(
            "text", "template", "pattern", "path", "model", "shape",
            "voice", "provider", "separator", "by", "condition", "mode", "tool", "script",
            "arguments", "portType",
        )
    }

    /** What a freshly-added step starts at. */
    private fun defaultsFor(kind: NodeKind): JsonObject = when (kind) {
        NodeKind.Input, NodeKind.LibraryItem ->
            JsonObject(mapOf("portType" to JsonPrimitive(PortType.TEXT.name)))
        NodeKind.RepeatStart, NodeKind.Batch ->
            JsonObject(mapOf("times" to JsonPrimitive(2)))
        NodeKind.TextSplit ->
            JsonObject(mapOf("by" to JsonPrimitive("paragraph")))
        NodeKind.TextJoin ->
            JsonObject(mapOf("separator" to JsonPrimitive("\n\n")))
        else -> JsonObject(emptyMap())
    }

    // ── running ──────────────────────────────────────────────────────────

    /** What the run will cost, worked out before anybody commits to it. */
    fun preview() {
        runScope.launch {
            val models = installedModels().associateBy { it.id }
            _state.value = _state.value.copy(
                plan = ResidencyPlanner.plan(_state.value.graph, models),
            )
        }
    }

    fun run() {
        if (_state.value.running) return
        val workflow = _state.value.editing ?: return
        val graph = _state.value.graph
        val runId = UUID.randomUUID().toString()

        _state.value = _state.value.copy(
            running = true,
            cancelling = false,
            runId = runId,
            nodeStates = emptyMap(),
            error = null,
            errorHint = null,
            finishedAt = null,
        )

        runJob = runScope.launch {
            // One wake lock for the whole graph. The bracket rather than the
            // pair, because the release has to happen on every way out.
            InferenceService.holdingWakeLock(context) {
                val recording = recorder.start(runScope)
                val live = runScope.launch {
                    recording.live.collect { trace ->
                        _state.value = _state.value.copy(liveTrace = trace)
                    }
                }
                db.workflows().upsertRun(
                    WorkflowRunEntity(
                        id = runId,
                        workflowId = workflow.id,
                        graphJson = graph.encode(),
                        state = "RUNNING",
                        startedAt = System.currentTimeMillis(),
                    ),
                )
                try {
                    val outcome = runner.run(runId, graph, reporter())
                    val failure = outcome.exceptionOrNull()
                    _state.value = _state.value.copy(
                        error = failure?.message,
                        errorHint = failure?.let { "The steps before it kept what they made." },
                        finishedAt = System.currentTimeMillis(),
                    )
                    db.workflows().upsertRun(
                        WorkflowRunEntity(
                            id = runId,
                            workflowId = workflow.id,
                            graphJson = graph.encode(),
                            state = if (failure == null) "DONE" else "FAILED",
                            startedAt = System.currentTimeMillis(),
                            finishedAt = System.currentTimeMillis(),
                            error = failure?.message,
                        ),
                    )
                    db.workflows().touch(workflow.id, System.currentTimeMillis())
                } finally {
                    live.cancel()
                    recording.stop()
                    _state.value = _state.value.copy(
                        running = false,
                        cancelling = false,
                        activeNodeId = null,
                        choosingNodeId = null,
                    )
                }
            }
        }
    }

    fun cancel() {
        if (!_state.value.running) return
        _state.value = _state.value.copy(cancelling = true)
        // Reach the native call first: cancelling the job does not.
        runner.activeCancel?.invoke()
        session.pending?.complete(null)
        runJob?.cancel()
    }

    /** Answer a Pick step. */
    fun choose(path: String?) {
        session.pending?.complete(path)
        session.pending = null
        _state.value = _state.value.copy(choosingNodeId = null, choices = emptyList())
    }

    private fun reporter() = object : RunReporter {
        override fun onNode(nodeId: String, progress: NodeProgress) {
            _state.value = _state.value.copy(
                nodeStates = _state.value.nodeStates + (nodeId to progress),
                activeNodeId = if (
                    progress.state == NodeRunState.RUNNING || progress.state == NodeRunState.LOADING
                ) {
                    nodeId
                } else {
                    _state.value.activeNodeId
                },
            )
        }

        override fun onLoading(what: List<String>, stage: String?) {
            _state.value = _state.value.copy(loadingWhat = what)
        }

        override fun onUnload(because: String) {
            _state.value = _state.value.copy(unloadReason = because)
        }

        override suspend fun awaitChoice(nodeId: String, options: List<String>): String? {
            val deferred = CompletableDeferred<String?>()
            session.pending = deferred
            _state.value = _state.value.copy(choosingNodeId = nodeId, choices = options)
            return deferred.await()
        }
    }
}
