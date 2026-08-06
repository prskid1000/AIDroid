package ai.ondevice.ui.vm

import androidx.lifecycle.ViewModel
import ai.ondevice.core.workflow.NodeKind
import ai.ondevice.core.workflow.NodeRecord
import ai.ondevice.core.workflow.PortType
import ai.ondevice.core.workflow.WorkflowGraph
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.WorkflowEntity
import ai.ondevice.engine.workflow.ResidencyPlanner
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.util.UUID
import javax.inject.Inject

@HiltViewModel
class WorkflowViewModel @Inject constructor(
    private val session: WorkflowSession,
    private val db: OnDeviceDatabase,
    private val launcher: ai.ondevice.engine.workflow.WorkflowLauncher,
    private val shortcuts: ai.ondevice.workflow.ShortcutPublisher,
    private val scheduler: ai.ondevice.engine.workflow.Scheduler,
    private val toolProviders: ai.ondevice.tools.ToolProviderFactory,
    private val prefs: ai.ondevice.data.prefs.AppPrefs,
) : ViewModel() {

    private val _state get() = session.state
    val state: StateFlow<WorkflowState> = session.state.asStateFlow()
    private val runScope get() = session.scope

    init {
        if (session.claimObservers()) {
            attachObservers()
            findInterruptedRun()
        }
    }

    /**
     * Look for a run the system stopped without telling anybody.
     *
     * A row still reading RUNNING when nothing is running means the process was
     * reclaimed mid-graph. Checked once, when the session is first observed,
     * because that is the moment the app has come back.
     */
    private fun findInterruptedRun() {
        runScope.launch {
            val stale = db.workflows().unfinishedRun() ?: return@launch
            if (_state.value.running) return@launch
            _state.value = _state.value.copy(interrupted = stale)
        }
    }

    /** Carry on from where the interrupted run stopped. */
    fun resumeInterrupted() {
        val stale = _state.value.interrupted ?: return
        _state.value = _state.value.copy(interrupted = null)
        launcher.launch(stale.workflowId, resumeFrom = stale)
    }

    /** Let it go, and stop the row claiming to be running. */
    fun discardInterrupted() {
        val stale = _state.value.interrupted ?: return
        _state.value = _state.value.copy(interrupted = null)
        runScope.launch {
            db.workflows().upsertRun(
                stale.copy(
                    state = "CANCELLED",
                    finishedAt = System.currentTimeMillis(),
                    error = "The app was closed before this finished.",
                ),
            )
        }
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

    /**
     * The tools a Tool step may call — the same set chat is offered.
     *
     * Asked of the registry rather than listed here, so a step can call an
     * MCP server's tool the moment that server is connected, with nothing in
     * this file to keep in step.
     */
    suspend fun availableTools(): List<ai.ondevice.engine.ToolSpec> = runCatching {
        toolProviders.registry(enabled = prefs.enabledToolProviders.first()).specs()
    }.getOrDefault(emptyList())

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

    // ── scheduling ───────────────────────────────────────────────────────

    /** The schedule on the workflow being edited. */
    fun schedule(): ai.ondevice.core.workflow.Schedule =
        ai.ondevice.core.workflow.Schedule.decode(_state.value.editing?.scheduleJson)

    /**
     * Save a schedule and set the alarm to match, in that order.
     *
     * Both, always: an alarm without the row is forgotten on the next reboot,
     * and a row without the alarm is a schedule that silently never fires —
     * which is the failure that looks exactly like the feature not existing.
     */
    fun setSchedule(schedule: ai.ondevice.core.workflow.Schedule) {
        val current = _state.value.editing ?: return
        edit { it.copy(scheduleJson = schedule.encode()) }
        scheduler.arm(current.id, schedule)
    }

    /** Whether a scheduled run may start on its own, or must wait for a tap. */
    fun canRunUnattended(): Boolean = scheduler.canRunUnattended

    /** Where to send somebody to allow it, or null when there is nothing to ask. */
    fun exactAlarmSettings(): android.content.Intent? = scheduler.permissionIntent()

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
        // Which share sheets this workflow belongs in is derived from the graph,
        // so any edit can change it — adding the Input that makes it a share
        // target, or deleting the one that made it one.
        runScope.launch { shortcuts.republish() }
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
            // Long-lived shortcuts survive being unpublished, so a deleted
            // workflow would otherwise keep its row in the share sheet and
            // start a run against a graph that is no longer there.
            shortcuts.forget(workflowId)
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

    /**
     * Run what is open in the editor.
     *
     * The body of this used to live here and started `val workflow =
     * _state.value.editing ?: return`, which made the editor the only door into
     * a run. It is now one caller of [WorkflowLauncher] among three — the other
     * two being a share from another app and a launcher shortcut, neither of
     * which has an editor or a screen behind it.
     */
    fun run(answers: Map<String, String> = emptyMap()) {
        _state.value.editing?.let { launcher.launch(it.id, answers = answers) }
    }

    /**
     * The steps that want to be asked before this runs, if any.
     *
     * Read by the screen so it can put a sheet in front of the run rather than
     * the run discovering mid-graph that it has nothing to work on.
     */
    fun askedInputs(): List<ai.ondevice.core.workflow.NodeRecord> =
        ai.ondevice.core.workflow.Triggers.askedInputs(_state.value.graph)

    fun cancel() = launcher.cancel()

    /** Answer a Pick step. */
    fun choose(path: String?) = launcher.choose(path)

    /** Send a result that was waiting for the app to come back. */
    fun deliver(handoff: ai.ondevice.engine.workflow.Handoff) = launcher.deliver(handoff)

    /**
     * Whether this workflow can be started by another app, and from where.
     *
     * Off is not stored: a workflow is reachable from outside exactly when it
     * has an Input marked *from another app*, because that is the only thing
     * that can receive what was shared. Turning it on means adding one; turning
     * it off means the Input goes back to being typed.
     */
    fun setInputSource(nodeId: String, from: String) {
        setParam(nodeId, ai.ondevice.core.workflow.Triggers.PARAM_FROM, from)
        // Republish, because what appears in a share sheet is derived from this.
        runScope.launch { shortcuts.republish() }
    }
}
