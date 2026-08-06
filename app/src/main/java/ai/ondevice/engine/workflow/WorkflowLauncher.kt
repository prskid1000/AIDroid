package ai.ondevice.engine.workflow

import ai.ondevice.core.workflow.PortType
import ai.ondevice.core.workflow.RunOutputs
import ai.ondevice.core.workflow.StoredValue
import ai.ondevice.core.workflow.TriggerPayload
import ai.ondevice.core.workflow.Triggers
import ai.ondevice.core.workflow.WorkflowGraph
import ai.ondevice.core.workflow.canResume
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.WorkflowRunEntity
import ai.ondevice.engine.InferenceService
import ai.ondevice.engine.ResourceRecorder
import ai.ondevice.ui.vm.WorkflowSession
import ai.ondevice.ui.vm.WorkflowState
import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Starting a run, from anywhere.
 *
 * This was the body of `WorkflowViewModel.run()`, and it began
 *
 * ```
 * val workflow = _state.value.editing ?: return
 * ```
 *
 * — so the only way to run a graph was to have it open in the editor, in a view
 * model scoped to a screen. A share arriving from another app has no editor and
 * no screen, and neither does a launcher shortcut. Moving the body here changes
 * nothing about the runner, the session or the reporter; it only stops the
 * editor being the sole door in. (It is also why there has never been a
 * rerun-from-history: the same door was missing.)
 */
@Singleton
class WorkflowLauncher @Inject constructor(
    @ApplicationContext private val context: Context,
    private val session: WorkflowSession,
    private val db: OnDeviceDatabase,
    private val runner: WorkflowRunner,
    private val recorder: ResourceRecorder,
    private val handoffs: HandoffDispatcher,
) {

    private val state get() = session.state

    /** Whether a run is in flight — one engine holds the weights, so one run. */
    val busy: Boolean get() = state.value.running

    /**
     * Run [workflowId], with [payload] written into any Input waiting for one.
     *
     * Returns false when something is already running. Refused rather than
     * queued: at these run lengths a translation that starts forty minutes late
     * has already been done by hand, and a silent queue is a worse answer than
     * a plain no.
     */
    fun launch(
        workflowId: String,
        payload: TriggerPayload? = null,
        /** Answers to Inputs marked *ask when it runs*, keyed by step id. */
        answers: Map<String, String> = emptyMap(),
        /**
         * An interrupted run to carry on from.
         *
         * Its id is reused so the artefacts already written under
         * `workflows/<runId>/` are the ones the resumed steps go on pointing at.
         */
        resumeFrom: WorkflowRunEntity? = null,
    ): Boolean {
        if (busy) return false
        val runId = resumeFrom?.id ?: UUID.randomUUID().toString()

        /*
         * Off the main thread, which it never was.
         *
         * WorkflowSession's scope is Dispatchers.Main.immediate — right for a
         * session that only holds state, wrong for the thing that runs the
         * graph. Model steps hid it, because each engine suspends into its own
         * dispatcher on the way to the JNI. A Script step does not: QuickJS
         * evaluates on whatever thread calls it, so a template with a loop in it
         * ran on the UI thread, and a long one ANR'd the app.
         *
         * The ANR was not the worst of it. A blocked main thread means the
         * activity lifecycle callbacks cannot run either, so the app still
         * believed it was on screen after Home had been pressed — and a Send
         * step then took the foreground path into a launch the platform
         * refuses.
         */
        session.runJob = session.scope.launch(Dispatchers.Default) {
            val workflow = db.workflows().get(workflowId) ?: return@launch
            // A resumed run works from the graph *that run* snapshotted, not
            // from the workflow as it stands now — the outputs being carried
            // over were made by those steps, and an edit in between would leave
            // them pointing at steps that no longer exist.
            val stored = WorkflowGraph.decode(resumeFrom?.graphJson ?: workflow.graphJson)
            // Filled before the run, not during it. The run stores a snapshot of
            // the graph it ran, so putting the shared value in here means the
            // history records what was actually worked on — and the runner needs
            // no notion of a trigger at all.
            val graph = Triggers.fillAsked(
                payload?.let { Triggers.fill(stored, it) } ?: stored,
                answers,
            )

            state.value = WorkflowState(
                workflows = state.value.workflows,
                models = state.value.models,
                editing = workflow,
                graph = graph,
                running = true,
                runId = runId,
                startedBy = payload?.fromPackage,
            )

            // One wake lock for the whole graph. The bracket rather than the
            // pair, because the release has to happen on every way out.
            InferenceService.holdingWakeLock(context) {
                val recording = recorder.start(session.scope)
                val live = session.scope.launch {
                    recording.live.collect { trace ->
                        state.value = state.value.copy(liveTrace = trace)
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
                val reporter = reporter(runId, workflow.id, graph)
                try {
                    // Only from a graph whose steps each run once. See
                    // WorkflowGraph.canResume for why a loop is refused rather
                    // than half-skipped.
                    val carried = resumeFrom
                        ?.takeIf { graph.canResume() }
                        ?.let { RunOutputs.decode(it.nodeStatesJson).toPortValues() }
                        .orEmpty()
                    val outcome = runner.run(runId, graph, reporter, carried)
                    val failure = outcome.exceptionOrNull()
                    state.value = state.value.copy(
                        error = failure?.message,
                        errorHint = failure?.let { "The steps before it kept what they made." },
                        finishedAt = System.currentTimeMillis(),
                        resultText = outcome.getOrNull()?.let { lastTextOf(graph, it) },
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
                            // Carried through rather than defaulted away: a run
                            // that failed at step five is the one most worth
                            // being able to pick up, and writing "{}" here would
                            // throw away the four steps that did work.
                            nodeStatesJson = reporter.snapshot(),
                        ),
                    )
                    db.workflows().touch(workflow.id, System.currentTimeMillis())
                } finally {
                    live.cancel()
                    recording.stop()
                    state.value = state.value.copy(
                        running = false,
                        cancelling = false,
                        activeNodeId = null,
                        choosingNodeId = null,
                    )
                }
            }
        }
        return true
    }

    fun cancel() {
        if (!state.value.running) return
        state.value = state.value.copy(cancelling = true)
        // Reach the native call first: cancelling the job does not.
        runner.activeCancel?.invoke()
        session.pending?.complete(null)
        session.runJob?.cancel()
    }

    /** Answer a Pick step. */
    fun choose(path: String?) {
        session.pending?.complete(path)
        session.pending = null
        state.value = state.value.copy(choosingNodeId = null, choices = emptyList())
    }

    /** Fire, or re-fire, a hand-off that is waiting on a tap. */
    fun deliver(handoff: Handoff) {
        handoffs.dispatch(handoff)
        state.value = state.value.copy(
            handoffs = state.value.handoffs.filterNot { it.nodeId == handoff.nodeId },
        )
    }

    /**
     * The last thing this graph said, in words.
     *
     * Walked backwards through the steps rather than taken from the outputs map
     * in insertion order, because a loop writes its collected list after the
     * steps inside it and the answer somebody is waiting for is the last one the
     * *author* wrote, not the last one the runner happened to store.
     */
    private fun lastTextOf(
        graph: WorkflowGraph,
        outputs: Map<String, PortValue>,
    ): String? = graph.nodes.asReversed().firstNotNullOfOrNull { node ->
        outputs[node.id]?.takeIf { it.type == PortType.TEXT && it.text.isNotBlank() }?.text
    }

    /** Everything a resumed run already has, back in the shape the runner uses. */
    private fun RunOutputs.toPortValues(): Map<String, PortValue> =
        values.mapValues { (_, v) ->
            PortValue(
                type = runCatching { PortType.valueOf(v.type) }.getOrDefault(PortType.TEXT),
                text = v.text,
                paths = v.paths,
                elementType = runCatching { PortType.valueOf(v.elementType) }
                    .getOrDefault(PortType.FILE),
            )
        }

    /** A reporter that can also hand back everything it has recorded. */
    private abstract class RecordingReporter : RunReporter {
        protected val produced = LinkedHashMap<String, StoredValue>()
        fun snapshot(): String = RunOutputs(produced.toMap()).encode()
    }

    private fun reporter(
        runId: String,
        workflowId: String,
        graph: WorkflowGraph,
    ) = object : RecordingReporter() {

        override fun onNode(nodeId: String, progress: NodeProgress) {
            state.value = state.value.copy(
                nodeStates = state.value.nodeStates + (nodeId to progress),
                activeNodeId = if (
                    progress.state == NodeRunState.RUNNING || progress.state == NodeRunState.LOADING
                ) {
                    nodeId
                } else {
                    state.value.activeNodeId
                },
            )
        }

        override fun onLoading(what: List<String>, stage: String?) {
            state.value = state.value.copy(loadingWhat = what)
        }

        override fun onUnload(because: String) {
            state.value = state.value.copy(unloadReason = because)
        }

        override suspend fun awaitChoice(nodeId: String, options: List<String>): String? {
            val deferred = CompletableDeferred<String?>()
            session.pending = deferred
            state.value = state.value.copy(choosingNodeId = nodeId, choices = options)
            return deferred.await()
        }

        /**
         * Fire it if the app is on screen; park it if it is not.
         *
         * Parked *and* listed, not only notified: the notification permission is
         * optional in this app and is asked for exactly once, so a person who
         * said no would otherwise lose every result a Send step produced with
         * nothing anywhere to show it had happened.
         */
        override fun onHandoff(handoff: Handoff) {
            val delivered = handoffs.dispatch(handoff)
            if (!delivered) {
                state.value = state.value.copy(handoffs = state.value.handoffs + handoff)
            }
        }

        /**
         * Write down what has been made, every time something is.
         *
         * One small row update per step. That is a cost worth paying against a
         * graph where a single step is minutes: the alternative is a row written
         * once at the end, which is never written at all for precisely the runs
         * that needed it.
         */
        override fun onProduced(nodeId: String, values: Map<String, PortValue>) {
            values.forEach { (key, value) ->
                produced[key] = StoredValue(
                    type = value.type.name,
                    text = value.text,
                    paths = value.paths,
                    elementType = value.elementType.name,
                )
            }
            val snapshot = snapshot()
            session.scope.launch {
                runCatching {
                    db.workflows().upsertRun(
                        WorkflowRunEntity(
                            id = runId,
                            workflowId = workflowId,
                            graphJson = graph.encode(),
                            state = "RUNNING",
                            startedAt = System.currentTimeMillis(),
                            nodeStatesJson = snapshot,
                        ),
                    )
                }
            }
        }
    }
}
