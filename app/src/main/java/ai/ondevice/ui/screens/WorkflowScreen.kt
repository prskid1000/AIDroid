package ai.ondevice.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.core.workflow.NodeKind
import ai.ondevice.core.workflow.RunOutputs
import ai.ondevice.core.workflow.Schedule
import ai.ondevice.core.workflow.WorkflowGraph
import ai.ondevice.core.workflow.canResume
import ai.ondevice.ui.BottomDestinations
import ai.ondevice.ui.components.NBottomBar
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCardKicker
import ai.ondevice.ui.components.NCardTitle
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.RootToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.ToolbarAction
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.vm.WorkflowViewModel

/**
 * S15 — the workflows there are.
 *
 * The shape of the Library screen rather than something new: a list of cards,
 * each saying what it is and when it last ran. Nothing here is a canvas.
 */
@Composable
fun WorkflowScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onOpen: (String) -> Unit,
    viewModel: WorkflowViewModel = activityWorkflowViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = {
            RootToolbar("Workflow") {
                ToolbarAction(NIcons.Plus, "New workflow", { viewModel.newWorkflow(); onOpen("") })
            }
        },
        bottomBar = { NBottomBar(BottomDestinations, currentRoute) { onNavigate(it.route) } },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            /*
             * A run the system stopped without telling anybody.
             *
             * Offered before anything else on this screen, because it is the one
             * thing here with work already sunk into it. What the finished steps
             * made is on disk and recorded, so carrying on skips them rather
             * than spending their model loads again — which for a graph of
             * several generations is most of an hour.
             */
            state.interrupted?.let { stale ->
                val graph = remember(stale.graphJson) { WorkflowGraph.decode(stale.graphJson) }
                val done = remember(stale.nodeStatesJson) {
                    RunOutputs.decode(stale.nodeStatesJson).completedNodes.size
                }
                NCard(
                    Modifier.fillMaxWidth().padding(top = 12.dp),
                    ring = NocturneColors.Accent800,
                ) {
                    NCardTitle("A run did not finish")
                    NHelp(
                        if (graph.canResume()) {
                            "It stopped after $done of ${graph.nodes.size} steps — the app was " +
                                "closed or the system reclaimed it. What those steps made was kept."
                        } else {
                            "It stopped after $done of ${graph.nodes.size} steps. This one has a " +
                                "loop in it, so it can only start again: a step inside a loop runs " +
                                "many times, and one pass finishing says nothing about the next."
                        },
                        Modifier.padding(top = 4.dp),
                    )
                    Row(
                        Modifier.fillMaxWidth().padding(top = 10.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        NButton(
                            "Forget it",
                            onClick = viewModel::discardInterrupted,
                            style = NButtonStyle.Secondary,
                            modifier = Modifier.weight(1f),
                            block = true,
                        )
                        NButton(
                            if (graph.canResume()) "Carry on" else "Start again",
                            onClick = viewModel::resumeInterrupted,
                            style = NButtonStyle.Primary,
                            modifier = Modifier.weight(1f),
                            block = true,
                        )
                    }
                }
            }

            if (state.workflows.isEmpty()) {
                NCard(Modifier.fillMaxWidth().padding(top = 12.dp)) {
                    NCardTitle("Nothing built yet")
                    NHelp(
                        "A workflow is a list of steps — bring something in, run a model over it, " +
                            "keep what comes out. The models are the ones already installed, and " +
                            "each step hands its result to the next.",
                        Modifier.padding(top = 6.dp),
                    )
                    NHelp(
                        "Start one with the + above.",
                        Modifier.padding(top = 8.dp),
                        color = NocturneColors.Accent300,
                    )
                }
                return@Column
            }

            SectionKicker(
                "Saved · ${state.workflows.size}",
                Modifier.padding(top = 14.dp, bottom = 8.dp),
            )

            state.workflows.forEach { workflow ->
                val graph = WorkflowGraph.decode(workflow.graphJson)
                val steps = graph.nodes.count { it.enabled }
                NCard(
                    Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp)
                        .nClickableFlat {
                            viewModel.open(workflow.id)
                            onOpen(workflow.id)
                        },
                ) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            NCardKicker(
                                if (steps == 1) "1 step" else "$steps steps",
                            )
                            NCardTitle(workflow.name)
                        }
                        // What it is made of, at a glance: one tag per family
                        // that appears in it.
                        graph.nodes.map { NodeKind.of(it.type).family }
                            .distinct()
                            .take(3)
                            .forEach { family ->
                                NTag(family.label, style = NTagStyle.Outline)
                            }
                    }
                    /*
                     * When it last ran, when it next will, and why it did not.
                     *
                     * The last of those is the point. A schedule that quietly
                     * skips is indistinguishable from one that was never saved,
                     * and this line is the only place that can tell them apart —
                     * so a skip is said here in the same breath as a success.
                     */
                    val schedule = remember(workflow.scheduleJson) {
                        Schedule.decode(workflow.scheduleJson)
                    }
                    val line = listOfNotNull(
                        workflow.lastRunAt?.let { "last run ${Fmt.relative(it)}" },
                        schedule.describe().takeIf { it.isNotBlank() },
                    ).joinToString(" · ")
                    if (line.isNotBlank()) {
                        Text(
                            line,
                            style = NocturneType.MonoXs,
                            color = NocturneColors.TextMuted,
                            modifier = Modifier.padding(top = 6.dp),
                        )
                    }
                    schedule.lastSkipReason?.takeIf { schedule.enabled }?.let { why ->
                        Text(
                            "skipped — $why",
                            style = NocturneType.MonoXs,
                            color = NocturneColors.Accent300,
                            modifier = Modifier.padding(top = 3.dp),
                        )
                    }
                }
            }
        }
    }
}
