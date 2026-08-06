package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil3.compose.AsyncImage
import ai.ondevice.core.Fmt
import ai.ondevice.core.workflow.NodeKind
import ai.ondevice.engine.workflow.NodeRunState
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NProgressBar
import ai.ondevice.ui.components.ResidentCard
import ai.ondevice.ui.components.ResourceBlock
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.WorkflowViewModel
import ai.ondevice.core.control

/**
 * S17 — a workflow running.
 *
 * Not the editor with one card highlighted. During a run the step being worked
 * on is the only thing worth the width, and the rest is a rail of dots — the
 * same decision the still and clip screens make, where the picture is large
 * and the settings are behind a sheet.
 *
 * Everything here is borrowed: the resource block, the resident card and the
 * progress bar are the ones the other tabs use, so a run looks like a run
 * wherever it is watched from.
 */
@Composable
fun WorkflowRunScreen(
    onBack: () -> Unit,
    viewModel: WorkflowViewModel = activityWorkflowViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val graph = state.graph

    // Work out the loads before anybody commits to them.
    LaunchedEffect(graph) { viewModel.preview() }

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = state.editing?.name ?: "Run",
                subtitle = when {
                    state.cancelling -> "stopping"
                    state.running -> "running"
                    state.error != null -> "stopped"
                    state.finishedAt != null -> "finished"
                    else -> "ready"
                },
                onBack = onBack,
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 20.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            // — the rail: one dot per step, tinted by what it is doing —
            Row(
                Modifier.fillMaxWidth().padding(top = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(5.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                graph.nodes.forEach { node ->
                    val runState = state.nodeStates[node.id]?.state ?: NodeRunState.WAITING
                    Box(
                        Modifier
                            .weight(1f)
                            .height(4.dp)
                            .background(colourFor(runState), Radius.Sm),
                    )
                }
            }
            NHelp(
                graph.nodes.count { state.nodeStates[it.id]?.state == NodeRunState.DONE }
                    .let { "$it of ${graph.nodes.size} done" },
                Modifier.padding(top = 6.dp),
            )

            // — what is being worked on now —
            val active = state.activeNodeId?.let { id -> graph.nodes.firstOrNull { it.id == id } }
            if (active != null) {
                val progress = state.nodeStates[active.id]
                NCard(Modifier.fillMaxWidth().padding(top = 12.dp), ring = NocturneColors.Accent800) {
                    Text(
                        active.label.ifBlank { NodeKind.of(active.type).title },
                        style = NocturneType.CardTitleSm,
                    )
                    Text(
                        listOfNotNull(
                            progress?.phaseLabel?.takeIf { it.isNotBlank() },
                            progress?.let { if (it.steps > 0) "step ${it.step}/${it.steps}" else null },
                            progress?.secondsPerStep?.takeIf { it > 0f }
                                ?.let { String.format("%.1f s/it", it) },
                        ).joinToString(" · "),
                        style = NocturneType.MonoXs,
                        color = NocturneColors.Accent200,
                        modifier = Modifier.padding(top = 4.dp),
                    )
                    progress?.takeIf { it.steps > 0 }?.let {
                        NProgressBar(
                            fraction = (it.step.toFloat() / it.steps).coerceIn(0f, 1f),
                            modifier = Modifier.padding(top = 8.dp),
                        )
                    }
                }
            }

            // — a step waiting on a person —
            if (state.choosingNodeId != null) {
                SectionKicker("Choose one", Modifier.padding(top = 18.dp, bottom = 6.dp))
                NHelp(
                    "The run is holding here until you pick. Everything after this step uses " +
                        "what you choose.",
                    Modifier.padding(bottom = 8.dp),
                )
                state.choices.forEach { option ->
                    // Shown as what it is. A list of transcript pieces
                    // rendered through an image loader is four blank boxes,
                    // and the step that exists to let somebody choose is the
                    // worst place to make them guess.
                    val isPicture = option.endsWith(".png", true) ||
                        option.endsWith(".jpg", true) ||
                        option.endsWith(".jpeg", true) ||
                        option.endsWith(".webp", true)
                    Box(
                        Modifier
                            .fillMaxWidth()
                            .padding(bottom = 8.dp)
                            .ring(NocturneColors.Divider, Radius.Md)
                            .nClickableFlat { viewModel.choose(option) },
                    ) {
                        if (isPicture) {
                            AsyncImage(
                                model = option,
                                contentDescription = null,
                                contentScale = ContentScale.Fit,
                                modifier = Modifier.fillMaxWidth(),
                            )
                        } else {
                            Text(
                                option,
                                style = NocturneType.CardBody,
                                modifier = Modifier.padding(12.dp),
                            )
                        }
                    }
                }
            }

            // — what the run will cost, said before it is spent —
            state.plan?.takeIf { !state.running }?.let { plan ->
                SectionKicker("Before you start", Modifier.padding(top = 18.dp, bottom = 6.dp))
                NCard(Modifier.fillMaxWidth()) {
                    Text(
                        listOfNotNull(
                            if (plan.loadCount == 1) "1 model load" else "${plan.loadCount} model loads",
                            plan.totalBytes.takeIf { it > 0 }?.let { "${Fmt.bytes(it)} read" },
                            plan.peakBytes.takeIf { it > 0 }?.let { "largest ${Fmt.bytes(it)}" },
                        ).joinToString(" · "),
                        style = NocturneType.MonoXs,
                        color = NocturneColors.Accent200,
                    )
                    NHelp(
                        "A load is tens of seconds and several gigabytes, and only one model can " +
                            "be resident at a time — so steps that share a model are run together " +
                            "and the one before is let go first.",
                        Modifier.padding(top = 6.dp),
                    )
                }
            }

            state.error?.let { message ->
                NCard(Modifier.fillMaxWidth().padding(top = 12.dp), ring = NocturneColors.Accent800) {
                    Text(message, style = NocturneType.CardTitleSm, color = NocturneColors.Accent200)
                    state.errorHint?.let { NHelp(it, Modifier.padding(top = 4.dp)) }
                }
            }

            /*
             * Results that finished while nobody was looking.
             *
             * A Send step cannot open a share sheet from the background —
             * Android refuses the activity start — so it posts a notification
             * and waits. Listed here as well, because the notification
             * permission is optional in this app and is asked for exactly once:
             * without the list, saying no to notifications would silently lose
             * every result a Send step ever made.
             */
            if (state.handoffs.isNotEmpty()) {
                SectionKicker("Waiting to be sent", Modifier.padding(top = 18.dp, bottom = 6.dp))
                state.handoffs.forEach { handoff ->
                    NCard(Modifier.fillMaxWidth().padding(bottom = 6.dp)) {
                        Text(handoff.describe, style = NocturneType.CardTitleSm)
                        NHelp(
                            "The run finished while the app was in the background, and Android " +
                                "does not let an app that is not on screen open another one.",
                            Modifier.padding(top = 4.dp),
                        )
                        NButton(
                            "Send it now",
                            onClick = { viewModel.deliver(handoff) },
                            style = NButtonStyle.Secondary,
                            block = true,
                            modifier = Modifier.padding(top = 10.dp),
                        )
                    }
                }
            }

            (state.liveTrace)?.let { trace ->
                var expanded by rememberSaveable { mutableStateOf(false) }
                ResourceBlock(
                    trace = trace,
                    expanded = expanded,
                    onToggle = { expanded = !expanded },
                    live = state.running,
                    modifier = Modifier.padding(top = 12.dp),
                )
            }

            if (state.loadingWhat.isNotEmpty() || state.runtimeBuffers.isNotEmpty()) {
                ResidentCard(
                    loadingNow = state.loadingWhat.isNotEmpty() && state.running,
                    loadingWhat = state.loadingWhat,
                    buffers = state.runtimeBuffers,
                    unloadReason = state.unloadReason,
                    modifier = Modifier.padding(top = 10.dp),
                )
            }

            // Same phase machine as the diffusion tabs: it decides the control
            // while the run exists, and the screen only names the idle case.
            val phase = ai.ondevice.core.runPhaseOf(
                stopping = state.cancelling,
                running = state.running,
            )
            val control = phase.control()
            NButton(
                control?.label ?: when {
                    state.finishedAt != null || state.error != null -> "Run again"
                    else -> "Run"
                },
                onClick = { if (phase.busy) viewModel.cancel() else viewModel.run() },
                style = if (phase.busy) NButtonStyle.Secondary else NButtonStyle.Primary,
                enabled = control?.enabled ?: true,
                block = true,
                modifier = Modifier.padding(top = 16.dp),
            )
            if (phase.showsProgress) {
                NHelp(
                    "This carries on when you leave the app — there is a notification while it " +
                        "runs, and opening it brings you back here.",
                    Modifier.padding(top = 4.dp),
                )
            }

            // — every step, small —
            SectionKicker("Steps", Modifier.padding(top = 20.dp, bottom = 6.dp))
            graph.nodes.forEachIndexed { index, node ->
                val runState = state.nodeStates[node.id]?.state ?: NodeRunState.WAITING
                Row(
                    Modifier.fillMaxWidth().padding(vertical = 5.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box(Modifier.size(8.dp).background(colourFor(runState), Radius.Sm))
                    Text(
                        "${index + 1}",
                        style = NocturneType.Mono2Xs,
                        color = NocturneColors.TextMuted,
                    )
                    Text(
                        node.label.ifBlank { NodeKind.of(node.type).title },
                        style = NocturneType.Row,
                        color = if (runState == NodeRunState.SKIPPED) {
                            NocturneColors.TextMuted
                        } else {
                            NocturneColors.Text
                        },
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        runState.name.lowercase(),
                        style = NocturneType.Mono2Xs,
                        color = NocturneColors.TextMuted,
                    )
                }
            }
        }
    }
}

private fun colourFor(state: NodeRunState) = when (state) {
    NodeRunState.DONE -> NocturneColors.Accent
    NodeRunState.RUNNING, NodeRunState.LOADING -> NocturneColors.Accent400
    NodeRunState.FAILED -> NocturneColors.Accent200
    NodeRunState.SKIPPED, NodeRunState.CANCELLED -> NocturneColors.Divider
    NodeRunState.WAITING -> NocturneColors.Neutral700
}
