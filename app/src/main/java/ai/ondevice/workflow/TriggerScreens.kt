package ai.ondevice.workflow

import ai.ondevice.core.workflow.NodeKind
import ai.ondevice.core.workflow.TriggerPayload
import ai.ondevice.core.workflow.Triggers
import ai.ondevice.core.workflow.WorkflowGraph
import ai.ondevice.data.db.WorkflowEntity
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.flow.StateFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties

/**
 * What the trampoline shows between another app's share sheet and a run.
 *
 * These use a sheet of their own rather than the app's `NBottomSheet`, and that
 * is worth explaining because sharing it was the obvious thing to do.
 *
 * `NBottomSheet` is a settings panel: a fixed nine tenths of the screen, with a
 * scrolling column stretched to fill it and the clearance for the gesture strip
 * living *inside* that scroll. Both choices are right there and wrong here.
 * These sheets ask one short question, so a fixed height drew a card and then
 * most of a phone's worth of empty surface under it — which reads as something
 * that failed to load — and they sit over another app, where covering the
 * screen to ask a yes-or-no is an overreach. Sizing that panel to its content
 * instead meant its scroll view ran to the very bottom of the display, taking
 * the buttons under the gesture strip with it, because trailing padding inside
 * a scroll container is scrollable content and not height.
 *
 * So: no scroll container in the common case, the clearance measured into the
 * sheet's own surface, and a shared component that keeps working for the twelve
 * screens already using it.
 */

/**
 * A sheet that is as tall as what it holds, and no taller.
 *
 * The bottom inset is added to the sheet's own padding rather than to anything
 * inside it, so the painted surface grows and the last control has real space
 * under it. `navigationBars` can report nothing at all on a gesture-navigation
 * device, so a fixed floor is added to whatever it says.
 */
@Composable
private fun TriggerSheet(
    title: String,
    onDismiss: () -> Unit,
    note: String? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            usePlatformDefaultWidth = false,
            /*
             * Left fitting system windows, which is the whole fix.
             *
             * `NBottomSheet` sets this false because it draws a full-height
             * panel and insets itself. Copying that here put the dialog window
             * over the navigation area, so a bottom-aligned sheet was laid out
             * with its lower edge below the display: the padding under the
             * buttons was measured, painted, and off the screen. Three
             * different attempts at adding more of it changed nothing, and the
             * node bounds said why — the buttons ended at exactly the display
             * height every time.
             *
             * Fitting means the window stops above the gesture strip and the
             * sheet's own padding is the only clearance it needs.
             */
            decorFitsSystemWindows = true,
        ),
    ) {
        Box(
            Modifier
                .fillMaxSize()
                .background(NocturneColors.Neutral900.copy(alpha = 0.55f))
                .nClickableFlat(onClick = onDismiss),
            contentAlignment = Alignment.BottomCenter,
        ) {
            // Whatever the window did not already take. Usually nothing now
            // that it fits system windows, so the constant below is what
            // actually separates the last button from the edge.
            val inset = WindowInsets.navigationBars.asPaddingValues().calculateBottomPadding()
            Column(
                Modifier
                    .fillMaxWidth()
                    .background(
                        NocturneColors.Surface,
                        RoundedCornerShape(20.dp, 20.dp, 0.dp, 0.dp),
                    )
                    // Swallowed so a tap inside does not reach the scrim behind
                    // it and close the thing being tapped.
                    .nClickableFlat { }
                    .padding(start = 18.dp, end = 18.dp, top = 10.dp),
            ) {
                Box(
                    Modifier
                        .align(Alignment.CenterHorizontally)
                        .padding(bottom = 12.dp)
                        .size(width = 34.dp, height = 4.dp)
                        .background(NocturneColors.Neutral700, Radius.Sm),
                )
                Row(
                    Modifier.fillMaxWidth().padding(bottom = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(9.dp),
                    verticalAlignment = Alignment.Bottom,
                ) {
                    Text(
                        title,
                        style = NocturneType.SheetTitle,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f),
                    )
                    if (note != null) {
                        Text(note, style = NocturneType.Meta, color = NocturneColors.TextMuted)
                    }
                }
                content()
            }
        }
    }
}

/**
 * The clear space under a sheet's last control.
 *
 * A child with a height, and emitted by each screen as the last thing in its
 * own content — both of which are the result of measurement rather than taste.
 *
 * Bottom padding does not survive here: on this dialog, whether or not it fits
 * system windows, four separate attempts at padding the sheet produced node
 * bounds identical to the pixel, with the buttons ending exactly at the display
 * height and the gesture strip across them. A Box with a size is laid out as
 * content and cannot be quietly dropped. Appending it inside [TriggerSheet]
 * after `content()` did not take either; inside the caller's own lambda it
 * does, and the bounds move the moment it goes in.
 */
@Composable
private fun ColumnScope.SheetFooterSpace() {
    val inset = WindowInsets.navigationBars.asPaddingValues().calculateBottomPadding()
    // Sized against the gesture pill rather than against a guideline: the pill
    // is drawn over this sheet, and a button it merely clears by a hair still
    // reads as one that is about to be swiped instead of tapped.
    Box(Modifier.fillMaxWidth().height(56.dp + maxOf(inset, 16.dp)))
}

/** Copying what arrived onto disk. Brief, unless it came from a cloud provider. */
@Composable
fun TriggerReading() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text(
            "Reading what was shared…",
            style = NocturneType.Row,
            color = NocturneColors.TextMuted,
        )
    }
}

/**
 * Which workflow, when more than one can take what arrived.
 *
 * Shown rather than guessed at. The share sheet's own rows are the fast path
 * and the system caps how many of them exist; this is the reliable one, and it
 * is where a workflow the cap left out is still reachable.
 */
@Composable
fun TriggerPick(
    payload: TriggerPayload,
    options: List<WorkflowEntity>,
    onPick: (WorkflowEntity) -> Unit,
    onDismiss: () -> Unit,
) {
    TriggerSheet("Run which workflow?", onDismiss, note = payload.describe()) {
        options.forEachIndexed { index, workflow ->
            val accepts = remember(workflow.graphJson) { acceptsOf(workflow) }
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(bottom = if (index == options.lastIndex) 2.dp else 6.dp)
                    .background(NocturneColors.Bg, Radius.Md)
                    .ring(NocturneColors.Divider, Radius.Md)
                    .nClickableFlat { onPick(workflow) }
                    .padding(horizontal = 12.dp, vertical = 11.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text(
                        workflow.name,
                        style = NocturneType.Row,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Text(
                        accepts.ifBlank { "takes nothing" },
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMuted,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                // A quiet affordance rather than a chevron glyph: this design
                // system has no icon set for rows, and a bare arrow character
                // sets at a different weight from everything around it.
                Box(
                    Modifier
                        .size(6.dp)
                        .background(NocturneColors.Accent, CircleShape),
                )
            }
        }
        SheetFooterSpace()
    }
}

/**
 * The consent sheet, and the reason this activity is not simply a launcher.
 *
 * Any app on the device can name any workflow. A run here is minutes of battery
 * and several gigabytes of reads, and a Send step at the end of one is a way for
 * content to leave. So what is about to happen is named — the caller, the
 * workflow, what it was handed, and every step that will send something out —
 * before a single weight is loaded.
 */
@Composable
fun TriggerConfirm(
    payload: TriggerPayload,
    workflow: WorkflowEntity,
    onRun: () -> Unit,
    onDismiss: () -> Unit,
) {
    val graph = remember(workflow.graphJson) { WorkflowGraph.decode(workflow.graphJson) }
    val sends = remember(graph) {
        graph.nodes
            .filter { it.enabled && it.type == NodeKind.Send.type }
            .map { it.label.ifBlank { NodeKind.Send.title } }
    }
    val steps = remember(graph) { graph.nodes.count { it.enabled } }

    TriggerSheet(workflow.name, onDismiss, note = if (steps == 1) "1 step" else "$steps steps") {
        /*
         * One framed block rather than three labelled sections.
         *
         * The earlier version gave each fact a section kicker of its own, which
         * is the grammar the editor uses for things you change. Nothing here is
         * editable — it is a summary read once, at a glance, under time
         * pressure — so it reads better as rows of label and value than as
         * three headings with one line under each.
         */
        Column(
            Modifier
                .fillMaxWidth()
                .background(NocturneColors.Bg, Radius.Md)
                .ring(NocturneColors.Divider, Radius.Md)
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            FactRow("From", payload.fromPackage ?: "another app")
            payload.values.forEachIndexed { index, value ->
                FactRow(
                    label = if (index == 0) "With" else "",
                    value = value.summary,
                    tag = value.type.label,
                )
            }
        }

        if (sends.isNotEmpty()) {
            // Named before the run, not discovered after it. A step that hands
            // content to another app is the one thing here somebody might not
            // want, and the moment to say so is before a model loads.
            Column(
                Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp)
                    .background(NocturneColors.Accent900, Radius.Md)
                    .ring(NocturneColors.Accent800, Radius.Md)
                    .padding(horizontal = 12.dp, vertical = 10.dp),
            ) {
                Text(
                    if (sends.size == 1) "Sends something out" else "Sends things out",
                    style = NocturneType.CardTitleSm,
                    color = NocturneColors.Accent200,
                )
                Text(
                    sends.joinToString(" · "),
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                    modifier = Modifier.padding(top = 3.dp),
                )
            }
        }

        Row(
            Modifier.fillMaxWidth().padding(top = 14.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            // Side by side rather than stacked. Two full-width buttons one above
            // the other is the shape of a destructive confirmation; this is an
            // ordinary yes-or-no and should not borrow that weight.
            NButton(
                "Cancel",
                onClick = onDismiss,
                style = NButtonStyle.Secondary,
                modifier = Modifier.weight(1f),
                block = true,
            )
            NButton(
                "Run",
                onClick = onRun,
                style = NButtonStyle.Primary,
                modifier = Modifier.weight(1f),
                block = true,
            )
        }
        SheetFooterSpace()
    }
}

/**
 * Held over the calling app while the run answers its selection.
 *
 * The one screen here that waits. It exists because `ACTION_PROCESS_TEXT`
 * replaces a selection only if the activity is alive to return a result, so the
 * run has to be watched rather than started and left — and because somebody is
 * standing there holding a phone, it says which step it is on and offers a way
 * out that does not cost them their selection.
 */
@Composable
fun TriggerWorking(
    workflow: WorkflowEntity,
    state: StateFlow<ai.ondevice.ui.vm.WorkflowState>,
    onCancel: () -> Unit,
    onDone: (String?) -> Unit,
) {
    val run by state.collectAsStateWithLifecycle()

    // Fires once the run has stopped, whichever way it stopped. Reading it in a
    // LaunchedEffect rather than in the composition keeps the result-and-finish
    // out of a frame.
    LaunchedEffect(run.running, run.finishedAt, run.error) {
        if (!run.running && run.finishedAt != null) onDone(run.resultText)
    }

    val graph = remember(workflow.graphJson) { WorkflowGraph.decode(workflow.graphJson) }
    val done = graph.nodes.count {
        run.nodeStates[it.id]?.state == ai.ondevice.engine.workflow.NodeRunState.DONE
    }
    val active = run.activeNodeId?.let { id -> graph.nodes.firstOrNull { it.id == id } }

    TriggerSheet(
        workflow.name,
        onDismiss = onCancel,
        note = "$done of ${graph.nodes.size}",
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .background(NocturneColors.Bg, Radius.Md)
                .ring(NocturneColors.Divider, Radius.Md)
                .padding(horizontal = 12.dp, vertical = 12.dp),
        ) {
            Text(
                when {
                    run.error != null -> "Stopped"
                    active != null -> active.label.ifBlank { NodeKind.of(active.type).title }
                    else -> "Starting…"
                },
                style = NocturneType.Row,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                run.error ?: "The result will replace what you selected.",
                style = NocturneType.MonoXs,
                color = NocturneColors.TextMuted,
                modifier = Modifier.padding(top = 3.dp),
            )
        }
        NButton(
            "Stop",
            onClick = onCancel,
            style = NButtonStyle.Secondary,
            block = true,
            modifier = Modifier.padding(top = 14.dp),
        )
        SheetFooterSpace()
    }
}

/** No, and why, and what would fix it — the voice the rest of the app refuses in. */
@Composable
fun TriggerRefuse(what: String, because: String, onDismiss: () -> Unit) {
    TriggerSheet(what, onDismiss) {
        NHelp(because)
        NButton(
            "Close",
            onClick = onDismiss,
            style = NButtonStyle.Secondary,
            block = true,
            modifier = Modifier.padding(top = 14.dp),
        )
        SheetFooterSpace()
    }
}

/** Label on the left, value on the right, at the density of a spec sheet. */
@Composable
private fun FactRow(label: String, value: String, tag: String? = null) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            label.uppercase(),
            style = NocturneType.Kicker,
            color = NocturneColors.TextMuted,
            modifier = Modifier.size(width = 44.dp, height = 14.dp),
        )
        if (tag != null) NTag(tag, style = NTagStyle.Outline)
        Text(
            value,
            style = NocturneType.MonoXs,
            color = NocturneColors.Text,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f),
        )
    }
}

/** What a workflow takes from outside, in words. */
private fun acceptsOf(workflow: WorkflowEntity): String =
    Triggers.accepts(WorkflowGraph.decode(workflow.graphJson))
        .joinToString(", ") { it.label.lowercase() }

/** One line naming what arrived, for the note under a sheet's title. */
private fun TriggerPayload.describe(): String =
    values.joinToString(", ") { it.type.label.lowercase() }.ifBlank { "nothing" }
