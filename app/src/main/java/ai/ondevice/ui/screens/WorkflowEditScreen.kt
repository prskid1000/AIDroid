package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.workflow.NodeContext
import ai.ondevice.core.workflow.NodeFamily
import ai.ondevice.core.workflow.NodeKind
import ai.ondevice.core.workflow.NodeRecord
import ai.ondevice.core.workflow.PortType
import ai.ondevice.core.workflow.SlotSpec
import ai.ondevice.core.workflow.WorkflowGraph
import ai.ondevice.ui.components.NBottomSheet
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.NTextArea
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.WorkflowViewModel

/**
 * S16 — a workflow, as a list of steps.
 *
 * Not a canvas, and the reasoning is worth keeping next to the code. A phone
 * gives about 360dp of width; a free-form graph needs pan, zoom, hit targets
 * below the minimum this app allows itself, and edge routing. The design
 * system has no canvas in it — ten component files whose whole grammar is a
 * vertical stack of cards. And the thing a canvas is *for* is showing branches
 * that run at the same time, which this runtime forbids: one engine holds the
 * weights, and everything is in sequence whether it is drawn that way or not.
 *
 * The list buys one more thing, quietly. A step may only read from steps above
 * it, so a cycle cannot be expressed — there is no sort, no cycle detector and
 * no error state for one, because the order on screen *is* the order it runs.
 */
@Composable
fun WorkflowEditScreen(
    onBack: () -> Unit,
    onRun: () -> Unit,
    viewModel: WorkflowViewModel = activityWorkflowViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val graph = state.graph
    var addOpen by rememberSaveable { mutableStateOf(false) }
    var slotFor by rememberSaveable { mutableStateOf<String?>(null) }

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = state.editing?.name ?: "Workflow",
                subtitle = "${graph.nodes.size} steps",
                onBack = onBack,
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 20.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            NInput(
                value = state.editing?.name.orEmpty(),
                onValueChange = viewModel::rename,
                placeholder = "Name this workflow",
                modifier = Modifier.padding(top = 12.dp),
            )

            // What is unfinished, said here rather than discovered four
            // minutes into a load. The same reasoning the media tabs use for
            // their component warnings: a run on this hardware is long enough
            // that finding out late is the expensive way to find out.
            val problems = remember(graph) { problemsIn(graph) }
            if (problems.isNotEmpty() && !state.running) {
                NCard(
                    Modifier.fillMaxWidth().padding(top = 12.dp),
                    ring = NocturneColors.Accent800,
                ) {
                    Text(
                        if (problems.size == 1) "One step is not ready" else "${problems.size} steps are not ready",
                        style = NocturneType.CardTitleSm,
                        color = NocturneColors.Accent200,
                    )
                    problems.forEach { NHelp(it, Modifier.padding(top = 4.dp)) }
                }
            }

            // While something is running this opens it rather than greying
            // out. A disabled button is the least useful thing to show
            // somebody who came here *because* a run is going.
            NButton(
                if (state.running) "Open the run" else "Run",
                onClick = onRun,
                style = if (state.running) NButtonStyle.Secondary else NButtonStyle.Primary,
                block = true,
                enabled = state.running || (graph.nodes.any { it.enabled } && problems.isEmpty()),
                modifier = Modifier.padding(top = 12.dp),
            )

            SectionKicker("Steps", Modifier.padding(top = 20.dp, bottom = 8.dp))

            if (graph.nodes.isEmpty()) {
                NCard(Modifier.fillMaxWidth()) {
                    Text("No steps yet", style = NocturneType.CardTitleSm)
                    NHelp(
                        "Every workflow starts with something to work on — a prompt you type, a " +
                            "picture you pick, or a recording. Add that first.",
                        Modifier.padding(top = 4.dp),
                    )
                }
            }

            graph.nodes.forEachIndexed { index, node ->
                StepCard(
                    index = index,
                    node = node,
                    graph = graph,
                    viewModel = viewModel,
                    models = state.models,
                    onBindSlot = { slot -> slotFor = "${node.id}/$slot" },
                    onMoveUp = { viewModel.moveNode(index, index - 1) }.takeIf { index > 0 },
                    onMoveDown = { viewModel.moveNode(index, index + 1) }
                        .takeIf { index < graph.nodes.lastIndex },
                )
            }

            NButton(
                "Add a step",
                onClick = { addOpen = true },
                style = NButtonStyle.Secondary,
                block = true,
                modifier = Modifier.padding(top = 12.dp),
            )

            // Two-step, like every other delete in this app: a workflow is
            // easier to lose than to rebuild.
            var confirmingDelete by rememberSaveable { mutableStateOf(false) }
            NButton(
                if (confirmingDelete) "Delete for good" else "Delete this workflow",
                onClick = {
                    if (confirmingDelete) {
                        state.editing?.let { viewModel.delete(it.id) }
                        onBack()
                    } else {
                        confirmingDelete = true
                    }
                },
                style = NButtonStyle.Ghost,
                block = true,
                enabled = state.editing != null && !state.running,
                modifier = Modifier.padding(top = 18.dp),
            )
        }
    }

    if (addOpen) {
        AddStepSheet(onDismiss = { addOpen = false }) { kind ->
            viewModel.addNode(kind)
            // A bracket without its closing half runs to the end of the list,
            // which is never what anybody meant. Both halves go in together
            // and the steps to repeat are dragged between them.
            when (kind) {
                NodeKind.RepeatStart, NodeKind.Batch -> viewModel.addNode(NodeKind.RepeatEnd)
                NodeKind.ForEachStart -> viewModel.addNode(NodeKind.ForEachEnd)
                else -> Unit
            }
            addOpen = false
        }
    }

    slotFor?.let { key ->
        val (nodeId, slot) = key.split('/', limit = 2).let { it[0] to it[1] }
        val node = graph.nodes.firstOrNull { it.id == nodeId }
        if (node != null) {
            BindSlotSheet(
                graph = graph,
                node = node,
                slot = slot,
                onDismiss = { slotFor = null },
                onPick = { reference ->
                    viewModel.bind(nodeId, slot, reference)
                    slotFor = null
                },
            )
        }
    }
}

/**
 * Everything that would stop this graph part-way, found before it starts.
 *
 * Not a type check — the editor already makes an invalid connection
 * unrepresentable. These are the things it cannot prevent: a required slot
 * nobody filled, a model step pointed at nothing, and a step written by a
 * build that knew a type this one does not.
 */
private fun problemsIn(graph: WorkflowGraph): List<String> = buildList {
    graph.nodes.filter { it.enabled }.forEachIndexed { index, node ->
        val kind = NodeKind.of(node.type)
        val name = node.label.ifBlank { kind.title }

        if (kind is NodeKind.Unknown) {
            add("Step ${index + 1}, $name — saved by a newer version and cannot run here.")
            return@forEachIndexed
        }
        if (kind == NodeKind.Processor &&
            (node.params["model"] as? kotlinx.serialization.json.JsonPrimitive)?.content.isNullOrBlank()
        ) {
            add("Step ${index + 1}, $name — no model chosen.")
            return@forEachIndexed
        }
        kind.slots(contextFor(node)).filter { it.required }.forEach { spec ->
            if (node.slots[spec.name].isNullOrBlank()) {
                add("Step ${index + 1}, $name — ${spec.label.lowercase()} is empty.")
            }
        }
    }
}

/** One step. The index leads, because the order is the meaning. */
@Composable
private fun StepCard(
    index: Int,
    node: NodeRecord,
    graph: WorkflowGraph,
    viewModel: WorkflowViewModel,
    models: List<ai.ondevice.data.db.ModelEntity>,
    onBindSlot: (String) -> Unit,
    onMoveUp: (() -> Unit)?,
    onMoveDown: (() -> Unit)?,
) {
    val kind = NodeKind.of(node.type)
    val context = contextFor(node)
    var expanded by rememberSaveable(node.id) { mutableStateOf(false) }

    NCard(
        Modifier.fillMaxWidth().padding(bottom = 8.dp),
        ring = if (node.enabled) NocturneColors.Neutral700 else NocturneColors.Divider,
    ) {
        Row(
            Modifier.fillMaxWidth().nClickableFlat { expanded = !expanded },
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier
                    .size(24.dp)
                    .background(NocturneColors.Accent900, Radius.Sm)
                    .ring(NocturneColors.Accent700, Radius.Sm),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    "${index + 1}",
                    style = NocturneType.Mono2Xs,
                    color = NocturneColors.Accent200,
                )
            }
            Column(Modifier.weight(1f)) {
                Text(
                    kind.family.label.uppercase(),
                    style = NocturneType.Kicker,
                    color = NocturneColors.TextMuted,
                )
                Text(
                    node.label.ifBlank { kind.title },
                    style = NocturneType.CardTitleSm,
                    color = if (node.enabled) NocturneColors.Text else NocturneColors.TextMuted,
                )
            }
            if (kind is NodeKind.Unknown) {
                NTag("cannot run", style = NTagStyle.Accent2)
            }
        }

        if (!expanded) return@NCard

        NHelp(kind.blurb, Modifier.padding(top = 6.dp))

        // What it reads. Only steps above this one can be offered — which is
        // what makes a cycle impossible rather than merely discouraged.
        val slots = kind.slots(context)
        if (slots.isNotEmpty()) {
            SectionKicker("Takes", Modifier.padding(top = 12.dp, bottom = 4.dp))
            slots.forEach { spec ->
                SlotRow(node, spec, graph, onBindSlot)
            }
        }

        StepSettings(node, kind, viewModel, models)

        Row(
            Modifier.fillMaxWidth().padding(top = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            onMoveUp?.let {
                NButton("Up", onClick = it, style = NButtonStyle.Ghost, modifier = Modifier.weight(1f))
            }
            onMoveDown?.let {
                NButton("Down", onClick = it, style = NButtonStyle.Ghost, modifier = Modifier.weight(1f))
            }
            NButton(
                if (node.enabled) "Skip" else "Use",
                onClick = { viewModel.setEnabled(node.id, !node.enabled) },
                style = NButtonStyle.Ghost,
                modifier = Modifier.weight(1f),
            )
            NButton(
                "Remove",
                onClick = { viewModel.removeNode(node.id) },
                style = NButtonStyle.Ghost,
                modifier = Modifier.weight(1f),
            )
        }
    }
}

/** One input, and what is plugged into it. */
@Composable
private fun SlotRow(
    node: NodeRecord,
    spec: SlotSpec,
    graph: WorkflowGraph,
    onBindSlot: (String) -> Unit,
) {
    val bound = node.slots[spec.name]
    val source = bound?.let { graph.producerOf(it) }
    val position = source?.let { graph.indexOf(it.id) + 1 }

    Row(
        Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp)
            .background(NocturneColors.Bg, Radius.Md)
            .ring(NocturneColors.Divider, Radius.Md)
            .nClickableFlat { onBindSlot(spec.name) }
            .padding(horizontal = 12.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(spec.label, style = NocturneType.Row)
            Text(
                when {
                    source != null -> "from step $position · ${source.label.ifBlank { NodeKind.of(source.type).title }}"
                    spec.required -> "nothing chosen"
                    else -> "optional"
                },
                style = NocturneType.MonoXs,
                color = if (source == null && spec.required) {
                    NocturneColors.Accent200
                } else {
                    NocturneColors.TextMuted
                },
            )
        }
        NTag(spec.type.label, style = NTagStyle.Outline)
    }
    if (spec.help.isNotBlank()) NHelp(spec.help, Modifier.padding(bottom = 4.dp))
}

/** The handful of settings a step has that are not model parameters. */
@Composable
private fun StepSettings(
    node: NodeRecord,
    kind: NodeKind,
    viewModel: WorkflowViewModel,
    models: List<ai.ondevice.data.db.ModelEntity>,
) {
    fun value(key: String, fallback: String = "") =
        (node.params[key] as? kotlinx.serialization.json.JsonPrimitive)?.content ?: fallback

    when (kind) {
        NodeKind.Input, NodeKind.LibraryItem -> {
            SectionKicker("Brings in", Modifier.padding(top = 12.dp, bottom = 4.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf(PortType.TEXT, PortType.IMAGE, PortType.AUDIO, PortType.FILE).forEach { type ->
                    val selected = value("portType", "TEXT") == type.name
                    NTag(
                        type.label,
                        style = if (selected) NTagStyle.Accent else NTagStyle.Outline,
                        modifier = Modifier.nClickableFlat {
                            viewModel.setParam(node.id, "portType", type.name)
                        },
                    )
                }
            }
            if (value("portType", "TEXT") == PortType.TEXT.name) {
                NTextArea(
                    value = value("text"),
                    onValueChange = { viewModel.setParam(node.id, "text", it) },
                    placeholder = "Type what this step brings in",
                    minHeight = 72.dp,
                    modifier = Modifier.padding(top = 8.dp),
                )
            } else {
                NInput(
                    value = value("path"),
                    onValueChange = { viewModel.setParam(node.id, "path", it) },
                    placeholder = "Path to the file",
                    modifier = Modifier.padding(top = 8.dp),
                )
            }
        }

        NodeKind.Script -> {
            SectionKicker("Template", Modifier.padding(top = 12.dp, bottom = 4.dp))
            NTextArea(
                value = value("template"),
                onValueChange = { viewModel.setParam(node.id, "template", it) },
                placeholder = "A summary of {{ 1.text }}",
                minHeight = 96.dp,
            )
            NHelp(
                "Refer to an earlier step by its number — {{ 2.text }}. There is trim, join, " +
                    "split, replace, match, slice, upper, lower and length, and arithmetic.",
                Modifier.padding(top = 4.dp),
            )
        }

        NodeKind.RepeatStart, NodeKind.Batch -> {
            SectionKicker("How many", Modifier.padding(top = 12.dp, bottom = 4.dp))
            NInput(
                value = value("times", "2"),
                onValueChange = { viewModel.setParam(node.id, "times", it.filter(Char::isDigit)) },
                placeholder = "2",
                keyboardType = androidx.compose.ui.text.input.KeyboardType.Number,
            )
            NHelp(
                "A count is required rather than a condition alone. One pass here can be " +
                    "minutes, so a loop that decides its own length is a way to spend an " +
                    "afternoon by accident.",
                Modifier.padding(top = 4.dp),
            )
        }

        NodeKind.Branch -> {
            SectionKicker("Only if", Modifier.padding(top = 12.dp, bottom = 4.dp))
            NInput(
                value = value("condition"),
                onValueChange = { viewModel.setParam(node.id, "condition", it) },
                placeholder = "{{ length(1.text) > 40 }}",
            )
        }

        NodeKind.Extract -> {
            SectionKicker("Pattern", Modifier.padding(top = 12.dp, bottom = 4.dp))
            NInput(
                value = value("pattern"),
                onValueChange = { viewModel.setParam(node.id, "pattern", it) },
                placeholder = "\\d+",
            )
        }

        NodeKind.TextSplit -> {
            SectionKicker("Split by", Modifier.padding(top = 12.dp, bottom = 4.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf("paragraph", "sentence", "line").forEach { by ->
                    NTag(
                        by,
                        style = if (value("by", "paragraph") == by) NTagStyle.Accent else NTagStyle.Outline,
                        modifier = Modifier.nClickableFlat { viewModel.setParam(node.id, "by", by) },
                    )
                }
            }
        }

        NodeKind.Processor -> {
            SectionKicker("Model", Modifier.padding(top = 12.dp, bottom = 4.dp))
            var picking by remember { mutableStateOf(false) }
            val chosen = models.firstOrNull { it.id == value("model") }
            Row(
                Modifier
                    .fillMaxWidth()
                    .background(NocturneColors.Bg, Radius.Md)
                    .ring(
                        if (chosen == null) NocturneColors.Accent800 else NocturneColors.Divider,
                        Radius.Md,
                    )
                    .nClickableFlat { picking = true }
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text(
                        chosen?.label ?: "Choose a model",
                        style = NocturneType.Row,
                        color = if (chosen == null) NocturneColors.Accent200 else NocturneColors.Text,
                    )
                    Text(
                        chosen?.let {
                            listOfNotNull(it.quant, it.architecture).joinToString(" · ")
                        } ?: "nothing chosen",
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMuted,
                    )
                }
                chosen?.let { NTag(it.modality.name.lowercase(), style = NTagStyle.Outline) }
            }
            NHelp(
                "What this step takes and gives follows from the model — a transcription model " +
                    "takes a recording and gives text, a diffusion model takes a prompt and " +
                    "gives a picture.",
                Modifier.padding(top = 4.dp),
            )
            if (picking) {
                ModelPickerSheet(
                    models = models,
                    chosenId = value("model"),
                    onDismiss = { picking = false },
                    onPick = {
                        viewModel.chooseModel(node.id, it)
                        picking = false
                    },
                )
            }
        }

        else -> Unit
    }
}

/** The palette, grouped the way the families are. */
@Composable
private fun AddStepSheet(onDismiss: () -> Unit, onPick: (NodeKind) -> Unit) {
    NBottomSheet("Add a step", onDismiss) {
        NodeFamily.entries.forEach { family ->
            // The closing brackets are added with their openers, never alone.
            val kinds = NodeKind.ALL.filter {
                it.family == family &&
                    it != NodeKind.RepeatEnd && it != NodeKind.ForEachEnd
            }
            if (kinds.isEmpty()) return@forEach
            SectionKicker(family.label, Modifier.padding(top = 14.dp, bottom = 6.dp))
            kinds.forEach { kind ->
                Column(
                    Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp)
                        .background(NocturneColors.Bg, Radius.Md)
                        .ring(NocturneColors.Divider, Radius.Md)
                        .nClickableFlat { onPick(kind) }
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                ) {
                    Text(kind.title, style = NocturneType.Row)
                    Text(
                        kind.blurb,
                        style = NocturneType.Help,
                        color = NocturneColors.TextMuted,
                        modifier = Modifier.padding(top = 2.dp),
                    )
                }
            }
        }
    }
}

/**
 * What may be plugged into a slot.
 *
 * Only steps above this one, and only those whose output type satisfies it —
 * so an invalid connection is not rejected, it is unrepresentable. Where
 * nothing qualifies the sheet says what is missing and what would fix it,
 * rather than showing an empty list.
 */
@Composable
private fun BindSlotSheet(
    graph: WorkflowGraph,
    node: NodeRecord,
    slot: String,
    onDismiss: () -> Unit,
    onPick: (String?) -> Unit,
) {
    val spec = NodeKind.of(node.type).slots(contextFor(node)).firstOrNull { it.name == slot }
        ?: return
    val candidates = remember(graph, node.id, slot) {
        graph.sourcesFor(node.id).flatMap { source ->
            NodeKind.of(source.type).outputs(contextFor(source))
                .filter { it.type.satisfies(spec.type) }
                .map { source to it }
        }
    }

    NBottomSheet(spec.label, onDismiss, note = spec.type.label.lowercase()) {
        if (candidates.isEmpty()) {
            NCard(Modifier.fillMaxWidth().padding(top = 8.dp)) {
                Text("Nothing above makes ${spec.type.label.lowercase()}", style = NocturneType.CardTitleSm)
                NHelp(
                    "A step can only read from steps before it. Add one that produces " +
                        "${spec.type.label.lowercase()} above this one, then come back.",
                    Modifier.padding(top = 4.dp),
                )
            }
            return@NBottomSheet
        }

        candidates.forEach { (source, output) ->
            val position = graph.indexOf(source.id) + 1
            val reference = "${source.id}:${output.name}"
            val selected = node.slots[slot] == reference
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(bottom = 6.dp)
                    .background(
                        if (selected) NocturneColors.Accent900 else NocturneColors.Bg,
                        Radius.Md,
                    )
                    .ring(if (selected) NocturneColors.Accent else NocturneColors.Divider, Radius.Md)
                    .nClickableFlat { onPick(reference) }
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    "$position",
                    style = NocturneType.Mono2Xs,
                    color = NocturneColors.TextMuted,
                    modifier = Modifier.width(18.dp),
                )
                Column(Modifier.weight(1f)) {
                    Text(
                        source.label.ifBlank { NodeKind.of(source.type).title },
                        style = NocturneType.Row,
                        color = if (selected) NocturneColors.Accent200 else NocturneColors.Text,
                    )
                    Text(output.label, style = NocturneType.MonoXs, color = NocturneColors.TextMuted)
                }
                NTag(output.type.label, style = NTagStyle.Outline)
            }
        }

        if (!spec.required) {
            NButton(
                "Leave empty",
                onClick = { onPick(null) },
                style = NButtonStyle.Ghost,
                block = true,
                modifier = Modifier.padding(top = 6.dp),
            )
        }
    }
}

/**
 * Every installed model, grouped by what it does.
 *
 * Grouped rather than listed flat because the group is the useful fact: a step
 * pointed at a transcription model is a different step from one pointed at a
 * diffusion model, and the heading says so before the name does.
 */
@Composable
private fun ModelPickerSheet(
    models: List<ai.ondevice.data.db.ModelEntity>,
    chosenId: String,
    onDismiss: () -> Unit,
    onPick: (ai.ondevice.data.db.ModelEntity) -> Unit,
) {
    NBottomSheet("Choose a model", onDismiss, note = "${'$'}{models.size} installed") {
        if (models.isEmpty()) {
            NCard(Modifier.fillMaxWidth().padding(top = 8.dp)) {
                Text("No models installed", style = NocturneType.CardTitleSm)
                NHelp(
                    "A step needs something to run. Add one under Settings → Models.",
                    Modifier.padding(top = 4.dp),
                )
            }
            return@NBottomSheet
        }
        models.groupBy { it.modality }.forEach { (modality, group) ->
            SectionKicker(modality.name.lowercase(), Modifier.padding(top = 14.dp, bottom = 6.dp))
            group.forEach { model ->
                val selected = model.id == chosenId
                Row(
                    Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp)
                        .background(
                            if (selected) NocturneColors.Accent900 else NocturneColors.Bg,
                            Radius.Md,
                        )
                        .ring(
                            if (selected) NocturneColors.Accent else NocturneColors.Divider,
                            Radius.Md,
                        )
                        .nClickableFlat { onPick(model) }
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(
                            model.label,
                            style = NocturneType.Row,
                            color = if (selected) NocturneColors.Accent200 else NocturneColors.Text,
                        )
                        Text(
                            listOfNotNull(model.quant, model.architecture).joinToString(" · "),
                            style = NocturneType.MonoXs,
                            color = NocturneColors.TextMuted,
                        )
                    }
                }
            }
        }
    }
}

/**
 * What the editor knows when asking a step for its shape.
 *
 * A Processor's slots depend on the model it points at, which the editor does
 * not resolve here — so until a model is chosen it shows the generic shape and
 * fills in once it can. The alternative, asking the database from a composable,
 * is worse.
 */
private fun contextFor(node: NodeRecord): NodeContext {
    val declared = (node.params["portType"] as? kotlinx.serialization.json.JsonPrimitive)?.content
    val shape = (node.params["shape"] as? kotlinx.serialization.json.JsonPrimitive)?.content
    return NodeContext(
        // NONE until a model is chosen, rather than guessing at image. A step
        // that has not been pointed at anything yet takes nothing, and saying
        // it takes a prompt and a picture is a promise about a decision
        // nobody has made.
        shape = shape?.let {
            runCatching { ai.ondevice.core.workflow.ProcessorShape.valueOf(it) }.getOrNull()
        } ?: ai.ondevice.core.workflow.ProcessorShape.NONE,
        declaredType = declared?.let { runCatching { PortType.valueOf(it) }.getOrNull() }
            ?: PortType.TEXT,
    )
}
