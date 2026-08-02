package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ai.ondevice.core.Fmt
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.ui.components.NBottomSheet
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NDot
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NSlider
import ai.ondevice.ui.components.NTextArea
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.theme.ruleBelow
import ai.ondevice.ui.vm.ChatState

/** **S7 — Chat settings.** A sheet, not a screen: model, system prompt, thinking, chat template, and the Basic tier inline. */
@Composable
fun ChatSettingsSheet(
    state: ChatState,
    onDismiss: () -> Unit,
    onSelectModel: (ModelEntity) -> Unit,
    onSystemPromptChange: (String) -> Unit,
    onChatTemplateChange: (String?) -> Unit,
    onTemplateKwargsChange: (String) -> Unit,
    onLiveParam: (String, Any?) -> Unit,
    onOpenParameters: () -> Unit,
) {
    // The same chrome Image and Voice use.
    NBottomSheet("This conversation", onDismiss, note = "reload not required") {
                    // — model —
                    SectionKicker("Model", Modifier.padding(bottom = 8.dp))
                    var modelsExpanded by remember { mutableStateOf(false) }
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .background(NocturneColors.Bg, Radius.Md)
                            .ring(NocturneColors.Accent700, Radius.Md)
                            // The row *is* the control. It has always looked
                            // like a dropdown; it now behaves like one.
                            .nClickableFlat { modelsExpanded = !modelsExpanded }
                            .padding(horizontal = 12.dp, vertical = 11.dp),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                state.model?.label ?: "No model selected",
                                style = NocturneType.CardTitleSm,
                            )
                            Text(
                                listOfNotNull(
                                    state.model?.quant,
                                    // Only claim "loaded" when the engine actually holds it.
                                    when {
                                        state.loadingModel -> "loading…"
                                        state.model != null && state.loadedModelId == state.model.id -> "loaded"
                                        state.model != null -> "not loaded"
                                        else -> null
                                    },
                                    state.tokensPerSecond.takeIf { it > 0 }?.let { Fmt.tokensPerSecond(it) },
                                ).joinToString(" · "),
                                style = NocturneType.MonoXs,
                                color = NocturneColors.TextMuted,
                            )
                        }
                        Text(
                            "${state.availableModels.size}",
                            style = NocturneType.MonoXs,
                            color = NocturneColors.TextMuted,
                        )
                        Icon(
                            NIcons.ChevronDown,
                            contentDescription = if (modelsExpanded) "Collapse" else "Choose a model",
                            tint = NocturneColors.Text,
                            modifier = Modifier
                                .size(16.dp)
                                .rotate(if (modelsExpanded) 180f else 0f),
                        )
                    }
                    NHelp(
                        "Switching unloads this model first — the app never holds two at once.",
                        Modifier.padding(top = 6.dp),
                    )

                    if (modelsExpanded) {
                        if (state.availableModels.isEmpty()) {
                            NHelp(
                                "No text models installed. Models → Add.",
                                Modifier.padding(top = 8.dp),
                            )
                        }
                        Column(
                            Modifier.fillMaxWidth().padding(top = 8.dp),
                            verticalArrangement = Arrangement.spacedBy(5.dp),
                        ) {
                            state.availableModels.forEach { model ->
                                val selected = model.id == state.model?.id
                                Row(
                                    Modifier
                                        .fillMaxWidth()
                                        .background(
                                            if (selected) NocturneColors.Accent900 else NocturneColors.Bg,
                                            Radius.Md,
                                        )
                                        .ring(
                                            if (selected) NocturneColors.Accent else NocturneColors.Divider,
                                            Radius.Md,
                                        )
                                        .nClickableFlat {
                                            onSelectModel(model)
                                            modelsExpanded = false
                                        }
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
                                            listOfNotNull(model.quant, model.architecture)
                                                .joinToString(" · "),
                                            style = NocturneType.MonoXs,
                                            color = NocturneColors.TextMuted,
                                        )
                                    }
                                    Text(
                                        Fmt.bytes(model.sizeBytes),
                                        style = NocturneType.MonoXs,
                                        color = NocturneColors.TextMuted,
                                    )
                                }
                            }
                        }
                    }

                    state.importSummary?.let { summary ->
                        NHelp(summary, Modifier.padding(top = 20.dp))
                    }

                    // — system prompt —
                    SectionKicker("System prompt", Modifier.padding(top = 20.dp, bottom = 8.dp))
                    NTextArea(
                        value = state.systemPrompt,
                        onValueChange = onSystemPromptChange,
                        minHeight = 74.dp,
                        textStyle = NocturneType.Row,
                    )

                    // — template arguments —
                    //
                    // A model card that says `--chat-template-kwargs '{"…":…}'`
                    // is naming exactly this, so it is offered whole rather
                    // than as a switch per key somebody has to add each time —
                    // enable_thinking included.
                    SectionKicker("Template arguments", Modifier.padding(top = 20.dp, bottom = 8.dp))
                    var kwargsDraft by remember(state.templateKwargsJson) {
                        mutableStateOf(state.templateKwargsJson)
                    }
                    NTextArea(
                        value = kwargsDraft,
                        onValueChange = { kwargsDraft = it },
                        minHeight = 56.dp,
                        textStyle = NocturneType.MonoXs,
                        placeholder = """{"enable_thinking": false}""",
                    )
                    NButton(
                        "Apply arguments",
                        onClick = { onTemplateKwargsChange(kwargsDraft) },
                        style = NButtonStyle.Secondary,
                        block = true,
                        enabled = kwargsDraft.isNotBlank() && kwargsDraft != state.templateKwargsJson,
                        modifier = Modifier.padding(top = 8.dp),
                    )
                    NHelp(
                        "The same JSON llama.cpp takes as --chat-template-kwargs, handed to the " +
                            "template as-is. Keys it does not read are ignored by the template, " +
                            "not by this app.",
                        Modifier.padding(top = 6.dp),
                    )

                    // — chat template —
                    SectionKicker(
                        "Chat template",
                        Modifier.padding(top = 20.dp, bottom = 8.dp),
                        trailing = {
                            Text(
                                if (state.templateSource == "override") "overridden" else "from the GGUF",
                                style = NocturneType.Help,
                                color = if (state.templateSource == "override") {
                                    NocturneColors.Accent300
                                } else {
                                    NocturneColors.TextMuted
                                },
                            )
                        },
                    )
                    var templateOpen by remember { mutableStateOf(false) }
                    var draft by remember(state.chatTemplate) {
                        mutableStateOf(state.chatTemplate.orEmpty())
                    }
                    if (!templateOpen) {
                        NButton(
                            if (state.chatTemplate == null) "No model loaded" else "Edit template",
                            onClick = { templateOpen = true },
                            style = NButtonStyle.Secondary,
                            block = true,
                            enabled = state.chatTemplate != null,
                        )
                        NHelp(
                            "The Jinja the runtime renders every turn through. Editing it is how " +
                                "a model with a broken or missing template gets a working one.",
                            Modifier.padding(top = 6.dp),
                        )
                    } else {
                        NTextArea(
                            value = draft,
                            onValueChange = { draft = it },
                            minHeight = 180.dp,
                            textStyle = NocturneType.MonoXs,
                        )
                        Row(
                            Modifier.fillMaxWidth().padding(top = 8.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            NButton(
                                "Apply and reload",
                                onClick = {
                                    onChatTemplateChange(draft)
                                    templateOpen = false
                                },
                                style = NButtonStyle.Primary,
                                modifier = Modifier.weight(1f),
                            )
                            NButton(
                                "Reset",
                                onClick = {
                                    onChatTemplateChange(null)
                                    templateOpen = false
                                },
                                style = NButtonStyle.Secondary,
                                modifier = Modifier.weight(1f),
                            )
                        }
                        NHelp(
                            "Applying reloads the model: llama.cpp builds its parser and its stop " +
                                "sequences from the template once, at load.",
                            Modifier.padding(top = 6.dp),
                        )
                    }

                    // — Basic tier, inline —
                    SectionKicker(
                        "Basic",
                        Modifier.padding(top = 20.dp, bottom = 4.dp),
                    )

                    BasicParamSlider(
                        label = "Temperature",
                        value = state.liveOverrides.float("temp") ?: 0.7f,
                        range = 0f..2f,
                        format = { String.format("%.2f", it) },
                        help = "Higher is more random. 0 is greedy.",
                        defaultLabel = "0.80",
                        modified = state.liveOverrides.float("temp") != null,
                        onChange = { onLiveParam("temp", it) },
                        onReset = { onLiveParam("temp", null) },
                    )

                    // One button, because there was only ever one screen: the
                    // other opened it at a different tier, which is the first
                    // control on it.
                    NButton(
                        "All Parameters",
                        onClick = { onOpenParameters() },
                        style = NButtonStyle.Primary,
                        block = true,
                        modifier = Modifier.padding(top = 14.dp),
                    )
    }
}

/** The inline slider row from S7. */
@Composable
private fun BasicParamSlider(
    label: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    format: (Float) -> String,
    help: String,
    defaultLabel: String,
    modified: Boolean,
    onChange: (Float) -> Unit,
    onReset: () -> Unit,
) {
    Column(Modifier.fillMaxWidth().ruleBelow().padding(vertical = 10.dp)) {
        Row(
            Modifier.fillMaxWidth().padding(bottom = 5.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(label, style = NocturneType.Row, modifier = Modifier.weight(1f))
            if (modified) NDot(size = 6.dp)
            Text(format(value), style = NocturneType.MonoValue, color = NocturneColors.Accent300)
        }
        NSlider(value = value, onValueChange = onChange, valueRange = range)
        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            NHelp(help)
            if (modified) {
                Text(
                    "reset to $defaultLabel",
                    style = NocturneType.Help,
                    color = NocturneColors.Accent,
                    modifier = Modifier.nClickableFlat(onClick = onReset),
                )
            }
        }
    }
}
