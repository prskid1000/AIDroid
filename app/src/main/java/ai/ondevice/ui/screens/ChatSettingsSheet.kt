package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
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
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import ai.ondevice.core.Fmt
import ai.ondevice.core.Tier
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.PersonaEntity
import ai.ondevice.ui.components.NBottomSheet
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NDot
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NSeg
import ai.ondevice.ui.components.NSheetHandle
import ai.ondevice.ui.components.NSlider
import ai.ondevice.ui.components.NTextArea
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.vm.ExportFormat
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.theme.ruleBelow
import ai.ondevice.ui.vm.ChatState

/**
 * **S7 — Chat settings.**
 *
 * A sheet, not a screen: model, persona, system prompt, preset, and the Basic
 * tier inline. The header says "reload not required" because everything on this
 * sheet is live-editable — parameters that *do* need a reload are batched on
 * the All-parameters screen instead (SPEC §9).
 */
@Composable
fun ChatSettingsSheet(
    state: ChatState,
    onDismiss: () -> Unit,
    onSelectModel: (ModelEntity) -> Unit,
    onSelectPreset: (String) -> Unit,
    onSelectPersona: (PersonaEntity) -> Unit,
    onSystemPromptChange: (String) -> Unit,
    onLiveParam: (String, Any?) -> Unit,
    onOpenParametersAtTier: (Tier) -> Unit,
    onExport: (ExportFormat) -> Unit,
    onExportAll: () -> Unit,
    onImport: () -> Unit,
) {
    // The same chrome Image and Voice use. It was written inline here first,
    // which is how all three ended up drawing their last row underneath the
    // navigation bar — one of them fixed is one of them fixed.
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
                                state.model?.displayName ?: "No model selected",
                                style = NocturneType.CardTitleSm,
                            )
                            Text(
                                listOfNotNull(
                                    state.model?.quant,
                                    // Only claim "loaded" when the engine
                                    // actually holds it. The conversation's
                                    // preferred model is not the resident one.
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
                                            model.displayName,
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

                    // — persona —
                    SectionKicker("Persona", Modifier.padding(top = 20.dp, bottom = 8.dp))
                    Row(
                        Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(7.dp),
                    ) {
                        state.personas.forEach { persona ->
                            val selected = persona.id == state.selectedPersonaId
                            Column(
                                Modifier
                                    .widthIn(min = 104.dp)
                                    .background(
                                        if (selected) NocturneColors.Accent900 else NocturneColors.Bg,
                                        Radius.Md,
                                    )
                                    .ring(
                                        if (selected) NocturneColors.Accent else NocturneColors.Divider,
                                        Radius.Md,
                                    )
                                    .nClickableFlat { onSelectPersona(persona) }
                                    .padding(horizontal = 12.dp, vertical = 9.dp),
                            ) {
                                Text(
                                    persona.name,
                                    style = NocturneType.Row.copy(fontWeight = androidx.compose.ui.text.font.FontWeight.Medium),
                                    color = if (selected) NocturneColors.Accent200 else NocturneColors.Text,
                                )
                                Text(
                                    listOfNotNull(persona.defaultVoice, persona.defaultPresetId?.substringAfter('-'))
                                        .joinToString(" · "),
                                    style = NocturneType.Help,
                                    color = if (selected) NocturneColors.Accent300 else NocturneColors.TextMuted,
                                )
                            }
                        }
                        Box(
                            Modifier
                                .widthIn(min = 70.dp)
                                .background(NocturneColors.Bg, Radius.Md)
                                .ring(NocturneColors.Divider, Radius.Md)
                                .padding(vertical = 13.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            Icon(
                                NIcons.PlusThin,
                                contentDescription = "New persona",
                                tint = NocturneColors.Text.copy(alpha = 0.6f),
                                modifier = Modifier.size(16.dp),
                            )
                        }
                    }

                    // — export and import —
                    //
                    // On the sheet rather than behind a menu because SPEC §13
                    // makes getting a conversation *out* a first-class action,
                    // not an advanced one.
                    SectionKicker("This conversation", Modifier.padding(top = 20.dp, bottom = 8.dp))
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(7.dp),
                    ) {
                        NButton(
                            "Export .zip",
                            { onExport(ExportFormat.ARCHIVE) },
                            modifier = Modifier.weight(1f),
                        )
                        NButton(
                            "Export .md",
                            { onExport(ExportFormat.MARKDOWN) },
                            modifier = Modifier.weight(1f),
                        )
                    }
                    Row(
                        Modifier.fillMaxWidth().padding(top = 7.dp),
                        horizontalArrangement = Arrangement.spacedBy(7.dp),
                    ) {
                        NButton("Export all", onExportAll, modifier = Modifier.weight(1f))
                        NButton("Import…", onImport, modifier = Modifier.weight(1f))
                    }
                    NHelp(
                        "The archive round-trips: parameters, tok/s, backend and attachments all come " +
                            "back. Markdown is for reading — it does not import.",
                        Modifier.padding(top = 6.dp),
                    )
                    state.importSummary?.let { summary ->
                        NHelp(summary, Modifier.padding(top = 6.dp))
                    }
                    state.lastExport?.let { path ->
                        NHelp("Last export: ${path.substringAfterLast('/')}", Modifier.padding(top = 6.dp))
                    }

                    // — system prompt —
                    SectionKicker("System prompt", Modifier.padding(top = 20.dp, bottom = 8.dp))
                    NTextArea(
                        value = state.systemPrompt,
                        onValueChange = onSystemPromptChange,
                        minHeight = 74.dp,
                        textStyle = NocturneType.Row,
                    )

                    // — preset —
                    SectionKicker(
                        "Preset",
                        Modifier.padding(top = 20.dp, bottom = 8.dp),
                        trailing = {
                            Text("Save as…", style = NocturneType.Meta, color = NocturneColors.Accent)
                        },
                    )
                    if (state.presets.isNotEmpty()) {
                        NSeg(
                            options = state.presets.map { it.name },
                            selectedIndex = state.presets.indexOfFirst { it.id == state.selectedPresetId }
                                .coerceAtLeast(0),
                            onSelect = { onSelectPreset(state.presets[it].id) },
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }

                    // — Basic tier, inline —
                    SectionKicker(
                        "Basic",
                        Modifier.padding(top = 20.dp, bottom = 4.dp),
                        trailing = {
                            Text(
                                "${Tier.BASIC.label.lowercase()} tier",
                                style = NocturneType.Help,
                                color = NocturneColors.TextMuted,
                            )
                        },
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

                    Column(Modifier.fillMaxWidth().ruleBelow().padding(vertical = 10.dp)) {
                        Row(
                            Modifier.fillMaxWidth().padding(bottom = 5.dp),
                            horizontalArrangement = Arrangement.spacedBy(7.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text("Max tokens", style = NocturneType.Row, modifier = Modifier.weight(1f))
                            Text(
                                (state.liveOverrides.int("n_predict") ?: -1).toString(),
                                style = NocturneType.MonoValue,
                                color = NocturneColors.Accent300,
                            )
                        }
                        NHelp("−1 generates until EOS or the context fills.")
                    }

                    NButton(
                        "Advanced parameters",
                        onClick = { onOpenParametersAtTier(Tier.ADVANCED) },
                        style = NButtonStyle.Secondary,
                        block = true,
                        modifier = Modifier.padding(top = 14.dp),
                    )
                    NButton(
                        "All parameters →",
                        onClick = { onOpenParametersAtTier(Tier.EXPERT) },
                        style = NButtonStyle.Primary,
                        block = true,
                    )
    }
}

/**
 * The inline slider row from S7. It looks like a parameter row but is not one:
 * the real rows are generated from the manifest on S8. This is the Basic tier's
 * hand-placed surface, which §9 explicitly allows ("inline in the generation
 * screen") — it still reads and writes through the same sparse map.
 */
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
