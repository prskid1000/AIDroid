package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.core.Modality
import ai.ondevice.core.VerdictTone
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NDialog
import ai.ondevice.ui.components.NDialogActions
import ai.ondevice.ui.components.NDialogBody
import ai.ondevice.ui.components.NDialogTitle
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NIconButton
import ai.ondevice.ui.components.NSlider
import ai.ondevice.ui.components.NTable
import ai.ondevice.ui.components.NTableHeaderCell
import ai.ondevice.ui.components.NTableRow
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ruleBelow
import ai.ondevice.ui.vm.ModelDetailViewModel

/** **S2 — Model detail.** The screen the canvas annotates "drag the context slider — the verdict recomputes". */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ModelDetailScreen(
    modelId: String,
    onBack: () -> Unit,
    onOpenParameters: (String) -> Unit,
    viewModel: ModelDetailViewModel = hiltViewModel(),
) {
    LaunchedEffect(modelId) { viewModel.bind(modelId) }
    val state by viewModel.state.collectAsStateWithLifecycle()
    val model = state.model

    var confirmingDelete by rememberSaveable { mutableStateOf(false) }

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = model?.let { "${it.label} · ${it.quant.orEmpty()}" } ?: "Model",
                onBack = onBack,
                trailing = {
                    // The actions live up here, as icons, the way every other push screen in the app puts them.
                    val isAddOn = model?.attachmentRole != null
                    if (model != null && !isAddOn) {
                        NIconButton(
                            NIcons.Pin,
                            if (model.pinned) "Unpin from RAM" else "Keep loaded in RAM",
                            onClick = viewModel::togglePin,
                            style = if (model.pinned) NButtonStyle.Primary else NButtonStyle.Secondary,
                            size = 34.dp,
                            iconSize = 15.dp,
                        )
                        // Only while there is something to unload.
                        if (state.loaded) {
                            NIconButton(
                                NIcons.Eject,
                                "Unload from RAM",
                                onClick = viewModel::unload,
                                size = 34.dp,
                                iconSize = 15.dp,
                            )
                        }
                        NIconButton(
                            NIcons.Settings,
                            "Parameters",
                            onClick = {
                                onOpenParameters(state.paramRuntimeId)
                            },
                            size = 34.dp,
                            iconSize = 15.dp,
                        )
                    }
                    // Delete is its own icon rather than a ⋮ menu.
                    NIconButton(
                        NIcons.Trash,
                        "Delete model",
                        onClick = { confirmingDelete = true },
                        size = 34.dp,
                        iconSize = 15.dp,
                    )
                },
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        if (model == null) {
            NHelp("Loading…")
            return@PhoneScaffold
        }

        Column(Modifier.verticalScroll(rememberScrollState())) {

            FlowRow(
                Modifier.fillMaxWidth().padding(bottom = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(5.dp),
                verticalArrangement = Arrangement.spacedBy(5.dp),
            ) {
                if (state.loaded) NTag("loaded", style = NTagStyle.Accent)
                model.revision?.let { NTag("rev ${it.take(7)}", style = NTagStyle.Neutral) }
                if (model.sha256 != null) NTag("sha256 ✓", style = NTagStyle.Neutral)
                if (model.pinned) NTag("pinned in RAM", style = NTagStyle.Outline)
            }

            // A name of your own, which every list then uses.
            //
            // The app can only qualify what it derived — a role, a quant, a
            // folder — and that tells two rows apart without saying which one
            // you meant. Naming it settles that outright, and this is the one
            // screen that is unambiguously about this model.
            SectionKicker("Name", Modifier.padding(bottom = 8.dp))
            var typedLabel by rememberSaveable(model.id) {
                mutableStateOf(model.customLabel.orEmpty())
            }
            NInput(
                value = typedLabel,
                onValueChange = { typedLabel = it },
                placeholder = model.displayName,
                minHeight = 42.dp,
            )
            Row(
                Modifier.fillMaxWidth().padding(top = 6.dp, bottom = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                NButton(
                    "Save name",
                    onClick = { viewModel.setCustomLabel(typedLabel) },
                    style = NButtonStyle.Primary,
                    enabled = typedLabel.trim() != model.customLabel.orEmpty(),
                    modifier = Modifier.weight(1f),
                )
                if (!model.customLabel.isNullOrBlank()) {
                    NButton(
                        "Use repo name",
                        onClick = {
                            typedLabel = ""
                            viewModel.setCustomLabel("")
                        },
                        modifier = Modifier.weight(1f),
                    )
                }
            }
            NHelp(
                "Shown wherever this model is listed or picked. Leave it empty and the repo's " +
                    "name is used, qualified with the role or quant only when something else " +
                    "shares it.",
                Modifier.padding(bottom = 18.dp),
            )

            // Not every model has every property, and this screen used to act as though they all did.
            val isAddOn = model.attachmentRole != null
            val hasContextWindow = !isAddOn &&
                (model.modality == Modality.TEXT || model.modality == Modality.VISION)

            if (hasContextWindow) {
            SectionKicker("Context window", Modifier.padding(bottom = 10.dp))
            Row(
                Modifier.fillMaxWidth().padding(bottom = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.Bottom,
            ) {
                Text(
                    Fmt.contextLabel(state.contextTokens),
                    style = NocturneType.Numeral,
                    color = NocturneColors.Accent200,
                )
                Text(
                    "${Fmt.grouped(state.contextTokens)} tokens · max ${Fmt.grouped(state.maxContext)}",
                    style = NocturneType.MonoSm,
                    color = NocturneColors.TextMuted,
                )
            }
            NSlider(
                value = state.contextTokens.toFloat(),
                onValueChange = { viewModel.setContext((it / 2048).toInt() * 2048) },
                onValueChangeFinished = viewModel::commitContext,
                valueRange = 2048f..state.maxContext.coerceAtMost(65_536).toFloat(),
                height = 28.dp,
            )

            // The verdict card, recomputed on every slider tick.
            state.estimate?.let { estimate ->
                val tone = state.verdict?.tone ?: VerdictTone.CAVEAT
                NCard(
                    Modifier.padding(top = 6.dp),
                    padding = PaddingValues(0.dp),
                    gap = 0.dp,
                ) {
                    Column(
                        Modifier
                            .fillMaxWidth()
                            .background(verdictGround(tone))
                            .padding(horizontal = 13.dp, vertical = 11.dp),
                    ) {
                        Row(
                            Modifier.fillMaxWidth().padding(bottom = 6.dp),
                            horizontalArrangement = Arrangement.spacedBy(7.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            state.verdict?.let { VerdictTag(it) }
                            Text(
                                "≈ ${Fmt.gb(estimate.totalBytes)} GB resident",
                                style = NocturneType.CardTitleSm,
                                color = verdictInk(tone),
                            )
                        }
                        estimate.longWorking().forEach { line ->
                            Text(
                                line,
                                style = NocturneType.MonoSm,
                                color = verdictInk(tone).copy(alpha = 0.85f),
                            )
                        }
                        Text(
                            buildString {
                                append(
                                    "You have ${Fmt.gb(state.availableRamBytes)} GB free of " +
                                        "${Fmt.gb(state.totalRamBytes)} GB.",
                                )
                            },
                            style = NocturneType.Help,
                            color = verdictInk(tone).copy(alpha = 0.7f),
                            modifier = Modifier.padding(top = 6.dp),
                        )
                    }
                }
            }
            }


            if (state.companions.isNotEmpty()) {
                SectionKicker("Companions · auto-paired", Modifier.padding(top = 22.dp, bottom = 8.dp))
                state.companions.forEach { (role, path) ->
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .ruleBelow()
                            .padding(vertical = 10.dp),
                        horizontalArrangement = Arrangement.spacedBy(9.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            NIcons.Image,
                            contentDescription = null,
                            tint = NocturneColors.Accent,
                            modifier = Modifier.size(16.dp),
                        )
                        Text(
                            path.substringAfterLast('/'),
                            style = NocturneType.MonoXs,
                            modifier = Modifier.weight(1f),
                        )
                        Text(role, style = NocturneType.Meta, color = NocturneColors.TextMuted)
                    }
                }
                NHelp(
                    "Detected in the repo and queued with the weights. Image input is unavailable " +
                        "without the projector.",
                    Modifier.padding(top = 9.dp),
                )
            }

            if (state.files.isNotEmpty()) {
                ai.ondevice.ui.components.SectionKicker(
                    "Files · ${state.files.size} · ${Fmt.bytes(state.filesTotalBytes)}",
                    Modifier.padding(top = 20.dp, bottom = 8.dp),
                )
                state.files.forEach { file ->
                    Row(
                        Modifier.fillMaxWidth().padding(vertical = 6.dp),
                        horizontalArrangement = Arrangement.spacedBy(9.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            file.name,
                            style = NocturneType.MonoXs,
                            color = if (file.isPrimary) NocturneColors.Accent200 else NocturneColors.Text,
                            modifier = Modifier.weight(1f),
                        )
                        Text(
                            Fmt.bytes(file.sizeBytes),
                            style = NocturneType.Meta,
                            color = NocturneColors.TextMuted,
                        )
                    }
                }
                NHelp(
                    "What is on disk right now, read from the folder rather than from the download " +
                        "record. The accented row is the file handed to the runtime; a model that " +
                        "arrived incomplete shows it here instead of at the moment it fails to load.",
                    Modifier.padding(top = 9.dp),
                )
            }

            // Nothing down here any more.
        }

        if (confirmingDelete) {
            NDialog(onDismissRequest = { confirmingDelete = false }) {
                // The name goes in the body, not the title.
                NDialogTitle("Delete this model?")
                // The size is the point of the sentence. "Are you sure?" is a
                // question nobody can answer; "801 MB, downloaded again" is.
                NDialogBody(
                    "${model.label} — " +
                        "${Fmt.bytes(state.filesTotalBytes.coerceAtLeast(model.sizeBytes))} " +
                        "removed from this device. Nothing else changes: conversations, images and " +
                        "transcripts made with it stay. Getting it back means downloading it again.",
                )
                NDialogActions {
                    NButton(
                        "Cancel",
                        onClick = { confirmingDelete = false },
                        style = NButtonStyle.Secondary,
                    )
                    NButton(
                        "Delete",
                        onClick = {
                            confirmingDelete = false
                            viewModel.delete(onBack)
                        },
                        style = NButtonStyle.Primary,
                    )
                }
            }
        }
    }
}

