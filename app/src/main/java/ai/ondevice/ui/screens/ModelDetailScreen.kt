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
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ruleBelow
import ai.ondevice.ui.vm.ModelDetailViewModel

/**
 * **S2 — Model detail.**
 *
 * The screen the canvas annotates "drag the context slider — the verdict
 * recomputes". That live recompute is the whole point: SPEC §3.3 requires the
 * KV term be recalculated as the user moves the slider and the arithmetic
 * shown, so a "won't fit" is something they can act on rather than a verdict
 * handed down.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ModelDetailScreen(
    modelId: String,
    onBack: () -> Unit,
    onOpenParameters: (ai.ondevice.core.Tier, String) -> Unit,
    viewModel: ModelDetailViewModel = hiltViewModel(),
) {
    LaunchedEffect(modelId) { viewModel.bind(modelId) }
    val state by viewModel.state.collectAsStateWithLifecycle()
    val model = state.model

    var confirmingDelete by rememberSaveable { mutableStateOf(false) }

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = model?.let { "${it.displayName} · ${it.quant.orEmpty()}" } ?: "Model",
                onBack = onBack,
                trailing = {
                    // The actions live up here, as icons, the way every other
                    // push screen in the app puts them. They used to be a row
                    // of full-width text buttons pinned under the file list —
                    // reachable only after scrolling past everything, and a
                    // different shape from the same actions elsewhere.
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
                        // Only while there is something to unload. The other
                        // half of "Keep loaded": until now a model could be
                        // pinned and never released, and the only ways to get
                        // several gigabytes back were to load a different model
                        // or to kill the app.
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
                                onOpenParameters(ai.ondevice.core.Tier.EXPERT, state.paramRuntimeId)
                            },
                            size = 34.dp,
                            iconSize = 15.dp,
                        )
                    }
                    // Delete is its own icon rather than a ⋮ menu. The overflow
                    // held exactly one item, which is a menu that exists to
                    // hide a button — and as a bare 20 dp Icon it was a smaller
                    // tap target than everything beside it, so it read as
                    // decoration whether or not it worked. The confirmation is
                    // what actually protects the action; the indirection did not.
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

            // Not every model has every property, and this screen used to act as
            // though they all did. A ControlNet has no context window and is not
            // something an engine loads on its own — yet it was offered a context
            // slider to drag and a "keep loaded" pin, both of which describe a
            // language model. An add-on is a file another model reads; the only
            // honest things to say about it are what it is, where it came from
            // and what is on disk.
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
                                if (estimate.exceedsHexagonSession) {
                                    append(
                                        " Past the ~3.5 GB Hexagon session cap — this needs a layer " +
                                            "split across HTP sessions, or it falls back to OpenCL.",
                                    )
                                }
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

            // Nothing down here any more. Pin, Unload and Parameters are
            // toolbar icons; Delete is behind the ⋮ with a confirmation, since
            // it is the one action on this screen that cannot be undone and
            // costs a multi-gigabyte re-download — and it used to sit between
            // two reversible buttons at thumb height with nothing in the way.
        }

        if (confirmingDelete) {
            NDialog(onDismissRequest = { confirmingDelete = false }) {
                NDialogTitle("Delete ${model.displayName}?")
                // The size is the point of the sentence. "Are you sure?" is a
                // question nobody can answer; "6.2 GB, downloaded again" is.
                NDialogBody(
                    "This removes ${Fmt.bytes(state.filesTotalBytes.coerceAtLeast(model.sizeBytes))} " +
                        "from this device. Nothing else changes — conversations, images and " +
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

