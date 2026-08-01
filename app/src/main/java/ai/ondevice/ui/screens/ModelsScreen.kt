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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.core.Modality
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCardMeta
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NIconButton
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NMetaText
import ai.ondevice.ui.components.NStackedBar
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.ModelsViewModel

/** **S3 — Models library.** Residency, storage and orphans on one screen. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ModelsScreen(
    onBack: () -> Unit,
    onAddModel: () -> Unit,
    onOpenModel: (String) -> Unit,
    onOpenDownloads: () -> Unit,
    viewModel: ModelsViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    PhoneScaffold(
        // A push off Settings rather than a sixth tab.
        toolbar = {
            PushToolbar(
                title = "Models",
                onBack = onBack,
                trailing = {
                    NIconButton(
                        NIcons.Plus,
                        "Add model",
                        onClick = onAddModel,
                        style = NButtonStyle.Primary,
                        size = 34.dp,
                        iconSize = 15.dp,
                    )
                },
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        NInput(
            value = state.filter,
            onValueChange = viewModel::onFilterChange,
            placeholder = "Filter installed models",
            minHeight = 40.dp,
            modifier = Modifier.padding(bottom = 10.dp),
        )

        Column(Modifier.verticalScroll(rememberScrollState())) {

            // The download queue, always reachable.
            if (state.hasDownloadNews) {
                Row(
                    Modifier
                        .fillMaxWidth()
                        .padding(bottom = 10.dp)
                        .background(NocturneColors.Surface, Radius.Md)
                        .ring(
                            if (state.failedDownloads > 0) NocturneColors.Neutral700 else NocturneColors.Accent700,
                            Radius.Md,
                        )
                        .nClickableFlat(onClick = onOpenDownloads)
                        .padding(horizontal = 12.dp, vertical = 11.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text("Downloads", style = NocturneType.Row)
                        Text(
                            state.downloadSummary,
                            style = NocturneType.MonoXs,
                            color = if (state.failedDownloads > 0) {
                                NocturneColors.Neutral300
                            } else {
                                NocturneColors.Accent300
                            },
                        )
                    }
                    Text("→", style = NocturneType.Row, color = NocturneColors.Accent)
                }
            }

            // Disk usage grouped by modality (SPEC §3.5).
            val byModality = state.byModality
            val textBytes = byModality.filterKeys {
                it == Modality.TEXT || it == Modality.EMBEDDING
            }.values.sum()
            val visionBytes = byModality[Modality.VISION] ?: 0
            val diffusionBytes = byModality[Modality.DIFFUSION] ?: 0
            val speechBytes = byModality.filterKeys {
                it == Modality.SPEECH_TO_TEXT || it == Modality.TEXT_TO_SPEECH
            }.values.sum()
            val otherBytes =
                (state.usedBytes - textBytes - visionBytes - diffusionBytes - speechBytes)
                    .coerceAtLeast(0)
            val capacity = (state.usedBytes + state.freeStorageBytes).coerceAtLeast(1)

            Row(
                Modifier.fillMaxWidth().padding(bottom = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(9.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                NStackedBar(
                    segments = listOf(
                        textBytes.toFloat() / capacity to NocturneColors.Accent500,
                        visionBytes.toFloat() / capacity to NocturneColors.Accent300,
                        diffusionBytes.toFloat() / capacity to NocturneColors.Accent700,
                        speechBytes.toFloat() / capacity to NocturneColors.Neutral700,
                        otherBytes.toFloat() / capacity to NocturneColors.Neutral500,
                    ),
                    modifier = Modifier.weight(1f),
                )
                Text(
                    "${Fmt.gb(state.usedBytes)} / ${Fmt.gb(capacity)} GB",
                    style = NocturneType.MonoSm,
                    color = NocturneColors.TextMuted,
                )
            }

            FlowRow(
                Modifier.fillMaxWidth().padding(bottom = 18.dp),
                horizontalArrangement = Arrangement.spacedBy(11.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                StorageLegend("Text", textBytes, NocturneColors.Accent500)
                StorageLegend("Vision", visionBytes, NocturneColors.Accent300)
                StorageLegend("Diffusion", diffusionBytes, NocturneColors.Accent700)
                StorageLegend("Speech", speechBytes, NocturneColors.Neutral700)
                // Shown only when there is something in it.
                if (otherBytes > 0) {
                    StorageLegend("Other", otherBytes, NocturneColors.Neutral500)
                }
            }

            if (state.groups.isEmpty()) {
                NCard {
                    Text("No models installed", style = NocturneType.CardTitleSm)
                    Text(
                        "Paste a Hugging Face ID on the Add screen. Any model whose artifacts match a " +
                            "bundled runtime and fits this device will resolve — no curated list, no " +
                            "app update needed.",
                        style = NocturneType.CardBody,
                        color = NocturneColors.Text.copy(alpha = 0.8f),
                    )
                    NButton("Add a model", onAddModel, style = NButtonStyle.Primary, block = true)
                }
            }

            state.groups.forEach { group ->
                ai.ondevice.ui.components.SectionKicker(
                    "${group.modality.label} · ${group.models.size}",
                    Modifier.padding(bottom = 9.dp),
                )
                group.models.forEach { model ->
                    val loaded = model.id == state.loadedModelId
                    NCard(
                        Modifier
                            .padding(bottom = 7.dp)
                            .nClickableFlat { onOpenModel(model.id) },
                        gap = 7.dp,
                        ring = if (loaded) NocturneColors.Accent700 else null,
                    ) {
                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.Top,
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text(model.displayName, style = NocturneType.CardTitle)
                                Text(
                                    listOfNotNull(
                                        model.quant,
                                        Fmt.bytes(model.sizeBytes),
                                        model.architecture,
                                    ).joinToString(" · "),
                                    style = NocturneType.MonoXs,
                                    color = NocturneColors.TextMuted,
                                    modifier = Modifier.padding(top = 2.dp),
                                )
                            }
                            val pending = state.pending[model.id]
                            when {
                                pending != null -> NTag(
                                    if (pending.paused) "paused" else "downloading",
                                    style = NTagStyle.Outline,
                                )
                                loaded -> NTag("loaded", style = NTagStyle.Accent)
                                // What it is *for*. Three rows all reading
                                // "stable-diffusion-3.5-fp8" were told apart
                                // only by a filename, and the one thing that
                                // distinguishes them — which slot each fills —
                                // was the one thing not on the row.
                                model.attachmentRole != null ->
                                    NTag(model.attachmentRole!!.label, style = NTagStyle.Outline)
                                model.modality == Modality.VISION -> NTag("vision", style = NTagStyle.Outline)
                                else -> Unit
                            }
                        }
                        // While the bytes are still arriving, say so instead of
                        // "never used" — which read as installed-but-unopened.
                        state.pending[model.id]?.let { pending: ai.ondevice.ui.vm.PendingInstall ->
                            ai.ondevice.ui.components.NProgressBar(
                                fraction = pending.fraction,
                                modifier = Modifier.padding(top = 8.dp),
                            )
                        }
                        NCardMeta(gap = 10.dp) {
                            state.pending[model.id]?.let { NMetaText(it.label) }
                                ?: model.lastUsedAt?.let { NMetaText("used ${Fmt.relative(it)}") }
                                ?: NMetaText("never used")
                            if (model.pinned) {
                                Box(Modifier.weight(1f))
                                Text("pinned", style = NocturneType.Meta, color = NocturneColors.Accent)
                            }
                        }
                    }
                }
                Box(Modifier.padding(bottom = 9.dp))
            }

            // Orphan cleanup — files with no record, records with no file.
            state.orphans?.takeIf { it.hasAny }?.let { report ->
                NCard(ring = NocturneColors.Neutral700) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            NIcons.InfoCircle,
                            contentDescription = null,
                            tint = NocturneColors.Neutral400,
                            modifier = Modifier.size(16.dp),
                        )
                        Text(
                            "${report.strayFiles.size + report.recordsWithoutFiles.size} orphaned item(s)",
                            style = NocturneType.CardTitleSm,
                        )
                    }
                    Text(
                        buildString {
                            if (report.strayFiles.isNotEmpty()) {
                                append(
                                    "${Fmt.bytes(report.strayBytes)} on disk with no library record — " +
                                        "left by an interrupted install. ",
                                )
                            }
                            if (report.recordsWithoutFiles.isNotEmpty()) {
                                append("${report.recordsWithoutFiles.size} record(s) whose file has gone.")
                            }
                        },
                        style = NocturneType.CardBody,
                        color = NocturneColors.Text.copy(alpha = 0.8f),
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                        NButton("Clean up", viewModel::sweepOrphans, modifier = Modifier.weight(1f))
                        NButton("Downloads", onOpenDownloads, modifier = Modifier.weight(1f))
                    }
                }
            }

            NHelp(
                "Model files sit in a normal folder you can open in any file manager. Nothing is in a " +
                    "private store.",
                Modifier.padding(top = 14.dp),
            )
        }
    }
}

@Composable
private fun StorageLegend(label: String, bytes: Long, color: androidx.compose.ui.graphics.Color) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(7.dp).background(color, RoundedCornerShape(2.dp)))
        Text("$label ${Fmt.bytes(bytes)}", style = NocturneType.Help)
    }
}
