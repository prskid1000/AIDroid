package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.core.VerdictTone
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NHelp
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
    onOpenParameters: (ai.ondevice.core.Tier) -> Unit,
    viewModel: ModelDetailViewModel = hiltViewModel(),
) {
    LaunchedEffect(modelId) { viewModel.bind(modelId) }
    val state by viewModel.state.collectAsStateWithLifecycle()
    val model = state.model

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = model?.let { "${it.displayName} · ${it.quant.orEmpty()}" } ?: "Model",
                onBack = onBack,
                trailing = {
                    Icon(
                        NIcons.MoreVertical,
                        contentDescription = "More",
                        tint = NocturneColors.Text,
                        modifier = Modifier.size(20.dp),
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

            // §8.2 — measured, not assumed.
            SectionKicker("Measured on this device", Modifier.padding(top = 22.dp, bottom = 8.dp))
            if (state.benchmarks.isEmpty()) {
                NHelp(
                    "No benchmark yet. Backend performance on this hardware is a measurement, not an " +
                        "assumption — run one and the app auto-selects the winner and shows the numbers.",
                )
            } else {
                val best = state.benchmarks.maxByOrNull { it.genTokPerSec }
                NTable(
                    header = {
                        NTableHeaderCell("Backend", Modifier.weight(1f))
                        NTableHeaderCell("Prompt", Modifier.weight(0.5f), TextAlign.End)
                        NTableHeaderCell("Gen", Modifier.weight(0.4f), TextAlign.End)
                    },
                ) {
                    state.benchmarks.forEach { row ->
                        NTableRow {
                            Row(
                                Modifier.weight(1f),
                                horizontalArrangement = Arrangement.spacedBy(5.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Text(row.backend.label, style = NocturneType.Row)
                                if (row.id == best?.id) {
                                    NTag(
                                        "auto",
                                        style = NTagStyle.Accent,
                                        contentPadding = PaddingValues(horizontal = 6.dp, vertical = 1.dp),
                                    )
                                }
                            }
                            Text(
                                Fmt.tokensPerSecond(row.promptTokPerSec),
                                style = NocturneType.MonoValue,
                                modifier = Modifier.weight(0.5f),
                                textAlign = TextAlign.End,
                            )
                            Text(
                                Fmt.tokensPerSecond(row.genTokPerSec),
                                style = NocturneType.MonoValue,
                                modifier = Modifier.weight(0.4f),
                                textAlign = TextAlign.End,
                            )
                        }
                    }
                }
            }
            NButton(
                text = state.benchmarkingBackend?.let { "Benchmarking ${it.label}…" }
                    ?: if (state.benchmarks.isEmpty()) "Run benchmark" else "Re-run benchmark",
                onClick = viewModel::runBenchmark,
                style = NButtonStyle.Secondary,
                block = true,
                enabled = !state.benchmarking,
                modifier = Modifier.padding(top = 12.dp),
            )

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

            Row(
                Modifier.fillMaxWidth().padding(top = 18.dp),
                horizontalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                NButton(
                    if (model.pinned) "Unpin" else "Keep loaded",
                    onClick = viewModel::togglePin,
                    style = NButtonStyle.Secondary,
                    modifier = Modifier.weight(1f),
                )
                NButton(
                    "Parameters",
                    onClick = { onOpenParameters(ai.ondevice.core.Tier.EXPERT) },
                    style = NButtonStyle.Secondary,
                    modifier = Modifier.weight(1f),
                )
                NButton(
                    "Delete",
                    onClick = { viewModel.delete(onBack) },
                    style = NButtonStyle.Secondary,
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }
}

private val ai.ondevice.data.db.BenchmarkEntity.id: String get() = "$modelId:${backend.name}"
