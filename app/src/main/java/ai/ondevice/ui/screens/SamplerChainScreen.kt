package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGesturesAfterLongPress
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCardKicker
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.ParamsViewModel

/**
 * **S9 — Sampler chain.**
 *
 * SPEC §4.2: "Sampler chain ordering deserves first-class UI. The order
 * materially changes output and most apps hide it." So it gets a screen, a
 * drag-to-reorder list, and per-sampler enable/disable.
 *
 * The chain string is written verbatim into `samplers` and stored with every
 * message, which is what makes a reply reproducible rather than merely
 * repeatable.
 */
@Composable
fun SamplerChainScreen(
    onBack: () -> Unit,
    viewModel: ParamsViewModel = activityParamsViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    var order by remember { mutableStateOf(emptyList<String>()) }
    var draggingIndex by remember { mutableStateOf<Int?>(null) }

    // The manifest loads asynchronously, so the working order has to be seeded
    // once it arrives — keying a `remember` on the (initially empty) value map
    // would latch the empty list forever.
    LaunchedEffect(state.allSpecs, state.samplerOrder) {
        if (draggingIndex == null) order = viewModel.samplerOrder()
    }
    val rowHeight = with(LocalDensity.current) { 52.dp.toPx() }
    val mirostat = viewModel.mirostatActive()

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = "Sampler chain",
                subtitle = "Order changes the output. Most apps hide it.",
                subtitleMono = false,
                onBack = onBack,
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            Column(
                Modifier.fillMaxWidth().alpha(if (mirostat) 0.45f else 1f),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                order.forEachIndexed { index, sampler ->
                    val disabled = sampler in state.disabledSamplers
                    val dragging = draggingIndex == index

                    Row(
                        Modifier
                            .fillMaxWidth()
                            .defaultMinSize(minHeight = 46.dp)
                            .background(
                                if (dragging) NocturneColors.Accent900 else NocturneColors.Surface,
                                Radius.Md,
                            )
                            .ring(
                                if (dragging) NocturneColors.Accent else NocturneColors.Divider,
                                Radius.Md,
                            )
                            .alpha(if (disabled) 0.45f else 1f)
                            .pointerInput(order, mirostat) {
                                if (mirostat) return@pointerInput
                                detectDragGesturesAfterLongPress(
                                    onDragStart = { draggingIndex = index },
                                    onDragEnd = {
                                        draggingIndex?.let { viewModel.setSamplerOrder(order) }
                                        draggingIndex = null
                                    },
                                    onDragCancel = { draggingIndex = null },
                                    onDrag = { _, dragAmount ->
                                        val from = draggingIndex ?: return@detectDragGesturesAfterLongPress
                                        val steps = (dragAmount.y / rowHeight).toInt()
                                        val to = (from + steps).coerceIn(0, order.lastIndex)
                                        if (to != from) {
                                            order = order.toMutableList().apply { add(to, removeAt(from)) }
                                            draggingIndex = to
                                        }
                                    },
                                )
                            }
                            .padding(horizontal = 12.dp, vertical = 11.dp),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            "${index + 1}",
                            style = NocturneType.MonoSm,
                            color = NocturneColors.Accent300,
                            modifier = Modifier.width(14.dp),
                        )
                        Column(Modifier.weight(1f)) {
                            Text(samplerLabel(sampler), style = NocturneType.CardTitleSm)
                            Text(
                                samplerDetail(sampler),
                                style = NocturneType.MonoXs,
                                color = NocturneColors.TextMuted,
                            )
                        }
                        Text(
                            if (disabled) "off" else "on",
                            style = NocturneType.Mono2Xs,
                            color = if (disabled) NocturneColors.TextMuted else NocturneColors.Accent,
                            modifier = Modifier
                                .ring(
                                    if (disabled) NocturneColors.Divider else NocturneColors.Accent700,
                                    Radius.Sm,
                                )
                                .nClickableFlat { viewModel.toggleSampler(sampler) }
                                .padding(horizontal = 8.dp, vertical = 4.dp),
                        )
                        Icon(
                            NIcons.Grip,
                            contentDescription = "Reorder",
                            tint = NocturneColors.Text.copy(alpha = 0.35f),
                            modifier = Modifier.size(16.dp),
                        )
                    }
                }
            }

            NCard(Modifier.padding(top = 16.dp)) {
                NCardKicker("Current chain")
                Text(
                    order.filterNot { it in state.disabledSamplers }.joinToString(" → "),
                    style = NocturneType.MonoCode,
                    color = NocturneColors.Accent300,
                )
                Text(
                    "Written verbatim into `samplers` and stored with every message, so any reply can " +
                        "be reproduced exactly.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            }

            NButton(
                "Reset to runtime default",
                onClick = viewModel::resetSamplerChain,
                style = NButtonStyle.Secondary,
                block = true,
                modifier = Modifier.padding(top = 12.dp),
            )

            NHelp(
                "Mirostat, when enabled, replaces this chain entirely — the app greys it out and says " +
                    "so rather than silently ignoring your order.",
                Modifier.padding(top = 14.dp),
            )
        }
    }
}

/**
 * Labels for upstream's sampler identifiers. These are display strings for
 * values that come *from the manifest*, not a hardcoded list of samplers — an
 * unrecognised name still renders, using the raw identifier.
 */
private fun samplerLabel(id: String): String = when (id) {
    "penalties" -> "Repetition penalties"
    "dry" -> "DRY"
    "top_n_sigma" -> "Top-N sigma"
    "top_k" -> "Top-K"
    "typ_p" -> "Typical-P"
    "top_p" -> "Top-P"
    "min_p" -> "Min-P"
    "xtc" -> "XTC"
    "temperature" -> "Temperature"
    "dist" -> "Distribution"
    else -> id
}

private fun samplerDetail(id: String): String = when (id) {
    "penalties" -> "repeat · presence · frequency"
    "dry" -> "don't repeat yourself"
    "top_n_sigma" -> "sigma truncation"
    "top_k" -> "keep K most likely"
    "typ_p" -> "locally typical"
    "top_p" -> "nucleus"
    "min_p" -> "relative probability floor"
    "xtc" -> "exclude top choices"
    "temperature" -> "final softmax scaling"
    "dist" -> "final sample"
    else -> "from manifest"
}
