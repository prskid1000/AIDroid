package ai.ondevice.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import ai.ondevice.core.Fmt
import ai.ondevice.engine.ResourceTrace
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring

/**
 * What a run cost, drawn.
 *
 * Two series on one canvas because they are read together: a run pinned at 100%
 * CPU with flat memory is compute-bound, and one at 30% with memory climbing to
 * the ceiling is neither — those are different problems with different fixes,
 * and separating them into two charts would make the comparison a memory task.
 *
 * CPU sits on a fixed 0–100% axis, so height means the same thing in every graph
 * on every screen. Memory cannot, and it took drawing one to see why: a 700 MB
 * model and a 6 GB one share no useful scale, but a 0-to-peak axis is worse than
 * no scale at all — a run that holds 2.5 GB steadily and adds 44 MB is a flat
 * line pinned to the ceiling, which is the shape of "at the limit" drawn over
 * data that says the opposite.
 *
 * So RAM is drawn between its own minimum and maximum, where the variation
 * actually lives, and **both** ends are stated in the legend. A rescaling axis
 * is only a lie when it is unlabelled.
 */
@Composable
fun ResourceGraph(
    trace: ResourceTrace,
    modifier: Modifier = Modifier,
    height: androidx.compose.ui.unit.Dp = 64.dp,
) {
    if (trace.isEmpty) return
    Canvas(
        modifier
            .fillMaxWidth()
            .height(height)
            .background(NocturneColors.Neutral900, Radius.Sm)
            .padding(horizontal = 6.dp, vertical = 5.dp),
    ) {
        // The 50% guide, so a filled area can be read as a number rather than
        // just as more or less than the one before it.
        drawLine(
            color = NocturneColors.Divider,
            start = Offset(0f, size.height / 2),
            end = Offset(size.width, size.height / 2),
            strokeWidth = 1f,
        )

        series(trace.cpuPercent, floor = 0, ceiling = 100)?.let { points ->
            // Filled, because CPU is a proportion of a fixed whole and the area
            // under it is the work done.
            drawPath(
                path = Path().apply {
                    moveTo(0f, size.height)
                    points.forEach { lineTo(it.x, it.y) }
                    lineTo(size.width, size.height)
                    close()
                },
                color = NocturneColors.Accent.copy(alpha = 0.22f),
            )
            drawPath(
                path = Path().apply {
                    moveTo(points.first().x, points.first().y)
                    points.drop(1).forEach { lineTo(it.x, it.y) }
                },
                color = NocturneColors.Accent,
                style = Stroke(width = 1.6.dp.toPx()),
            )
        }

        // Memory is a level, not a quantity of work, so it is a line and never
        // a fill — nothing meaningful sits underneath it.
        series(trace.rssMb, floor = trace.floorRssMb, ceiling = trace.peakRssMb)?.let { points ->
            drawPath(
                path = Path().apply {
                    moveTo(points.first().x, points.first().y)
                    points.drop(1).forEach { lineTo(it.x, it.y) }
                },
                color = NocturneColors.Accent2300,
                style = Stroke(width = 1.4.dp.toPx()),
            )
        }
    }
}

/**
 * Map a series onto the canvas between [floor] and [ceiling], or null when there
 * is nothing to draw.
 *
 * A single sample has no line to make. A flat series — floor equal to ceiling —
 * is drawn down the middle rather than divided by zero, which is also the right
 * picture: a value that never changed has no shape to show.
 */
private fun DrawScope.series(values: List<Int>, floor: Int, ceiling: Int): List<Offset>? {
    if (values.size < 2 || ceiling < floor) return null
    val span = (ceiling - floor).toFloat()
    val step = size.width / (values.size - 1)
    return values.mapIndexed { index, value ->
        val fraction = if (span <= 0f) 0.5f else ((value - floor) / span).coerceIn(0f, 1f)
        Offset(index * step, size.height - fraction * size.height)
    }
}

/**
 * The collapsed form, for a screen where the graph is not the point.
 *
 * Deliberately the same shape as the chat screen's thinking block: a one-line
 * row you can tap open. Both are "the machinery behind this answer", both are
 * worth having and neither is worth a permanent third of the screen.
 */
@Composable
fun ResourceBlock(
    trace: ResourceTrace,
    expanded: Boolean,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier,
    live: Boolean = false,
) {
    if (trace.isEmpty) return
    Column(modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(
            Modifier
                .fillMaxWidth()
                .background(NocturneColors.Neutral900, Radius.Md)
                .ring(NocturneColors.Divider, Radius.Md)
                .nClickableFlat(onClick = onToggle)
                .padding(horizontal = 10.dp, vertical = 7.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                NIcons.Activity,
                contentDescription = null,
                tint = NocturneColors.Accent,
                modifier = Modifier.size(13.dp),
            )
            Text(
                if (live) {
                    "CPU ${trace.cpuPercent.lastOrNull() ?: 0}% · " +
                        "RAM ${Fmt.bytes(trace.rssMb.lastOrNull().orZeroMb())}"
                } else {
                    "CPU ${trace.peakCpuPercent}% peak · RAM ${Fmt.bytes(trace.peakRssBytes)} peak"
                },
                style = NocturneType.Meta.copy(fontSize = NocturneType.Row.fontSize),
                color = NocturneColors.Accent300,
                modifier = Modifier.weight(1f),
            )
            Text(
                if (expanded) "⌃" else "⌄",
                style = NocturneType.Row,
                color = NocturneColors.Accent300,
            )
        }
        if (expanded) ResourceDetail(trace)
    }
}

/** The graph plus every number behind it. Used expanded and on the detail screen. */
@Composable
fun ResourceDetail(trace: ResourceTrace, modifier: Modifier = Modifier) {
    if (trace.isEmpty) return
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        ResourceGraph(trace)
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            GraphLegend("CPU", "0–100%", NocturneColors.Accent)
            val floor = Fmt.bytes(trace.floorRssMb.toLong() * ResourceTrace.BYTES_PER_MB)
            val peak = Fmt.bytes(trace.peakRssBytes)
            // "2.50 GB–2.50 GB" is a range that is not one. A run whose memory
            // never moved says so with a single number.
            GraphLegend("RAM", if (floor == peak) peak else "$floor–$peak", NocturneColors.Accent2300)
        }
        NTable {
            val rows = buildList {
                add("duration" to Fmt.duration(trace.elapsedMillis, tenths = true))
                add("cpu peak" to "${trace.peakCpuPercent}% of ${trace.cores} cores")
                add("cpu mean" to "${trace.meanCpuPercent}%")
                add("ram peak" to Fmt.bytes(trace.peakRssBytes))
                // The model's own footprint, separated from whatever the app was
                // already holding — the number that decides whether a bigger
                // quant would fit.
                add("ram added" to Fmt.bytes(trace.deltaRssMb.toLong() * ResourceTrace.BYTES_PER_MB))
                add(
                    "device free" to "${Fmt.bytes(trace.minAvailMb.toLong() * ResourceTrace.BYTES_PER_MB)} " +
                        "of ${Fmt.bytes(trace.totalRamMb.toLong() * ResourceTrace.BYTES_PER_MB)}",
                )
                add("sampled" to "${trace.cpuPercent.size} × ${trace.intervalMillis} ms")
            }
            rows.forEach { (key, value) ->
                NTableRow {
                    Text(
                        key,
                        style = NocturneType.Row,
                        color = NocturneColors.TextMuted,
                        modifier = Modifier.weight(0.44f),
                    )
                    Text(value, style = NocturneType.MonoValue, modifier = Modifier.weight(0.56f))
                }
            }
        }
    }
}

@Composable
private fun GraphLegend(label: String, scale: String, color: androidx.compose.ui.graphics.Color) {
    Row(horizontalArrangement = Arrangement.spacedBy(5.dp), verticalAlignment = Alignment.CenterVertically) {
        androidx.compose.foundation.layout.Box(
            Modifier.size(width = 9.dp, height = 2.dp).background(color),
        )
        Text("$label $scale", style = NocturneType.Mono2Xs, color = NocturneColors.TextMeta)
    }
}

private fun Int?.orZeroMb(): Long = (this ?: 0).toLong() * ResourceTrace.BYTES_PER_MB
