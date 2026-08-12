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

/** What a run cost, drawn. */
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

        series(trace.gpuPercent, floor = 0, ceiling = 100)?.let { points ->
            drawPath(
                path = Path().apply {
                    moveTo(points.first().x, points.first().y)
                    points.drop(1).forEach { lineTo(it.x, it.y) }
                },
                color = NocturneColors.Accent500,
                style = Stroke(
                    width = 1.6.dp.toPx(),
                    pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(
                        floatArrayOf(6f, 4f),
                    ),
                ),
            )
        }

        // Clock, on the same canvas as the rest and scaled against its own peak.
        //
        // **Against the run's own peak rather than the chip's rated ceiling**,
        // because the rated figure is not readable from here and guessing one
        // would put this line on an axis it does not share with anything. What
        // it therefore shows is *shape*: whether the platform held a rate or
        // kept backing off. The absolute values are in the captions below, which
        // is where a number belongs anyway.
        //
        // Dotted, and third in the drawing order, so it reads as background
        // context behind CPU and GPU rather than as a fourth thing competing
        // with them. A run where this line sags while CPU stays high is the
        // whole reason it was added: busy is a measure of *time*, and says
        // nothing about the rate the work was done at.
        series(trace.clockMhz, floor = 0, ceiling = trace.peakClockMhz ?: 0)?.let { points ->
            drawPath(
                path = Path().apply {
                    moveTo(points.first().x, points.first().y)
                    points.drop(1).forEach { lineTo(it.x, it.y) }
                },
                color = NocturneColors.TextMeta,
                style = Stroke(
                    width = 1.2.dp.toPx(),
                    pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(
                        floatArrayOf(2f, 5f),
                    ),
                ),
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

/** Map a series onto the canvas between [floor] and [ceiling], or null when there is nothing to draw. */
private fun DrawScope.series(values: List<Int>, floor: Int, ceiling: Int): List<Offset>? {
    if (values.size < 2 || ceiling < floor) return null
    val span = (ceiling - floor).toFloat()
    val step = size.width / (values.size - 1)
    return values.mapIndexed { index, value ->
        val fraction = if (span <= 0f) 0.5f else ((value - floor) / span).coerceIn(0f, 1f)
        Offset(index * step, size.height - fraction * size.height)
    }
}

/** The collapsed form, for a screen where the graph is not the point. */
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
                    "CPU ${trace.cpuPercent.lastOrNull() ?: 0}%" +
                        (trace.gpuPercent.lastOrNull()?.let { " · GPU $it%" } ?: "") + " · " +
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
            if (trace.gpuPercent.isNotEmpty()) {
                // "device" because the counter is: the kernel counts every
                // client of the GPU, this app included but not only.
                GraphLegend("GPU", "device 0–100%", NocturneColors.Accent500)
            }
            trace.peakClockMhz?.let { peak ->
                // **"peak" said out loud.** The other two legends name a range a
                // reader already understands -- CPU is 0-100%, RAM is a span of
                // bytes -- but a lone clock figure reads as a live value, and this
                // one is the highest of the run. It was reported as a bug for
                // exactly that reason: 2433 MHz sitting still while the phone was
                // thermally capped to a mean of 1924, which is impossible as a
                // current reading and correct as a maximum.
                GraphLegend("CLK", "peak $peak MHz", NocturneColors.TextMeta)
            }
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
                // The model's own footprint, separated from whatever the app was already holding — the number that decides whether a bigger quant would fit.
                add("ram added" to Fmt.bytes(trace.deltaRssMb.toLong() * ResourceTrace.BYTES_PER_MB))
                add(
                    "device free" to "${Fmt.bytes(trace.minAvailMb.toLong() * ResourceTrace.BYTES_PER_MB)} " +
                        "of ${Fmt.bytes(trace.totalRamMb.toLong() * ResourceTrace.BYTES_PER_MB)}",
                )
                trace.peakGpuPercent?.let { add("gpu peak" to "$it% device-wide") }
                trace.meanGpuPercent?.let { add("gpu mean" to "$it%") }
                // Mean first: it is what the run actually ran at, where the peak
                // is only what the device was briefly willing to give. A wide gap
                // between the two is a platform that kept backing off.
                trace.meanClockMhz?.let { add("cpu clock mean" to "$it MHz") }
                trace.peakClockMhz?.let { add("cpu clock peak" to "$it MHz") }
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
