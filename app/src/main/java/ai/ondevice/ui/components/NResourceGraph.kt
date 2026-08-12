package ai.ondevice.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.pm.ActivityInfo
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.platform.LocalContext
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.WindowInsetsCompat
import androidx.compose.ui.window.DialogWindowProvider
import androidx.compose.ui.platform.LocalView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.window.Dialog
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
    val measurer = rememberTextMeasurer()
    val axis = NocturneType.Mono2Xs.copy(color = NocturneColors.TextMeta)

    // **Both axes are labelled, and only one of them can be.**
    //
    // X is time and every series shares it, so it is honest: the run is
    // `elapsedMillis` long and the samples are evenly spaced across it.
    //
    // Y is a percentage, and *only* CPU and GPU are on that scale. Memory is
    // drawn against its own floor-to-peak span and the clock against the run's
    // own peak, because a phone's RAM in MB and a core's MHz share no axis with
    // a percentage and forcing them onto one would flatten both to nothing. So
    // the left-hand numbers are marked `%`, and the two series they do not
    // describe are named in the captions with their own ranges. A graph that
    // labelled a single Y axis and drew four units against it would be worse
    // than one that admits which one the numbers are for.
    Canvas(
        modifier
            .fillMaxWidth()
            .then(if (height > 0.dp) Modifier.height(height) else Modifier)
            .background(NocturneColors.Neutral900, Radius.Sm)
            .padding(horizontal = 6.dp, vertical = 5.dp),
    ) {
        // Measured, not guessed: a flat gutter clips any label wider than it,
        // and the leading character is the one that goes. `100%` is the widest
        // here, but taking it from the text is what stops the next unit added to
        // this graph from silently losing a digit.
        val widest = measurer.measure("100%", axis)
        val labelled = size.height > 90f
        // The same breathing room on all four sides: the left and bottom get
        // theirs from the labels, and without this the top and right had none —
        // a line at 100% sat on the border and the last sample ran off the edge.
        val inset = if (labelled) 4.dp.toPx() else 0f
        val yGutter = if (labelled) widest.size.width + 6.dp.toPx() else 0f
        val xGutter = if (labelled) widest.size.height + 2.dp.toPx() else 0f
        val plot = Size(size.width - yGutter - inset, size.height - xGutter - inset)

        // Percentage guides. Four lines rather than one: the midline alone said
        // "more or less than half", which is the question you ask of a graph you
        // are not really reading.
        for (pct in intArrayOf(0, 25, 50, 75, 100)) {
            val y = inset + plot.height - plot.height * pct / 100f
            drawLine(
                color = NocturneColors.Divider.copy(alpha = if (pct == 50) 1f else 0.45f),
                start = Offset(yGutter, y),
                end = Offset(yGutter + plot.width, y),
                strokeWidth = 1f,
            )
            if (yGutter > 0f && pct % 50 == 0) {
                val label = measurer.measure("$pct%", axis)
                drawText(
                    label,
                    topLeft = Offset(
                        yGutter - label.size.width - 4.dp.toPx(),
                        (y - label.size.height / 2f).coerceIn(0f, size.height - label.size.height),
                    ),
                )
            }
        }

        // X is seconds from the start of the run. The right-hand label is the
        // run's own length rather than a round number, because that is the fact
        // somebody reading this wants and rounding it would invent one.
        if (xGutter > 0f) {
            // A run under a second would otherwise label every tick `0s`. The
            // sample count is the honest fallback: it says how much history the
            // line covers, which is what the axis is for.
            val seconds = (trace.elapsedMillis / 1000).toInt()
            val ticks = if (seconds > 1) {
                listOf(0f to "0s", 0.5f to "${seconds / 2}s", 1f to "${seconds}s")
            } else {
                listOf(0f to "0", 1f to "${trace.cpuPercent.size} samples")
            }
            for ((frac, text) in ticks) {
                val label = measurer.measure(text, axis)
                val x = yGutter + (plot.width - label.size.width) * frac
                drawText(label, topLeft = Offset(x, inset + plot.height + 1.dp.toPx()))
            }
        }

        series(trace.cpuPercent, 0, 100, yGutter, inset, plot)?.let { points ->
            // Filled, because CPU is a proportion of a fixed whole and the area
            // under it is the work done.
            drawPath(
                path = Path().apply {
                    moveTo(yGutter, inset + plot.height)
                    points.forEach { lineTo(it.x, it.y) }
                    lineTo(yGutter + plot.width, inset + plot.height)
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

        series(trace.gpuPercent, 0, 100, yGutter, inset, plot)?.let { points ->
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
        series(trace.clockMhz, 0, trace.peakClockMhz ?: 0, yGutter, inset, plot)?.let { points ->
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
        series(trace.rssMb, trace.floorRssMb, trace.peakRssMb, yGutter, inset, plot)?.let { points ->
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
private fun DrawScope.series(
    values: List<Int>,
    floor: Int,
    ceiling: Int,
    left: Float = 0f,
    top: Float = 0f,
    plot: Size = size,
): List<Offset>? {
    if (values.size < 2 || ceiling < floor) return null
    val span = (ceiling - floor).toFloat()
    val step = plot.width / (values.size - 1)
    return values.mapIndexed { index, value ->
        val fraction = if (span <= 0f) 0.5f else ((value - floor) / span).coerceIn(0f, 1f)
        Offset(left + index * step, top + plot.height - fraction * plot.height)
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
        if (expanded) {
            // **Tap the chart to fill the screen with it.**
            //
            // Inline it is about 70 dp tall in a chat bubble, which is enough to
            // see that a line sagged and not enough to see where. The numbers
            // underneath were doing all the work and the picture almost none.
            //
            // On the graph rather than on a button: the thing you want bigger is
            // the thing you tap, and there is no room beside it for a control
            // that would only ever mean "bigger".
            ResourceDetail(trace)
        }
    }
}

/**
 * The graph alone, landscape, edge to edge.
 *
 * **Only the chart.** The first version put the legends and the numbers under
 * it, which is the detail view again at a larger size — and the detail view is
 * what you already have inline. What is missing at 64 dp is the *shape*: where a
 * line sagged, how long it held, whether a dip was one sample or twenty. That
 * needs pixels along the time axis and nothing else competing for them.
 *
 * **Forced landscape.** The axis that matters is time and the phone is twice as
 * long as it is wide; in portrait this is the inline graph with more height,
 * which is the one dimension it did not need. The host activity's
 * `requestedOrientation` is set for as long as the dialog lives and put back
 * exactly as it was found — restored rather than set to portrait, because the
 * app may have been in landscape already or following the sensor, and forcing a
 * value it never asked for is its own bug.
 *
 * A tap anywhere closes it. There is no chrome to hang a button on, and adding
 * some would spend the pixels this exists to free.
 */
@Composable
private fun ResourceGraphDialog(trace: ResourceTrace, onDismiss: () -> Unit) {
    val context = LocalContext.current
    DisposableEffect(Unit) {
        val activity = context.findHostActivity()
        val previous = activity?.requestedOrientation
        activity?.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        onDispose { previous?.let { activity?.requestedOrientation = it } }
    }
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        // **The bars are chrome and the chart is the content.** A status bar
        // with a battery icon over a resource graph is thirty pixels of the time
        // axis spent on something the graph is not about. Hidden for as long as
        // the dialog lives and shown again on dispose — the dialog has its own
        // window, so this cannot leak into the screen behind it.
        val view = LocalView.current
        DisposableEffect(Unit) {
            val window = (view.parent as? DialogWindowProvider)?.window
            val bars = window?.let { WindowInsetsControllerCompat(it, it.decorView) }
            bars?.hide(WindowInsetsCompat.Type.systemBars())
            bars?.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            onDispose { bars?.show(WindowInsetsCompat.Type.systemBars()) }
        }
        Box(
            Modifier
                .fillMaxSize()
                .background(NocturneColors.Bg)
                .nClickableFlat(onClick = onDismiss)
                .padding(12.dp),
        ) {
            ResourceGraph(trace, Modifier.fillMaxSize(), height = 0.dp)
        }
    }
}

/** The Activity behind a Compose context, through any number of wrappers. */
private tailrec fun Context.findHostActivity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.findHostActivity()
    else -> null
}

/** The graph plus every number behind it. Used expanded and on the detail screen. */
@Composable
fun ResourceDetail(trace: ResourceTrace, modifier: Modifier = Modifier) {
    if (trace.isEmpty) return
    // **The zoom lives here, not at the call sites.** It was a callback the
    // caller passed, so `ResourceBlock` got it and the Library screen — which
    // draws this directly — did not, and tapping its chart did nothing. A
    // component that behaves differently depending on which screen drew it has a
    // bug in every screen that forgot the argument.
    var full by rememberSaveable { mutableStateOf(false) }
    if (full) ResourceGraphDialog(trace) { full = false }
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        ResourceGraph(trace, Modifier.nClickableFlat(onClick = { full = true }))
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
