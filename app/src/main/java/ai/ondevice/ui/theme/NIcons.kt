package ai.ondevice.ui.theme

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.PathParser
import androidx.compose.ui.unit.dp

/** The interface icon set, transcribed path-for-path from the design canvas. */
object NIcons {

    // — path-data helpers, so the transcriptions below read like the SVG —

    private fun circle(cx: Float, cy: Float, r: Float): String =
        "M${cx - r} ${cy}a$r $r 0 1 0 ${r * 2} 0a$r $r 0 1 0 ${-r * 2} 0"

    private fun rrect(x: Float, y: Float, w: Float, h: Float, r: Float): String =
        "M${x + r} ${y}h${w - 2 * r}a$r $r 0 0 1 $r ${r}v${h - 2 * r}" +
            "a$r $r 0 0 1 ${-r} ${r}h${-(w - 2 * r)}a$r $r 0 0 1 ${-r} ${-r}" +
            "v${-(h - 2 * r)}a$r $r 0 0 1 $r ${-r}z"

    private fun icon(
        strokeWidth: Float = 1.7f,
        cap: StrokeCap = StrokeCap.Round,
        join: StrokeJoin = StrokeJoin.Round,
        stroked: List<String> = emptyList(),
        filled: List<String> = emptyList(),
    ): ImageVector {
        val b = ImageVector.Builder(
            defaultWidth = 24.dp,
            defaultHeight = 24.dp,
            viewportWidth = 24f,
            viewportHeight = 24f,
        )
        stroked.forEach { d ->
            b.addPath(
                pathData = PathParser().parsePathString(d).toNodes(),
                stroke = SolidColor(Color.Black),
                strokeLineWidth = strokeWidth,
                strokeLineCap = cap,
                strokeLineJoin = join,
            )
        }
        filled.forEach { d ->
            b.addPath(
                pathData = PathParser().parsePathString(d).toNodes(),
                fill = SolidColor(Color.Black),
            )
        }
        return b.build()
    }

    // — bottom bar, six destinations —

    /** Chat: a speech bubble with a flat tail. */
    val Chat: ImageVector by lazy { icon(1.7f, stroked = listOf("M4 5.5h16v11H9l-5 4z")) }

    /** Image: framed picture with a sun and a ridge line. */
    val Image: ImageVector by lazy {
        icon(
            1.7f,
            stroked = listOf(
                rrect(3.5f, 4.5f, 17f, 15f, 2.5f),
                circle(9f, 10f, 1.6f),
                "M4 17l5-4.5 4 3.5 3-2.5 4 3.5",
            ),
        )
    }

    /** Voice: a five-bar waveform. */
    val Voice: ImageVector by lazy {
        icon(1.8f, cap = StrokeCap.Round, stroked = listOf("M4 11v2M8 8v8M12 4.5v15M16 8v8M20 11v2"))
    }

    /** Library: three spines on a shelf. */
    val Library: ImageVector by lazy {
        icon(
            1.7f,
            stroked = listOf(
                rrect(4f, 5f, 4f, 15f, 1f),
                rrect(10f, 5f, 4f, 15f, 1f),
                "M16.5 6.2l3.4 1 -3 13.4 -3.4-1z",
            ),
        )
    }

    /** Models: stacked layers — also the launcher mark. */
    val Models: ImageVector by lazy {
        icon(1.7f, stroked = listOf("M12 3.5l8 4-8 4-8-4z", "M4 12l8 4 8-4M4 16.5l8 4 8-4"))
    }

    /** Settings: two sliders. */
    val Settings: ImageVector by lazy {
        icon(
            1.7f,
            stroked = listOf(
                "M3 7h11M18 7h3M3 17h3M10 17h11",
                circle(16f, 7f, 2.2f),
                circle(8f, 17f, 2.2f),
            ),
        )
    }

    // — navigation and toolbar —

    /** Pin: "keep this in RAM". A push pin seen from the side. */
    val Pin: ImageVector by lazy {
        icon(1.7f, stroked = listOf("M9 4h6l-1 5 3 3v2H7v-2l3-3z", "M12 14v6"))
    }

    /** Eject: the counterpart to Pin — take it back out of RAM. */
    val Eject: ImageVector by lazy {
        icon(1.7f, stroked = listOf("M12 5l7 9H5z", "M5 18h14"))
    }

    /** Tools: a plug, because that is what an MCP server is. */
    val Tools: ImageVector by lazy {
        icon(1.7f, stroked = listOf("M9 3v4M15 3v4", rrect(7f, 7f, 10f, 7f, 2f), "M12 14v6"))
    }

    /**
     * Endpoint: this device, with two arcs leaving it. What other machines reach.
     *
     * A new glyph rather than a reused one, because nothing here meant "a
     * server other machines connect to". [Tools] is a plug and already means
     * MCP; [Runtime] is a chip and means silicon; [Wifi] is status-bar chrome at
     * a 14x10 viewport rather than an interface icon at 24.
     *
     * Two arcs, not three. At the 15 dp the Settings toolbar draws these at, a
     * third collapses into the second — which is the same reason [Wifi] has two.
     */
    val Endpoint: ImageVector by lazy {
        icon(
            1.7f,
            stroked = listOf(
                rrect(4f, 5f, 9f, 14f, 2f),
                "M15.5 9.5a4 4 0 0 1 0 5",
                "M18.5 7a7.5 7.5 0 0 1 0 10",
            ),
        )
    }

    /** Runtimes: a chip with pins — the engines are the silicon-facing part. */
    val Runtime: ImageVector by lazy {
        icon(
            1.6f,
            stroked = listOf(
                rrect(7f, 7f, 10f, 10f, 1.6f),
                "M9.8 3v4M14.2 3v4M9.8 17v4M14.2 17v4",
                "M3 9.8h4M3 14.2h4M17 9.8h4M17 14.2h4",
            ),
        )
    }

    val ChevronLeft: ImageVector by lazy { icon(1.8f, stroked = listOf("M15 5l-7 7 7 7")) }
    val ChevronDown: ImageVector by lazy { icon(2f, stroked = listOf("M8 10l4 4 4-4")) }
    val Menu: ImageVector by lazy { icon(1.8f, stroked = listOf("M4 7h16M4 12h16M4 17h16")) }
    val Plus: ImageVector by lazy { icon(2.2f, stroked = listOf("M12 5v14M5 12h14")) }
    val PlusThin: ImageVector by lazy { icon(1.8f, stroked = listOf("M12 5v14M5 12h14")) }

    /** The overflow affordance on the model-detail toolbar — filled discs. */
    val MoreVertical: ImageVector by lazy {
        icon(filled = listOf(circle(12f, 5f, 1.7f), circle(12f, 12f, 1.7f), circle(12f, 19f, 1.7f)))
    }

    // — verdict and state marks — Nocturne carries no red or green.

    /** The heavy check inside a `.tag-accent` — stroke 3, deliberately blunt. */
    val Check: ImageVector by lazy { icon(3f, stroked = listOf("M4 12.5l5 5L20 6.5")) }

    /** The cross inside a neutral disc: won't fit / unsupported / not runnable. */
    val Cross: ImageVector by lazy { icon(2.6f, cap = StrokeCap.Round, stroked = listOf("M6 6l12 12")) }

    /** A bang, for the caveat cards (gated repo). */
    val Bang: ImageVector by lazy { icon(2.4f, stroked = listOf("M12 5v9M12 18h.01")) }

    /** Circle-slash: the checksum failure mark. */
    val SlashCircle: ImageVector by lazy {
        icon(1.8f, stroked = listOf(circle(12f, 12f, 9f), "M8.5 15.5l7-7"))
    }

    /** Circle-i: the orphaned-file notice. */
    val InfoCircle: ImageVector by lazy {
        icon(1.7f, stroked = listOf(circle(12f, 12f, 9f), "M12 8v5M12 16h.01"))
    }

    /** Shield: the unscanned-files warning. */
    val Shield: ImageVector by lazy {
        icon(2f, stroked = listOf("M12 3l8 4v5c0 5-3.4 8-8 9-4.6-1-8-4-8-9V7z"))
    }

    /** Triangle-bang: the memory-envelope guardrail on the Image screen. */
    val TriangleAlert: ImageVector by lazy {
        icon(1.8f, stroked = listOf("M12 4l9 16H3z", "M12 10v4M12 17h.01"))
    }

    // — message actions —

    val Copy: ImageVector by lazy {
        icon(1.7f, stroked = listOf(rrect(9f, 3f, 12f, 14f, 2f), "M15 21H5a2 2 0 0 1-2-2V7"))
    }

    /** Regenerate — also Undo in the mask editor, and the manifest re-check. */
    val Rotate: ImageVector by lazy { icon(1.7f, stroked = listOf("M4 12a8 8 0 1 1 3 6.2M4 8v4h4")) }
    val RotateBack: ImageVector by lazy { icon(1.7f, stroked = listOf("M20 12a8 8 0 1 0-3 6.2M20 8v4h-4")) }

    /** Token-probability inspector. */
    val Activity: ImageVector by lazy { icon(1.7f, stroked = listOf("M4 12h5l2-4 3 8 2-4h4")) }

    /**
     * Workflow: three steps and the line between them.
     *
     * Not the Activity trace, which is close and already means "what the
     * device is doing" — a tab icon that means two things means neither.
     */
    val Flow: ImageVector by lazy {
        icon(
            1.7f,
            stroked = listOf(
                "M6 5h5v4H6zM13 15h5v4h-5z",
                "M8.5 9v3.5h7V15",
            ),
        )
    }

    /** Read-aloud. */
    val Speaker: ImageVector by lazy {
        icon(1.7f, stroked = listOf("M4 9v6h3l5 4V5L7 9zM16 8.5a5 5 0 0 1 0 7"))
    }

    /** Transcribe, as the counterpart to [Speaker]. */
    val Mic: ImageVector by lazy {
        icon(
            1.7f,
            stroked = listOf(
                "M12 3.5a2.5 2.5 0 0 1 2.5 2.5v5a2.5 2.5 0 0 1-5 0V6A2.5 2.5 0 0 1 12 3.5z",
                "M5.5 11a6.5 6.5 0 0 0 13 0",
                "M12 17.5V21M9 21h6",
            ),
        )
    }

    /** A film frame: a rounded rectangle with the sprocket rail down one side. */
    val Video: ImageVector by lazy {
        icon(
            1.8f,
            stroked = listOf(
                rrect(3f, 5f, 18f, 14f, 2f),
                "M8 5v14",
                "M5.5 8.5h0M5.5 12h0M5.5 15.5h0",
            ),
        )
    }

    val Play: ImageVector by lazy { icon(2f, stroked = listOf("M8 5l11 7-11 7z")) }
    val Pause: ImageVector by lazy { icon(1.8f, stroked = listOf("M9 6v12M15 6v12")) }

    /** The send button becomes a stop square while generating. */
    val Stop: ImageVector by lazy { icon(2f, stroked = listOf(rrect(6f, 6f, 12f, 12f, 2f))) }
    val Send: ImageVector by lazy { icon(2f, stroked = listOf("M12 19V5M6 11l6-6 6 6")) }

    /** Import: an arrow coming down onto a floor. */
    val Import: ImageVector by lazy {
        icon(1.8f, stroked = listOf("M12 4v9M8 9.5l4 4 4-4", "M5 19h14"))
    }

    /** Thinking block: a lightbulb-as-brain. */
    val Think: ImageVector by lazy {
        icon(1.8f, stroked = listOf("M12 3a6 6 0 0 1 3 11v3H9v-3a6 6 0 0 1 3-11zM9.5 21h5"))
    }

    // — gallery, mask editor —

    val Share: ImageVector by lazy {
        icon(1.7f, stroked = listOf("M12 16V4M8 8l4-4 4 4M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"))
    }
    val Trash: ImageVector by lazy {
        icon(1.7f, stroked = listOf("M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"))
    }
    /** A page with a folded corner — a document attachment. */
    val File: ImageVector by lazy {
        icon(1.7f, stroked = listOf("M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z", "M14 3v5h5"))
    }

    /** Sound wave — read aloud. */
    val Waveform: ImageVector by lazy {
        icon(1.8f, stroked = listOf("M4 10v4M8 6v12M12 3v18M16 7v10M20 10v4"))
    }

    val Brush: ImageVector by lazy { icon(1.7f, stroked = listOf("M15.5 4.5l4 4-9 9H6v-4.5")) }

    /** Outpainting: four corners opening outwards. */
    val Expand: ImageVector by lazy {
        icon(
            1.7f,
            stroked = listOf("M4 9V4h5", "M15 4h5v5", "M20 15v5h-5", "M9 20H4v-5"),
        )
    }
    val Erase: ImageVector by lazy { icon(1.7f, stroked = listOf("M9 19H5l-1.5-4 11-11 4 4-11 11z", "M13 21h8")) }

    /**
     * Edit by instruction: a wand, not a brush.
     *
     * Erase and Brush are both a diagonal stroke and are indistinguishable at
     * toolbar size, and this is a different act anyway — you say what to change
     * rather than painting where.
     */
    val Wand: ImageVector by lazy {
        icon(
            1.7f,
            stroked = listOf(
                "M4 20l9-9",
                "M17 3l1.1 2.6L20.7 6.7l-2.6 1.1L17 10.4l-1.1-2.6L13.3 6.7l2.6-1.1z",
            ),
        )
    }

    /** Invert: a circle with one half filled. */
    val Invert: ImageVector by lazy {
        icon(
            1.7f,
            stroked = listOf(circle(12f, 12f, 8.5f), "M12 3.5v17"),
            filled = listOf("M12 3.5a8.5 8.5 0 0 1 0 17z"),
        )
    }

    // — status bar chrome —

    val Cellular: ImageVector by lazy {
        ImageVector.Builder(defaultWidth = 13.dp, defaultHeight = 10.dp, viewportWidth = 13f, viewportHeight = 10f)
            .apply {
                listOf(
                    rrect(0f, 7f, 2f, 3f, 1f),
                    rrect(3.5f, 5f, 2f, 5f, 1f),
                    rrect(7f, 2.5f, 2f, 7.5f, 1f),
                    rrect(10.5f, 0f, 2f, 10f, 1f),
                ).forEach { addPath(PathParser().parsePathString(it).toNodes(), fill = SolidColor(Color.Black)) }
            }.build()
    }

    val Wifi: ImageVector by lazy {
        ImageVector.Builder(defaultWidth = 14.dp, defaultHeight = 10.dp, viewportWidth = 14f, viewportHeight = 10f)
            .apply {
                addPath(
                    PathParser().parsePathString("M1 3.4a8 8 0 0 1 12 0M3.4 6a5 5 0 0 1 7.2 0").toNodes(),
                    stroke = SolidColor(Color.Black),
                    strokeLineWidth = 1.3f,
                    strokeLineCap = StrokeCap.Round,
                )
                addPath(PathParser().parsePathString(circle(7f, 8.6f, 0.8f)).toNodes(), fill = SolidColor(Color.Black))
            }.build()
    }

    val Battery: ImageVector by lazy {
        ImageVector.Builder(defaultWidth = 20.dp, defaultHeight = 10.dp, viewportWidth = 20f, viewportHeight = 10f)
            .apply {
                addPath(
                    PathParser().parsePathString(rrect(0.5f, 0.5f, 16f, 9f, 2.4f)).toNodes(),
                    stroke = SolidColor(Color.Black),
                    strokeLineWidth = 1.1f,
                )
                addPath(PathParser().parsePathString(rrect(2f, 2f, 11f, 6f, 1.4f)).toNodes(), fill = SolidColor(Color.Black))
                addPath(
                    PathParser().parsePathString("M18.4 3.6v2.8").toNodes(),
                    stroke = SolidColor(Color.Black),
                    strokeLineWidth = 1.1f,
                    strokeLineCap = StrokeCap.Round,
                )
            }.build()
    }

    /** Drag handle on the sampler-chain rows — six filled dots. */
    val Grip: ImageVector by lazy {
        icon(
            filled = listOf(
                circle(9f, 7f, 1.4f), circle(15f, 7f, 1.4f),
                circle(9f, 12f, 1.4f), circle(15f, 12f, 1.4f),
                circle(9f, 17f, 1.4f), circle(15f, 17f, 1.4f),
            ),
        )
    }
}
