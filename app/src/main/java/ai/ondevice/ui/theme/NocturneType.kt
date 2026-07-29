package ai.ondevice.ui.theme

import androidx.compose.runtime.Immutable
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import ai.ondevice.R

/**
 * Inter for headings over Inter for body — `--font-heading` / `--font-body` are
 * the same family in this system; hierarchy is size and space, never weight
 * past 500.
 */
val Inter = FontFamily(
    Font(R.font.inter_regular, FontWeight.Normal),
    Font(R.font.inter_medium, FontWeight.Medium),
    Font(R.font.inter_semibold, FontWeight.SemiBold),
    Font(R.font.inter_bold, FontWeight.Bold),
)

/**
 * The canvas uses `ui-monospace,Menlo,monospace` for every technical value:
 * model IDs, quant names, byte counts, token counts, the fit arithmetic, the
 * prompt inspector. Android's system mono is the equivalent of `ui-monospace`.
 */
val Mono = FontFamily.Monospace

/** `--font-heading-weight: 500`. Never bolder — the readme is explicit. */
val HeadingWeight = FontWeight.Medium

/**
 * The type scale from `styles.css`. `h1`…`h6` plus the body defaults, and the
 * handful of sizes the canvas uses repeatedly at screen scale.
 *
 * Headings: `line-height: 1.12`, `letter-spacing: -0.015em`.
 * Body: `font-size: 15px`, `line-height: 1.55`, `font-weight: 400`.
 */
@Immutable
object NocturneType {

    private fun heading(sizeSp: Int) = TextStyle(
        fontFamily = Inter,
        fontWeight = HeadingWeight,
        fontSize = sizeSp.sp,
        lineHeight = (sizeSp * 1.12f).sp,
        letterSpacing = (-0.015).em,
    )

    val H1 = heading(42)
    val H2 = heading(32)
    val H3 = heading(25)
    val H4 = heading(20)
    val H5 = heading(16)

    /** `h6` additionally: uppercase, `letter-spacing: 0.08em`. */
    val H6 = TextStyle(
        fontFamily = Inter,
        fontWeight = HeadingWeight,
        fontSize = 13.sp,
        lineHeight = 14.56.sp,
        letterSpacing = 0.08.em,
    )

    /** `body` — the document default. */
    val Body = TextStyle(
        fontFamily = Inter,
        fontWeight = FontWeight.Normal,
        fontSize = 15.sp,
        lineHeight = 23.25.sp,
    )

    /** `a` — accent, 3px underline offset. */
    val Link = Body.copy(color = NocturneColors.Accent, textDecoration = TextDecoration.Underline)

    // — the sizes the phone screens actually use —

    /** Screen title in a pushed toolbar: `font:500 17px var(--font-heading)`. */
    val ScreenTitle = TextStyle(fontFamily = Inter, fontWeight = HeadingWeight, fontSize = 17.sp, lineHeight = 20.sp)

    /** Root destination title: `font:500 21px`. */
    val RootTitle = TextStyle(fontFamily = Inter, fontWeight = HeadingWeight, fontSize = 21.sp, lineHeight = 24.sp)

    /** Bottom-sheet title: `font:500 19px`. */
    val SheetTitle = TextStyle(fontFamily = Inter, fontWeight = HeadingWeight, fontSize = 19.sp, lineHeight = 22.sp)

    /** `.card-title` at screen scale — the canvas uses 13–15px variants. */
    val CardTitle = TextStyle(fontFamily = Inter, fontWeight = HeadingWeight, fontSize = 14.sp, lineHeight = 17.sp)
    val CardTitleLg = TextStyle(fontFamily = Inter, fontWeight = HeadingWeight, fontSize = 15.sp, lineHeight = 19.sp)
    val CardTitleSm = TextStyle(fontFamily = Inter, fontWeight = HeadingWeight, fontSize = 13.sp, lineHeight = 16.sp)

    /** `.card-body` — 13px at 80% opacity, applied by the caller. */
    val CardBody = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Normal, fontSize = 13.sp, lineHeight = 20.sp)

    /** Message body in chat: `font-size:13.5px;line-height:1.6`. */
    val Message = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Normal, fontSize = 13.5.sp, lineHeight = 21.6.sp)

    /** Ordinary interface row label: 12.5px. */
    val Row = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Normal, fontSize = 12.5.sp, lineHeight = 17.sp)

    /** `.btn` / `.input` — both 14px, so they align in a row. */
    val Control = TextStyle(fontFamily = Inter, fontWeight = HeadingWeight, fontSize = 14.sp, lineHeight = 16.8.sp)
    val Input = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Normal, fontSize = 14.sp, lineHeight = 20.sp)

    /** `.field > label` — 12px. */
    val FieldLabel = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Normal, fontSize = 12.sp, lineHeight = 15.sp)

    /** `.tag` — 11px, `letter-spacing: 0.02em`. */
    val Tag = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Normal, fontSize = 11.sp, lineHeight = 14.sp, letterSpacing = 0.02.em)

    /** `.card-meta` — 11px. */
    val Meta = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Normal, fontSize = 11.sp, lineHeight = 14.sp)

    /** The recurring muted footnote under a control: 10.5px. */
    val Help = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Normal, fontSize = 10.5.sp, lineHeight = 14.sp)

    /**
     * `.card-kicker` — 10px, `letter-spacing: 0.1em`, uppercase, accent.
     * The canvas also uses this shape as a standalone section heading in mono;
     * see [SectionKicker].
     */
    val Kicker = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Normal, fontSize = 10.sp, lineHeight = 13.sp, letterSpacing = 0.1.em)

    /**
     * The mono section rule that separates every group on every screen:
     * `font:600 10px ui-monospace;letter-spacing:.1em;color:neutral-500`.
     */
    val SectionKicker = TextStyle(fontFamily = Mono, fontWeight = FontWeight.SemiBold, fontSize = 10.sp, lineHeight = 13.sp, letterSpacing = 0.1.em)

    // — mono variants, for every technical value on screen —
    val MonoValue = TextStyle(fontFamily = Mono, fontWeight = HeadingWeight, fontSize = 12.sp, lineHeight = 16.sp)
    val MonoSm = TextStyle(fontFamily = Mono, fontWeight = FontWeight.Normal, fontSize = 11.sp, lineHeight = 18.sp)
    val MonoXs = TextStyle(fontFamily = Mono, fontWeight = FontWeight.Normal, fontSize = 10.5.sp, lineHeight = 16.sp)
    val Mono2Xs = TextStyle(fontFamily = Mono, fontWeight = FontWeight.Normal, fontSize = 9.5.sp, lineHeight = 13.sp)
    val MonoBody = TextStyle(fontFamily = Mono, fontWeight = FontWeight.Normal, fontSize = 12.sp, lineHeight = 22.sp)
    val MonoCode = TextStyle(fontFamily = Mono, fontWeight = FontWeight.Normal, fontSize = 11.5.sp, lineHeight = 19.sp)
    val MonoTimestamp = TextStyle(fontFamily = Mono, fontWeight = HeadingWeight, fontSize = 11.sp, lineHeight = 15.sp)

    /** The big accent numeral on the model-detail context readout: 26px. */
    val Numeral = TextStyle(fontFamily = Inter, fontWeight = HeadingWeight, fontSize = 26.sp, lineHeight = 30.sp)

    /** Status bar / bottom-bar label: 10–11px. */
    val NavLabel = TextStyle(fontFamily = Inter, fontWeight = FontWeight.Normal, fontSize = 10.sp, lineHeight = 12.sp)
    val StatusBar = TextStyle(fontFamily = Inter, fontWeight = HeadingWeight, fontSize = 11.sp, lineHeight = 13.sp)
}
