package ai.ondevice.ui.theme

import androidx.compose.foundation.LocalIndication
import androidx.compose.foundation.text.selection.LocalTextSelectionColors
import androidx.compose.foundation.text.selection.TextSelectionColors
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density

/** Nocturne is a dark-only, mono-accent system. */
private val NocturneM3 = darkColorScheme(
    primary = NocturneColors.Accent,
    onPrimary = NocturneColors.Accent100,
    primaryContainer = NocturneColors.Accent900,
    onPrimaryContainer = NocturneColors.Accent200,
    secondary = NocturneColors.Accent2,
    onSecondary = NocturneColors.Accent2100,
    secondaryContainer = NocturneColors.Accent2900,
    onSecondaryContainer = NocturneColors.Accent2200,
    tertiary = NocturneColors.Accent,
    onTertiary = NocturneColors.Accent100,
    background = NocturneColors.Bg,
    onBackground = NocturneColors.Text,
    surface = NocturneColors.Surface,
    onSurface = NocturneColors.Text,
    surfaceVariant = NocturneColors.Neutral900,
    onSurfaceVariant = NocturneColors.TextMuted,
    surfaceContainerLowest = NocturneColors.Bg,
    surfaceContainerLow = NocturneColors.Neutral900,
    surfaceContainer = NocturneColors.Surface,
    surfaceContainerHigh = NocturneColors.Surface,
    surfaceContainerHighest = NocturneColors.Neutral800,
    outline = NocturneColors.Neutral700,
    outlineVariant = NocturneColors.Neutral800,
    scrim = NocturneColors.Neutral900,
    // Nocturne carries no red.
    error = NocturneColors.Neutral400,
    onError = NocturneColors.Neutral900,
    errorContainer = NocturneColors.Neutral800,
    onErrorContainer = NocturneColors.Neutral200,
)

/** `::selection { background: color-mix(in srgb, var(--color-accent) 30%, transparent); }` */
private val NocturneSelection = TextSelectionColors(
    handleColor = NocturneColors.Accent,
    backgroundColor = NocturneColors.Selection,
)

/** True when the app is drawing at phone scale inside the design canvas' 392×824 frame. */
val LocalShowsMockChrome = staticCompositionLocalOf { false }

@Composable
fun NocturneTheme(
    showsMockChrome: Boolean = false,
    content: @Composable () -> Unit,
) {
    // Every dp in this app was written against a phone, and on a desktop-mirrored,
    // freeform or tablet window Android hands us more dp at the *same* density:
    // a 44 dp control stays 44 dp, the window is three times as wide, and the
    // difference becomes empty padding. Scaling density is what turns the extra
    // dp back into a bigger interface.
    //
    // `uiScaleFor` is deliberately a plain function rather than a @Composable —
    // a composable cannot be called from inside `remember { }`, so the width is
    // read here, in the composable body, and passed in as an Int.
    val base = LocalDensity.current
    val widthDp = LocalConfiguration.current.screenWidthDp
    val scale = uiScaleFor(widthDp)
    val scaled = remember(base, scale) {
        // fontScale is carried through untouched: the user's font-size setting
        // is theirs, and multiplying it here would compound with the density.
        if (scale == 1f) base else Density(base.density * scale, base.fontScale)
    }
    MaterialTheme(
        colorScheme = NocturneM3,
        typography = MaterialTheme.typography, // unused: every call site passes an explicit style
    ) {
        CompositionLocalProvider(
            LocalDensity provides scaled,
            // Both of these matter, and for different reasons.
            LocalContentColor provides NocturneColors.Text,
            LocalTextStyle provides NocturneType.Body.copy(color = NocturneColors.Text),
            LocalTextSelectionColors provides NocturneSelection,
            LocalIndication provides ripple(color = NocturneColors.Accent),
            LocalShowsMockChrome provides showsMockChrome,
            content = content,
        )
    }
}

/** Convenience: a colour at an explicit alpha, matching CSS `color-mix(... N%, transparent)`. */
fun Color.mix(percent: Int): Color = copy(alpha = percent / 100f)

/**
 * How much to enlarge the interface, from the width of the window.
 *
 * **Density, not a redesign.** Multiplying density scales every dp in the app
 * at once and preserves every relationship between elements, so it cannot
 * produce a layout that was never tested — it produces the tested layout,
 * larger. A genuinely adaptive design, one that spent a wide window on a second
 * column or a wider inspector panel, would be better, and it is a separate and
 * much larger piece of work than this.
 *
 * The steps are coarse on purpose. Continuous scaling reflows text at widths
 * nobody ever looked at; the claim being made here is only that a 44 dp control
 * on a mirrored desktop should not read as a postage stamp.
 *
 * Capped at 1.8x, because past that the interface stops looking scaled and
 * starts looking zoomed — touch targets are sized for fingers, not for how much
 * screen happens to be available.
 *
 * Not a @Composable: it is called from inside `remember { }`, which is an
 * ordinary lambda and cannot host composable calls. The caller reads
 * `LocalConfiguration.current.screenWidthDp` and hands the Int over.
 *
 * Below [WIDE_DP] this returns 1.0f and nothing happens at all, so a phone
 * renders exactly as it did before this existed.
 */
private fun uiScaleFor(widthDp: Int): Float = when {
    widthDp >= HUGE_DP -> 1.8f
    widthDp >= LARGE_DP -> 1.5f
    widthDp >= WIDE_DP -> 1.25f
    else -> 1.0f
}

/**
 * Above any phone this app targets, so a phone is never scaled.
 *
 * Measured on the target device: 1264x2780 at density 480 is 3.0x, which is
 * 421 dp in portrait and **927 dp in landscape**. An earlier attempt used a
 * 900 dp floor, which 927 clears — so the phone scaled *itself* by 1.25x on the
 * one display every metric in this theme was chosen against, and the UI came
 * out too large. 1100 clears 927 with headroom for a taller phone. Do not
 * lower it.
 */
private const val WIDE_DP = 1100

/** A tablet, or a window given most of a laptop screen. */
private const val LARGE_DP = 1500

/** A desktop-sized window, where phone dp are simply unreadable. */
private const val HUGE_DP = 2000
