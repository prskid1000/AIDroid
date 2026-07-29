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
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color

/**
 * Nocturne is a dark-only, mono-accent system. There is no light scheme and no
 * dynamic colour: "exact same colouring" means the tokens in `styles.css` and
 * nothing else, so Material You is deliberately not wired up.
 *
 * Material 3's own colour roles are mapped onto Nocturne rather than left at
 * their defaults, so that any Material primitive still reachable (a `Text` with
 * no explicit colour, a ripple, a text-field cursor) cannot paint a colour that
 * isn't in the system.
 */
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
    // Nocturne carries no red. A failure is a neutral-800 disc with a slash,
    // not an error colour — so `error` is mapped to the neutral ramp on purpose
    // and no screen should reach for it.
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

/**
 * True when the app is drawing at phone scale inside the design canvas' 392×824
 * frame. Screens read it only to decide whether to draw the mock status bar.
 */
val LocalShowsMockChrome = staticCompositionLocalOf { false }

@Composable
fun NocturneTheme(
    showsMockChrome: Boolean = false,
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = NocturneM3,
        typography = MaterialTheme.typography, // unused: every call site passes an explicit style
    ) {
        CompositionLocalProvider(
            // Both of these matter, and for different reasons.
            //
            // LocalTextStyle covers a `Text` that passes no style at all.
            // LocalContentColor covers the far more common case here: a `Text`
            // that passes one of the NocturneType styles, none of which carry a
            // colour. Material 3 resolves that to LocalContentColor, which
            // defaults to *black* outside a `Surface` — and this app draws its
            // own grounds rather than using Surface, so without this every
            // heading, card title and label would render near-black on a
            // near-black ground.
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
