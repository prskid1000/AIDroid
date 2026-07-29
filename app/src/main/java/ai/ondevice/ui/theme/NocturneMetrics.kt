package ai.ondevice.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Immutable
import androidx.compose.ui.unit.dp

/**
 * `--space-*` — density 0.70×, already baked into the scale. The system is dense
 * on purpose; use these rather than raw numbers.
 *
 * The CSS carries fractional pixels (2.8, 5.6, 8.4…). Compose dp is a float, so
 * they survive the port unrounded.
 */
@Immutable
object Space {
    val s1 = 2.8.dp
    val s2 = 5.6.dp
    val s3 = 8.4.dp
    val s4 = 11.2.dp
    val s6 = 16.8.dp
    val s8 = 22.4.dp

    /**
     * The canvas lays its phone screens out on an 18px horizontal gutter, which
     * is not a `--space-*` step — it is the screen margin the mockups use
     * throughout. Named so it stays consistent across all 15 screens.
     */
    val ScreenGutter = 18.dp
}

/** `--radius-*`. */
@Immutable
object Radius {
    val sm = 4.dp
    val md = 8.dp
    val lg = 14.dp

    val Sm = RoundedCornerShape(sm)
    val Md = RoundedCornerShape(md)
    val Lg = RoundedCornerShape(lg)

    /** `.tag` — `calc(var(--radius-md) * 0.75)`. */
    val Tag = RoundedCornerShape(6.dp)
}

/**
 * `--shadow-*`. On a dark ground elevation is "an edge plus ambient darkness",
 * so every step starts with a 1px hairline ring; only md and lg add a drop.
 *
 * Compose has no multi-layer box-shadow, so these are expressed as a ring
 * colour plus an optional ambient blur — see `elev()` in NocturneComponents.kt.
 * The readme's rule is that shadows are never stacked, so one step per surface.
 */
@Immutable
data class NocturneShadow(
    val ringColor: androidx.compose.ui.graphics.Color,
    val ambientRadius: androidx.compose.ui.unit.Dp,
    val ambientAlpha: Float,
    val ambientOffsetY: androidx.compose.ui.unit.Dp,
)

@Immutable
object Elevation {
    /** `0 0 0 1px #3f424d` */
    val sm = NocturneShadow(NocturneColors.Neutral800, 0.dp, 0f, 0.dp)

    /** `0 0 0 1px #595d6c, 0 6px 18px rgba(0,0,0,0.55)` */
    val md = NocturneShadow(NocturneColors.Neutral700, 18.dp, 0.55f, 6.dp)

    /** `0 0 0 1px #9397ab, 0 16px 40px rgba(0,0,0,0.65)` */
    val lg = NocturneShadow(NocturneColors.Neutral500, 40.dp, 0.65f, 16.dp)
}

/**
 * Touch target floor. The canvas gives every interactive row `min-height:44px`
 * and the primary actions 46–48px; that is the number the screens honour rather
 * than Material's 48dp default, because the mockups are explicit about it.
 */
object Touch {
    val Min = 44.dp
    val Primary = 46.dp
    val Tall = 48.dp
}
