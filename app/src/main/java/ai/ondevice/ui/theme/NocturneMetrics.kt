package ai.ondevice.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Immutable
import androidx.compose.ui.unit.dp

/** `--space-*` — density 0.70×, already baked into the scale. */
@Immutable
object Space {
    val s1 = 2.8.dp
    val s2 = 5.6.dp
    val s3 = 8.4.dp
    val s4 = 11.2.dp
    val s6 = 16.8.dp
    val s8 = 22.4.dp

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

/** `--shadow-*`. */
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

/** Touch target floor. */
object Touch {
    val Min = 44.dp
    val Primary = 46.dp
    val Tall = 48.dp
}
