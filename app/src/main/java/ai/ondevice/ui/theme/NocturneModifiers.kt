package ai.ondevice.ui.theme

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * `.elev-sm` / `.elev-md` / `.elev-lg`.
 *
 * On a dark ground elevation is a hairline edge plus ambient darkness, never a
 * stack of shadows — so each step draws exactly one ring and at most one drop.
 */
fun Modifier.elev(
    shadow: NocturneShadow,
    shape: Shape = Radius.Md,
): Modifier {
    val withAmbient = if (shadow.ambientRadius > 0.dp) {
        this.shadow(
            elevation = shadow.ambientOffsetY,
            shape = shape,
            clip = false,
            ambientColor = Color.Black.copy(alpha = shadow.ambientAlpha),
            spotColor = Color.Black.copy(alpha = shadow.ambientAlpha),
        )
    } else {
        this
    }
    return withAmbient.border(1.dp, shadow.ringColor, shape)
}

/**
 * `box-shadow: inset 0 0 0 1px <color>` — the canvas' single most-used
 * treatment. It marks the selected quant, the loaded model, the active persona,
 * the refusal cards, the running download. Inset means it does not grow the
 * box, which is why the mockups reach for it instead of a border.
 */
fun Modifier.ring(
    color: Color,
    shape: Shape = Radius.Md,
    width: Dp = 1.dp,
): Modifier = border(width, color, shape)

/** A ring plus a fill, in the order the canvas paints them. */
fun Modifier.ringedSurface(
    fill: Color,
    ring: Color,
    shape: Shape = Radius.Md,
): Modifier = background(fill, shape).border(1.dp, ring, shape)

/**
 * `box-shadow: 0 1px 0 var(--color-divider)` — an in-control separator under a
 * row. Solid, not faded: the readme reserves the end-fade for freestanding
 * rules and table rows.
 */
fun Modifier.ruleBelow(color: Color = NocturneColors.Divider): Modifier = drawBehind {
    val y = size.height - 0.5f
    drawLine(color, Offset(0f, y), Offset(size.width, y), strokeWidth = 1f)
}

/** The same rule above a row — used by the chat composer and the bottom bar. */
fun Modifier.ruleAbove(color: Color = NocturneColors.Divider): Modifier = drawBehind {
    drawLine(color, Offset(0f, 0.5f), Offset(size.width, 0.5f), strokeWidth = 1f)
}

/**
 * The Nocturne signature: a rule that fades to transparent over 48px at each
 * end rather than stopping cleanly. Used by `.hr` and by table row rules.
 *
 * When the element is narrower than 96px the two ramps would overlap, so the
 * fade collapses to a symmetric gradient rather than producing a hard edge.
 */
fun Modifier.fadingRule(
    color: Color = NocturneColors.Divider,
    fadeWidth: Dp = 48.dp,
    atBottom: Boolean = true,
): Modifier = drawBehind {
    val fade = fadeWidth.toPx().coerceAtMost(size.width / 2f)
    val stops = if (fade * 2 >= size.width) {
        arrayOf(0f to Color.Transparent, 0.5f to color, 1f to Color.Transparent)
    } else {
        arrayOf(
            0f to Color.Transparent,
            fade / size.width to color,
            1f - fade / size.width to color,
            1f to Color.Transparent,
        )
    }
    val y = if (atBottom) size.height - 0.5f else 0.5f
    drawLine(
        brush = Brush.horizontalGradient(colorStops = stops),
        start = Offset(0f, y),
        end = Offset(size.width, y),
        strokeWidth = 1f,
    )
}

/**
 * The chat composer's pill and several inset panels want a fill plus an inset
 * hairline in one call, on an arbitrary corner radius.
 */
fun Modifier.panel(
    fill: Color = NocturneColors.Surface,
    ring: Color = NocturneColors.Divider,
    radius: Dp = Radius.md,
): Modifier = ringedSurface(fill, ring, RoundedCornerShape(radius))

/**
 * The vertical scrim the canvas paints under the TAESD preview's progress
 * readout: `linear-gradient(transparent, rgba(0,0,0,.6))`.
 */
fun Modifier.bottomScrim(alpha: Float = 0.6f): Modifier = drawBehind {
    drawRect(
        brush = Brush.verticalGradient(
            listOf(Color.Transparent, Color.Black.copy(alpha = alpha)),
        ),
        topLeft = Offset.Zero,
        size = Size(size.width, size.height),
    )
}
