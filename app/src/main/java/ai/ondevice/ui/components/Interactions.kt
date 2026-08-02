package ai.ondevice.ui.components
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.SpringSpec
import androidx.compose.animation.core.TweenSpec
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.unit.IntSize

import androidx.compose.foundation.LocalIndication
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed

/** The one clickable used across the system, so that hover/press/focus behaviour is defined once. */
fun Modifier.nClickable(
    enabled: Boolean = true,
    interactionSource: MutableInteractionSource? = null,
    role: androidx.compose.ui.semantics.Role? = null,
    onClick: () -> Unit,
): Modifier = composed {
    val source = interactionSource ?: remember { MutableInteractionSource() }
    clickable(
        interactionSource = source,
        indication = if (interactionSource == null) LocalIndication.current else null,
        enabled = enabled,
        role = role,
        onClick = onClick,
    )
}

/** A clickable that never draws indication — for rows that tint themselves. */
@Composable
fun Modifier.nClickableFlat(enabled: Boolean = true, onClick: () -> Unit): Modifier {
    val source = remember { MutableInteractionSource() }
    // Every tappable row in the app comes through here, so this is where they
    // all learn to answer a finger. Drawing no indication is the point of this
    // modifier; a scale is not indication drawn *over* the row, it is the row
    // itself moving, so the two do not fight.
    val scale = pressScale(source)
    return graphicsLayer { scaleX = scale; scaleY = scale }
        .clickable(interactionSource = source, indication = null, enabled = enabled, onClick = onClick)
}

/**
 * How this app moves.
 *
 * One place, because motion that varies per screen reads as several apps. The
 * durations are short on purpose: a press has to answer within the time a
 * finger is still down, or the feedback arrives after the thing it was feedback
 * for. Springs rather than curves, so an interrupted animation continues from
 * where it was instead of jumping back to the start — which is what happens
 * when a list is tapped twice quickly.
 */
object Motion {

    /** A press: fast, slightly under-damped, no visible bounce at this scale. */
    val press: SpringSpec<Float> = spring(
        dampingRatio = Spring.DampingRatioNoBouncy,
        stiffness = Spring.StiffnessHigh,
    )

    /** A colour crossing between two states. */
    val colour: TweenSpec<Color> = tween(durationMillis = 140)

    /** Something growing or shrinking — a card that gained a line, a list that filtered. */
    val resize: SpringSpec<IntSize> = spring(
        dampingRatio = Spring.DampingRatioNoBouncy,
        stiffness = Spring.StiffnessMediumLow,
    )

    /** A value the user did not set — a progress bar catching up to its number. */
    val value: TweenSpec<Float> = tween(durationMillis = 220)

    /** How far a pressed surface shrinks. Enough to feel, too little to notice. */
    const val PRESSED_SCALE = 0.975f
}

/**
 * The scale a surface takes while pressed, animated.
 *
 * Kept next to [Motion] rather than repeated at each call site: every tappable
 * thing in this app should answer a finger the same way, and the way to
 * guarantee that is for there to be one implementation of it.
 */
@Composable
fun pressScale(source: MutableInteractionSource): Float {
    val pressed by source.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (pressed) Motion.PRESSED_SCALE else 1f,
        animationSpec = Motion.press,
        label = "press",
    )
    return scale
}
