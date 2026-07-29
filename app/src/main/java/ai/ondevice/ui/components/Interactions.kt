package ai.ondevice.ui.components

import androidx.compose.foundation.LocalIndication
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed

/**
 * The one clickable used across the system, so that hover/press/focus behaviour
 * is defined once. The readme is explicit: interactive states are themed, never
 * browser (or platform) defaults — pressed states come from the accent ramp and
 * keyboard focus is the 2px accent ring.
 *
 * Callers that paint their own pressed tint pass their own [interactionSource]
 * and get no ripple on top; callers that don't get the accent ripple from
 * [LocalIndication], which `NocturneTheme` sets.
 */
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
    return clickable(interactionSource = source, indication = null, enabled = enabled, onClick = onClick)
}
