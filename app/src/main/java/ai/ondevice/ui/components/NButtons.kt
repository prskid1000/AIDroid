package ai.ondevice.ui.components
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.ui.graphics.graphicsLayer

import androidx.compose.foundation.background
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.Space
import ai.ondevice.ui.theme.Touch
import ai.ondevice.ui.theme.ring

/** `.btn` and its variants. */
enum class NButtonStyle { Primary, Secondary, Ghost }

@Composable
fun NButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    style: NButtonStyle = NButtonStyle.Secondary,
    leadingIcon: ImageVector? = null,
    enabled: Boolean = true,
    block: Boolean = false,
    minHeight: Dp = Touch.Min,
    textStyle: androidx.compose.ui.text.TextStyle = NocturneType.Control,
) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()

    val contentColor = when (style) {
        NButtonStyle.Primary, NButtonStyle.Ghost -> NocturneColors.Accent
        NButtonStyle.Secondary -> NocturneColors.Text
    }
    val borderColor = when (style) {
        NButtonStyle.Primary -> NocturneColors.Accent
        NButtonStyle.Secondary -> NocturneColors.Divider
        NButtonStyle.Ghost -> Color.Transparent
    }
    // Pressed states come from the accent ramp, per the readme's interaction
    // rules — a colour-mix tint for outlined and ghost variants.
    val fillTarget = when {
        !pressed -> Color.Transparent
        style == NButtonStyle.Primary -> NocturneColors.AccentPressed
        style == NButtonStyle.Ghost -> NocturneColors.AccentGhostPressed
        else -> NocturneColors.NeutralPressed
    }
    // Crossed rather than switched: at 140 ms the fill arrives with the press
    // instead of after it, and a double tap picks up from wherever it was.
    val fill by animateColorAsState(fillTarget, Motion.colour, label = "fill")
    val scale by animateFloatAsState(
        if (pressed) Motion.PRESSED_SCALE else 1f,
        Motion.press,
        label = "press",
    )
    val fade by animateFloatAsState(
        if (enabled) 1f else NocturneColors.DisabledAlpha,
        Motion.press,
        label = "enabled",
    )
    val hPad = if (style == NButtonStyle.Ghost) Space.s1 else 10.08.dp

    Row(
        modifier = modifier
            .then(if (block) Modifier.fillMaxWidth() else Modifier)
            .graphicsLayer { scaleX = scale; scaleY = scale; alpha = fade }
            .defaultMinSize(minHeight = minHeight)
            .background(fill, Radius.Md)
            .ring(borderColor, Radius.Md)
            .nClickable(enabled = enabled, interactionSource = interaction, onClick = onClick)
            .padding(horizontal = hPad, vertical = Space.s2),
        horizontalArrangement = Arrangement.spacedBy(6.dp, Alignment.CenterHorizontally),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (leadingIcon != null) {
            Icon(leadingIcon, contentDescription = null, tint = contentColor, modifier = Modifier.size(14.dp))
        }
        Text(
            text = text,
            style = textStyle,
            color = contentColor,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            textAlign = TextAlign.Center,
        )
    }
}

/** `.btn-icon` — a 36×36 square with no padding. */
@Composable
fun NIconButton(
    icon: ImageVector,
    contentDescription: String?,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    style: NButtonStyle = NButtonStyle.Secondary,
    enabled: Boolean = true,
    size: Dp = 36.dp,
    iconSize: Dp = 17.dp,
    shape: Shape = Radius.Md,
    tint: Color? = null,
) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val contentColor = tint ?: when (style) {
        NButtonStyle.Primary, NButtonStyle.Ghost -> NocturneColors.Accent
        NButtonStyle.Secondary -> NocturneColors.Text
    }
    val borderColor = when (style) {
        NButtonStyle.Primary -> NocturneColors.Accent
        NButtonStyle.Secondary -> NocturneColors.Divider
        NButtonStyle.Ghost -> Color.Transparent
    }
    val fillTarget = when {
        !pressed -> Color.Transparent
        style == NButtonStyle.Primary -> NocturneColors.AccentPressed
        style == NButtonStyle.Ghost -> NocturneColors.AccentGhostPressed
        else -> NocturneColors.NeutralPressed
    }
    // Crossed rather than switched: at 140 ms the fill arrives with the press
    // instead of after it, and a double tap picks up from wherever it was.
    val fill by animateColorAsState(fillTarget, Motion.colour, label = "fill")
    val scale by animateFloatAsState(
        if (pressed) Motion.PRESSED_SCALE else 1f,
        Motion.press,
        label = "press",
    )
    val fade by animateFloatAsState(
        if (enabled) 1f else NocturneColors.DisabledAlpha,
        Motion.press,
        label = "enabled",
    )

    Box(
        modifier = modifier
            .size(size)
            .alpha(if (enabled) 1f else NocturneColors.DisabledAlpha)
            .background(fill, shape)
            .ring(borderColor, shape)
            .nClickable(enabled = enabled, interactionSource = interaction, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, contentDescription = contentDescription, tint = contentColor, modifier = Modifier.size(iconSize))
    }
}

/** The chat send/stop control: a 44dp accent-outlined circle. */
@Composable
fun NCircleButton(
    icon: ImageVector,
    contentDescription: String?,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) = NIconButton(
    icon = icon,
    contentDescription = contentDescription,
    onClick = onClick,
    modifier = modifier,
    style = NButtonStyle.Primary,
    enabled = enabled,
    size = 44.dp,
    iconSize = 17.dp,
    shape = CircleShape,
)

/** `.tag` — small labels tinted from the ramps. */
enum class NTagStyle { Accent, Accent2, Neutral, Outline }

@Composable
fun NTag(
    text: String,
    modifier: Modifier = Modifier,
    style: NTagStyle = NTagStyle.Neutral,
    leadingIcon: ImageVector? = null,
    iconSize: Dp = 11.dp,
    textStyle: androidx.compose.ui.text.TextStyle = NocturneType.Tag,
    contentPadding: PaddingValues = PaddingValues(horizontal = 10.dp, vertical = 3.dp),
) {
    val (bg, fg, border) = when (style) {
        NTagStyle.Accent -> Triple(NocturneColors.Accent800, NocturneColors.Accent100, Color.Transparent)
        NTagStyle.Accent2 -> Triple(NocturneColors.Accent2800, NocturneColors.Accent2100, Color.Transparent)
        NTagStyle.Neutral -> Triple(NocturneColors.Neutral800, NocturneColors.Neutral100, Color.Transparent)
        NTagStyle.Outline -> Triple(Color.Transparent, NocturneColors.Accent, NocturneColors.Accent)
    }
    Row(
        modifier = modifier
            .background(bg, Radius.Tag)
            .then(if (border != Color.Transparent) Modifier.ring(border, Radius.Tag) else Modifier)
            .padding(contentPadding),
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (leadingIcon != null) {
            Icon(leadingIcon, contentDescription = null, tint = fg, modifier = Modifier.size(iconSize))
        }
        Text(text, style = textStyle, color = fg, maxLines = 1)
    }
}

/** A tab/pill row — the shape the canvas uses for the parameter tiers (S8), the image modes (S11) and the transcribe modes (S14). */
@Composable
fun NPills(
    options: List<String>,
    selectedIndex: Int,
    onSelect: (Int) -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(modifier = modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        options.forEachIndexed { i, label ->
            val selected = i == selectedIndex
            Box(
                modifier = Modifier
                    .weight(1f)
                    .background(
                        if (selected) NocturneColors.Accent900 else NocturneColors.Bg,
                        Radius.Md,
                    )
                    .ring(if (selected) NocturneColors.Accent else NocturneColors.Divider, Radius.Md)
                    .selectable(selected = selected, onClick = { onSelect(i) })
                    .padding(vertical = 8.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    label,
                    style = NocturneType.Control.copy(fontSize = 12.sp),
                    color = if (selected) NocturneColors.Accent200 else NocturneColors.Text,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}
