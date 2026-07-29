package ai.ondevice.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsFocusedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring

/** `.field > label` — 12px at 70% text, 5px below. */
@Composable
fun NFieldLabel(text: String, modifier: Modifier = Modifier) {
    Text(text, style = NocturneType.FieldLabel, color = NocturneColors.TextLabel, modifier = modifier.padding(bottom = 5.dp))
}

/**
 * `.input`.
 *
 * Surface fill, divider hairline, accent caret, and an accent border on focus —
 * `:focus-visible { border-color: var(--color-accent); outline-offset: 0 }`, so
 * the ring replaces the border rather than sitting outside it.
 */
@Composable
fun NInput(
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    placeholder: String? = null,
    enabled: Boolean = true,
    singleLine: Boolean = true,
    minHeight: Dp = 36.dp,
    textStyle: TextStyle = NocturneType.Input,
    keyboardType: KeyboardType = KeyboardType.Text,
    trailing: (@Composable () -> Unit)? = null,
) {
    val interaction = remember { MutableInteractionSource() }
    val focused by interaction.collectIsFocusedAsState()

    Row(
        modifier = modifier
            .fillMaxWidth()
            .defaultMinSize(minHeight = minHeight)
            .background(NocturneColors.Surface, Radius.Md)
            .ring(if (focused) NocturneColors.Accent else NocturneColors.Divider, Radius.Md)
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.weight(1f)) {
            if (value.isEmpty() && placeholder != null) {
                Text(placeholder, style = textStyle, color = NocturneColors.TextMuted)
            }
            BasicTextField(
                value = value,
                onValueChange = onValueChange,
                enabled = enabled,
                singleLine = singleLine,
                textStyle = textStyle.copy(color = NocturneColors.Text),
                cursorBrush = SolidColor(NocturneColors.Accent),
                interactionSource = interaction,
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(keyboardType = keyboardType),
                modifier = Modifier.fillMaxWidth(),
            )
        }
        if (trailing != null) trailing()
    }
}

/** `textarea.input` — `min-height: 90px`, vertically resizable in CSS; here a floor. */
@Composable
fun NTextArea(
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    minHeight: Dp = 90.dp,
    placeholder: String? = null,
    textStyle: TextStyle = NocturneType.Input,
) = NInput(
    value = value,
    onValueChange = onValueChange,
    modifier = modifier,
    placeholder = placeholder,
    singleLine = false,
    minHeight = minHeight,
    textStyle = textStyle,
)

/** `.field` — label above input. */
@Composable
fun NField(
    label: String,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit,
) {
    Column(modifier) {
        NFieldLabel(label)
        content()
    }
}

/**
 * `.radio` + `.dot` — a 16px ring that fills with the accent and punches a
 * 4px hole of the ground through the middle when checked
 * (`box-shadow: inset 0 0 0 4px var(--color-bg)`).
 */
@Composable
fun NRadio(
    label: String,
    selected: Boolean,
    onSelect: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier.selectable(selected = selected, onClick = onSelect),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier
                .size(16.dp)
                .background(if (selected) NocturneColors.Accent else Color.Transparent, CircleShape)
                .ring(if (selected) NocturneColors.Accent else NocturneColors.Divider, CircleShape, 1.5.dp),
            contentAlignment = Alignment.Center,
        ) {
            if (selected) Box(Modifier.size(8.dp).background(NocturneColors.Bg, CircleShape))
        }
        Text(label, style = NocturneType.Input)
    }
}

/**
 * `.seg` + `.seg-opt` — a joined segmented control. Selected reads as accent
 * text plus an accent ring inset over the whole option; the divider between
 * options stays solid because it is an in-control separator.
 */
@Composable
fun NSeg(
    options: List<String>,
    selectedIndex: Int,
    onSelect: (Int) -> Unit,
    modifier: Modifier = Modifier,
    textStyle: TextStyle = NocturneType.Input.copy(fontSize = 13.sp),
) {
    Row(
        modifier = modifier
            .clip(Radius.Md)
            .ring(NocturneColors.Divider, Radius.Md),
    ) {
        options.forEachIndexed { i, label ->
            val selected = i == selectedIndex
            Box(
                modifier = Modifier
                    .weight(1f)
                    .then(
                        if (i > 0) Modifier.drawLeftDivider() else Modifier,
                    )
                    .then(if (selected) Modifier.ring(NocturneColors.Accent, RoundedCornerShape(0.dp)) else Modifier)
                    .selectable(selected = selected, onClick = { onSelect(i) })
                    .padding(horizontal = 12.dp, vertical = 7.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    label,
                    style = textStyle,
                    color = if (selected) NocturneColors.Accent else NocturneColors.Text,
                    maxLines = 1,
                )
            }
        }
    }
}

/** `.seg-opt + .seg-opt { border-left: 1px solid var(--color-divider) }`. */
private fun Modifier.drawLeftDivider(): Modifier = drawBehind {
    drawLine(
        color = NocturneColors.Divider,
        start = Offset(0.5f, 0f),
        end = Offset(0.5f, size.height),
        strokeWidth = 1f,
    )
}

/**
 * The range control. The canvas styles `input[type=range]` with
 * `accent-color: #9184d9` and gives it a 22–28px tap height, so the track is
 * thin and the thumb is the only accent mass.
 */
@OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
fun NSlider(
    value: Float,
    onValueChange: (Float) -> Unit,
    valueRange: ClosedFloatingPointRange<Float>,
    modifier: Modifier = Modifier,
    steps: Int = 0,
    enabled: Boolean = true,
    height: Dp = 22.dp,
    onValueChangeFinished: (() -> Unit)? = null,
) {
    androidx.compose.material3.Slider(
        value = value.coerceIn(valueRange.start, valueRange.endInclusive),
        onValueChange = onValueChange,
        onValueChangeFinished = onValueChangeFinished,
        valueRange = valueRange,
        steps = steps,
        enabled = enabled,
        modifier = modifier.fillMaxWidth().height(height),
        colors = androidx.compose.material3.SliderDefaults.colors(
            thumbColor = NocturneColors.Accent,
            activeTrackColor = NocturneColors.Accent,
            inactiveTrackColor = NocturneColors.Neutral800,
            activeTickColor = Color.Transparent,
            inactiveTickColor = Color.Transparent,
            disabledThumbColor = NocturneColors.Neutral600,
            disabledActiveTrackColor = NocturneColors.Neutral700,
            disabledInactiveTrackColor = NocturneColors.Neutral900,
        ),
        thumb = {
            Box(
                Modifier
                    .size(16.dp)
                    .background(if (enabled) NocturneColors.Accent else NocturneColors.Neutral600, CircleShape),
            )
        },
        track = { state ->
            val fraction = if (state.valueRange.endInclusive > state.valueRange.start) {
                (state.value - state.valueRange.start) /
                    (state.valueRange.endInclusive - state.valueRange.start)
            } else {
                0f
            }
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(4.dp)
                    .background(NocturneColors.Neutral800, CircleShape),
            ) {
                Box(
                    Modifier
                        .fillMaxWidth(fraction.coerceIn(0f, 1f))
                        .height(4.dp)
                        .background(
                            if (enabled) NocturneColors.Accent else NocturneColors.Neutral700,
                            CircleShape,
                        ),
                )
            }
        },
    )
}

/**
 * The pill toggle from S8 and S11 — 40×23 track, 17px knob, divider hairline.
 * On is an accent-700 track with an accent-200 knob; off is the neutral ramp.
 * Explicitly not `material3.Switch`, whose proportions are different.
 */
@Composable
fun NSwitch(
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    val offset by animateFloatAsState(if (checked) 1f else 0f, label = "switch")
    Box(
        modifier = modifier
            .width(40.dp)
            .height(23.dp)
            .background(
                if (checked) NocturneColors.Accent700 else NocturneColors.Neutral900,
                CircleShape,
            )
            .ring(NocturneColors.Divider, CircleShape)
            .nClickableFlat(enabled) { onCheckedChange(!checked) }
            .padding(2.dp),
        contentAlignment = Alignment.CenterStart,
    ) {
        Box(
            Modifier
                .offset(x = (offset * 17).dp)
                .size(17.dp)
                .background(
                    if (checked) NocturneColors.Accent200 else NocturneColors.Neutral600,
                    CircleShape,
                ),
        )
    }
}

/**
 * The chip editor used for `stop`, `dry_sequence_breakers` and the manifest's
 * `string[]` type. A chip is neutral-900 with a divider ring; the add
 * affordance is accent text inside an accent-700 ring.
 */
@OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
fun NChipRow(
    chips: List<String>,
    onRemove: ((Int) -> Unit)? = null,
    onAdd: (() -> Unit)? = null,
    modifier: Modifier = Modifier,
) {
    androidx.compose.foundation.layout.FlowRow(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        chips.forEachIndexed { i, chip ->
            Box(
                Modifier
                    .background(NocturneColors.Neutral900, Radius.Sm)
                    .ring(NocturneColors.Divider, Radius.Sm)
                    .then(if (onRemove != null) Modifier.nClickableFlat { onRemove(i) } else Modifier)
                    .padding(horizontal = 8.dp, vertical = 3.dp),
            ) {
                Text(chip, style = NocturneType.MonoSm.copy(fontWeight = FontWeight.Medium))
            }
        }
        if (onAdd != null) {
            Box(
                Modifier
                    .ring(NocturneColors.Accent700, Radius.Sm)
                    .nClickableFlat(onClick = onAdd)
                    .padding(horizontal = 8.dp, vertical = 3.dp),
            ) {
                Text("+ add", style = NocturneType.Meta, color = NocturneColors.Accent)
            }
        }
    }
}

/**
 * The enum picker from S8: a wrapped row of mono option chips. Selected is an
 * accent-900 fill with an accent ring.
 */
@OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
fun NEnumRow(
    options: List<String>,
    selected: String?,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    androidx.compose.foundation.layout.FlowRow(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        options.forEach { opt ->
            val isSelected = opt == selected
            Box(
                Modifier
                    .background(if (isSelected) NocturneColors.Accent900 else Color.Transparent, Radius.Sm)
                    .ring(if (isSelected) NocturneColors.Accent else NocturneColors.Divider, Radius.Sm)
                    .selectable(selected = isSelected, onClick = { onSelect(opt) })
                    .padding(horizontal = 9.dp, vertical = 5.dp),
            ) {
                Text(
                    opt,
                    style = NocturneType.MonoSm.copy(fontWeight = FontWeight.Medium),
                    color = if (isSelected) NocturneColors.Accent200 else NocturneColors.Text,
                )
            }
        }
    }
}
