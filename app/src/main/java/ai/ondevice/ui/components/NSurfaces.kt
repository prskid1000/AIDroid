package ai.ondevice.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import ai.ondevice.ui.theme.Elevation
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneShadow
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.Space
import ai.ondevice.ui.theme.elev
import ai.ondevice.ui.theme.fadingRule
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.theme.ruleBelow

/**
 * `.card` — a surface-filled content card at `--space-3` padding with
 * `--space-2` between children.
 *
 * [ring] replaces the elevation ring when set, which is how the canvas marks
 * state: accent-700 for the loaded model and the running download, neutral-700
 * for a refusal, accent-800 for the escape hatch.
 */
@Composable
fun NCard(
    modifier: Modifier = Modifier,
    elevation: NocturneShadow? = Elevation.sm,
    ring: Color? = null,
    fill: Color = NocturneColors.Surface,
    shape: Shape = Radius.Md,
    padding: PaddingValues = PaddingValues(Space.s3),
    gap: Dp = Space.s2,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier = modifier
            .background(fill, shape)
            .then(
                when {
                    ring != null -> Modifier.ring(ring, shape)
                    elevation != null -> Modifier.elev(elevation, shape)
                    else -> Modifier
                },
            )
            .padding(padding),
        verticalArrangement = Arrangement.spacedBy(gap),
        content = content,
    )
}

/** `.card-kicker` — 10px uppercase accent, letter-spaced. */
@Composable
fun NCardKicker(text: String, modifier: Modifier = Modifier) {
    Text(text.uppercase(), style = NocturneType.Kicker, color = NocturneColors.Accent, modifier = modifier)
}

/** `.card-title`. */
@Composable
fun NCardTitle(text: String, modifier: Modifier = Modifier, style: TextStyle = NocturneType.CardTitle) {
    Text(text, style = style, modifier = modifier)
}

/** `.card-body` — 13px at 80% opacity. */
@Composable
fun NCardBody(text: String, modifier: Modifier = Modifier) {
    Text(text, style = NocturneType.CardBody, color = NocturneColors.Text.copy(alpha = 0.8f), modifier = modifier)
}

/** `.card-meta` — 11px at 50% text, 6px gaps. */
@Composable
fun NCardMeta(
    modifier: Modifier = Modifier,
    gap: Dp = 6.dp,
    content: @Composable RowScope.() -> Unit,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(gap),
        verticalAlignment = Alignment.CenterVertically,
        content = content,
    )
}

@Composable
fun NMetaText(text: String, modifier: Modifier = Modifier, color: Color = NocturneColors.TextMeta) {
    Text(text, style = NocturneType.Meta, color = color, modifier = modifier)
}

/**
 * The mono section rule that heads every group on every screen. Not `.h6` —
 * the canvas uses a monospace 10px semibold in neutral-500, which is a
 * different mark from the stylesheet's `h6`.
 */
@Composable
fun SectionKicker(
    text: String,
    modifier: Modifier = Modifier,
    color: Color = NocturneColors.Neutral500,
    trailing: (@Composable () -> Unit)? = null,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(text.uppercase(), style = NocturneType.SectionKicker, color = color, modifier = Modifier.weight(1f))
        if (trailing != null) trailing()
    }
}

/** The recurring 10.5px muted footnote under a control or section. */
@Composable
fun NHelp(text: String, modifier: Modifier = Modifier, color: Color = NocturneColors.TextMuted) {
    Text(text, style = NocturneType.Help, color = color, modifier = modifier)
}

/**
 * `.hr` — the Nocturne signature rule, fading to transparent over 48px an end.
 *
 * Present, but the system prefers whitespace and the readme says to avoid it;
 * the phone screens use [NRowRule] instead. Kept because the system defines it.
 */
@Composable
fun NHr(modifier: Modifier = Modifier) {
    Box(modifier.fillMaxWidth().height(1.dp).fadingRule())
}

/** An in-control separator under a row: solid, per the readme's rule. */
@Composable
fun NRowRule(modifier: Modifier = Modifier) {
    Box(modifier.fillMaxWidth().height(1.dp).background(NocturneColors.Divider))
}

/**
 * A list row that carries its own bottom rule — the shape used by the parameter
 * list, the transcript segments and the companions list.
 */
@Composable
fun NRuledRow(
    modifier: Modifier = Modifier,
    verticalPadding: Dp = 11.dp,
    ruleColor: Color = NocturneColors.Divider,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .ruleBelow(ruleColor)
            .padding(vertical = verticalPadding),
        content = content,
    )
}

/**
 * `.table` — themed header and row rules. The rules are drawn at row level, not
 * cell level, so the 48px end-fade spans the whole row exactly as the CSS
 * background-gradient does.
 */
@Composable
fun NTable(
    modifier: Modifier = Modifier,
    header: (@Composable RowScope.() -> Unit)? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(modifier.fillMaxWidth()) {
        if (header != null) {
            Row(
                Modifier
                    .fillMaxWidth()
                    .fadingRule(NocturneColors.Divider)
                    .padding(Space.s2),
                verticalAlignment = Alignment.CenterVertically,
                content = header,
            )
        }
        content()
    }
}

@Composable
fun NTableRow(
    modifier: Modifier = Modifier,
    content: @Composable RowScope.() -> Unit,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .fadingRule(NocturneColors.TableRowRule)
            .padding(Space.s2),
        verticalAlignment = Alignment.CenterVertically,
        content = content,
    )
}

@Composable
fun NTableHeaderCell(
    text: String,
    modifier: Modifier = Modifier,
    align: TextAlign = TextAlign.Start,
) {
    Text(
        text.uppercase(),
        style = NocturneType.Meta.copy(letterSpacing = 0.08.em, textAlign = align),
        color = NocturneColors.TextTableHead,
        modifier = modifier,
    )
}

/**
 * `.dialog-backdrop` + `.dialog` — a modal at the top elevation over a
 * neutral-900 50% scrim.
 */
@Composable
fun NDialog(
    onDismissRequest: () -> Unit,
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit,
) {
    androidx.compose.ui.window.Dialog(onDismissRequest = onDismissRequest) {
        Box(
            Modifier
                .fillMaxSize()
                .background(NocturneColors.DialogScrim)
                .nClickableFlat(onClick = onDismissRequest)
                .padding(Space.s4),
            contentAlignment = Alignment.Center,
        ) {
            Column(
                modifier = modifier
                    .widthIn(max = 440.dp)
                    .fillMaxWidth()
                    .background(NocturneColors.Surface, Radius.Lg)
                    .elev(Elevation.lg, Radius.Lg)
                    .nClickableFlat { /* swallow: taps inside must not dismiss */ }
                    .padding(Space.s4),
                verticalArrangement = Arrangement.spacedBy(Space.s3),
                content = content,
            )
        }
    }
}

@Composable
fun NDialogTitle(text: String) {
    Text(text, style = NocturneType.H4)
}

@Composable
fun NDialogBody(text: String) {
    Text(text, style = NocturneType.Input, color = NocturneColors.Text.copy(alpha = 0.85f))
}

@Composable
fun NDialogActions(content: @Composable RowScope.() -> Unit) {
    Row(
        Modifier.fillMaxWidth().padding(top = Space.s2),
        horizontalArrangement = Arrangement.spacedBy(Space.s2, Alignment.End),
        content = content,
    )
}

/**
 * A thin progress bar. The canvas draws these as a neutral-900 track with an
 * accent-500 fill — 5px in download cards, 4px over the TAESD preview, 7px in
 * the storage meter.
 */
@Composable
fun NProgressBar(
    fraction: Float,
    modifier: Modifier = Modifier,
    height: Dp = 5.dp,
    fill: Color = NocturneColors.Accent500,
    track: Color = NocturneColors.Neutral900,
) {
    Box(
        modifier
            .fillMaxWidth()
            .height(height)
            .background(track, Radius.Sm),
    ) {
        Box(
            Modifier
                .fillMaxWidth(fraction.coerceIn(0f, 1f))
                .height(height)
                .background(fill, Radius.Sm),
        )
    }
}

/**
 * The multi-segment storage meter on the Models screen — one strip, several
 * ramp steps, no gaps.
 */
@Composable
fun NStackedBar(
    segments: List<Pair<Float, Color>>,
    modifier: Modifier = Modifier,
    height: Dp = 7.dp,
) {
    val used = segments.sumOf { it.first.toDouble() }.toFloat().coerceIn(0f, 1f)
    Row(
        modifier
            .fillMaxWidth()
            .height(height)
            .clip(Radius.Sm)
            .background(NocturneColors.Neutral900),
    ) {
        segments.forEach { (weight, color) ->
            if (weight > 0f) Box(Modifier.weight(weight).fillMaxHeight().background(color))
        }
        if (used < 1f) Box(Modifier.weight(1f - used).fillMaxHeight())
    }
}
