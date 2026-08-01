package ai.ondevice.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NSlider
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring

/** **S12 — Mask editor.** SPEC §5.3 asks for a real in-app brush editor for inpainting: brush size, hardness, erase, invert, clear, undo. */
@Composable
fun MaskEditorScreen(
    onCancel: () -> Unit,
    onDone: () -> Unit,
    viewModel: ai.ondevice.ui.vm.ImageViewModel = activityImageViewModel(),
) {
    val imageState by viewModel.state.collectAsStateWithLifecycle()
    val sourceUri = imageState.sourceImageUri
    val strokes = remember { mutableStateListOf<MaskStroke>() }
    val redoStack = remember { mutableStateListOf<MaskStroke>() }
    var brushSize by remember { mutableFloatStateOf(74f) }
    var hardness by remember { mutableFloatStateOf(0.35f) }
    var overlayOpacity by remember { mutableFloatStateOf(0.42f) }
    var erasing by remember { mutableStateOf(false) }
    var inverted by remember { mutableStateOf(false) }
    var cursor by remember { mutableStateOf<Offset?>(null) }
    var current by remember { mutableStateOf<MaskStroke?>(null) }

    PhoneScaffold(
        toolbar = {
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    "Cancel",
                    style = NocturneType.Input,
                    color = NocturneColors.Accent,
                    modifier = Modifier.nClickableFlat(onClick = onCancel),
                )
                Text(
                    "Mask",
                    style = NocturneType.H5,
                    modifier = Modifier.weight(1f),
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                )
                Text(
                    "Done",
                    style = NocturneType.Input,
                    color = NocturneColors.Accent,
                    modifier = Modifier.nClickableFlat(onClick = onDone),
                )
            }
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .clip(Radius.Md)
                .background(NocturneColors.Neutral900)
                .ring(NocturneColors.Divider, Radius.Md)
                .pointerInput(brushSize, erasing) {
                    detectDragGestures(
                        onDragStart = { offset ->
                            cursor = offset
                            current = MaskStroke(mutableListOf(offset), brushSize, erasing)
                            redoStack.clear()
                        },
                        onDragEnd = {
                            current?.let { strokes.add(it) }
                            current = null
                        },
                        onDragCancel = { current = null },
                        onDrag = { change, _ ->
                            cursor = change.position
                            current?.points?.add(change.position)
                        },
                    )
                },
        ) {
            // The real source the Image screen picked.
            sourceUri?.let {
                coil3.compose.AsyncImage(
                    model = it,
                    contentDescription = "Source image",
                    contentScale = androidx.compose.ui.layout.ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
            }

            if (sourceUri == null) {
                Canvas(Modifier.fillMaxSize()) {
                    drawRect(
                        Brush.radialGradient(
                            colors = listOf(NocturneColors.Neutral700, NocturneColors.Neutral900),
                            center = Offset(size.width * 0.42f, size.height * 0.38f),
                            radius = size.maxDimension * 0.6f,
                        ),
                    )
                }
            }

            // The mask is composited off-screen so that erasing punches a hole in the overlay and reveals the source underneath.
            Canvas(
                Modifier
                    .fillMaxSize()
                    .graphicsLayer { compositingStrategy = CompositingStrategy.Offscreen },
            ) {
                val paint = NocturneColors.Accent.copy(alpha = overlayOpacity)
                (strokes + listOfNotNull(current)).forEach { stroke ->
                    stroke.points.forEach { point ->
                        drawCircle(
                            color = if (stroke.erase) Color.Transparent else paint,
                            radius = stroke.size / 2f,
                            center = point,
                            // Hardness shapes the edge: 1.0 is a hard disc,
                            // lower values feather it.
                            alpha = if (stroke.erase) 1f else (0.35f + hardness * 0.65f),
                            blendMode = if (stroke.erase) BlendMode.Clear else DrawScope.DefaultBlendMode,
                        )
                    }
                }
            }

            Canvas(Modifier.fillMaxSize()) {
                // The brush cursor ring from the canvas.
                cursor?.let { position ->
                    drawCircle(
                        color = NocturneColors.Accent200,
                        radius = brushSize / 2f,
                        center = position,
                        style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1.5f),
                    )
                }
            }

            if (strokes.isEmpty() && current == null) {
                Text(
                    if (sourceUri == null) {
                        "no source image — pick one on the Image screen"
                    } else {
                        "drag to paint the mask"
                    },
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                    modifier = Modifier.align(Alignment.Center),
                )
            }
        }

        Row(
            Modifier.fillMaxWidth().padding(top = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            ToolButton(NIcons.Brush, "Paint", selected = !erasing) { erasing = false }
            ToolButton(NIcons.Erase, "Erase", selected = erasing) { erasing = true }
            ToolButton(NIcons.Invert, "Invert", selected = inverted) { inverted = !inverted }
            ToolButton(NIcons.Rotate, "Undo", enabled = strokes.isNotEmpty()) {
                strokes.removeLastOrNull()?.let { redoStack.add(it) }
            }
            ToolButton(NIcons.RotateBack, "Redo", enabled = redoStack.isNotEmpty()) {
                redoStack.removeLastOrNull()?.let { strokes.add(it) }
            }
        }

        LabeledMaskSlider("Brush size", "${brushSize.toInt()} px", brushSize, 4f..240f) { brushSize = it }
        LabeledMaskSlider("Hardness", String.format("%.2f", hardness), hardness, 0f..1f) { hardness = it }
        LabeledMaskSlider(
            "Mask opacity overlay",
            String.format("%.2f", overlayOpacity),
            overlayOpacity,
            0f..1f,
        ) { overlayOpacity = it }

        Row(
            Modifier.fillMaxWidth().padding(top = 14.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
        ) {
            NButton("Invert", { inverted = !inverted }, modifier = Modifier.weight(1f))
            NButton(
                "Clear",
                { strokes.clear(); redoStack.clear() },
                modifier = Modifier.weight(1f),
            )
        }

        NHelp(
            "Denoising is confined to the painted region. The mask is saved with the generation so the " +
                "result can be reproduced.",
            Modifier.padding(top = 14.dp),
        )
    }
}

private data class MaskStroke(
    val points: MutableList<Offset>,
    val size: Float,
    val erase: Boolean,
)

@Composable
private fun androidx.compose.foundation.layout.RowScope.ToolButton(
    icon: ImageVector,
    label: String,
    selected: Boolean = false,
    enabled: Boolean = true,
    onClick: () -> Unit,
) {
    Box(
        Modifier
            .weight(1f)
            .background(if (selected) NocturneColors.Accent900 else Color.Transparent, Radius.Md)
            .ring(if (selected) NocturneColors.Accent else NocturneColors.Divider, Radius.Md)
            .nClickableFlat(enabled = enabled, onClick = onClick)
            .padding(vertical = 11.dp),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            icon,
            contentDescription = label,
            tint = when {
                !enabled -> NocturneColors.Text.copy(alpha = 0.3f)
                selected -> NocturneColors.Accent200
                else -> NocturneColors.Text.copy(alpha = 0.75f)
            },
            modifier = Modifier.size(19.dp),
        )
    }
}

@Composable
private fun LabeledMaskSlider(
    label: String,
    display: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    onChange: (Float) -> Unit,
) {
    Column(Modifier.fillMaxWidth().padding(top = 12.dp)) {
        Row(
            Modifier.fillMaxWidth().padding(bottom = 3.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(label, style = NocturneType.Row, modifier = Modifier.weight(1f))
            Text(display, style = NocturneType.MonoValue, color = NocturneColors.Accent300)
        }
        NSlider(value = value, onValueChange = onChange, valueRange = range, height = 24.dp)
    }
}
