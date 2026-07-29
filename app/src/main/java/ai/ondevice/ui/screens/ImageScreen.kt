package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.ui.BottomDestinations
import ai.ondevice.ui.components.NBottomBar
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NField
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NPills
import ai.ondevice.ui.components.NProgressBar
import ai.ondevice.ui.components.NSeg
import ai.ondevice.ui.components.NSlider
import ai.ondevice.ui.components.NSwitch
import ai.ondevice.ui.components.NTextArea
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.RootToolbar
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.bottomScrim
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.ImageAction
import ai.ondevice.ui.vm.ImageMode
import ai.ondevice.ui.vm.ImageState
import ai.ondevice.ui.vm.ImageViewModel

/**
 * **S11 — Image.**
 *
 * SPEC §5.4's obligations, made visible: a live TAESD preview instead of a
 * spinner, a cancel that says out loud that it frees native memory, and the
 * memory guardrail that suggests `vae_tiling` rather than letting a 1024×1024
 * run take the process down.
 *
 * The screen is honest about the hardware too: sd.cpp on Adreno 829 is
 * unproven (§15), so the readout names the backend actually in use.
 */
@Composable
fun ImageScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onOpenMask: () -> Unit,
    onOpenGallery: () -> Unit,
    onOpenRuntimes: () -> Unit,
    onAddModel: () -> Unit,
    viewModel: ImageViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = {
            RootToolbar("Image") {
                Icon(
                    NIcons.Image,
                    contentDescription = "Gallery",
                    tint = NocturneColors.Text,
                    modifier = Modifier.size(20.dp).nClickableFlat(onClick = onOpenGallery),
                )
            }
        },
        bottomBar = { NBottomBar(BottomDestinations, currentRoute) { onNavigate(it.route) } },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        NPills(
            options = ImageMode.entries.map { it.label },
            selectedIndex = ImageMode.entries.indexOf(state.mode),
            onSelect = { viewModel.setMode(ImageMode.entries[it]) },
            modifier = Modifier.padding(bottom = 10.dp),
        )

        Column(Modifier.verticalScroll(rememberScrollState())) {

            TaesdPreview(state)

            NButton(
                text = state.actionLabel,
                onClick = {
                    when (state.action) {
                        ImageAction.CANCEL -> viewModel.cancel()
                        ImageAction.GENERATE -> viewModel.generate()
                        ImageAction.INSTALL_RUNTIME -> onOpenRuntimes()
                        ImageAction.ADD_MODEL -> onAddModel()
                    }
                },
                style = if (state.action == ImageAction.CANCEL) {
                    NButtonStyle.Secondary
                } else {
                    NButtonStyle.Primary
                },
                block = true,
                modifier = Modifier.padding(top = 8.dp),
            )
            state.actionHint?.let { NHelp(it, Modifier.padding(top = 6.dp)) }

            NField("Prompt", Modifier.padding(top = 16.dp)) {
                NTextArea(
                    value = state.prompt,
                    onValueChange = viewModel::setPrompt,
                    minHeight = 66.dp,
                    textStyle = NocturneType.Row,
                )
            }
            NField("Negative prompt", Modifier.padding(top = 8.dp)) {
                NInput(
                    value = state.negativePrompt,
                    onValueChange = viewModel::setNegativePrompt,
                )
            }

            // The key dial in img2img and inpaint, surfaced only in those modes.
            if (state.showStrength) {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .padding(top = 12.dp)
                        .background(NocturneColors.Accent900, Radius.Md)
                        .padding(horizontal = 12.dp, vertical = 11.dp),
                ) {
                    Row(
                        Modifier.fillMaxWidth().padding(bottom = 4.dp),
                        horizontalArrangement = Arrangement.spacedBy(7.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            "Denoise strength",
                            style = NocturneType.Row,
                            color = NocturneColors.Accent200,
                            modifier = Modifier.weight(1f),
                        )
                        Text(
                            String.format("%.2f", state.strength),
                            style = NocturneType.MonoValue,
                            color = NocturneColors.Accent200,
                        )
                    }
                    NSlider(
                        value = state.strength,
                        onValueChange = viewModel::setStrength,
                        valueRange = 0f..1f,
                    )
                    NHelp(
                        "The key dial in this mode. 0 leaves the source untouched, 1 ignores it.",
                        color = NocturneColors.Accent300,
                    )
                }
                if (state.mode == ImageMode.INPAINT) {
                    NButton(
                        "Edit mask",
                        onClick = onOpenMask,
                        style = NButtonStyle.Secondary,
                        block = true,
                        modifier = Modifier.padding(top = 8.dp),
                    )
                }
            }

            Row(
                Modifier.fillMaxWidth().padding(top = 14.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                LabeledSlider(
                    label = "Steps",
                    value = state.steps.toFloat(),
                    display = state.steps.toString(),
                    range = 1f..60f,
                    onChange = { viewModel.setSteps(it.toInt()) },
                    modifier = Modifier.weight(1f),
                )
                LabeledSlider(
                    label = "CFG",
                    value = state.cfgScale,
                    display = String.format("%.1f", state.cfgScale),
                    range = 1f..20f,
                    onChange = viewModel::setCfg,
                    modifier = Modifier.weight(1f),
                )
            }

            Row(
                Modifier.fillMaxWidth().padding(top = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Column(Modifier.weight(1f)) {
                    NHelp("Size", Modifier.padding(bottom = 4.dp))
                    val sizes = listOf(512, 768, 1024)
                    NSeg(
                        options = sizes.map { it.toString() },
                        selectedIndex = sizes.indexOf(state.width).coerceAtLeast(0),
                        onSelect = { viewModel.setSize(sizes[it]) },
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                Column(Modifier.width(118.dp)) {
                    NHelp("Seed", Modifier.padding(bottom = 4.dp))
                    NInput(
                        value = state.seed.toString(),
                        onValueChange = { it.toLongOrNull()?.let(viewModel::setSeed) },
                        textStyle = NocturneType.MonoValue,
                        keyboardType = androidx.compose.ui.text.input.KeyboardType.Number,
                    )
                }
            }

            // §5.4 — memory guardrail with the fix inline, not a toast.
            if (state.exceedsEnvelope) {
                NCard(Modifier.padding(top = 14.dp), ring = NocturneColors.Neutral700) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            NIcons.TriangleAlert,
                            contentDescription = null,
                            tint = NocturneColors.Neutral300,
                            modifier = Modifier.size(15.dp),
                        )
                        Text(
                            "${state.width} × ${state.height} exceeds the measured envelope",
                            style = NocturneType.CardTitleSm,
                            modifier = Modifier.weight(1f),
                        )
                    }
                    Text(
                        "Turn on vae_tiling or drop to 768. sd.cpp on Adreno 829 is unproven — this " +
                            "runs on CPU.",
                        style = NocturneType.CardBody,
                        color = NocturneColors.Text.copy(alpha = 0.8f),
                    )
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(9.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text("vae_tiling", style = NocturneType.Row, modifier = Modifier.weight(1f))
                        NSwitch(state.vaeTiling, viewModel::setVaeTiling)
                    }
                }
            }

            NButton(
                "Advanced · schedule, clip_skip, SLG",
                onClick = { },
                style = NButtonStyle.Secondary,
                block = true,
                modifier = Modifier.padding(top = 14.dp),
            )

            state.usedSeed?.let { seed ->
                NHelp(
                    "Last run used seed $seed. Every generation writes its full parameter set into the " +
                        "PNG, so any image in the gallery can be reproduced exactly.",
                    Modifier.padding(top = 12.dp),
                )
            }
        }
    }
}

/**
 * The live preview. SPEC §5.4: "show intermediate latents, not just a spinner".
 * Until sd.cpp is wired in, this paints the same accent-and-section-glow field
 * the canvas mocks, over a scanline texture, with the real step/ETA readout.
 */
@Composable
private fun TaesdPreview(state: ImageState) {
    Box(
        Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .clip(Radius.Md)
            .background(NocturneColors.Neutral900)
            .ring(NocturneColors.Divider, Radius.Md),
        contentAlignment = Alignment.Center,
    ) {
        // The latent field only blooms while sampling. Idle, it stays on the
        // neutral ramp — the readme forbids flooding a large area with a
        // saturated fill, and an idle preview has nothing to say anyway.
        val bloom = if (state.generating) 1f else 0.12f

        androidx.compose.foundation.Canvas(Modifier.fillMaxSize()) {
            drawRect(
                Brush.radialGradient(
                    colors = listOf(NocturneColors.Accent700.copy(alpha = bloom), Color.Transparent),
                    center = Offset(size.width * 0.28f, size.height * 0.22f),
                    radius = size.maxDimension * 0.62f,
                ),
            )
            drawRect(
                Brush.radialGradient(
                    colors = listOf(NocturneColors.SectionGlow.copy(alpha = bloom), Color.Transparent),
                    center = Offset(size.width * 0.76f, size.height * 0.78f),
                    radius = size.maxDimension * 0.58f,
                ),
            )
            // Scanlines — the canvas' `repeating-linear-gradient`.
            var y = 0f
            while (y < size.height) {
                drawRect(
                    color = Color.Black.copy(alpha = 0.16f),
                    topLeft = Offset(0f, y),
                    size = Size(size.width, 1f),
                )
                y += 3f
            }
        }

        Text(
            if (state.generating) "TAESD preview" else "No preview yet",
            style = NocturneType.MonoSm,
            color = if (state.generating) {
                NocturneColors.Accent200.copy(alpha = 0.9f)
            } else {
                NocturneColors.TextMuted
            },
        )

        if (state.generating) {
            Column(
                Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .bottomScrim()
                    .padding(horizontal = 11.dp, vertical = 9.dp),
            ) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        "step ${state.step}/${state.steps}",
                        style = NocturneType.MonoSm,
                        color = NocturneColors.Accent200,
                    )
                    Text("·", style = NocturneType.MonoSm, color = Color.White.copy(alpha = 0.7f))
                    Text(
                        "CPU · ${String.format("%.1f", state.secondsPerStep)} s/it",
                        style = NocturneType.MonoSm,
                        color = Color.White.copy(alpha = 0.7f),
                    )
                    Box(Modifier.weight(1f))
                    Text(
                        Fmt.eta(state.etaSeconds),
                        style = NocturneType.MonoSm,
                        color = Color.White.copy(alpha = 0.7f),
                    )
                }
                NProgressBar(
                    fraction = state.progress,
                    modifier = Modifier.padding(top = 6.dp),
                    height = 4.dp,
                    fill = NocturneColors.Accent400,
                    track = Color.Black.copy(alpha = 0.45f),
                )
            }
        }
    }
}

@Composable
private fun LabeledSlider(
    label: String,
    value: Float,
    display: String,
    range: ClosedFloatingPointRange<Float>,
    onChange: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(label, style = NocturneType.Row, modifier = Modifier.weight(1f))
            Text(display, style = NocturneType.MonoValue, color = NocturneColors.Accent300)
        }
        NSlider(value = value, onValueChange = onChange, valueRange = range)
    }
}
