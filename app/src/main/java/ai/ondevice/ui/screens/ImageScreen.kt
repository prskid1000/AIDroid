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
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil3.compose.AsyncImage
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
import ai.ondevice.ui.components.SectionKicker
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
    onOpenAdvanced: () -> Unit,
    viewModel: ImageViewModel = activityImageViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val pickSource = rememberSourceImagePicker(viewModel::setSourceImage)
    val pickControl = rememberSourceImagePicker(viewModel::setControlImage)

    // The Advanced screen writes to the same diffusion model row, so pick up
    // anything it changed on the way back rather than showing a stale form.
    LaunchedEffect(Unit) { viewModel.refreshFromOverrides() }

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

            // §1.2 — a model that cannot run says why, in the runtime's own
            // words, instead of leaving a blank frame.
            state.error?.let { message ->
                NCard(Modifier.padding(top = 10.dp), ring = NocturneColors.Neutral700) {
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
                        Text(message, style = NocturneType.CardTitleSm, modifier = Modifier.weight(1f))
                    }
                    state.errorHint?.let {
                        Text(
                            it,
                            style = NocturneType.CardBody,
                            color = NocturneColors.Text.copy(alpha = 0.8f),
                        )
                    }
                }
            }

            NButton(
                text = state.actionLabel,
                onClick = {
                    when (state.action) {
                        ImageAction.CANCEL -> viewModel.cancel()
                        ImageAction.GENERATE -> viewModel.generate()
                        ImageAction.INSTALL_RUNTIME -> onOpenRuntimes()
                        ImageAction.ADD_MODEL -> onAddModel()
                        ImageAction.PICK_SOURCE -> pickSource()
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

            // The source image. Optional in Generate — attaching one is what
            // makes it img2img — and required by Inpaint and Extend.
            SourceImageField(
                label = if (state.requiresSource) "Source image" else "Source image · optional",
                uri = state.sourceImageUri,
                emptyLabel = if (state.requiresSource) {
                    "Choose the image to edit"
                } else {
                    "Add a source image to transform one instead"
                },
                onPick = pickSource,
                onClear = { viewModel.setSourceImage(null) },
            )

            // ControlNet's reference is a different input, not a second source:
            // it contributes structure, never pixels.
            SourceImageField(
                label = "Control image · optional",
                uri = state.controlImageUri,
                emptyLabel = "Add a pose, depth or edge map to steer composition",
                onPick = pickControl,
                onClear = { viewModel.setControlImage(null) },
            )
            if (state.controlImageUri != null) {
                LabeledSlider(
                    label = "Control strength",
                    value = state.controlStrength,
                    display = String.format("%.2f", state.controlStrength),
                    range = 0f..1f,
                    onChange = viewModel::setControlStrength,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }

            if (state.mode == ImageMode.OUTPAINT) {
                ExtendField(state, viewModel)
            }

            // The key dial: only meaningful once there is something to denoise.
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

            AttachmentsSection(state, viewModel)

            // The same manifest renderer S8 uses, pointed at the sd.cpp block —
            // schedule and clip_skip are Advanced there, SLG is Expert, and all
            // 41 of them come from the manifest rather than this file.
            NButton(
                "Advanced · schedule, clip_skip, SLG",
                onClick = onOpenAdvanced,
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
 * The system photo picker.
 *
 * `PickVisualMedia` is the right contract even at minSdk 31: where the platform
 * picker exists it runs without any storage permission at all, and where it
 * does not it falls back to `OPEN_DOCUMENT` on its own. Either way the app
 * never asks for READ_MEDIA_IMAGES, which is the whole point — SPEC §13 says
 * the app should hold no permission it can avoid.
 */
@Composable
private fun rememberSourceImagePicker(onPicked: (String) -> Unit): () -> Unit {
    val context = LocalContext.current
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
    ) { uri ->
        uri ?: return@rememberLauncherForActivityResult
        // The picker's grant dies with the process. Persisting it when the
        // provider allows keeps a chosen source valid across a restart; when it
        // does not, the URI still works for this session.
        runCatching {
            context.contentResolver.takePersistableUriPermission(
                uri,
                android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION,
            )
        }
        onPicked(uri.toString())
    }
    return {
        launcher.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
        )
    }
}

/**
 * The img2img / inpaint source. SPEC §5.2 — the mode is meaningless without one,
 * so it sits in the form above the dial that acts on it rather than being
 * discovered at generate time.
 */
@Composable
private fun SourceImageField(
    label: String,
    uri: String?,
    emptyLabel: String,
    onPick: () -> Unit,
    onClear: () -> Unit,
) {
    NField(label, Modifier.padding(top = 12.dp)) {
        if (uri == null) {
            NButton(
                emptyLabel,
                onClick = onPick,
                style = NButtonStyle.Secondary,
                block = true,
            )
        } else {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                AsyncImage(
                    model = uri,
                    contentDescription = "Source image",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier
                        .size(56.dp)
                        .clip(Radius.Sm)
                        .background(NocturneColors.Neutral900)
                        .ring(NocturneColors.Divider, Radius.Sm),
                )
                Column(Modifier.weight(1f)) {
                    Text("Picked", style = NocturneType.Row)
                    Text(
                        uri.substringAfterLast('/'),
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMuted,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Text(
                    "Replace",
                    style = NocturneType.Meta,
                    color = NocturneColors.Accent,
                    modifier = Modifier.nClickableFlat(onClick = onPick).padding(6.dp),
                )
                Text(
                    "Remove",
                    style = NocturneType.Meta,
                    color = NocturneColors.TextMuted,
                    modifier = Modifier.nClickableFlat(onClick = onClear).padding(6.dp),
                )
            }
        }
    }
}

/**
 * LoRAs, ControlNets, IP-Adapters, VAEs — whatever is installed.
 *
 * There is no per-family logic here and there is not supposed to be. Each
 * installed auxiliary declares a *role*, the role knows which manifest key
 * carries its path, and the runtime decides whether the file is usable against
 * the loaded base model. That is what makes this work for SD 1.5, SDXL, Flux
 * and something released next month without an app update — the same bargain
 * §1.5 strikes for parameters, one level up.
 */
@Composable
private fun AttachmentsSection(
    state: ImageState,
    viewModel: ImageViewModel,
) {
    if (state.availableAttachments.isEmpty()) {
        NHelp(
            "No LoRAs or ControlNets installed. Add one on the Models screen and it appears here — " +
                "the app pairs it with a role rather than a model family, so anything the runtime " +
                "can load will work.",
            Modifier.padding(top = 14.dp),
        )
        return
    }

    SectionKicker(
        "Attachments · ${state.attachments.size} of ${state.availableAttachments.size} on",
        Modifier.padding(top = 20.dp, bottom = 8.dp),
    )

    state.availableAttachments
        .groupBy { it.role }
        .forEach { (role, items) ->
            NHelp(role.label, Modifier.padding(top = 4.dp, bottom = 4.dp))
            items.forEach { attachment ->
                Column(
                    Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp)
                        .background(
                            if (attachment.enabled) NocturneColors.Accent900 else NocturneColors.Surface,
                            Radius.Md,
                        )
                        .ring(
                            if (attachment.enabled) NocturneColors.Accent else NocturneColors.Divider,
                            Radius.Md,
                        )
                        .nClickableFlat { viewModel.toggleAttachment(attachment.modelId) }
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                ) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(9.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            attachment.displayName,
                            style = NocturneType.Row,
                            color = if (attachment.enabled) NocturneColors.Accent200 else NocturneColors.Text,
                            modifier = Modifier.weight(1f),
                        )
                        Text(
                            if (attachment.enabled) "on" else "off",
                            style = NocturneType.Mono2Xs,
                            color = if (attachment.enabled) NocturneColors.Accent else NocturneColors.TextMuted,
                        )
                    }
                    // Only the roles the runtime actually weights get a dial.
                    if (attachment.enabled && role.weighted) {
                        LabeledSlider(
                            label = "Weight",
                            value = attachment.weight,
                            display = String.format("%.2f", attachment.weight),
                            range = 0f..2f,
                            onChange = { viewModel.setAttachmentWeight(attachment.modelId, it) },
                            modifier = Modifier.padding(top = 6.dp),
                        )
                    }
                }
            }
        }
}

/**
 * Outpainting margins.
 *
 * The mask is derived, not painted: the original is composited into a larger
 * canvas and the new border is what gets filled. So the control the user needs
 * is "how much further, on which edges" — and the output size is shown because
 * an extend that quietly doubles the pixel count is the sort of thing that
 * turns a 30-second run into a five-minute one.
 */
@Composable
private fun ExtendField(
    state: ImageState,
    viewModel: ImageViewModel,
) {
    NField("Extend by", Modifier.padding(top = 12.dp)) {
        val presets = listOf(0, 64, 128, 256)
        NSeg(
            options = presets.map { if (it == 0) "none" else "$it px" },
            selectedIndex = presets.indexOfFirst { it == state.extendLeft }.coerceAtLeast(0),
            onSelect = { viewModel.setExtendAll(presets[it]) },
            modifier = Modifier.fillMaxWidth(),
        )
        Row(
            Modifier.fillMaxWidth().padding(top = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            EdgeStepper("L", state.extendLeft) { viewModel.setExtend(left = it) }
            EdgeStepper("T", state.extendTop) { viewModel.setExtend(top = it) }
            EdgeStepper("R", state.extendRight) { viewModel.setExtend(right = it) }
            EdgeStepper("B", state.extendBottom) { viewModel.setExtend(bottom = it) }
        }
        NHelp(
            if (state.extendLeft + state.extendTop + state.extendRight + state.extendBottom == 0) {
                "Set a margin on at least one edge — there is nothing to fill otherwise."
            } else {
                "Output ${state.outputWidth} × ${state.outputHeight}. The original is kept and only " +
                    "the new border is generated."
            },
            Modifier.padding(top = 6.dp),
        )
    }
}

@Composable
private fun androidx.compose.foundation.layout.RowScope.EdgeStepper(
    label: String,
    value: Int,
    onChange: (Int) -> Unit,
) {
    Row(
        Modifier
            .weight(1f)
            .ring(NocturneColors.Divider, Radius.Md)
            .padding(horizontal = 8.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = NocturneType.Mono2Xs, color = NocturneColors.TextMuted)
        Text(
            "−",
            style = NocturneType.Row,
            color = NocturneColors.Accent,
            modifier = Modifier.nClickableFlat { onChange((value - 64).coerceAtLeast(0)) },
        )
        Text(
            "$value",
            style = NocturneType.MonoXs,
            modifier = Modifier.weight(1f),
            textAlign = androidx.compose.ui.text.style.TextAlign.Center,
        )
        Text(
            "+",
            style = NocturneType.Row,
            color = NocturneColors.Accent,
            modifier = Modifier.nClickableFlat { onChange((value + 64).coerceAtMost(512)) },
        )
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
        // The real thing, in priority order: the decoded latent while sampling,
        // then the finished image, then the source it will act on. Every one of
        // these is actual pixels — SPEC §5.4's objection is to a spinner
        // standing in for state the engine already has.
        val preview = state.previewBitmap
        val showingSource = preview == null && state.sourceImageUri != null
        if (preview != null) {
            androidx.compose.foundation.Image(
                bitmap = preview.asImageBitmap(),
                contentDescription = if (state.generating) "Live preview" else "Generated image",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize(),
            )
        } else if (showingSource) {
            AsyncImage(
                model = state.sourceImageUri,
                contentDescription = "Source image",
                contentScale = ContentScale.Crop,
                modifier = Modifier.fillMaxSize(),
            )
        }

        // The latent field only blooms while sampling. Idle, it stays on the
        // neutral ramp — the readme forbids flooding a large area with a
        // saturated fill, and an idle preview has nothing to say anyway.
        val bloom = when {
            preview != null || showingSource -> 0f
            state.generating -> 1f
            else -> 0.12f
        }

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

        if (!showingSource && preview == null) {
            Text(
                if (state.generating) "warming up…" else "No preview yet",
                style = NocturneType.MonoSm,
                color = if (state.generating) {
                    NocturneColors.Accent200.copy(alpha = 0.9f)
                } else {
                    NocturneColors.TextMuted
                },
            )
        }

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
