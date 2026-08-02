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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
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
import ai.ondevice.ui.labelFor
import ai.ondevice.ui.pickerLabels
import ai.ondevice.ui.components.NBottomBar
import ai.ondevice.ui.components.NBottomSheet
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NDropdown
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
import ai.ondevice.ui.components.ResourceBlock
import ai.ondevice.ui.components.RootToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.ToolbarAction
import ai.ondevice.ui.components.ToolbarToggle
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.bottomScrim
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.ImageAction
import ai.ondevice.ui.vm.ImageUse
import ai.ondevice.ui.vm.ImageState
import ai.ondevice.ui.vm.ImageViewModel

@Composable
fun ImageScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onOpenMask: () -> Unit,
    onOpenRuntimes: () -> Unit,
    onAddModel: () -> Unit,
    onOpenAdvanced: () -> Unit,
    viewModel: ImageViewModel = activityImageViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val pickSource = rememberSourceImagePicker(viewModel::setSourceImage)
    val pickControl = rememberSourceImagePicker(viewModel::setControlImage)
    val pickStyle = rememberSourceImagePicker(viewModel::setStyleImage)

    // The Advanced screen writes to the same diffusion model row, so pick up
    // anything it changed on the way back rather than showing a stale form.
    LaunchedEffect(Unit) { viewModel.refreshFromOverrides() }

    var settingsOpen by rememberSaveable { mutableStateOf(false) }

    PhoneScaffold(
        toolbar = {
            // The same pair Chat and Voice carry.
            RootToolbar("Image") {
                ToolbarAction(NIcons.Plus, "New image", viewModel::reset)
                ToolbarAction(NIcons.Settings, "Image settings", { settingsOpen = true })
            }
        },
        bottomBar = { NBottomBar(BottomDestinations, currentRoute) { onNavigate(it.route) } },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            LivePreview(state)

            // Live while sampling, then the finished run's.
            (state.liveTrace ?: state.lastTrace)?.let { trace ->
                var traceExpanded by rememberSaveable { mutableStateOf(false) }
                ResourceBlock(
                    trace = trace,
                    expanded = traceExpanded,
                    onToggle = { traceExpanded = !traceExpanded },
                    live = state.generating,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }

            // Loading four gigabytes is minutes of one opaque call. Name what
            // is going in, let the loader say where it has got to — and then
            // keep saying what is in there.
            //
            // This used to appear only while `loadingModel` was true, which
            // meant it was on screen for exactly as long as the load and gone
            // the instant it mattered most: the list of what a run is actually
            // holding is worth reading *after* the load, when there is time to
            // read it and a picture to explain. A load that hits a warm context
            // never showed it at all.
            val loadingNow = state.loadingModel && state.loadingWhat.isNotEmpty()
            val resident = state.residentComponents
            if (loadingNow || resident.isNotEmpty()) {
                NCard(
                    Modifier.padding(top = 10.dp),
                    ring = if (loadingNow) NocturneColors.Accent800 else NocturneColors.Neutral700,
                ) {
                    Text(
                        if (loadingNow) "Loading into memory" else "In memory",
                        style = NocturneType.CardTitleSm,
                    )
                    (if (loadingNow) state.loadingWhat else resident).forEach {
                        Text(it, style = NocturneType.MonoXs, color = NocturneColors.Accent300)
                    }
                    // While loading this is the loader's progress; during a run
                    // it is the sampler's or the decoder's. Either way it is the
                    // runtime's own sentence and not a guess at one.
                    (state.loadingStage ?: state.runStage)?.let {
                        Text(
                            it,
                            style = NocturneType.Help,
                            color = NocturneColors.TextMuted,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
            }

            // A LoRA can fail without failing: it loads, costs its time and
            // matches nothing, and the picture looks like one where it worked.
            state.loraOutcome.forEach { note ->
                NCard(Modifier.padding(top = 10.dp), ring = NocturneColors.Accent800) {
                    Text("LoRA had no effect", style = NocturneType.CardTitleSm,
                        color = NocturneColors.Accent200)
                    Text(
                        note,
                        style = NocturneType.CardBody,
                        color = NocturneColors.Text.copy(alpha = 0.8f),
                    )
                }
            }

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
                        ImageAction.INSTALLING -> Unit
                        ImageAction.ADD_MODEL -> onAddModel()
                        ImageAction.PICK_SOURCE -> pickSource()
                    }
                },
                enabled = state.actionEnabled,
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
            // it contributes structure, never pixels. It appears with a
            // ControlNet armed and not otherwise, because nothing else reads it.
            if (state.usesControlImage) {
                SourceImageField(
                    label = "Control image · ControlNet",
                    uri = state.controlImageUri,
                    emptyLabel = "Add a pose, depth or edge map to steer composition",
                    onPick = pickControl,
                    onClear = { viewModel.setControlImage(null) },
                )
            }

            // A third picture, and a third thing: the IP-Adapter reads style
            // from it through CLIP-Vision — colour, texture, treatment — where
            // a ControlNet reads structure and img2img reads pixels. It appears
            // only with an IP-Adapter armed, because it goes nowhere else.
            if (state.usesStyleReference) {
                SourceImageField(
                    label = "Style reference · IP-Adapter",
                    uri = state.styleImageUri,
                    emptyLabel = "Add a picture to take the look from",
                    onPick = pickStyle,
                    onClear = { viewModel.setStyleImage(null) },
                )
            }

            // What the attached picture is for. Four screens once, and they
            // were one screen: a prompt in, a picture out, and a picture
            // optionally in. With nothing attached there is nothing to decide.
            if (state.sourceImageUri != null) {
                SectionKicker("This picture is to", Modifier.padding(top = 14.dp, bottom = 8.dp))
                val uses = ImageUse.entries
                NSeg(
                    options = uses.map { it.label },
                    selectedIndex = uses.indexOf(state.use).coerceAtLeast(0),
                    onSelect = { viewModel.setUse(uses[it]) },
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            if (state.use == ImageUse.EXTEND && state.sourceImageUri != null) {
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
                if (state.use == ImageUse.REPAINT) {
                    NButton(
                        "Edit mask",
                        onClick = onOpenMask,
                        style = NButtonStyle.Secondary,
                        block = true,
                        modifier = Modifier.padding(top = 8.dp),
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
                        "Turn on vae_tiling or drop to 768.",
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

            // Only offered when there is something to upscale and something to upscale it with.
            state.lastImage?.let { last ->
                val hasUpscaler = state.availableAttachments.any {
                    it.role == ai.ondevice.core.AttachmentRole.UPSCALER
                }
                if (hasUpscaler) {
                    NButton(
                        if (state.generating) "Working…" else "Upscale ${last.width}×${last.height} with ESRGAN",
                        onClick = { viewModel.upscale() },
                        style = NButtonStyle.Secondary,
                        block = true,
                        modifier = Modifier.padding(top = 14.dp),
                    )
                    NHelp(
                        "Writes a new gallery entry rather than replacing this one — the seed and " +
                            "parameters recorded here reproduce the original size.",
                        Modifier.padding(top = 6.dp),
                    )
                }
            }

            state.usedSeed?.let { seed ->
                NHelp(
                    "Last run used seed $seed. Every generation writes its full parameter set into the " +
                        "PNG, so any image in the gallery can be reproduced exactly.",
                    Modifier.padding(top = 12.dp),
                )
            }
        }
    }

    // Note what is *not* here: closing the sheet on the way to Advanced.
    if (settingsOpen) {
        ImageSettingsSheet(
            state = state,
            viewModel = viewModel,
            onDismiss = { settingsOpen = false },
            onOpenAdvanced = onOpenAdvanced,
        )
    }
}

/** Everything that decides *how* the next image is made. */
@Composable
private fun ImageSettingsSheet(
    state: ImageState,
    viewModel: ImageViewModel,
    onDismiss: () -> Unit,
    onOpenAdvanced: () -> Unit,
) {
    NBottomSheet("Image settings", onDismiss, note = "applies to the next run") {
        // Which model runs is the user's choice.
        if (state.availableModels.isNotEmpty()) {
            SectionKicker("Model", Modifier.padding(bottom = 8.dp))
            // Labels have to be unique here, not merely readable: the dropdown
            // hands back the label, so two models sharing a display name meant
            // choosing the second selected the first.
            val modelLabels = state.availableModels.pickerLabels()
            NDropdown(
                options = modelLabels,
                selected = state.availableModels.labelFor(state.model),
                onSelect = { label ->
                    modelLabels.indexOf(label)
                        .takeIf { it >= 0 }
                        ?.let { viewModel.selectModel(state.availableModels[it]) }
                },
                placeholder = "Choose a model…",
                modifier = Modifier.padding(bottom = 14.dp),
            )
        }

        SectionKicker("Output", Modifier.padding(bottom = 8.dp))
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Column(Modifier.weight(1f)) {
                NHelp("Size", Modifier.padding(bottom = 4.dp))
                // Every multiple-of-64 square worth having, in a dropdown
                // rather than a segmented control: eight choices do not fit
                // across a phone, and the ends of the range are the ones worth
                // reaching — 64 to see what a prompt does in seconds, 4096 when
                // there is time to spend.
                val sizes = listOf(64, 128, 256, 512, 768, 1024, 2048, 4096)
                NDropdown(
                    options = sizes.map { "$it × $it" },
                    selected = "${state.width} × ${state.width}",
                    onSelect = { label ->
                        label.substringBefore(' ').trim().toIntOrNull()?.let(viewModel::setSize)
                    },
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

        Row(
            Modifier.fillMaxWidth().padding(top = 12.dp),
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

        AttachmentsSection(state, viewModel)

        NButton(
            "All Parameters",
            onClick = onOpenAdvanced,
            style = NButtonStyle.Secondary,
            block = true,
            modifier = Modifier.padding(top = 14.dp),
        )
    }
}

/** The system photo picker. */
@Composable
private fun rememberSourceImagePicker(onPicked: (String) -> Unit): () -> Unit {
    val context = LocalContext.current
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
    ) { uri ->
        uri ?: return@rememberLauncherForActivityResult
        // The picker's grant dies with the process.
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

/** The img2img / inpaint source. */
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
 * What this run will pass alongside the model, and nothing else.
 *
 * Only components that have a file *chosen* appear here, each as one switch.
 * Choosing which file fills a role is a per-model decision and belongs on the
 * All Parameters screen; this is the per-run one.
 */
@Composable
private fun AttachmentsSection(
    state: ImageState,
    viewModel: ImageViewModel,
) {
    if (state.availableAttachments.isEmpty()) {
        SectionKicker("Components", Modifier.padding(top = 20.dp, bottom = 8.dp))

        // A required component with nothing chosen is the difference between
        // this screen being empty because there is nothing to add and being
        // empty because the run is about to fail.
        state.missingComponents.forEach { missing ->
            NCard(Modifier.padding(bottom = 8.dp)) {
                // A substitution is a choice, not a fault, and reads as one.
                Text(
                    missing.what,
                    style = NocturneType.CardTitleSm,
                    color = if (missing.state ==
                        ai.ondevice.core.MissingComponent.State.SUBSTITUTES
                    ) {
                        NocturneColors.TextMuted
                    } else {
                        NocturneColors.Accent200
                    },
                )
                Text(
                    "${missing.because}. ${missing.remedy}.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            }
        }

        NCard {
            // Two different situations wore the same sentence: a library with
            // no add-ons in it, and a library holding several of one role with
            // nobody having said which. Only the second is a decision waiting.
            if (state.unchosenRoles.isEmpty()) {
                Text("Nothing chosen for this model", style = NocturneType.CardTitleSm)
                Text(
                    "A prompt encoder, a VAE, a LoRA, a ControlNet — whichever of them this " +
                        "architecture can take. Downloading one is enough for the parts a run " +
                        "cannot do without; the rest are chosen under All Parameters.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
                Text(
                    "Add model lists a few that are known to work — look under Image add-ons.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            } else {
                Text("Installed, not chosen", style = NocturneType.CardTitleSm)
                Text(
                    state.unchosenRoles.joinToString(", ") { it.label },
                    style = NocturneType.CardBody,
                    color = NocturneColors.Accent200,
                )
                Text(
                    "Each of these fits this model and more than one file could fill it, so the " +
                        "choice is yours: pick one per role under All Parameters and it appears " +
                        "here as a switch.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            }
        }
        return
    }

    SectionKicker(
        "Components · ${state.attachments.size} of ${state.availableAttachments.size} on",
        Modifier.padding(top = 20.dp, bottom = 8.dp),
    )

    // Said here, next to the switches that fix it, rather than after Generate
    // has spent a minute finding out.
    state.missingComponents.forEach { missing ->
        NCard(Modifier.padding(bottom = 8.dp)) {
            // A substitution is a choice, not a fault, and reads as one.
            Text(
                missing.what,
                style = NocturneType.CardTitleSm,
                color = if (missing.state ==
                    ai.ondevice.core.MissingComponent.State.SUBSTITUTES
                ) {
                    NocturneColors.TextMuted
                } else {
                    NocturneColors.Accent200
                },
            )
            Text(
                "${missing.because}. ${missing.remedy}.",
                style = NocturneType.CardBody,
                color = NocturneColors.Text.copy(alpha = 0.8f),
            )
        }
    }

    // Grouped by what the thing is *for*, in a fixed order, so the four ways of
    // encoding a prompt sit under one heading instead of reading as four
    // unrelated files scattered through the list.
    state.availableAttachments
        .groupBy { it.role.family }
        .toList()
        .sortedBy { (family, _) -> family.ordinal }
        .forEach { (family, inFamily) ->
            NHelp(family.label, Modifier.padding(top = 10.dp, bottom = 4.dp))
            inFamily.forEach { attachment ->
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
                        .then(
                            if (attachment.applicable) {
                                Modifier.nClickableFlat { viewModel.toggleAttachment(attachment.modelId) }
                            } else {
                                Modifier
                            },
                        )
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                ) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(9.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        // The role leads, because the role is the slot; the
                        // file is which one is in it.
                        Text(
                            attachment.role.label,
                            style = NocturneType.Row,
                            color = if (attachment.enabled) NocturneColors.Accent200 else NocturneColors.Text,
                            modifier = Modifier.weight(1f),
                        )
                        Text(
                            when {
                                !attachment.applicable -> "n/a"
                                attachment.enabled -> "on"
                                else -> "off"
                            },
                            style = NocturneType.Mono2Xs,
                            color = if (attachment.enabled) NocturneColors.Accent else NocturneColors.TextMuted,
                        )
                    }
                    // Chosen, but this model has no use for it — so it is not
                    // armed, not passed to the loader, and says which.
                    if (!attachment.applicable) {
                        Text(
                            "Not used by ${state.recognisedAs ?: state.model?.architecture ?: "this model"}",
                            style = NocturneType.Help,
                            color = NocturneColors.TextMuted,
                            modifier = Modifier.padding(top = 2.dp),
                        )
                    }
                    // Which file is in the slot — by the name it is known by,
                    // which is the one given by hand where there is one. This
                    // read the path directly and so never saw a rename, in the
                    // one place a renamed component is most likely to be read.
                    Text(
                        attachment.displayName,
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMuted,
                        modifier = Modifier.padding(top = 2.dp),
                    )
                    // Only the roles the runtime actually weights get a dial,
                    // and weight is a per-run thought, so it lives here rather
                    // than beside the choice of file.
                    if (attachment.enabled && attachment.role.weighted) {
                        LabeledSlider(
                            label = "Weight",
                            value = attachment.weight,
                            display = String.format("%.2f", attachment.weight),
                            range = 0f..2f,
                            onChange = { viewModel.setAttachmentWeight(attachment.modelId, it) },
                            modifier = Modifier.padding(top = 6.dp),
                        )
                    }
                    // A ControlNet and an IP-Adapter take one strength each,
                    // and it is a number about the run rather than about the
                    // file — so it is not the per-attachment weight above, and
                    // it belongs beside its component all the same. It lived
                    // under the picture pickers on the main screen, which is
                    // where the picture is chosen and not where the component
                    // is; two dials in two places for one idea.
                    if (attachment.enabled) {
                        when (attachment.role) {
                            ai.ondevice.core.AttachmentRole.CONTROLNET -> LabeledSlider(
                                label = "Strength",
                                value = state.controlStrength,
                                display = String.format("%.2f", state.controlStrength),
                                range = 0f..1f,
                                onChange = viewModel::setControlStrength,
                                modifier = Modifier.padding(top = 6.dp),
                            )
                            ai.ondevice.core.AttachmentRole.IP_ADAPTER -> LabeledSlider(
                                label = "Strength",
                                value = state.styleStrength,
                                display = String.format("%.2f", state.styleStrength),
                                range = 0f..1f,
                                onChange = viewModel::setStyleStrength,
                                modifier = Modifier.padding(top = 6.dp),
                            )
                            else -> Unit
                        }
                    }
                }
            }
        }
}

/** Outpainting margins. */
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

/** The live preview. */
@Composable
private fun LivePreview(state: ImageState) {
    Box(
        Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .clip(Radius.Md)
            .background(NocturneColors.Neutral900)
            .ring(NocturneColors.Divider, Radius.Md),
        contentAlignment = Alignment.Center,
    ) {
        // This frame is the output and only ever the output: the decoded
        // latent while sampling, then the finished picture. It used to fall
        // back to the source image, which has its own field below with its own
        // thumbnail — the same picture in the result frame reads as a result.
        val preview = state.previewBitmap
        if (preview != null) {
            androidx.compose.foundation.Image(
                bitmap = preview.asImageBitmap(),
                contentDescription = if (state.generating) "Live preview" else "Generated image",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize(),
            )
        }

        // The latent field only blooms while sampling.
        val bloom = when {
            preview != null -> 0f
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

        // "Warming up" is only true before the first step.
        if (preview == null) {
            Text(
                // Driven by the phase, because the phase is the thing that knows.
                when {
                    state.loadingModel -> "loading model…"
                    !state.generating -> "No preview yet"
                    state.phase == ai.ondevice.engine.DiffusionPhase.PREPARING ->
                        "preparing · loading weights, no steps to count yet"
                    state.phase == ai.ondevice.engine.DiffusionPhase.DECODING ->
                        "decoding the latent to pixels · almost done"
                    state.step <= 0 -> "warming up…"
                    else -> "sampling · no preview decoder installed"
                },
                style = NocturneType.MonoSm,
                color = if (state.generating) {
                    NocturneColors.Accent200.copy(alpha = 0.9f)
                } else {
                    NocturneColors.TextMuted
                },
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                modifier = Modifier.padding(horizontal = 24.dp),
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
                        // Only the sampling phase has steps worth counting.
                        if (state.phase == ai.ondevice.engine.DiffusionPhase.SAMPLING) {
                            "step ${state.step}/${state.progressSteps}"
                        } else {
                            state.phase.label
                        },
                        style = NocturneType.MonoSm,
                        color = NocturneColors.Accent200,
                    )
                    if (state.secondsPerStep > 0f) {
                        Text("·", style = NocturneType.MonoSm, color = Color.White.copy(alpha = 0.7f))
                        // The rate, not the device: there is one device.
                        Text(
                            "${String.format("%.1f", state.secondsPerStep)} s/it",
                            style = NocturneType.MonoSm,
                            color = Color.White.copy(alpha = 0.7f),
                        )
                    }
                    Box(Modifier.weight(1f))
                    // An ETA computed from a rate we do not have yet is a
                    // guess dressed as a number. Better to show nothing.
                    if (state.secondsPerStep > 0f && state.etaSeconds > 0) {
                        Text(
                            Fmt.eta(state.etaSeconds),
                            style = NocturneType.MonoSm,
                            color = Color.White.copy(alpha = 0.7f),
                        )
                    }
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
