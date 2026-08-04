package ai.ondevice.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil3.compose.AsyncImage
import ai.ondevice.core.Fmt
import ai.ondevice.ui.BottomDestinations
import ai.ondevice.ui.components.GenerationProgress
import ai.ondevice.ui.components.PickedImageField
import ai.ondevice.ui.components.ResidentCard
import ai.ondevice.ui.components.ResourceBlock
import ai.ondevice.ui.components.NBottomBar
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NDropdown
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NProgressBar
import ai.ondevice.ui.components.NSlider
import ai.ondevice.ui.components.NSwitch
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.NTextArea
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.RootToolbar
import ai.ondevice.ui.components.ToolbarToggle
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.ToolbarAction
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.VideoState
import ai.ondevice.ui.vm.VideoViewModel
import androidx.compose.ui.draw.clip
import androidx.hilt.navigation.compose.hiltViewModel

/**
 * S14 — video, on the same runtime and the same models as stills.
 *
 * Laid out like the Image screen because it is the same act with one more
 * dimension: a prompt, optionally a picture to start from, and a result. What
 * it does not carry is the three slots `sd_vid_gen_params_t` has no field for —
 * no IP-Adapter, no mask, no reference image for an edit model — because a
 * control that reaches nothing is worse than an absent one.
 */
@Composable
fun VideoScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onBack: () -> Unit,
    onAddModel: () -> Unit,
    onOpenAdvanced: (String?) -> Unit,
    viewModel: VideoViewModel = activityVideoViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    var settingsOpen by rememberSaveable { mutableStateOf(false) }

    val pickFirst = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
    ) { uri -> uri?.let { viewModel.setFirstFrame(it.toString()) } }
    val pickLast = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
    ) { uri -> uri?.let { viewModel.setLastFrame(it.toString()) } }
    val pickControl = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
    ) { uri -> uri?.let { viewModel.setControlImage(it.toString()) } }

    PhoneScaffold(
        toolbar = {
            // The same shape Voice uses for Transcribe and Speak: two modes of
            // one thing, switched in the toolbar, with the tab bar left alone.
            // Video was a push, so opening it hid the bottom bar — which said
            // "you have left the app's main screens" about a screen that runs
            // the same runtime on the same model as the one before it.
            RootToolbar("Video") {
                // The same pair Chat, Image and Voice carry. It was missing
                // here, so Video was the one screen with no way to start over.
                ToolbarAction(NIcons.Plus, "New clip", viewModel::reset)
                ToolbarToggle(NIcons.Image, "Stills", selected = false, onClick = onBack)
                ToolbarToggle(NIcons.Video, "Video", selected = true, onClick = {})
                ToolbarAction(NIcons.Settings, "Video settings", { settingsOpen = true })
            }
        },
        bottomBar = { NBottomBar(BottomDestinations, currentRoute) { onNavigate(it.route) } },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            ClipStage(state, viewModel)

            // — go —
            //
            // Under the preview, where the still screen keeps it. It used to
            // sit at the very bottom, past the prompt, both frame pickers, the
            // control frame and the length sliders — so starting a clip meant
            // scrolling past every setting, and Cancel was somewhere off-screen
            // for the whole run.
            val busy = state.generating || state.loadingModel
            NButton(
                when {
                    state.cancelling -> "Stopping…"
                    busy -> "Cancel"
                    else -> "Generate clip"
                },
                onClick = { if (busy) viewModel.cancel() else viewModel.generate() },
                style = if (busy) NButtonStyle.Secondary else NButtonStyle.Primary,
                enabled = state.model != null && (busy || state.runtimeInstalled),
                block = true,
                modifier = Modifier.padding(top = 12.dp),
            )

            if (!state.runtimeInstalled) {
                NCard(Modifier.padding(top = 10.dp), ring = NocturneColors.Accent800) {
                    Text("The diffusion runtime is not installed", style = NocturneType.CardTitleSm)
                    NHelp("This build carries no diffusion runtime, so there is nothing to load it with.")
                }
            }

            // What is in memory, and what the loader is doing while it fills.
            val loadingNow = state.loadingModel && state.loadingWhat.isNotEmpty()
            if (loadingNow || state.residentComponents.isNotEmpty()) {
                ResidentCard(
                    loadingNow = loadingNow,
                    loadingWhat = state.loadingWhat,
                    buffers = state.runtimeBuffers,
                    loaded = state.residentComponents.isNotEmpty(),
                    stage = state.loadingStage ?: state.runStage,
                    modifier = Modifier.padding(top = 10.dp),
                )
            }

            state.loraOutcome.forEach { note ->
                NCard(Modifier.padding(top = 10.dp), ring = NocturneColors.Accent800) {
                    Text("LoRA had no effect", style = NocturneType.CardTitleSm,
                        color = NocturneColors.Accent200)
                    Text(note, style = NocturneType.CardBody)
                }
            }

            state.error?.let { message ->
                NCard(Modifier.padding(top = 10.dp), ring = NocturneColors.Accent800) {
                    Text(message, style = NocturneType.CardTitleSm, color = NocturneColors.Accent200)
                    state.errorHint?.let { NHelp(it, Modifier.padding(top = 4.dp)) }
                }
            }

            // — the prompt —

            SectionKicker("Prompt", Modifier.padding(top = 18.dp, bottom = 6.dp))
            NTextArea(
                value = state.prompt,
                onValueChange = viewModel::setPrompt,
                minHeight = 84.dp,
                placeholder = "What should happen, and how the camera should move.",
            )
            NHelp(
                "Motion is described here as much as subject — \"slowly pans left\", " +
                    "\"leaves drifting down\" — because nothing else on this screen says it.",
                Modifier.padding(top = 5.dp),
            )

            // — the two ends —

            SectionKicker("Frames you supply", Modifier.padding(top = 18.dp, bottom = 6.dp))
            NHelp(
                "Both optional. With a first frame the clip starts from your picture; with " +
                    "both, the model travels from one to the other.",
                Modifier.padding(bottom = 8.dp),
            )
            PickedImageField(
                label = "First frame",
                uri = state.firstFrameUri,
                emptyLabel = "Start the clip from a picture",
                onPick = {
                    pickFirst.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                    )
                },
                onClear = { viewModel.setFirstFrame(null) },
            )
            PickedImageField(
                label = "Last frame",
                uri = state.lastFrameUri,
                emptyLabel = "End the clip on a picture",
                onPick = {
                    pickLast.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                    )
                },
                onClear = { viewModel.setLastFrame(null) },
            )

            // The pose or depth map the motion follows.
            //
            // The engine has taken one since video generation was added and
            // nothing on this screen offered it, so `vace_strength` — the dial
            // that weights it — sat behind a ControlNet row that a clip never
            // reads. VACE is what a clip uses instead, and it reads this.
            SectionKicker("Control frame", Modifier.padding(top = 18.dp, bottom = 6.dp))
            PickedImageField(
                label = "Control",
                uri = state.controlImageUri,
                emptyLabel = "Add a pose, depth or edge map to steer the motion",
                onPick = {
                    pickControl.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                    )
                },
                onClear = { viewModel.setControlImage(null) },
            )
            if (state.controlImageUri != null) {
                LabelledSlider(
                    label = "VACE strength",
                    value = String.format("%.2f", state.controlStrength),
                    position = state.controlStrength,
                    range = 0f..2f,
                    step = 0.05f,
                    onChange = viewModel::setControlStrength,
                )
            }
            NHelp(
                "Applied to every frame — the app has one map to give, not one per frame. " +
                    "Only a VACE-capable checkpoint reads it, and LTX-AV does not read it at all.",
                Modifier.padding(top = 4.dp),
            )

            // Length lives in the settings sheet now, with the size and the
            // sampler — the three sliders are set once for a run and then read,
            // and they were standing between the prompt and the button.
            //
            // What stays here is the consequence, because it is the number that
            // decides whether the run is possible at all and it is not derivable
            // from any one slider.
            NHelp(
                "${String.format("%.1f", state.requestedSeconds)} s at ${state.fps} fps · about " +
                    "${state.estimatedFrameMegabytes} MB of frames held while it finishes",
                Modifier.padding(top = 12.dp),
                color = if (state.estimatedFrameMegabytes > 1500) {
                    NocturneColors.Accent300
                } else {
                    NocturneColors.TextMuted
                },
            )

        }
    }

    if (settingsOpen) {
        VideoSettingsSheet(
            state = state,
            viewModel = viewModel,
            onOpenAdvanced = { settingsOpen = false; onOpenAdvanced(state.model?.id) },
            onAddModel = { settingsOpen = false; onAddModel() },
            onClose = { settingsOpen = false },
        )
    }
}

/**
 * The clip, one frame at a time.
 *
 * Frames live on disk and are loaded as they are shown — a 129-frame clip at
 * 512² is about 400 MB decoded, and exactly one of them is ever on screen.
 */
@Composable
private fun ClipStage(state: VideoState, viewModel: VideoViewModel) {
    val clip = state.clip
    Box(
        Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            // The image tab's viewport, to the pixel. These were `Radius.Lg`
            // over `Surface` against the still screen's `Radius.Md` over
            // `Neutral900` — a lighter box with rounder corners, side by side
            // with the one it is a sibling of, for no reason either screen
            // could give.
            .clip(Radius.Md)
            .background(NocturneColors.Neutral900)
            .ring(NocturneColors.Divider, Radius.Md),
        contentAlignment = Alignment.Center,
    ) {
        when {
            clip != null && state.currentFramePath != null ->
                AsyncImage(
                    model = state.currentFramePath,
                    contentDescription = "Frame ${state.frameIndex + 1}",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit,
                )

            state.previewBitmap != null ->
                androidx.compose.foundation.Image(
                    bitmap = state.previewBitmap.asImageBitmap(),
                    contentDescription = "Preview",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit,
                )

            else -> Text(
                if (state.generating) state.phase.label else "No clip yet",
                style = NocturneType.Help,
                color = NocturneColors.TextMuted,
            )
        }

        // Over the frame, not under it — the same block the still screen
        // draws, so one run does not describe itself in two shapes depending
        // on which tab is open.
        if (state.generating) {
            GenerationProgress(
                phase = state.phase,
                step = state.step,
                steps = state.progressSteps,
                secondsPerStep = state.secondsPerStep,
                etaSeconds = state.etaSeconds,
            )
        }
    }

    // Live while it renders, then the finished run's — the same block the
    // image tab shows, on the tab that works the phone hardest.
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

    if (clip != null) {
        Row(
            Modifier.fillMaxWidth().padding(top = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            NButton(
                if (state.playing) "Pause" else "Play",
                onClick = { if (state.playing) viewModel.pause() else viewModel.play() },
                style = NButtonStyle.Secondary,
            )
            Text(
                "${state.frameIndex + 1}/${clip.frames.size}",
                style = NocturneType.MonoXs,
                color = NocturneColors.TextMuted,
            )
            NSlider(
                value = state.frameIndex.toFloat(),
                onValueChange = { viewModel.seekTo(it.toInt()) },
                valueRange = 0f..(clip.frames.size - 1).coerceAtLeast(1).toFloat(),
                modifier = Modifier.weight(1f),
            )
        }
        Row(
            Modifier.padding(top = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            NTag("${clip.width}×${clip.height}", style = NTagStyle.Outline)
            NTag("${clip.fps} fps", style = NTagStyle.Outline)
            NTag("${String.format("%.1f", clip.durationSeconds)} s", style = NTagStyle.Outline)
            // Only LTX-AV returns one, so its presence is worth saying.
            if (clip.audioPath != null) NTag("with audio", style = NTagStyle.Neutral)
        }
        Text(
            "Discard clip",
            style = NocturneType.Help,
            color = NocturneColors.Accent,
            modifier = Modifier.padding(top = 6.dp).nClickableFlat { viewModel.discard() },
        )
    }
}

@Composable
private fun LabelledSlider(
    label: String,
    value: String,
    position: Float,
    range: ClosedFloatingPointRange<Float>,
    onChange: (Float) -> Unit,
    step: Float = 1f,
) {
    Column(Modifier.padding(top = 8.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, style = NocturneType.Row)
            Text(value, style = NocturneType.MonoValue, color = NocturneColors.Accent300)
        }
        ai.ondevice.ui.components.NNudgeSlider(
            value = position,
            onValueChange = onChange,
            valueRange = range,
            step = step,
            modifier = Modifier.padding(top = 2.dp),
        )
    }
}

/** Seed, and the components this clip will be built from. */
@Composable
private fun VideoSettingsSheet(
    state: VideoState,
    viewModel: VideoViewModel,
    onOpenAdvanced: () -> Unit,
    onAddModel: () -> Unit,
    onClose: () -> Unit,
) {
    ai.ondevice.ui.components.NBottomSheet(onDismiss = onClose, title = "Video settings") {
        // `weight(1f)` is load-bearing, not spacing. A Column measures its
        // children with unbounded height, and a scrolling child given infinite
        // height is a hard crash in Compose rather than a layout oddity — the
        // weight is what bounds it to the sheet.
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
            // Size, steps, CFG and tiling live here rather than on the main
            // screen, the way the Image screen keeps them: the screen itself is
            // the prompt, the two end frames and the result, and these are what
            // the run is made of rather than what it is.
            // The model, here rather than on the screen behind — the same place
            // the Image screen keeps it. What the screen itself is is a prompt,
            // two end frames and a result; which checkpoint makes them is a
            // setting about the run, and it was taking the top third of the
            // screen to say so.
            SectionKicker("Model", Modifier.padding(bottom = 6.dp))
            if (state.models.isEmpty()) {
                NCard {
                    Text("No video model installed", style = NocturneType.CardTitleSm)
                    NHelp(
                        "Wan, Hunyuan and LTX-AV generate video directly. An SD 1.5 checkpoint " +
                            "does too, once a motion module is attached to it. Checkpoints that " +
                            "only make stills are not listed here.",
                        Modifier.padding(top = 4.dp),
                    )
                    NButton(
                        "Add a model",
                        onClick = onAddModel,
                        style = NButtonStyle.Secondary,
                        block = true,
                        modifier = Modifier.padding(top = 8.dp),
                    )
                }
            } else {
                NDropdown(
                    options = state.models.map { it.label },
                    selected = state.model?.label,
                    onSelect = { label ->
                        state.models.firstOrNull { it.label == label }?.let(viewModel::selectModel)
                    },
                    placeholder = "Choose a model…",
                )
                // Whether it can make video at all is the loader's answer, and
                // nothing before the first load can know it — for SD 1.x it
                // depends on whether a motion module was attached.
                state.recognisedAs?.let {
                    Row(
                        Modifier.padding(top = 6.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        NTag(it, style = NTagStyle.Outline)
                        NTag(
                            if (state.supportsVideo) "makes video" else "stills only",
                            style = if (state.supportsVideo) NTagStyle.Neutral else NTagStyle.Outline,
                        )
                    }
                }
            }

            SectionKicker("Length", Modifier.padding(top = 18.dp, bottom = 6.dp))
            LabelledSlider(
                label = "Frames",
                value = "${state.frames}",
                position = state.frames.toFloat(),
                range = 1f..129f,
                onChange = { viewModel.setFrames(Math.round(it)) },
            )
            LabelledSlider(
                label = "Frames per second",
                value = "${state.fps}",
                position = state.fps.toFloat(),
                range = 1f..60f,
                onChange = { viewModel.setFps(Math.round(it)) },
            )
            NHelp(
                "${String.format("%.1f", state.requestedSeconds)} s · about " +
                    "${state.estimatedFrameMegabytes} MB of frames held while it finishes",
                Modifier.padding(top = 4.dp),
                color = if (state.estimatedFrameMegabytes > 1500) {
                    NocturneColors.Accent300
                } else {
                    NocturneColors.TextMuted
                },
            )

            SectionKicker("Sampling", Modifier.padding(top = 18.dp, bottom = 6.dp))
            LabelledSlider(
                label = "Size",
                value = "${state.width}²",
                position = state.width.toFloat(),
                range = 64f..768f,
                step = 64f,
                onChange = { viewModel.setSize(Math.round(it)) },
            )
            LabelledSlider(
                label = "Steps",
                value = "${state.steps}",
                position = state.steps.toFloat(),
                range = 1f..60f,
                onChange = { viewModel.setSteps(Math.round(it)) },
            )
            LabelledSlider(
                label = "CFG scale",
                value = String.format("%.1f", state.cfgScale),
                position = state.cfgScale,
                range = 1f..15f,
                step = 0.1f,
                onChange = viewModel::setCfg,
            )
            Row(
                Modifier.fillMaxWidth().padding(top = 10.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text("Tiled decode", style = NocturneType.Row)
                    NHelp("Decodes in tiles. On a clip this is usually what makes it fit at all.")
                }
                NSwitch(checked = state.vaeTiling, onCheckedChange = viewModel::setVaeTiling)
            }

            SectionKicker("Seed", Modifier.padding(top = 18.dp, bottom = 6.dp))
            NInput(
                value = if (state.seed < 0) "" else state.seed.toString(),
                onValueChange = { viewModel.setSeed(it.toLongOrNull() ?: -1L) },
                placeholder = "Random each run",
                keyboardType = androidx.compose.ui.text.input.KeyboardType.Number,
            )
            state.usedSeed?.let {
                NHelp("Last run used $it.", Modifier.padding(top = 4.dp))
            }

            AttachmentsPicker(
                available = state.availableAttachments,
                armedCount = state.attachments.size,
                missing = state.missingComponents,
                unchosenRoles = state.unchosenRoles,
                architectureLabel = state.recognisedAs ?: state.model?.architecture,
                onToggle = viewModel::toggleAttachment,
                onWeight = viewModel::setAttachmentWeight,
                // The control frame has its own section above, with VACE's dial
                // beside it. Nothing in this list carries a strength.
                strengthFor = { null },
                onStrength = { _, _ -> },
                emptyHelp = "A motion module is what turns an SD 1.5 checkpoint into one that " +
                    "animates. Wan, Hunyuan and LTX-AV need their encoder and decoder instead. " +
                    "IP-Adapters, the identity adapters and ControlNets are not listed at all: " +
                    "the runtime's video path has no field for them, so one attached here would " +
                    "cost its weights and never be read. An upscaler is listed because LTX-AV's " +
                    "hi-res stage is a separate latent upsampler, and that is where it goes.",
            )

            // Unloading is not something to do by accident mid-run.
            //
            // The button was live whenever anything was resident, including
            // while sampling — and freeing the weights under a running
            // generation ends it, badly, with the native side reading memory
            // that has gone. Now that a run outlives the screen it is easier
            // than ever to arrive here with one going, so a run in flight
            // turns this into a two-step: say it, then mean it.
            var confirmingUnload by rememberSaveable { mutableStateOf(false) }
            val busyNow = state.generating || state.loadingModel
            NButton(
                when {
                    !busyNow -> "Unload model"
                    confirmingUnload -> "Unload and stop the run"
                    else -> "Unload model…"
                },
                onClick = {
                    when {
                        !busyNow -> viewModel.unloadModel()
                        confirmingUnload -> { confirmingUnload = false; viewModel.unloadModel() }
                        else -> confirmingUnload = true
                    }
                },
                style = NButtonStyle.Ghost,
                block = true,
                enabled = state.residentComponents.isNotEmpty(),
                modifier = Modifier.padding(top = 18.dp),
            )
            NHelp(
                if (busyNow) {
                    "A run is in progress. Unloading frees its weights and stops it."
                } else {
                    "Frees the weights now. Generating again reloads them."
                },
                Modifier.padding(top = 4.dp),
            )

            // Where the Image screen keeps it too: the sheet is for what this
            // run is made of, and the full set is one step further in.
            NButton(
                "All Parameters",
                onClick = onOpenAdvanced,
                style = NButtonStyle.Secondary,
                block = true,
                modifier = Modifier.padding(top = 12.dp),
            )
            NHelp(
                "Every setting the runtime reports, including the hi-res stage and Wan 2.2's " +
                    "second denoiser.",
                Modifier.padding(top = 4.dp),
            )
        }
    }
}
