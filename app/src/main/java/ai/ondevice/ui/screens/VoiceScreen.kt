package ai.ondevice.ui.screens

import android.Manifest
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.core.TranscriptFormat
import ai.ondevice.ui.BottomDestinations
import ai.ondevice.ui.components.NBottomBar
import ai.ondevice.ui.components.NAudioPlayer
import ai.ondevice.ui.components.NBottomSheet
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCardMeta
import ai.ondevice.ui.components.NDropdown
import ai.ondevice.ui.components.NEnumRow
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NIconButton
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NMetaText
import ai.ondevice.ui.components.NPills
import ai.ondevice.ui.components.NProgressBar
import ai.ondevice.ui.components.NSlider
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
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
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.theme.ruleBelow
import ai.ondevice.ui.components.NSeg
import ai.ondevice.ui.vm.VoiceMode
import ai.ondevice.ui.vm.VoiceState
import ai.ondevice.ui.vm.VoiceViewModel

/**
 * **S14 — Voice.**
 *
 * Two tabs, and they are inverses of each other:
 *
 *  - **Transcribe** (whisper.cpp, SPEC §6) — audio in, text out. The audio
 *    comes from the microphone or a file; both go through the same decoder and
 *    produce the same timed, confidence-scored segments.
 *  - **Speak** (SPEC §7) — text in, audio out. The text is typed or loaded from
 *    a file; the audio is played aloud and can be saved as a WAV.
 *
 * The detail the canvas calls out is the confidence shading: a partial
 * transcript fades by per-token confidence, and the caption says plainly that
 * faded text may still change as the window slides. That is the honest-refusal
 * principle applied to a streaming decoder — whisper genuinely re-decodes the
 * window each pass, so earlier words really can change.
 */
@Composable
fun VoiceScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onOpenAdvanced: (String) -> Unit,
    viewModel: VoiceViewModel = activityVoiceViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val context = LocalContext.current

    // The Advanced screen edits the same model rows this screen surfaces, so
    // pick its changes up on the way back — otherwise "Chunk pattern" and
    // "Trim silence" are written and never read.
    androidx.compose.runtime.LaunchedEffect(Unit) { viewModel.refreshFromOverrides() }

    // Scripts are documents, so this is the document picker rather than the
    // photo picker — the same choice the chat composer makes.
    val scriptLauncher = rememberLauncherForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.OpenDocument(),
    ) { uri -> uri?.let(viewModel::loadScript) }
    val pickScript = {
        scriptLauncher.launch(arrayOf("text/*", "application/json", "application/pdf"))
    }

    val audioLauncher = rememberLauncherForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.OpenDocument(),
    ) { uri -> uri?.let(viewModel::chooseFile) }
    val pickAudio = { audioLauncher.launch(arrayOf("audio/*", "video/*")) }

    // A separate launcher from the one above: both take a recording, but one is
    // audio to read out and the other is a voice to copy, and routing them
    // through one callback would make which happened depend on the tab.
    val referenceLauncher = rememberLauncherForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.OpenDocument(),
    ) { uri -> uri?.let(viewModel::useReferenceClip) }
    val pickReference = { referenceLauncher.launch(arrayOf("audio/*", "video/*")) }

    var hasMicPermission by remember {
        mutableStateOf(
            context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
                android.content.pm.PackageManager.PERMISSION_GRANTED,
        )
    }
    val micPermission = rememberLauncherForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.RequestPermission(),
    ) { granted ->
        hasMicPermission = granted
        if (granted) viewModel.startRecording()
    }

    var settingsOpen by rememberSaveable { mutableStateOf(false) }

    PhoneScaffold(
        toolbar = {
            // No engine tag up here. Which model does the work depends on the
            // tab, so a single toolbar badge was either wrong on one of them or
            // duplicating the card below it — and the sheet behind the sliders
            // names it in one place for both.
            // The mode lives here rather than in the body. It is the one
            // question this screen asks that changes everything below it — text
            // out of audio, or audio out of text — and as a pill row it cost a
            // full band of the screen to say something the toolbar can hold.
            // Both icons carry a content description, because a microphone and
            // a speaker are only obvious once you already know.
            RootToolbar("Voice") {
                // Iterated, not listed — see the note on Image's toolbar. A mode
                // added to the enum has to be given an icon before this builds.
                VoiceMode.entries.forEach { mode ->
                    ToolbarToggle(
                        when (mode) {
                            VoiceMode.TRANSCRIBE -> NIcons.Mic
                            VoiceMode.SPEAK -> NIcons.Speaker
                        },
                        mode.label,
                        selected = state.mode == mode,
                        onClick = { viewModel.setMode(mode) },
                    )
                }
                ToolbarAction(NIcons.Plus, "Start over", viewModel::reset)
                ToolbarAction(NIcons.Settings, "Voice settings", { settingsOpen = true })
            }
        },
        bottomBar = { NBottomBar(BottomDestinations, currentRoute) { onNavigate(it.route) } },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        // Where the input comes from is no longer a mode. "Type" and "File"
        // never differed in anything but where the characters arrived from, so
        // a tab for it was a wall between one text box and the button that
        // fills it; "Microphone" and "File" both end in the same transcript, so
        // the two ways in sit side by side above it. Each panel now offers both
        // and the screen has one row of chrome where it had two.
        Column(Modifier.verticalScroll(rememberScrollState())) {

            // §1.2 — a refusal names what went wrong and what to do about it.
            state.error?.let { message ->
                NCard(Modifier.padding(bottom = 10.dp), ring = NocturneColors.Neutral700) {
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

            // Transcribe's engine card, the mirror of Speak's. Both tabs now
            // name the model doing the work in the same place and the same
            // shape, which is what the toolbar tag was standing in for.
            if (state.mode == VoiceMode.TRANSCRIBE) {
                NCard(
                    Modifier.padding(bottom = 10.dp),
                    ring = if (state.sttModel != null) NocturneColors.Accent700 else NocturneColors.Divider,
                ) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            NIcons.Waveform,
                            contentDescription = null,
                            tint = NocturneColors.Accent300,
                            modifier = Modifier.size(16.dp),
                        )
                        // No engine name here. Transcribe has exactly one —
                        // whisper.cpp — so naming it says nothing the user can
                        // act on, and it duplicated the model shown immediately
                        // below. Speak names its engine because it genuinely has
                        // three to choose between.
                        Text(
                            if (state.sttModel != null) "Speech model" else "No speech model",
                            style = NocturneType.CardTitleSm,
                            modifier = Modifier.weight(1f),
                        )
                        NTag(
                            if (state.sttModel != null) "on-device" else "missing",
                            style = if (state.sttModel != null) NTagStyle.Accent else NTagStyle.Outline,
                        )
                    }
                    if (state.sttModel == null) {
                        Text(
                            "Nothing to transcribe with yet. Add model lists whisper under Speech — " +
                                "base or small suits a phone.",
                            style = NocturneType.CardBody,
                            color = NocturneColors.Text.copy(alpha = 0.8f),
                        )
                    }
                    // Shown whenever anything is installed, not only when there
                    // are two. Gating it on a choice being available hid the
                    // control on every single-model device, so there was no way
                    // to confirm which model was about to run — and Chat has
                    // always shown its model row regardless.
                    if (state.sttModels.isNotEmpty()) {
                        // Labelled by quant, since a whisper library is normally
                        // several sizes of the same repo — "tiny-q5_1" and
                        // "small" distinguish them, the repo name does not.
                        val labels = state.sttModels.map { it.quant ?: it.displayName }
                        NDropdown(
                            options = labels,
                            selected = state.sttModel?.let { it.quant ?: it.displayName },
                            onSelect = { label ->
                                state.sttModels
                                    .firstOrNull { (it.quant ?: it.displayName) == label }
                                    ?.let(viewModel::selectSttModel)
                            },
                            modifier = Modifier.padding(top = 4.dp),
                        )
                    }
                }
            }

            when (state.mode) {
                VoiceMode.TRANSCRIBE -> TranscribePanel(
                    state = state,
                    viewModel = viewModel,
                    onRecord = {
                        when {
                            state.recording -> viewModel.stopRecording()
                            // The permission is asked for at the moment it is
                            // needed, with the reason on screen — not at launch,
                            // before the user has any context for the request.
                            !hasMicPermission ->
                                micPermission.launch(Manifest.permission.RECORD_AUDIO)
                            else -> viewModel.startRecording()
                        }
                    },
                    onPickAudio = pickAudio,
                )

                VoiceMode.SPEAK -> SpeakPanel(
                    state = state,
                    viewModel = viewModel,
                    onPickScript = pickScript,
                    onPickReference = pickReference,
                )
            }

            // Outside the mode switch, because there is one runtime and one
            // trace: transcribing and speaking never overlap, so a block per
            // panel would be the same block written twice.
            (state.liveTrace ?: state.lastTrace)?.let { trace ->
                var traceExpanded by rememberSaveable { mutableStateOf(false) }
                ResourceBlock(
                    trace = trace,
                    expanded = traceExpanded,
                    onToggle = { traceExpanded = !traceExpanded },
                    live = state.recording || state.speaking || state.loading || state.rendering,
                    modifier = Modifier.padding(top = 12.dp),
                )
            }
        }
    }

    // Left open on the way to Advanced on purpose — see the note on the Image
    // screen's sheet. Back from Parameters returns to the panel it was opened
    // from rather than to a bare screen.
    if (settingsOpen) {
        VoiceSettingsSheet(
            state = state,
            viewModel = viewModel,
            onDismiss = { settingsOpen = false },
            onOpenAdvanced = onOpenAdvanced,
        )
    }
}

/**
 * **Transcribe — SPEC §6.**
 *
 * One clip, however it arrived.
 *
 * This was two panels behind a source switch, and the switch was a wall: the
 * file button took you to a different screen, with a "Record instead" button to
 * get back, as though picking a file and recording one were different jobs. They
 * are not — both end as a decodable file on disk, and everything after that
 * point is identical. So there is one recorder, a file button beside it, and a
 * clip that behaves the same way whichever button produced it: play it, run it,
 * or put it down.
 *
 * Picking a file no longer starts a decode on its own. It stages the clip, the
 * way stopping the recorder does, and **Process** is the one thing that spends
 * compute — which also means a picked file can be listened to before it is run.
 */
@Composable
private fun TranscribePanel(
    state: VoiceState,
    viewModel: VoiceViewModel,
    onRecord: () -> Unit,
    onPickAudio: () -> Unit,
) {
    // — the recorder —
    //
    // The waveform stays on screen when idle rather than appearing on the first
    // press: it is the face of the recorder, and a control that materialises
    // under your thumb moves everything below it.
    Row(
        Modifier.fillMaxWidth().height(56.dp).padding(horizontal = 2.dp),
        horizontalArrangement = Arrangement.spacedBy(2.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        state.waveform.forEach { level ->
            Box(
                Modifier
                    .weight(1f)
                    .fillMaxHeight(if (state.recording) level.coerceIn(0.05f, 1f) else 0.05f)
                    .background(
                        if (state.recording && level > 0.45f) {
                            NocturneColors.Accent500
                        } else {
                            NocturneColors.Neutral700
                        },
                        RoundedCornerShape(2.dp),
                    ),
            )
        }
    }

    Row(
        Modifier.fillMaxWidth().padding(top = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(9.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            when {
                state.paused -> "❙❙ held ${Fmt.duration(state.elapsedMillis)}"
                state.recording -> "● REC ${Fmt.duration(state.elapsedMillis)}"
                else -> "○ idle"
            },
            style = NocturneType.MonoSm,
            color = if (state.recording) NocturneColors.Accent else NocturneColors.TextMuted,
        )
        Text(
            "transcribes on stop",
            style = NocturneType.MonoSm,
            color = NocturneColors.TextMuted,
        )
    }

    // Record, hold, stop — the three states a recorder has — and the other way
    // in, as an icon beside them rather than as a second screen. Pause keeps the
    // microphone, so resuming does not risk losing the device to another app in
    // the handover.
    Row(
        Modifier.fillMaxWidth().padding(top = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(7.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        NButton(
            // Not "Loading model…" any more. `loading` is also true while
            // Process runs, so the recorder announced a model load that was
            // really a transcription happening in the card below it. The
            // recorder says what the recorder is doing; it is simply
            // unavailable while the decoder has the runtime.
            if (state.recording) "Stop" else "Record",
            onClick = onRecord,
            style = NButtonStyle.Primary,
            enabled = !state.loading || state.recording,
            modifier = Modifier.weight(1f),
            minHeight = 48.dp,
        )
        if (state.recording) {
            NIconButton(
                if (state.paused) NIcons.Play else NIcons.Pause,
                if (state.paused) "Resume recording" else "Pause recording",
                onClick = {
                    if (state.paused) viewModel.resumeRecording() else viewModel.pauseRecording()
                },
                size = 48.dp,
            )
        }
        NIconButton(
            NIcons.File,
            if (state.sourcePath != null) "Choose a different file" else "Choose an audio file",
            onClick = onPickAudio,
            size = 48.dp,
        )
    }

    // No live transcript panel. There is nothing to put in it: recording
    // decodes nothing now, and a panel saying "listening" over a decoder that
    // is not running is the kind of claim this screen keeps having to retract.
    if (state.recording) {
        NHelp(
            "Recording. The take is transcribed in one pass when you stop, which is what " +
                "gives the segments their timings.",
            Modifier.padding(top = 12.dp),
        )
    }

    // — the clip —
    val clip = state.sourcePath
    if (clip == null) {
        // Not while the microphone is open: it describes how to get a clip, and
        // by then you are getting one.
        if (!state.recording) NHelp(
            "Record, or pick a file — anything Android can decode works, so m4a, mp3, wav " +
                "and opus are all fine. Either way you get the clip back to listen to before " +
                "it is transcribed.",
            Modifier.padding(top = 12.dp),
        )
    } else {
        SectionKicker("Clip", Modifier.padding(top = 18.dp, bottom = 8.dp))
        NCard(gap = 8.dp) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    if (state.sourceIsRecording) NIcons.Mic else NIcons.File,
                    contentDescription = null,
                    tint = NocturneColors.Accent300,
                    modifier = Modifier.size(15.dp),
                )
                Text(
                    state.sourceName ?: clip.substringAfterLast('/'),
                    style = NocturneType.CardTitleSm,
                    maxLines = 1,
                    modifier = Modifier.weight(1f),
                )
                NTag(if (state.sourceIsRecording) "recorded" else "file", style = NTagStyle.Outline)
                NIconButton(
                    NIcons.Trash,
                    "Remove the clip",
                    onClick = viewModel::clearSource,
                    enabled = !state.loading,
                )
            }
            // Hear it before deciding to spend the decode on it.
            NAudioPlayer(file = java.io.File(clip))
            NButton(
                if (state.loading) "Transcribing…" else "Process",
                onClick = viewModel::process,
                style = NButtonStyle.Primary,
                block = true,
                enabled = !state.loading && !state.recording,
                minHeight = 46.dp,
            )
        }
        if (state.sourceIsRecording && state.segments.isEmpty()) {
            NHelp(
                "The live text above came from a sliding window. Processing decodes the whole " +
                    "recording, which is what produces timed segments.",
                Modifier.padding(top = 6.dp),
            )
        }
    }

    // — the transcript —
    if (state.segments.isNotEmpty()) {
        SectionKicker("Transcript", Modifier.padding(top = 18.dp, bottom = 8.dp))
        NCard(gap = 8.dp) {
            Text(state.title, style = NocturneType.CardTitleSm)
            Text(
                "${Fmt.duration(state.segments.maxOf { it.endMillis })} · " +
                    "${state.segments.size} segments",
                style = NocturneType.MonoXs,
                color = NocturneColors.TextMuted,
            )
            NProgressBar(fraction = state.fileProgress)
            NCardMeta(gap = 8.dp) {
                Text(
                    Fmt.percent(state.fileProgress),
                    style = NocturneType.MonoSm,
                    color = NocturneColors.Accent300,
                )
                NMetaText("·")
                NMetaText(state.sttModel?.displayName ?: "no model")
                Box(Modifier.weight(1f))
                // Measured, not asserted (§8.2).
                NMetaText(String.format("%.1f× realtime", state.realtimeFactor))
            }
        }

        state.segments.forEachIndexed { _, segment ->
            Row(
                Modifier
                    .fillMaxWidth()
                    .ruleBelow()
                    .padding(vertical = 10.dp)
                    // Faded by the decoder's own confidence, not by position — a
                    // late segment the model is sure of reads at full strength.
                    .alpha(0.35f + segment.confidence * 0.65f),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    timestamp(segment.startMillis),
                    style = NocturneType.MonoTimestamp,
                    color = if (segment.confidence < 0.7f) {
                        NocturneColors.TextMuted
                    } else {
                        NocturneColors.Accent300
                    },
                    modifier = Modifier.width(52.dp),
                )
                Text(segment.text, style = NocturneType.Row, modifier = Modifier.weight(1f))
            }
        }

        NHelp(
            "Opacity tracks the decoder's own confidence for each segment.",
            Modifier.padding(top = 8.dp),
        )
        // The four format buttons moved to Library, where every artifact is
        // saved and shared the same way and the file lands in a folder you chose
        // rather than in app-private storage no file manager will show you.
        NHelp(
            "Saved to the library. Open it there to export as TXT, SRT, VTT or JSON into a " +
                "folder of your choosing.",
            Modifier.padding(top = 10.dp),
        )
    }
}

/**
 * **Read aloud — SPEC §7.**
 *
 * A script, a voice, an expression, and a file at the end. The order is the
 * order you work in, which is why this is a mode and not a card: choosing a
 * voice for a passage you have not written yet is backwards.
 *
 * The engine in use is named at the top of the panel, always. The app will fall
 * back to the system synthesiser when Kokoro is not installed — that is a
 * feature, but it is not something to be quiet about.
 */
@Composable
private fun SpeakPanel(
    state: ai.ondevice.ui.vm.VoiceState,
    viewModel: VoiceViewModel,
    onPickScript: () -> Unit,
    onPickReference: () -> Unit,
) {
    // Which engine will speak, stated but not chosen here — the picker moved to
    // the settings sheet with the voice list and the dials, so this panel is the
    // script and the act, not the configuration.
    val provider = state.selectedVoice?.provider ?: ai.ondevice.speech.SynthProvider.SYSTEM
    NCard(ring = if (provider != ai.ondevice.speech.SynthProvider.SYSTEM) NocturneColors.Accent700 else NocturneColors.Divider) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                NIcons.Waveform,
                contentDescription = null,
                tint = NocturneColors.Accent300,
                modifier = Modifier.size(16.dp),
            )
            Text(
                listOfNotNull(
                    when (provider) {
                        ai.ondevice.speech.SynthProvider.KOKORO -> "Kokoro"
                        ai.ondevice.speech.SynthProvider.OMNIVOICE -> "OmniVoice"
                        ai.ondevice.speech.SynthProvider.SYSTEM -> "System engine"
                    },
                    state.selectedVoice?.displayName,
                ).joinToString(" · "),
                style = NocturneType.CardTitleSm,
                modifier = Modifier.weight(1f),
            )
            NTag(
                if (provider == ai.ondevice.speech.SynthProvider.SYSTEM) "fallback" else "neural",
                style = if (provider == ai.ondevice.speech.SynthProvider.SYSTEM) NTagStyle.Outline else NTagStyle.Accent,
            )
        }
        // Before Speak rather than after: with no packs the model loads and then
        // fails inside the graph, which reads as a broken model.
        state.missingVoiceComponent?.let { missing ->
            Text(
                "${missing.what} — ${missing.because}. ${missing.remedy}.",
                style = NocturneType.CardBody,
                color = NocturneColors.Text.copy(alpha = 0.8f),
            )
        }
    }

    // — the script —
    SectionKicker("Script", Modifier.padding(top = 18.dp, bottom = 8.dp))

    // OmniVoice takes direction inline, and there is no way to discover that
    // from a text box. These are not conventions this app invented: the tag
    // list and both pronunciation forms are upstream's, quoted so a typo in a
    // tag reads as a typo rather than as the model ignoring you.
    if (provider == ai.ondevice.speech.SynthProvider.OMNIVOICE) {
        NCard(gap = 6.dp) {
            Text("Direction goes in the text", style = NocturneType.CardTitleSm)
            Text(
                "Non-verbal sounds are tags written where you want them: " +
                    "\"[laughter] You really got me.\" Supported are [laughter], [sigh], " +
                    "[confirmation-en], [question-en], [question-ah], [question-oh], " +
                    "[question-ei], [question-yi], [surprise-ah], [surprise-oh], [surprise-wa], " +
                    "[surprise-yo] and [dissatisfaction-hnn]. Anything else in brackets is read " +
                    "as an English pronunciation.",
                style = NocturneType.CardBody,
                color = NocturneColors.Text.copy(alpha = 0.8f),
            )
            Text(
                "Fix a pronunciation in English with CMU dictionary symbols in capitals: " +
                    "\"He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.\" " +
                    "In Chinese, write the pinyin with a tone number in capitals after the " +
                    "character: \"打ZHE2出售\".",
                style = NocturneType.CardBody,
                color = NocturneColors.Text.copy(alpha = 0.8f),
            )
            Text(
                "Who is speaking is not set here — describe them under Advanced · voice design, " +
                    "e.g. \"female, low pitch, british accent\", or copy a real voice below.",
                style = NocturneType.CardBody,
                color = NocturneColors.Text.copy(alpha = 0.8f),
            )
        }

        // — the voice to copy —
        NCard(gap = 8.dp, modifier = Modifier.padding(top = 8.dp)) {
            // The same shape as Transcribe's clip: a title row that carries the
            // verbs as icons, and the clip itself playable underneath. A
            // reference you can only read the filename of is one you cannot
            // judge — whether it is clean enough to copy is a question about
            // the sound.
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Copy a voice", style = NocturneType.CardTitleSm, modifier = Modifier.weight(1f))
                if (state.cloningAvailable) {
                    NIconButton(
                        NIcons.File,
                        if (state.referenceSamples != null) "Replace the recording" else "Choose a recording",
                        onClick = onPickReference,
                    )
                    NIconButton(
                        NIcons.Trash,
                        "Remove the recording",
                        onClick = viewModel::clearReferenceClip,
                        enabled = state.referenceSamples != null,
                    )
                }
            }
            if (!state.cloningAvailable) {
                Text(
                    "This OmniVoice install does not have the three encoders that turn a " +
                        "recording into something the model can copy. Reinstalling it adds them; " +
                        "describing a voice in words works either way.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            } else if (state.referenceSamples == null) {
                Text(
                    "Give it a few seconds of someone speaking and it will read your text in " +
                        "that voice. Three to ten seconds is the sweet spot — past twenty it gets " +
                        "slower and worse, not better.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            } else {
                Text(
                    "${state.referenceName} · ${"%.1f".format(state.referenceSeconds)} s",
                    style = NocturneType.Row,
                )
                state.referencePath?.let { NAudioPlayer(file = java.io.File(it)) }
                if (state.referenceSeconds > 20f) {
                    NHelp(
                        "Longer than twenty seconds. Upstream's own warning is that this makes " +
                            "generation slower and the copy worse — trimming to a clean few " +
                            "seconds usually sounds better.",
                    )
                }
                Text(
                    if (state.transcribingReference) {
                        "Working out what it says…"
                    } else {
                        // Not a nicety: the model is given the reference's words
                        // and its sound and asked to carry on, so a wrong
                        // transcript pulls the new speech towards the wrong thing.
                        "What the recording says. Filled in by the speech model where one is " +
                            "installed — correct it if it got a name wrong, because the copy " +
                            "follows these words as well as the voice."
                    },
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
                NTextArea(
                    value = state.referenceTranscript,
                    onValueChange = viewModel::setReferenceTranscript,
                    minHeight = 64.dp,
                    textStyle = NocturneType.Row,
                )
            }
        }
    }

    // One script box, and a button that fills it.
    //
    // These were two tabs, and the difference between them was where the
    // characters came from — the "File" tab already showed the same editable
    // text area, because a file you cannot fix a typo in before it is read
    // aloud is a worse file. So the tab was a wall between a text box and the
    // button that fills it. Loading a file now just types into it.
    // Two verbs on one row of icons rather than two full-width buttons. They
    // are things you do *to* the script, so they sit with it rather than
    // competing with Speak for the width of the screen.
    Row(
        Modifier.fillMaxWidth().padding(bottom = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        NMetaText(
            state.scriptSource?.substringAfterLast('/')
                ?: "${state.script.length} characters",
        )
        Box(Modifier.weight(1f))
        NIconButton(
            NIcons.File,
            if (state.scriptSource != null) "Replace with a file" else "Load from file",
            onClick = onPickScript,
        )
        NIconButton(
            NIcons.Trash,
            "Clear the script",
            onClick = { viewModel.setScript("") },
            enabled = state.script.isNotEmpty(),
        )
    }
    run {
        NTextArea(
            value = state.script,
            onValueChange = viewModel::setScript,
            placeholder = "Type or paste what should be read aloud.",
            minHeight = 130.dp,
            textStyle = NocturneType.Row,
        )
    }
    NHelp(
        buildString {
            state.scriptSource?.let { append("$it · ") }
            append("${Fmt.grouped(state.script.length)} characters")
            if (state.estimatedSeconds > 0) {
                append(" · about ${Fmt.duration(state.estimatedSeconds * 1000L)} at ${
                    String.format("%.2f", state.speed)
                }×")
            }
        },
        Modifier.padding(top = 6.dp),
    )

    state.speakError?.let { error ->
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
                Text(error, style = NocturneType.CardBody, modifier = Modifier.weight(1f))
            }
        }
    }

    // — act —
    NButton(
        // One button, because there is one act: rendering *is* speaking now,
        // and the take that comes out is stored like an image or a transcript.
        if (state.rendering) "Stop" else "Speak",
        onClick = { if (state.rendering) viewModel.stopSpeaking() else viewModel.speak() },
        style = if (state.rendering) NButtonStyle.Secondary else NButtonStyle.Primary,
        block = true,
        modifier = Modifier.padding(top = 16.dp),
    )
    // Rendering still writes the WAV — it has to, the library lists it — but
    // the two buttons that used to sit here are gone. "Save as WAV" wrote into
    // app-private storage and called that a save; Library does it properly, to
    // a folder you picked, and does it the same way for every artifact.
    // The take itself, playable. Before this the only way to hear a render was
    // to render it again, which for OmniVoice is minutes of compute to replay
    // four seconds of audio.
    state.lastAudioPath?.let { path ->
        NAudioPlayer(
            file = java.io.File(path),
            label = path.substringAfterLast('/'),
            autoPlay = state.autoPlay,
            onAutoPlayed = viewModel::autoPlayHandled,
            modifier = Modifier.padding(top = 8.dp),
        )
        NHelp(
            "Listed in the library — open it there to save it to a folder or share it.",
            Modifier.padding(top = 6.dp),
        )
    }
}

@Composable
private fun SpeakSlider(
    label: String,
    display: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    onChange: (Float) -> Unit,
) {
    Row(
        Modifier.fillMaxWidth().padding(top = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(7.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = NocturneType.Row, modifier = Modifier.weight(1f))
        Text(display, style = NocturneType.MonoValue, color = NocturneColors.Accent300)
    }
    NSlider(value = value, onValueChange = onChange, valueRange = range)
}

/**
 * Five.
 *
 * The catalogue is 338 voices and the list is not the point of the screen — the
 * script above it is. Fourteen rows pushed Expression and the read-aloud button
 * off the bottom of a phone, which made the voice picker look like the whole
 * feature. Five is enough to browse and short enough that searching is the
 * obvious next move, which for a list this long it should be.
 */
private const val VOICE_ROWS = 5

private fun voiceDescription(voice: String, speed: Float): String {
    val region = when (voice.take(1)) {
        "a" -> "US"
        "b" -> "UK"
        "j" -> "Japanese"
        "z" -> "Chinese"
        "e" -> "Spanish"
        "f" -> "French"
        "h" -> "Hindi"
        "i" -> "Italian"
        "p" -> "Portuguese"
        else -> "—"
    }
    val gender = if (voice.getOrNull(1) == 'f') "female" else "male"
    return "$region · $gender · ${String.format("%.1f", speed)}×"
}

/**
 * Which engine, which model, which voice, and how it is shaped.
 *
 * All of it used to sit in the same scroll as the script and the Speak button —
 * the engine card, a fifty-row voice list, three sliders and a blend picker
 * between what you wrote and the button that reads it. None of that is about
 * this passage; it is how this device is set up to speak.
 */
@Composable
private fun VoiceSettingsSheet(
    state: ai.ondevice.ui.vm.VoiceState,
    viewModel: VoiceViewModel,
    onDismiss: () -> Unit,
    onOpenAdvanced: (String) -> Unit,
) {
    NBottomSheet("Voice settings", onDismiss, note = "applies to the next run") {
        if (state.mode == VoiceMode.TRANSCRIBE) {
            TranscribeSettings(state, viewModel)
            return@NBottomSheet
        }

        // A switch rather than an automatic choice. The two neural engines are
        // not interchangeable: Kokoro is an order of magnitude faster, OmniVoice
        // can do things Kokoro cannot do at all. Picking on the user's behalf
        // would mean either surprising them with a minute of compute or silently
        // dropping the feature they came for.
        val provider = state.selectedVoice?.provider ?: ai.ondevice.speech.SynthProvider.SYSTEM
        val engines = ai.ondevice.speech.SynthProvider.entries
        SectionKicker("Engine", Modifier.padding(bottom = 8.dp))
        NSeg(
            options = engines.map { it.label },
            selectedIndex = engines.indexOf(provider).coerceAtLeast(0),
            onSelect = { viewModel.selectProvider(engines[it]) },
            modifier = Modifier.fillMaxWidth(),
        )

        // Which *model* provides the engine, as distinct from which engine. Only
        // the models this engine can load, and only when the engine loads one at
        // all: Android's synthesiser is part of the OS and has no file to
        // choose, so a picker over it is a control with nothing behind it. The
        // unfiltered list was worse than useless — the OmniVoice tab listed
        // Kokoro and showed it as the selection, which is a promise the engine
        // cannot keep.
        val engineModels = state.ttsModels.filter { state.ttsModelProviders[it.id] == provider }
        if (provider != ai.ondevice.speech.SynthProvider.SYSTEM && engineModels.isNotEmpty()) {
            NDropdown(
                options = engineModels.map { it.displayName },
                selected = engineModels.firstOrNull { it.id == state.ttsModel?.id }?.displayName
                    ?: engineModels.first().displayName,
                onSelect = { name ->
                    engineModels.firstOrNull { it.displayName == name }
                        ?.let(viewModel::selectTtsModel)
                },
                modifier = Modifier.padding(top = 8.dp),
            )
        }

        NHelp(
            when (provider) {
                ai.ondevice.speech.SynthProvider.KOKORO ->
                    "The quick one — short lines come back near enough to straight away. " +
                        "Fifty voices, six languages, no emotion tags."
                // Both descriptions used to quote figures — "half a second of work per second
                // of speech", "six to seven times slower", "around a minute for a short
                // sentence". None was ever timed on a device. The ordering is real and worth
                // saying; the numbers go back in when they have been measured.
                ai.ondevice.speech.SynthProvider.OMNIVOICE ->
                    "Reads any language with no phonemiser, takes [laughter] and [sigh], and can " +
                        "voice several speakers. Much slower than Kokoro — it runs the whole " +
                        "model twice per step and there are thirty-two of them, so expect to " +
                        "wait even for a short sentence."
                ai.ondevice.speech.SynthProvider.SYSTEM ->
                    "Android's own synthesiser. A different voice with different prosody — the app " +
                        "says so rather than passing it off as one of the neural engines."
            },
            Modifier.padding(top = 8.dp),
        )

        if (!state.omniVoiceAvailable && provider != ai.ondevice.speech.SynthProvider.OMNIVOICE) {
            NHelp(
                "OmniVoice is not installed — Add model lists it under Voice: " +
                    (ai.ondevice.core.StarterModels.installHint(
                        ai.ondevice.core.StarterModels.OMNIVOICE_REPO,
                    ) ?: ai.ondevice.core.StarterModels.OMNIVOICE_REPO) + ".",
                Modifier.padding(top = 4.dp),
            )
        }

        // — the voice —
        // Counted over this engine's voices, since those are the only ones
        // listed. "51 available of 339" above a list of one engine's voices
        // described the library, not the choice on offer.
        val engineVoices = state.voices.filter { it.provider == state.selectedProvider }
        SectionKicker(
            "Voice · ${engineVoices.count { it.available }} available of ${engineVoices.size}",
            Modifier.padding(top = 20.dp, bottom = 8.dp),
        )
        state.missingVoiceComponent?.let { missing ->
            NCard(Modifier.padding(bottom = 8.dp)) {
                Text(missing.what, style = NocturneType.CardTitleSm, color = NocturneColors.Accent200)
                Text(
                    "${missing.because}. ${missing.remedy}.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            }
        }
        NInput(
            value = state.voiceQuery,
            onValueChange = viewModel::setVoiceQuery,
            placeholder = "Search by name or language",
            minHeight = 40.dp,
        )
        Column(
            Modifier.fillMaxWidth().padding(top = 8.dp),
            verticalArrangement = Arrangement.spacedBy(5.dp),
        ) {
            // Capped, with the count stated — a silent truncation would read as
            // "these are all of them".
            val shown = state.filteredVoices.take(VOICE_ROWS)
            shown.forEach { voice ->
                val selected = voice.id == state.voice
                Row(
                    Modifier
                        .fillMaxWidth()
                        .background(
                            if (selected) NocturneColors.Accent900 else NocturneColors.Surface,
                            Radius.Md,
                        )
                        .ring(if (selected) NocturneColors.Accent else NocturneColors.Divider, Radius.Md)
                        .alpha(if (voice.available) 1f else 0.45f)
                        .nClickableFlat { viewModel.selectVoice(voice.id) }
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                    horizontalArrangement = Arrangement.spacedBy(9.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(
                            voice.displayName,
                            style = NocturneType.Row,
                            color = if (selected) NocturneColors.Accent200 else NocturneColors.Text,
                        )
                        Text(voice.localeLabel, style = NocturneType.MonoXs, color = NocturneColors.TextMuted)
                    }
                    if (!voice.available) {
                        NTag("not installed", style = NTagStyle.Outline)
                    } else {
                        NTag(voice.provider.label, style = NTagStyle.Neutral)
                    }
                }
            }
            if (state.filteredVoices.size > shown.size) {
                NHelp("${state.filteredVoices.size - shown.size} more — narrow the search to see them.")
            }
        }

        // — expression —
        //
        // Which dials exist depends on the engine, because the two engines
        // genuinely differ. Kokoro's graph takes token ids, a style vector and a
        // speed — there is no pitch input at all. Showing a Pitch slider for it
        // would leave the user adjusting something inert, so it is absent rather
        // than ignored.
        val kokoroSelected = state.selectedVoice?.provider == ai.ondevice.speech.SynthProvider.KOKORO
        SectionKicker("Expression", Modifier.padding(top = 20.dp, bottom = 8.dp))
        NCard(gap = 4.dp) {
            SpeakSlider("Speed", String.format("%.2f×", state.speed), state.speed, 0.5f..2f, viewModel::setSpeed)
            if (!kokoroSelected) {
                SpeakSlider("Pitch", String.format("%.2f", state.pitch), state.pitch, 0.5f..2f, viewModel::setPitch)
            }
            SpeakSlider("Volume", Fmt.percent(state.volume), state.volume, 0f..1f, viewModel::setVolume)
            Text(
                if (kokoroSelected) {
                    "Speed is applied during synthesis, not by resampling, so the voice does not " +
                        "turn chipmunk when you raise it. Kokoro has no pitch input, so there is " +
                        "no pitch dial here — the system engine does, and shows one."
                } else {
                    "Speed and pitch are applied by the engine, not by resampling, so the voice " +
                        "does not turn chipmunk when you raise it."
                },
                style = NocturneType.CardBody,
                color = NocturneColors.Text.copy(alpha = 0.8f),
            )
        }

        // — blend, Kokoro only —
        //
        // Gated on Kokoro being *selected*, not merely installed. Blending mixes
        // two Kokoro style vectors, which is a thing only Kokoro's graph takes;
        // keyed on availability it sat under OmniVoice offering af_heart and
        // bm_george to an engine with no style input at all.
        if (kokoroSelected) {
            SectionKicker("Blend", Modifier.padding(top = 20.dp, bottom = 8.dp))
            NCard(gap = 6.dp) {
                Text("Mix a second voice", style = NocturneType.CardTitleSm)
                NEnumRow(
                    options = state.voices
                        .filter { it.provider == ai.ondevice.speech.SynthProvider.KOKORO }
                        .take(12).map { it.id },
                    selected = state.blendVoice,
                    onSelect = { viewModel.setBlendVoice(if (it == state.blendVoice) null else it) },
                )
                if (state.blendVoice != null) {
                    SpeakSlider(
                        "Mix",
                        "${Fmt.percent(1f - state.blendRatio)} / ${Fmt.percent(state.blendRatio)}",
                        state.blendRatio,
                        0f..1f,
                        viewModel::setBlendRatio,
                    )
                }
            }
        }

        // Each engine's own parameter set. Kokoro and OmniVoice are both
        // text-to-speech but share almost no controls, which is why the label
        // names what is behind the button rather than saying "Advanced".
        NButton(
            when (provider) {
                ai.ondevice.speech.SynthProvider.OMNIVOICE -> "Advanced · voice design, language, steps"
                ai.ondevice.speech.SynthProvider.KOKORO -> "Advanced · language, chunking, trim, gain"
                else -> "Advanced · system engine"
            },
            onClick = {
                onOpenAdvanced(
                    when (provider) {
                        ai.ondevice.speech.SynthProvider.OMNIVOICE ->
                            ai.ondevice.engine.RuntimeRegistry.OMNIVOICE
                        else -> ai.ondevice.engine.RuntimeRegistry.KOKORO
                    },
                )
            },
            style = NButtonStyle.Secondary,
            block = true,
            modifier = Modifier.padding(top = 16.dp),
        )
    }
}

/** Transcribe has one engine and one choice to make: which whisper. */
@Composable
private fun TranscribeSettings(
    state: ai.ondevice.ui.vm.VoiceState,
    viewModel: VoiceViewModel,
) {
    SectionKicker("Speech model", Modifier.padding(bottom = 8.dp))
    if (state.sttModels.isEmpty()) {
        NHelp(
            "Nothing to transcribe with yet. Add model lists whisper under Speech — base or " +
                "small suits a phone.",
        )
        return
    }
    // Labelled by quant, since a whisper library is normally several sizes of
    // the same repo — "tiny-q5_1" and "small" distinguish them, the repo name
    // does not.
    val labels = state.sttModels.map { it.quant ?: it.displayName }
    NDropdown(
        options = labels,
        selected = state.sttModel?.let { it.quant ?: it.displayName },
        onSelect = { label ->
            state.sttModels.firstOrNull { (it.quant ?: it.displayName) == label }
                ?.let(viewModel::selectSttModel)
        },
    )
    NHelp(
        "Recording decodes nothing — the take is transcribed once, when you stop.",
        Modifier.padding(top = 8.dp),
    )
}

/** `MM:SS.d` — the canvas' segment clock. */
private fun timestamp(millis: Long): String =
    String.format("%02d:%02d.%d", millis / 60_000, (millis % 60_000) / 1000, (millis % 1000) / 100)

/**
 * The export leaves through the system share sheet rather than a bespoke
 * "saved to…" toast: the file is already in a folder the user can open, and the
 * chooser is how a transcript actually reaches a player or an editor.
 */
/** The "send audio" half of §7 — the rendered WAV, out through the share sheet. */
private fun shareAudio(context: android.content.Context, file: java.io.File) {
    val uri = androidx.core.content.FileProvider.getUriForFile(
        context,
        "${context.packageName}.fileprovider",
        file,
    )
    val intent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
        type = "audio/wav"
        putExtra(android.content.Intent.EXTRA_STREAM, uri)
        putExtra(android.content.Intent.EXTRA_TITLE, file.name)
        addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(android.content.Intent.createChooser(intent, "Send audio"))
}

private fun shareTranscript(
    context: android.content.Context,
    file: java.io.File,
    format: TranscriptFormat,
) {
    val uri = androidx.core.content.FileProvider.getUriForFile(
        context,
        "${context.packageName}.fileprovider",
        file,
    )
    val intent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
        type = format.mime
        putExtra(android.content.Intent.EXTRA_STREAM, uri)
        putExtra(android.content.Intent.EXTRA_TITLE, file.name)
        addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(android.content.Intent.createChooser(intent, "Export ${format.label}"))
}
