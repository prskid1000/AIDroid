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

/** **S14 — Voice.** Two tabs, and they are inverses of each other: - **Transcribe** (whisper.cpp, SPEC §6) — audio in, text out. */
@Composable
fun VoiceScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onOpenAdvanced: (String) -> Unit,
    viewModel: VoiceViewModel = activityVoiceViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val context = LocalContext.current

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
            // No engine tag up here.
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
        // Where the input comes from is no longer a mode.
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

            // Transcribe's engine card, the mirror of Speak's.
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
                        // No engine name here.
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
                    // Shown whenever anything is installed, not only when there are two.
                    if (state.sttModels.isNotEmpty()) {
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
                            // The permission is asked for at the moment it is needed, with the reason on screen — not at launch, before the user has any context for the request.
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

    // Left open on the way to Advanced on purpose — see the note on the Image screen's sheet.
    if (settingsOpen) {
        VoiceSettingsSheet(
            state = state,
            viewModel = viewModel,
            onDismiss = { settingsOpen = false },
            onOpenAdvanced = onOpenAdvanced,
        )
    }
}

/** **Transcribe — SPEC §6.** One clip, however it arrived. */
@Composable
private fun TranscribePanel(
    state: VoiceState,
    viewModel: VoiceViewModel,
    onRecord: () -> Unit,
    onPickAudio: () -> Unit,
) {
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
        // "idle" is wrong while the take is being decoded, and that is exactly when it used to say so.
        val transcribing = state.loading && !state.recording
        Text(
            when {
                state.paused -> "❙❙ held ${Fmt.duration(state.elapsedMillis)}"
                state.recording -> "● REC ${Fmt.duration(state.elapsedMillis)}"
                transcribing -> "◐ transcribing"
                else -> "○ idle"
            },
            style = NocturneType.MonoSm,
            color = if (state.recording || transcribing) {
                NocturneColors.Accent
            } else {
                NocturneColors.TextMuted
            },
        )
        Text(
            if (transcribing) "the whole take, in one pass" else "transcribes on stop",
            style = NocturneType.MonoSm,
            color = NocturneColors.TextMuted,
        )
    }

    // Record, hold, stop — the three states a recorder has — and the other way in, as an icon beside them rather than as a second screen.
    Row(
        Modifier.fillMaxWidth().padding(top = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(7.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        NButton(
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

    // No live transcript panel.
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
        NHelp(
            "Saved to the library. Open it there to export as TXT, SRT, VTT or JSON into a " +
                "folder of your choosing.",
            Modifier.padding(top = 10.dp),
        )
    }
}

/** **Read aloud — SPEC §7.** A script, a voice, an expression, and a file at the end. */
@Composable
private fun SpeakPanel(
    state: ai.ondevice.ui.vm.VoiceState,
    viewModel: VoiceViewModel,
    onPickScript: () -> Unit,
    onPickReference: () -> Unit,
) {
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

    // OmniVoice takes direction inline, and there is no way to discover that from a text box.
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
            // The same shape as Transcribe's clip: a title row that carries the verbs as icons, and the clip itself playable underneath.
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
    // Rendering still writes the WAV — it has to, the library lists it — but the two buttons that used to sit here are gone.
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

/** Five. */
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

/** Which engine, which model, which voice, and how it is shaped. */
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

        // A switch rather than an automatic choice.
        val provider = state.selectedVoice?.provider ?: ai.ondevice.speech.SynthProvider.SYSTEM
        val engines = ai.ondevice.speech.SynthProvider.entries
        SectionKicker("Engine", Modifier.padding(bottom = 8.dp))
        NSeg(
            options = engines.map { it.label },
            selectedIndex = engines.indexOf(provider).coerceAtLeast(0),
            onSelect = { viewModel.selectProvider(engines[it]) },
            modifier = Modifier.fillMaxWidth(),
        )

        // Which *model* provides the engine, as distinct from which engine.
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

        // — the voice — Counted over this engine's voices, since those are the only ones listed.
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

        // — expression — Which dials exist depends on the engine, because the two engines genuinely differ.
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

        // — blend, Kokoro only — Gated on Kokoro being *selected*, not merely installed.
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

        // Each engine's own parameter set.
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
