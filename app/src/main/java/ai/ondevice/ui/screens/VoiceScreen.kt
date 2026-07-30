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
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCardMeta
import ai.ondevice.ui.components.NDropdown
import ai.ondevice.ui.components.NEnumRow
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NMetaText
import ai.ondevice.ui.components.NPills
import ai.ondevice.ui.components.NProgressBar
import ai.ondevice.ui.components.NSlider
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.NTextArea
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.RootToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.theme.ruleBelow
import ai.ondevice.ui.components.NSeg
import ai.ondevice.ui.vm.SpeakSource
import ai.ondevice.ui.vm.TranscribeSource
import ai.ondevice.ui.vm.VoiceMode
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
    onOpenAdvanced: () -> Unit,
    viewModel: VoiceViewModel = hiltViewModel(),
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
    ) { uri -> uri?.let(viewModel::transcribeFile) }
    val pickAudio = { audioLauncher.launch(arrayOf("audio/*", "video/*")) }

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

    PhoneScaffold(
        toolbar = {
            // No engine tag up here. Which model does the work depends on the
            // tab, so a single toolbar badge was either wrong on one of them or
            // duplicating the card below it. Each tab names its own engine in
            // its own panel instead, in the same shape, so the two tabs read
            // alike.
            RootToolbar("Voice")
        },
        bottomBar = { NBottomBar(BottomDestinations, currentRoute) { onNavigate(it.route) } },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        NPills(
            options = VoiceMode.entries.map { it.label },
            selectedIndex = VoiceMode.entries.indexOf(state.mode),
            onSelect = { viewModel.setMode(VoiceMode.entries[it]) },
            modifier = Modifier.padding(bottom = 8.dp),
        )

        // The input source, one level down from the mode. Both tabs have the
        // same two answers — live from the device, or from a file — so they
        // read as the mirror image they are.
        NSeg(
            options = when (state.mode) {
                VoiceMode.TRANSCRIBE -> TranscribeSource.entries.map { it.label }
                VoiceMode.SPEAK -> SpeakSource.entries.map { it.label }
            },
            selectedIndex = when (state.mode) {
                VoiceMode.TRANSCRIBE -> TranscribeSource.entries.indexOf(state.source)
                VoiceMode.SPEAK -> SpeakSource.entries.indexOf(state.speakSource)
            },
            onSelect = { index ->
                when (state.mode) {
                    VoiceMode.TRANSCRIBE -> viewModel.setSource(TranscribeSource.entries[index])
                    VoiceMode.SPEAK -> viewModel.setSpeakSource(SpeakSource.entries[index])
                }
            },
            modifier = Modifier.fillMaxWidth().padding(bottom = 10.dp),
        )

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

            when {
                state.mode == VoiceMode.TRANSCRIBE &&
                    state.source == TranscribeSource.MICROPHONE -> {
                    // The waveform: one bar per window slot, accent where the
                    // VAD says speech, neutral where it doesn't.
                    Row(
                        Modifier.fillMaxWidth().height(56.dp).padding(horizontal = 2.dp),
                        horizontalArrangement = Arrangement.spacedBy(2.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        state.waveform.forEach { level ->
                            Box(
                                Modifier
                                    .weight(1f)
                                    .fillMaxHeight(level.coerceIn(0.05f, 1f))
                                    .background(
                                        if (level > 0.45f) NocturneColors.Accent500 else NocturneColors.Neutral700,
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
                            if (state.recording) {
                                "● REC ${Fmt.duration(state.elapsedMillis)}"
                            } else {
                                "○ idle"
                            },
                            style = NocturneType.MonoSm,
                            color = if (state.recording) NocturneColors.Accent else NocturneColors.TextMuted,
                        )
                        Text(
                            "VAD ${if (state.vadEnabled) "on" else "off"} · step ${Fmt.grouped(state.stepMs)} ms",
                            style = NocturneType.MonoSm,
                            color = NocturneColors.TextMuted,
                        )
                    }

                    Box(
                        Modifier
                            .fillMaxWidth()
                            .padding(top = 12.dp)
                            .background(NocturneColors.Surface, Radius.Md)
                            .padding(12.dp),
                    ) {
                        if (state.partial.isEmpty()) {
                            Text(
                                "Start recording and partials appear here, shaded by confidence.",
                                style = NocturneType.Message,
                                color = NocturneColors.TextMuted,
                            )
                        } else {
                            Text(
                                buildAnnotatedTranscript(state.partial),
                                style = NocturneType.Message,
                            )
                        }
                    }

                    NHelp(
                        "Opacity tracks per-token confidence. Faded text may still change as the " +
                            "window slides.",
                        Modifier.padding(top = 8.dp),
                    )

                    NButton(
                        when {
                            state.loading -> "Loading model…"
                            state.recording -> "Stop and keep"
                            else -> "Start recording"
                        },
                        onClick = {
                            when {
                                state.recording -> viewModel.stopRecording()
                                // The permission is asked for at the moment it
                                // is needed, with the reason on screen — not at
                                // launch, before the user has any context.
                                !hasMicPermission -> micPermission.launch(Manifest.permission.RECORD_AUDIO)
                                else -> viewModel.startRecording()
                            }
                        },
                        style = NButtonStyle.Primary,
                        block = true,
                        minHeight = 48.dp,
                        modifier = Modifier.padding(top = 12.dp),
                    )
                }

                state.mode == VoiceMode.TRANSCRIBE -> {
                    if (state.segments.isEmpty()) {
                        NHelp(
                            "Pick an audio file and whisper.cpp transcribes it on this device. " +
                                "Anything Android can decode works — m4a, mp3, wav, opus.",
                            Modifier.padding(bottom = 10.dp),
                        )
                    } else {
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
                    }

                    NButton(
                        if (state.loading) "Transcribing…" else "Choose an audio file",
                        onClick = pickAudio,
                        style = NButtonStyle.Primary,
                        block = true,
                        minHeight = 46.dp,
                        modifier = Modifier.padding(bottom = 8.dp),
                    )

                    // The transcript panel is always here, the way Microphone's
                    // is. It used to appear only once segments existed, so File
                    // mode jumped from a line of help text to four live export
                    // buttons with nothing to export — the one mode that reads
                    // its confidence off a finished decode looked like it did not
                    // report confidence at all.
                    if (state.segments.isEmpty()) {
                        Box(
                            Modifier
                                .fillMaxWidth()
                                .background(NocturneColors.Surface, Radius.Md)
                                .padding(12.dp),
                        ) {
                            Text(
                                "Choose a file and its segments appear here with timings, shaded by " +
                                    "confidence.",
                                style = NocturneType.Message,
                                color = NocturneColors.TextMuted,
                            )
                        }
                    }

                    state.segments.forEachIndexed { index, segment ->
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .ruleBelow()
                                .padding(vertical = 10.dp)
                                // Faded by the decoder's own confidence, not by
                                // position — a late segment the model is sure of
                                // reads at full strength.
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

                    if (state.segments.isNotEmpty()) {
                        NHelp(
                            "Opacity tracks the decoder's own confidence for each segment.",
                            Modifier.padding(top = 8.dp),
                        )
                    }

                    // Exports only once there is a transcript behind them.
                    if (state.segments.isNotEmpty()) Row(
                        Modifier.fillMaxWidth().padding(top = 12.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        TranscriptFormat.entries.forEach { format ->
                            NButton(
                                format.label,
                                onClick = {
                                    viewModel.export(format) { file ->
                                        shareTranscript(context, file, format)
                                    }
                                },
                                modifier = Modifier.weight(1f),
                                minHeight = 42.dp,
                            )
                        }
                    }
                }

                else -> SpeakPanel(
                    state = state,
                    viewModel = viewModel,
                    onPickScript = pickScript,
                    onShareAudio = { file -> shareAudio(context, file) },
                    onOpenAdvanced = onOpenAdvanced,
                )
            }
        }
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
    onShareAudio: (java.io.File) -> Unit,
    onOpenAdvanced: () -> Unit,
) {
    // — which engine is actually speaking —
    //
    // A switch rather than an automatic choice. The two neural engines are not
    // interchangeable: Kokoro is six or seven times faster, OmniVoice can do
    // things Kokoro cannot do at all. Picking on the user's behalf would mean
    // either surprising them with a minute of compute or silently dropping the
    // feature they came for.
    val provider = state.selectedVoice?.provider ?: ai.ondevice.speech.SynthProvider.SYSTEM
    val engines = ai.ondevice.speech.SynthProvider.entries
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
                when (provider) {
                    ai.ondevice.speech.SynthProvider.KOKORO -> "Kokoro · on-device neural"
                    ai.ondevice.speech.SynthProvider.OMNIVOICE -> "OmniVoice · expressive, slow"
                    ai.ondevice.speech.SynthProvider.SYSTEM -> "System engine"
                },
                style = NocturneType.CardTitleSm,
                modifier = Modifier.weight(1f),
            )
            NTag(
                if (provider == ai.ondevice.speech.SynthProvider.SYSTEM) "fallback" else "neural",
                style = if (provider == ai.ondevice.speech.SynthProvider.SYSTEM) NTagStyle.Outline else NTagStyle.Accent,
            )
        }

        NSeg(
            options = engines.map { it.label },
            selectedIndex = engines.indexOf(provider).coerceAtLeast(0),
            onSelect = { viewModel.selectProvider(engines[it]) },
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
        )

        // Which *model* provides the engine, as distinct from which engine. Two
        // are ordinary — Kokoro and OmniVoice are both voice models — and until
        // now the app picked by scanning directories and never said which it had
        // landed on.
        //
        // Only the models this engine can load, and only when the engine loads
        // one at all: Android's synthesiser is part of the OS and has no file to
        // choose, so a picker over it is a control with nothing behind it. The
        // unfiltered list was worse than useless — the OmniVoice tab listed
        // Kokoro and showed it as the selection, which is a promise the engine
        // cannot keep.
        val engineModels = state.ttsModels.filter {
            state.ttsModelProviders[it.id] == provider
        }
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

        Text(
            when (provider) {
                ai.ondevice.speech.SynthProvider.KOKORO ->
                    "Fast enough to use for anything: about half a second of work per second of " +
                        "speech. Fifty voices, six languages, no emotion tags."
                ai.ondevice.speech.SynthProvider.OMNIVOICE ->
                    "Reads any language with no phonemiser, takes [laughter] and [sigh], and can " +
                        "voice several speakers. It is six to seven times slower than Kokoro — " +
                        "expect around a minute of work for a short sentence on a phone."
                ai.ondevice.speech.SynthProvider.SYSTEM ->
                    "Android's own synthesiser. A different voice with different prosody — the app " +
                        "says so rather than passing it off as one of the neural engines."
            },
            style = NocturneType.CardBody,
            color = NocturneColors.Text.copy(alpha = 0.8f),
        )

        if (!state.omniVoiceAvailable && provider != ai.ondevice.speech.SynthProvider.OMNIVOICE) {
            NHelp("OmniVoice is not installed — Add model lists it under Voice, about 683 MB.")
        }
    }

    // — the script —
    SectionKicker("Script", Modifier.padding(top = 18.dp, bottom = 8.dp))
    if (state.speakSource == ai.ondevice.ui.vm.SpeakSource.FILE) {
        NButton(
            if (state.scriptSource != null) "Replace script file" else "Choose a script file",
            onClick = onPickScript,
            style = NButtonStyle.Primary,
            block = true,
            minHeight = 46.dp,
        )
        // The loaded text is still shown and still editable. A file you cannot
        // correct a typo in before it is read aloud is a worse file.
        if (state.script.isNotEmpty()) {
            NTextArea(
                value = state.script,
                onValueChange = viewModel::setScript,
                minHeight = 130.dp,
                textStyle = NocturneType.Row,
                modifier = Modifier.padding(top = 8.dp),
            )
        }
    } else {
        NTextArea(
            value = state.script,
            onValueChange = viewModel::setScript,
            placeholder = "Type or paste what should be read aloud.",
            minHeight = 130.dp,
            textStyle = NocturneType.Row,
        )
        NButton(
            "Clear",
            onClick = { viewModel.setScript("") },
            block = true,
            minHeight = 44.dp,
            modifier = Modifier.padding(top = 8.dp),
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

    // — the voice —
    SectionKicker(
        "Voice · ${state.voices.count { it.available }} available of ${state.voices.size}",
        Modifier.padding(top = 20.dp, bottom = 8.dp),
    )
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
                    Text(
                        voice.localeLabel,
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMuted,
                    )
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
    // Which dials exist depends on the engine, because the two engines genuinely
    // differ. Kokoro's graph takes token ids, a style vector and a speed — there
    // is no pitch input at all. Showing a Pitch slider for it would leave the
    // user adjusting something inert, so it is absent rather than ignored.
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
                "Speed is applied during synthesis, not by resampling, so the voice does not turn " +
                    "chipmunk when you raise it. Kokoro has no pitch input, so there is no pitch " +
                    "dial here — the system engine does, and shows one."
            } else {
                "Speed and pitch are applied by the engine, not by resampling, so the voice does " +
                    "not turn chipmunk when you raise it."
            },
            style = NocturneType.CardBody,
            color = NocturneColors.Text.copy(alpha = 0.8f),
        )
    }

    // — blend, Kokoro only —
    if (state.kokoroAvailable) {
        SectionKicker("Blend", Modifier.padding(top = 20.dp, bottom = 8.dp))
        NCard(gap = 6.dp) {
            Text("Mix a second voice", style = NocturneType.CardTitleSm)
            NEnumRow(
                options = state.voices.filter { it.provider == ai.ondevice.speech.SynthProvider.KOKORO }
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
        if (state.speaking) "Stop" else "Read aloud",
        onClick = { if (state.speaking) viewModel.stopSpeaking() else viewModel.speak() },
        style = if (state.speaking) NButtonStyle.Secondary else NButtonStyle.Primary,
        block = true,
        modifier = Modifier.padding(top = 16.dp),
    )
    Row(
        Modifier.fillMaxWidth().padding(top = 7.dp),
        horizontalArrangement = Arrangement.spacedBy(7.dp),
    ) {
        NButton(
            if (state.rendering) "Rendering…" else "Save as WAV",
            onClick = { viewModel.exportAudio { } },
            modifier = Modifier.weight(1f),
            minHeight = 44.dp,
        )
        NButton(
            "Send audio",
            onClick = { viewModel.exportAudio(onShareAudio) },
            modifier = Modifier.weight(1f),
            minHeight = 44.dp,
        )
    }
    state.lastAudioPath?.let {
        NHelp(
            "Saved to ${it.substringAfterLast('/')} in the transcripts folder — an ordinary file you " +
                "can open in any player.",
            Modifier.padding(top = 6.dp),
        )
    }

    NButton(
        "Advanced · language, chunking, trim, gain",
        onClick = onOpenAdvanced,
        style = NButtonStyle.Secondary,
        block = true,
        modifier = Modifier.padding(top = 14.dp),
    )
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

/** Per-token confidence expressed as opacity — value, not hue. */
private fun buildAnnotatedTranscript(
    partials: List<ai.ondevice.ui.vm.PartialSegment>,
): AnnotatedString = buildAnnotatedString {
    partials.forEachIndexed { index, segment ->
        withStyle(
            SpanStyle(color = NocturneColors.Text.copy(alpha = segment.confidence.coerceIn(0.25f, 1f))),
        ) {
            append(segment.text)
            if (index != partials.lastIndex) append(" ")
        }
    }
}

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
