package ai.ondevice.ui.screens

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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.ui.BottomDestinations
import ai.ondevice.ui.components.NBottomBar
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCardMeta
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NMetaText
import ai.ondevice.ui.components.NPills
import ai.ondevice.ui.components.NProgressBar
import ai.ondevice.ui.components.NSlider
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.RootToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ruleBelow
import ai.ondevice.ui.vm.VoiceMode
import ai.ondevice.ui.vm.VoiceViewModel

/**
 * **S14 — Voice.**
 *
 * Live and file transcription (whisper.cpp, SPEC §6) plus the Kokoro read-aloud
 * panel (§7). The detail the canvas calls out is the confidence shading: a
 * partial transcript fades by per-token confidence, and the caption says
 * plainly that faded text may still change as the window slides. That is the
 * honest-refusal principle applied to a streaming decoder.
 */
@Composable
fun VoiceScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    viewModel: VoiceViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = {
            RootToolbar("Voice") {
                state.sttModel?.let { NTag(it.displayName, style = NTagStyle.Neutral) }
            }
        },
        bottomBar = { NBottomBar(BottomDestinations, currentRoute) { onNavigate(it.route) } },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        NPills(
            options = VoiceMode.entries.map { it.label },
            selectedIndex = VoiceMode.entries.indexOf(state.mode),
            onSelect = { viewModel.setMode(VoiceMode.entries[it]) },
            modifier = Modifier.padding(bottom = 10.dp),
        )

        Column(Modifier.verticalScroll(rememberScrollState())) {

            when (state.mode) {
                VoiceMode.LIVE -> {
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
                        if (state.recording) "Stop and keep" else "Start recording",
                        onClick = { if (state.recording) viewModel.stopRecording() else viewModel.startRecording() },
                        style = NButtonStyle.Primary,
                        block = true,
                        minHeight = 48.dp,
                        modifier = Modifier.padding(top = 12.dp),
                    )
                }

                VoiceMode.FILE -> {
                    NCard(gap = 8.dp) {
                        Text("standup-2026-07-28.m4a", style = NocturneType.CardTitleSm)
                        Text(
                            "18:42 · 21.4 MB · shared from Recorder",
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
                            NMetaText("13:51 of 18:42")
                            Box(Modifier.weight(1f))
                            NMetaText("4.1× realtime")
                        }
                    }

                    SampleSegments.forEachIndexed { index, (timestamp, text) ->
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .ruleBelow()
                                .padding(vertical = 10.dp)
                                .alpha(if (index == SampleSegments.lastIndex) 0.5f else 1f),
                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                        ) {
                            Text(
                                timestamp,
                                style = NocturneType.MonoTimestamp,
                                color = if (index == SampleSegments.lastIndex) {
                                    NocturneColors.TextMuted
                                } else {
                                    NocturneColors.Accent300
                                },
                                modifier = Modifier.width(52.dp),
                            )
                            Text(text, style = NocturneType.Row, modifier = Modifier.weight(1f))
                        }
                    }

                    Row(
                        Modifier.fillMaxWidth().padding(top = 12.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        listOf("TXT", "SRT", "VTT", "JSON").forEach { format ->
                            NButton(
                                format,
                                onClick = { },
                                modifier = Modifier.weight(1f),
                                minHeight = 42.dp,
                            )
                        }
                    }
                }
            }

            // — Kokoro read-aloud (SPEC §7) —
            SectionKicker("Read aloud · Kokoro", Modifier.padding(top = 20.dp, bottom = 8.dp))
            NCard(gap = 9.dp) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(9.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box(
                        Modifier.size(30.dp).background(NocturneColors.Accent800, CircleShape),
                        contentAlignment = Alignment.Center,
                    ) {
                        Icon(
                            NIcons.Play,
                            contentDescription = "Preview voice",
                            tint = NocturneColors.Accent100,
                            modifier = Modifier.size(14.dp),
                        )
                    }
                    Column(Modifier.weight(1f)) {
                        Text(state.voice, style = NocturneType.CardTitleSm)
                        Text(
                            voiceDescription(state.voice, state.speed),
                            style = NocturneType.Help,
                            color = NocturneColors.TextMuted,
                        )
                    }
                    NTag("blend", style = NTagStyle.Outline)
                }
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(7.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("Speed", style = NocturneType.Row, modifier = Modifier.weight(1f))
                    Text(
                        String.format("%.2f×", state.speed),
                        style = NocturneType.MonoValue,
                        color = NocturneColors.Accent300,
                    )
                }
                NSlider(
                    value = state.speed,
                    onValueChange = viewModel::setSpeed,
                    valueRange = 0.5f..2f,
                )
                Text(
                    "Playback starts on the first synthesised chunk, not after the whole passage. " +
                        "54 voices, each with a preview.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            }
        }
    }
}

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

private val SampleSegments = listOf(
    "00:12.4" to "Right, the Hexagon path is capped at three and a half gigs a session.",
    "00:19.1" to "So anything bigger has to be layer-split, or we just fall back to OpenCL.",
    "00:27.8" to "Falling back is fine as long as we say we fell back.",
    "00:34.2" to "Yeah — and the number goes in the model sheet, not a toast.",
)
