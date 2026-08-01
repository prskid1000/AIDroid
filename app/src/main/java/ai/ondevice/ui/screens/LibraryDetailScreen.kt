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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.core.PredictionKind
import ai.ondevice.core.SparseParams
import ai.ondevice.core.TranscriptSegments
import ai.ondevice.core.displayValue
import ai.ondevice.ui.components.NAudioPlayer
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NDropdown
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NIconButton
import ai.ondevice.ui.components.NTable
import ai.ondevice.ui.components.NTableRow
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.ResourceDetail
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.LibraryDetailState
import ai.ondevice.ui.vm.LibraryDetailViewModel

/** One library item, opened. */
@Composable
fun LibraryDetailScreen(
    onBack: () -> Unit,
    onOpenChat: () -> Unit,
    onOpenImage: () -> Unit,
    onOpenVoice: () -> Unit,
    viewModel: LibraryDetailViewModel = hiltViewModel(),
    chatViewModel: ai.ondevice.ui.vm.ChatViewModel = activityChatViewModel(),
    imageViewModel: ai.ondevice.ui.vm.ImageViewModel = activityImageViewModel(),
    voiceViewModel: ai.ondevice.ui.vm.VoiceViewModel = activityVoiceViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val pickFolder = ai.ondevice.ui.rememberFolderPicker(viewModel::folderPicked)

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = state.title.ifBlank { state.kind.label },
                onBack = onBack,
                subtitle = if (state.createdAt > 0) Fmt.relative(state.createdAt) else null,
                subtitleMono = false,
                trailing = {
                    NIconButton(
                        NIcons.Trash,
                        "Delete",
                        onClick = { viewModel.delete(onDone = onBack) },
                        size = 34.dp,
                        iconSize = 15.dp,
                    )
                },
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        if (state.missing) {
            NHelp("This item is no longer here. It was deleted somewhere else.")
            return@PhoneScaffold
        }

        Column(Modifier.verticalScroll(rememberScrollState())) {
            when (state.kind) {
                PredictionKind.CHAT -> ConversationDetail(state, onOpen = {
                    state.conversation?.let { chatViewModel.openConversation(it.id) }
                    onOpenChat()
                })

                PredictionKind.IMAGE -> ImageDetail(
                    state = state,
                    onOpen = {
                        state.image?.let { imageViewModel.reuseParameters(it) }
                        onOpenImage()
                    },
                )

                PredictionKind.SPEECH -> SynthesisDetail(
                    state = state,
                    onOpen = {
                        state.synthesis?.let { voiceViewModel.loadSynthesis(it) }
                        onOpenVoice()
                    },
                )

                PredictionKind.TRANSCRIBE -> TranscriptDetail(state, onOpen = {
                    state.transcript?.let { voiceViewModel.loadTranscript(it) }
                    onOpenVoice()
                })
            }

            ExportSection(state, viewModel, pickFolder)
            ParameterTable(state)
            RunSection(state)
        }
    }
}

// — the part that differs per kind —

@Composable
private fun ConversationDetail(state: LibraryDetailState, onOpen: () -> Unit) {
    val conversation = state.conversation ?: return
    NButton("Open in Chat", onClick = onOpen, style = NButtonStyle.Primary, block = true)

    SectionKicker("Thread", Modifier.padding(top = 18.dp, bottom = 8.dp))
    NTable {
        val assistant = state.messages.count { it.role == ai.ondevice.core.MessageRole.ASSISTANT }
        DetailRow("messages", "${state.messages.size} · $assistant from the model")
        // No model row here.
        DetailRow("updated", Fmt.relative(conversation.updatedAt))
        // The measured rate across every turn in the thread, weighted by nothing — each turn's own figure, averaged.
        val rates = state.messages.mapNotNull { it.tokensPerSecond?.takeIf { rate -> rate > 0 } }
        if (rates.isNotEmpty()) {
            DetailRow("speed", "${Fmt.tokensPerSecond(rates.average().toFloat())} over ${rates.size} turns")
        }
    }

    conversation.systemPrompt?.takeIf { it.isNotBlank() }?.let { prompt ->
        SectionKicker("System prompt", Modifier.padding(top = 18.dp, bottom = 8.dp))
        Text(
            prompt,
            style = NocturneType.MonoCode,
            color = NocturneColors.Text.copy(alpha = 0.72f),
            modifier = Modifier
                .fillMaxWidth()
                .background(NocturneColors.Neutral900, Radius.Md)
                .padding(horizontal = 11.dp, vertical = 9.dp),
        )
    }
}

@Composable
private fun ImageDetail(state: LibraryDetailState, onOpen: () -> Unit) {
    val image = state.image ?: return
    Box(
        Modifier
            .fillMaxWidth()
            .aspectRatio(image.width.toFloat() / image.height.coerceAtLeast(1))
            .clip(Radius.Md)
            .background(NocturneColors.Neutral900)
            .ring(NocturneColors.Divider, Radius.Md),
        contentAlignment = Alignment.Center,
    ) {
        // The file itself.
        coil3.compose.AsyncImage(
            model = image.path,
            contentDescription = image.prompt,
            contentScale = ContentScale.Fit,
            modifier = Modifier.fillMaxSize(),
        )
    }

    Row(
        Modifier.fillMaxWidth().padding(top = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        NButton(
            "Open in Image",
            onClick = onOpen,
            style = NButtonStyle.Primary,
            modifier = Modifier.weight(1f),
        )
    }

    SectionKicker("Prompt", Modifier.padding(top = 18.dp, bottom = 8.dp))
    Text(image.prompt, style = NocturneType.Message)
    image.negativePrompt?.takeIf { it.isNotBlank() }?.let {
        Text(
            "negative: $it",
            style = NocturneType.MonoXs,
            color = NocturneColors.TextMuted,
            modifier = Modifier.padding(top = 6.dp),
        )
    }
}

@Composable
private fun SynthesisDetail(state: LibraryDetailState, onOpen: () -> Unit) {
    val synthesis = state.synthesis ?: return

    // The point of opening a synthesis is to hear it. Until now this screen
    // described a WAV in a table and offered no way to play it.
    NAudioPlayer(
        file = java.io.File(synthesis.path),
        label = synthesis.path.substringAfterLast('/'),
        modifier = Modifier.padding(bottom = 10.dp),
    )

    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        NButton(
            "Open in Voice",
            onClick = onOpen,
            style = NButtonStyle.Primary,
            modifier = Modifier.weight(1f),
        )
    }

    SectionKicker("Take", Modifier.padding(top = 18.dp, bottom = 8.dp))
    NTable {
        DetailRow("engine", synthesis.engineId)
        DetailRow("voice", synthesis.voice ?: "—")
        DetailRow("duration", Fmt.duration(synthesis.durationMillis, tenths = true))
        DetailRow("sample rate", "${synthesis.sampleRate} Hz")
        DetailRow("file", synthesis.path.substringAfterLast('/'))
    }

    SectionKicker("Script", Modifier.padding(top = 18.dp, bottom = 8.dp))
    Text(
        synthesis.text,
        style = NocturneType.Message,
        modifier = Modifier
            .fillMaxWidth()
            .background(NocturneColors.Neutral900, Radius.Md)
            .padding(horizontal = 11.dp, vertical = 9.dp),
    )
}

@Composable
private fun TranscriptDetail(state: LibraryDetailState, onOpen: () -> Unit) {
    val transcript = state.transcript ?: return
    val segments = TranscriptSegments.parse(transcript.segmentsJson)

    NButton("Open in Voice", onClick = onOpen, style = NButtonStyle.Primary, block = true)
    NHelp(
        "Opens in Transcribe with these segments loaded, where they can be exported as TXT, " +
            "SRT, VTT or JSON.",
        Modifier.padding(top = 6.dp),
    )

    // Present for a file transcription, and for a recording made since the capture path started keeping its WAV.
    transcript.sourcePath?.let { path ->
        val audio = java.io.File(path)
        if (audio.isFile) {
            NAudioPlayer(
                file = audio,
                label = path.substringAfterLast('/'),
                modifier = Modifier.padding(top = 12.dp),
            )
        }
    }

    SectionKicker("Recording", Modifier.padding(top = 18.dp, bottom = 8.dp))
    NTable {
        DetailRow("duration", Fmt.duration(transcript.durationMillis, tenths = true))
        DetailRow("segments", segments.size.toString())
        DetailRow("source", transcript.sourcePath?.substringAfterLast('/') ?: "microphone")
    }

    SectionKicker("Transcript", Modifier.padding(top = 18.dp, bottom = 8.dp))
    Column(
        Modifier
            .fillMaxWidth()
            .background(NocturneColors.Neutral900, Radius.Md)
            .padding(horizontal = 11.dp, vertical = 9.dp),
        verticalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        if (segments.isEmpty()) {
            Text("No segments were recorded.", style = NocturneType.Row, color = NocturneColors.TextMuted)
        }
        segments.forEach { segment ->
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                // A live capture has no boundaries to report, so its cue times are genuinely zero and saying "0:00" for every line would be noise dressed as data.
                if (segment.endMillis > 0) {
                    Text(
                        Fmt.duration(segment.startMillis),
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMeta,
                    )
                }
                Text(segment.text, style = NocturneType.Row, modifier = Modifier.weight(1f))
            }
        }
    }
}

// — the part every kind shares —

@Composable
private fun ParameterTable(state: LibraryDetailState) {
    val params = state.params
    if (params.isEmpty && state.modelId == null) return
    SectionKicker("Parameters", Modifier.padding(top = 20.dp, bottom = 8.dp))
    NTable {
        state.modelId?.let { DetailRow("model", it) }
        params.keys
            .filterNot { it == "prompt" || it == "negative_prompt" }
            .sorted()
            .forEach { key -> DetailRow(key, params[key]?.displayValue() ?: "—") }
    }
    if (state.kind == PredictionKind.IMAGE) {
        NHelp(
            "This set is also written into the PNG itself, so the file reproduces the image " +
                "without the app.",
            Modifier.padding(top = 8.dp),
        )
    }
}

/** What it cost to make. */
@Composable
private fun RunSection(state: LibraryDetailState) {
    val traces = state.traces
    if (traces.isEmpty()) {
        SectionKicker("Cost", Modifier.padding(top = 20.dp, bottom = 8.dp))
        NHelp(
            "No usage was recorded for this one. Traces are kept from the version that " +
                "introduced them onwards; anything made before has none.",
        )
        return
    }
    SectionKicker(
        if (traces.size == 1) "Cost" else "Cost · ${traces.size} runs",
        Modifier.padding(top = 20.dp, bottom = 8.dp),
    )
    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        traces.forEach { trace -> ResourceDetail(trace) }
    }
}

@Composable
private fun DetailRow(key: String, value: String) {
    NTableRow {
        Text(
            key,
            style = NocturneType.Row,
            color = NocturneColors.TextMuted,
            modifier = Modifier.weight(0.44f),
        )
        Text(value, style = NocturneType.MonoValue, modifier = Modifier.weight(0.56f))
    }
}

/** Share the file when it exists, and its parameter set when it does not. */
internal fun shareImage(
    context: android.content.Context,
    image: ai.ondevice.data.db.GeneratedImageEntity,
) {
    val file = java.io.File(image.path)
    val params = SparseParams.parse(image.paramsJson)
    val text = buildString {
        appendLine(image.prompt)
        image.negativePrompt?.let { appendLine("negative: $it") }
        appendLine("seed: ${image.seed} · ${image.width}×${image.height}")
        image.modelId?.let { appendLine("model: $it") }
        params.keys.sorted().forEach { key ->
            if (key != "prompt" && key != "negative_prompt" && key != "seed") {
                appendLine("$key: ${params[key]?.displayValue() ?: "—"}")
            }
        }
    }

    val intent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
        putExtra(android.content.Intent.EXTRA_TEXT, text)
        if (file.exists()) {
            val uri = androidx.core.content.FileProvider.getUriForFile(
                context,
                "${context.packageName}.fileprovider",
                file,
            )
            type = "image/png"
            putExtra(android.content.Intent.EXTRA_STREAM, uri)
            addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } else {
            type = "text/plain"
        }
    }
    context.startActivity(android.content.Intent.createChooser(intent, "Share image"))
}

/** Save and Share, for every kind of artifact, in one place. */
@Composable
private fun ExportSection(
    state: LibraryDetailState,
    viewModel: LibraryDetailViewModel,
    onNeedFolder: () -> Unit,
) {
    val context = LocalContext.current

    SectionKicker("Export", Modifier.padding(top = 18.dp, bottom = 8.dp))

    // Only where there is a genuine choice. A picture is a PNG and a synthesis
    // is a WAV; a dropdown with one entry is furniture.
    if (viewModel.formats.size > 1) {
        NDropdown(
            options = viewModel.formats,
            selected = state.format,
            onSelect = viewModel::selectFormat,
            modifier = Modifier.padding(bottom = 8.dp),
        )
    }

    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        NButton(
            "Save",
            onClick = { viewModel.save(onNeedFolder = onNeedFolder) },
            style = NButtonStyle.Primary,
            modifier = Modifier.weight(1f),
        )
        NButton(
            "Share",
            onClick = { viewModel.share { ai.ondevice.ui.shareExport(context, it) } },
            style = NButtonStyle.Secondary,
            modifier = Modifier.weight(1f),
        )
    }

    // Where it went, named. A save that reports nothing but success is the
    // failure this change exists to end.
    state.exportMessage?.let { message ->
        NHelp(message, Modifier.padding(top = 8.dp))
    }
}
