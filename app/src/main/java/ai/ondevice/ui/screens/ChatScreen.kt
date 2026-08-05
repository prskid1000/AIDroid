package ai.ondevice.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.foundation.layout.widthIn
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.core.MessageRole
import ai.ondevice.core.SparseParams
import ai.ondevice.data.db.MessageEntity
import ai.ondevice.ui.BottomDestinations
import ai.ondevice.ui.components.NBottomBar
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCircleButton
import ai.ondevice.ui.components.NDot
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NProgressBar
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.ResourceBlock
import ai.ondevice.ui.components.RootToolbar
import ai.ondevice.ui.components.ToolbarAction
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.theme.ruleAbove
import ai.ondevice.ui.vm.ChatState
import ai.ondevice.ui.vm.ChatViewModel
import ai.ondevice.ui.vm.StreamingMessage

/** **S6 — Chat**, with **S7** as its settings sheet. */
@Composable
fun ChatScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onOpenParameters: (String?) -> Unit,
    onOpenPromptInspector: () -> Unit,
    onOpenModels: () -> Unit,
    // Activity-scoped so the prompt inspector inspects *this* conversation.
    viewModel: ChatViewModel = activityChatViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val listState = rememberLazyListState()
    var sheetOpen by rememberSaveable { mutableStateOf(false) }

    // One picker for all three kinds, and it takes several at once: a question
    // about two photographs is one message, not two.
    val context = LocalContext.current
    val attachLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments(),
    ) { uris ->
        uris.forEach { uri ->
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION,
                )
            }
            viewModel.attach(uri)
        }
    }
    // What this model can actually receive, not everything the composer knows how to file.
    val pickAttachment = {
        attachLauncher.launch(
            buildList {
                if (state.acceptsImages) add("image/*")
                add("text/*")
                add("application/pdf")
                add("application/json")
            }.toTypedArray(),
        )
    }

    // Import takes a zip written by this app's export.
    val importLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri -> uri?.let(viewModel::import) }

    LaunchedEffect(state.messages.size, state.streaming?.content) {
        val target = state.messages.size
        if (target > 0) listState.animateScrollToItem(target)
    }

    Box(Modifier.fillMaxWidth()) {
        PhoneScaffold(
            toolbar = {
                ChatToolbar(
                    state,
                    onOpenSettings = { sheetOpen = true },
                    onNewConversation = viewModel::startNewConversation,
                    onImport = { importLauncher.launch(arrayOf("application/zip", "*/*")) },
                )
            },
            bottomBar = {
                Column {
                    ChatComposer(
                        state = state,
                        onInputChange = viewModel::onInputChange,
                        onSend = viewModel::send,
                        onStop = viewModel::stop,
                        onAttach = { pickAttachment() },
                        onRemoveAttachment = viewModel::removeAttachment,
                    )
                    NBottomBar(BottomDestinations, currentRoute) { onNavigate(it.route) }
                }
            },
            contentPadding = PaddingValues(horizontal = 16.dp),
        ) {
            state.error?.let { error ->
                NCard(
                    Modifier.padding(vertical = 8.dp),
                    ring = NocturneColors.Neutral700,
                ) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            NIcons.InfoCircle,
                            contentDescription = null,
                            tint = NocturneColors.Neutral300,
                            modifier = Modifier.size(16.dp),
                        )
                        Text(error, style = NocturneType.CardTitleSm, modifier = Modifier.weight(1f))
                    }
                    state.errorSuggestion?.let {
                        Text(it, style = NocturneType.CardBody, color = NocturneColors.Text.copy(alpha = 0.8f))
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                        NButton("Dismiss", viewModel::dismissError, modifier = Modifier.weight(1f))
                        NButton("Models", onOpenModels, modifier = Modifier.weight(1f))
                    }
                }
            }

            if (state.messages.isEmpty() && state.streaming == null) {
                EmptyChat(state, onOpenModels)
            }

            LazyColumn(
                state = listState,
                modifier = Modifier.fillMaxWidth(),
                contentPadding = PaddingValues(top = 8.dp, bottom = 12.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                items(state.messages, key = { it.id }) { message ->
                    MessageBubble(
                        message = message,
                        expanded = message.id in state.expandedThinking,
                        onToggleThinking = { viewModel.toggleThinking(message.id) },
                        onRegenerate = { viewModel.regenerate(message) },
                        onInspect = onOpenPromptInspector,
                        trace = state.traces[message.id],
                    )
                }
                state.streaming?.let { streaming ->
                    item(key = "streaming") {
                        StreamingBubble(
                            streaming = streaming,
                            expanded = streaming.id in state.expandedThinking,
                            onToggleThinking = { viewModel.toggleThinking(streaming.id) },
                            tokensPerSecond = state.tokensPerSecond,
                            trace = state.liveTrace,
                        )
                    }
                }
                // A tool call can take seconds against a network the user believed was never touched.
                state.runningTool?.let { name ->
                    item(key = "running-tool") {
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .background(NocturneColors.Neutral900, Radius.Md)
                                .ring(NocturneColors.Divider, Radius.Md)
                                .padding(horizontal = 10.dp, vertical = 8.dp),
                            horizontalArrangement = Arrangement.spacedBy(7.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            NDot(color = NocturneColors.Accent)
                            Text(
                                "Running $name…",
                                style = NocturneType.Meta.copy(fontSize = NocturneType.Row.fontSize),
                                color = NocturneColors.Accent300,
                            )
                        }
                    }
                }
            }
        }

        if (sheetOpen) {
            ChatSettingsSheet(
                state = state,
                onDismiss = { sheetOpen = false },
                onSelectModel = viewModel::setModel,
                onSystemPromptChange = viewModel::setSystemPrompt,
                onLiveParam = viewModel::setLiveParam,
                onVisionEnabledChange = viewModel::setVisionEnabled,
                onOpenParameters = { onOpenParameters(state.model?.id) },
                onUnloadModel = { viewModel.unloadModel(); sheetOpen = false },
            )
        }
    }
}

/** The live context readout. Which model it is belongs in chat settings. */
@Composable
private fun ChatToolbar(
    state: ChatState,
    onOpenSettings: () -> Unit,
    onNewConversation: () -> Unit,
    onImport: () -> Unit,
) {
    RootToolbar(
        title = if (state.model == null) "No model" else "Chat",
        subtitle = {
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.alpha(0.6f),
            ) {
                NDot(
                    color = if (state.generating || state.loadingModel) {
                        NocturneColors.Accent
                    } else {
                        NocturneColors.Neutral500
                    },
                )
                Text(
                    // A context readout of a context that does not exist yet.
                    //
                    // Loading a few gigabytes is the longest wait in the app,
                    // and for all of it this said "0 / 8192 ctx" — a live
                    // figure about a model that is not there, next to a dot
                    // that says nothing is happening. Both were true and
                    // neither was the answer to "what is it doing".
                    when {
                        state.loadingModel -> "loading…"
                        state.loadedModelId == null ->
                            "not loaded · ${Fmt.grouped(state.contextLimit)} ctx"
                        else ->
                            "${Fmt.grouped(state.contextUsed)} / ${Fmt.grouped(state.contextLimit)} ctx"
                    },
                    style = NocturneType.NavLabel,
                )
            }
        },
    ) {
        // A new conversation, not a wiped one — the old thread stays whole in the library, which is why this is a plus rather than a bin.
        ToolbarAction(NIcons.Import, "Import a conversation", onImport)
        ToolbarAction(NIcons.Plus, "New conversation", onNewConversation)
        ToolbarAction(NIcons.Settings, "Chat settings", onOpenSettings)
    }
}

@Composable
private fun EmptyChat(state: ChatState, onOpenModels: () -> Unit) {
    // Three states, not two: nothing here, something on its way, something ready.
    val arriving = state.installing.takeIf { state.model == null }.orEmpty()
    NCard(Modifier.padding(top = 12.dp)) {
        Text(
            when {
                arriving.isNotEmpty() -> "Downloading"
                state.model == null -> "No model loaded"
                else -> "Ready"
            },
            style = NocturneType.CardTitleSm,
        )
        if (arriving.isNotEmpty()) {
            arriving.forEach { job ->
                Text(job.label, style = NocturneType.CardBody, color = NocturneColors.Accent200)
                NProgressBar(job.fraction)
            }
            Text(
                "The library row is written when a download starts, so it is here already — it " +
                    "becomes usable the moment the last byte verifies.",
                style = NocturneType.CardBody,
                color = NocturneColors.Text.copy(alpha = 0.8f),
            )
        } else {
            Text(
                if (state.model == null) {
                    "Nothing is loaded yet. Add a model and the chat runs entirely on this device — after " +
                        "download there is no network at all."
                } else {
                    "${state.model.label} is installed. Send a message and it loads on first use; " +
                        "the KV cache is reused across turns so follow-ups don't reprocess the prompt."
                },
                style = NocturneType.CardBody,
                color = NocturneColors.Text.copy(alpha = 0.8f),
            )
        }
        if (state.model == null) {
            NButton(
                if (arriving.isEmpty()) "Add a model" else "Open Models",
                onOpenModels,
                style = NButtonStyle.Primary,
                block = true,
            )
        }
    }
}

/** A user bubble, or an assistant reply with its thinking block and actions. */
/**
 * An attached document, as a row that can be opened rather than a wall of its
 * own text. The model still receives every byte of it.
 */
@Composable
private fun DocumentChip(document: ai.ondevice.core.MessageAttachments.Document) {
    var open by rememberSaveable(document.path) { mutableStateOf(false) }
    Column(
        Modifier
            .fillMaxWidth(0.8f)
            .background(NocturneColors.Neutral900, Radius.Md)
            .ring(NocturneColors.Divider, Radius.Md)
            .nClickableFlat { open = !open }
            .padding(horizontal = 11.dp, vertical = 9.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                NIcons.File,
                contentDescription = null,
                tint = NocturneColors.TextMuted,
                modifier = Modifier.size(14.dp),
            )
            Text(
                document.name,
                style = NocturneType.CardTitleSm,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )
            Text(
                "${Fmt.grouped(document.text.length)} chars",
                style = NocturneType.MonoXs,
                color = NocturneColors.TextMuted,
            )
            Icon(
                NIcons.ChevronDown,
                contentDescription = null,
                tint = NocturneColors.TextMuted,
                modifier = Modifier.size(14.dp).alpha(if (open) 1f else 0.5f),
            )
        }
        if (open) {
            Text(
                document.text.take(DOCUMENT_PREVIEW_CHARS) +
                    if (document.text.length > DOCUMENT_PREVIEW_CHARS) "\n…" else "",
                style = NocturneType.MonoXs,
                color = NocturneColors.Text.copy(alpha = 0.75f),
            )
        }
    }
}

/** Enough to recognise the file by, not enough to bury the conversation. */
private const val DOCUMENT_PREVIEW_CHARS = 2000

@Composable
private fun MessageBubble(
    message: MessageEntity,
    expanded: Boolean,
    onToggleThinking: () -> Unit,
    onRegenerate: () -> Unit,
    onInspect: () -> Unit,
    trace: ai.ondevice.engine.ResourceTrace?,
) {
    when (message.role) {
        MessageRole.USER -> Column(
            Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.End,
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (message.content.isNotBlank()) {
                Box(
                    Modifier
                        .fillMaxWidth(0.8f)
                        .background(
                            NocturneColors.Accent900,
                            RoundedCornerShape(14.dp, 14.dp, 4.dp, 14.dp),
                        )
                        .padding(horizontal = 12.dp, vertical = 9.dp),
                ) {
                    Text(message.content, style = NocturneType.Message)
                }
            }
            val attachments = ai.ondevice.core.MessageAttachments.of(message.imagePathsJson)
            attachments.documents.forEach { document ->
                DocumentChip(document)
            }
            val images = attachments.images
            images.forEach { path ->
                // The picture itself. The file name is a copy-in id and a
                // sanitised original, which reads as a hash and tells the reader
                // nothing they cannot see by looking.
                coil3.compose.AsyncImage(
                    model = path,
                    contentDescription = null,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier
                        .fillMaxWidth(0.8f)
                        .heightIn(max = 220.dp)
                        .clip(Radius.Md)
                        .background(NocturneColors.Neutral900, Radius.Md)
                        .ring(NocturneColors.Divider, Radius.Md),
                )
            }
            // Images consume context fast, and the cost is per turn, not per
            // image (SPEC §4.5).
            if (images.isNotEmpty()) {
                Text(
                    "${Fmt.grouped(message.imageTokenCount ?: 0)} image tokens",
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                )
            }
        }

        // A tool result is not the assistant speaking.
        MessageRole.TOOL_RESULT -> ToolResultBlock(message, expanded, onToggleThinking)

        else -> Column(
            Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            message.thinking?.let { thinking ->
                ThinkingBlock(
                    thinking = thinking,
                    millis = message.thinkingMillis,
                    tokens = message.thinkingTokens,
                    expanded = expanded,
                    onToggle = onToggleThinking,
                )
            }
            if (message.content.isNotBlank()) {
                Text(message.content, style = NocturneType.Message)
            }
            ToolCallList(message.toolCallsJson)
            MessageActions(
                tokensPerSecond = message.tokensPerSecond,
                onRegenerate = onRegenerate,
                onInspect = onInspect,
            )
            // Collapsed, and under the actions rather than above them: what a
            // reply cost is worth having and is not what you came to read.
            trace?.let {
                var traceExpanded by rememberSaveable(message.id) { mutableStateOf(false) }
                ResourceBlock(
                    trace = it,
                    expanded = traceExpanded,
                    onToggle = { traceExpanded = !traceExpanded },
                )
            }
        }
    }
}

/** What the model asked for, before anything ran it. */
@Composable
private fun ToolCallList(toolCallsJson: String?) {
    val calls = remember(toolCallsJson) { parseToolCalls(toolCallsJson) }
    if (calls.isEmpty()) return

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        calls.forEach { call -> ToolCallCard(call) }
    }
}

@Composable
private fun ToolCallCard(call: RenderedToolCall) {
    val hasArguments = call.arguments.isNotBlank() && call.arguments != "{}"
    var expanded by rememberSaveable(call.id, call.name) { mutableStateOf(false) }
    val preview = remember(call.arguments) { previewArguments(call.arguments) }
    val pretty = remember(call.arguments) { prettyArguments(call.arguments) }

    Column(
        Modifier
            .fillMaxWidth()
            .background(NocturneColors.Neutral900, Radius.Md)
            .ring(NocturneColors.Divider, Radius.Md),
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .then(if (hasArguments) Modifier.nClickableFlat(onClick = { expanded = !expanded }) else Modifier)
                .padding(horizontal = 10.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                NIcons.Activity,
                contentDescription = null,
                tint = NocturneColors.Accent,
                modifier = Modifier.size(13.dp),
            )
            Text(
                call.name,
                style = NocturneType.MonoCode,
                color = NocturneColors.Accent300,
            )
            if (hasArguments) {
                Text(
                    preview,
                    style = NocturneType.MonoCode,
                    color = NocturneColors.TextMuted,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f),
                )
                Text(
                    if (expanded) "⌃" else "⌄",
                    style = NocturneType.Row,
                    color = NocturneColors.Accent300,
                )
            }
        }
        if (expanded && hasArguments) {
            Text(
                pretty,
                style = NocturneType.MonoCode,
                color = NocturneColors.Text.copy(alpha = 0.72f),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 11.dp)
                    .padding(bottom = 9.dp),
            )
        }
    }
}

/** `timezone: Asia/Tokyo` — enough of the arguments to read at a glance. */
private fun previewArguments(raw: String): String = runCatching {
    kotlinx.serialization.json.Json.parseToJsonElement(raw).jsonObject.entries
        .joinToString(" · ") { (key, value) ->
            "$key: ${(value as? kotlinx.serialization.json.JsonPrimitive)?.content ?: value}"
        }
}.getOrDefault(raw).ifBlank { raw }

/** One argument per line. Models write these on one line; people don't read that way. */
private fun prettyArguments(raw: String): String = runCatching {
    kotlinx.serialization.json.Json.parseToJsonElement(raw).jsonObject.entries
        .joinToString("\n") { (key, value) -> "$key = $value" }
}.getOrDefault(raw).ifBlank { raw }

/** The result that came back, collapsed — they are routinely long. */
@Composable
private fun ToolResultBlock(
    message: MessageEntity,
    expanded: Boolean,
    onToggle: () -> Unit,
) {
    val meta = remember(message.toolCallsJson) { SparseParams.parse(message.toolCallsJson ?: "{}") }
    val name = meta.string("tool_name") ?: "tool"
    val isError = meta.bool("is_error") == true

    Column(
        Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .background(NocturneColors.Neutral900, Radius.Md)
                .ring(if (isError) NocturneColors.Neutral700 else NocturneColors.Divider, Radius.Md)
                .nClickableFlat(onClick = onToggle)
                .padding(horizontal = 10.dp, vertical = 7.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                if (isError) NIcons.TriangleAlert else NIcons.File,
                contentDescription = null,
                tint = if (isError) NocturneColors.Neutral300 else NocturneColors.Accent,
                modifier = Modifier.size(13.dp),
            )
            Text(
                if (isError) "$name failed" else message.content.lineSequence().firstOrNull()
                    ?.takeIf { it.isNotBlank() } ?: "$name returned nothing",
                style = NocturneType.MonoCode,
                color = if (isError) NocturneColors.Neutral300 else NocturneColors.Accent300,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )
            Text(if (expanded) "⌃" else "⌄", style = NocturneType.Row, color = NocturneColors.Accent300)
        }
        if (expanded) {
            Text(
                message.content,
                style = NocturneType.MonoCode,
                color = NocturneColors.Text.copy(alpha = 0.72f),
                modifier = Modifier
                    .fillMaxWidth()
                    .background(NocturneColors.Neutral900, Radius.Md)
                    .padding(horizontal = 11.dp, vertical = 9.dp),
            )
        }
    }
}

/** Hand the exported file to the chooser. */
internal fun shareFile(context: android.content.Context, file: java.io.File, mime: String) {
    val uri = androidx.core.content.FileProvider.getUriForFile(
        context,
        "${context.packageName}.fileprovider",
        file,
    )
    val intent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
        type = mime
        putExtra(android.content.Intent.EXTRA_STREAM, uri)
        putExtra(android.content.Intent.EXTRA_TITLE, file.name)
        addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(android.content.Intent.createChooser(intent, "Send conversation"))
}

private data class RenderedToolCall(val name: String, val arguments: String, val id: String)

private fun parseToolCalls(raw: String?): List<RenderedToolCall> {
    if (raw.isNullOrBlank()) return emptyList()
    return runCatching {
        kotlinx.serialization.json.Json.parseToJsonElement(raw)
            .let { it as? kotlinx.serialization.json.JsonArray ?: return emptyList() }
            .map { element ->
                val obj = element.jsonObject
                RenderedToolCall(
                    name = obj["name"]?.jsonPrimitive?.content.orEmpty(),
                    arguments = obj["arguments"]?.jsonPrimitive?.content.orEmpty(),
                    id = obj["id"]?.jsonPrimitive?.content.orEmpty(),
                )
            }
    }.getOrDefault(emptyList())
}

@Composable
private fun StreamingBubble(
    streaming: StreamingMessage,
    expanded: Boolean,
    onToggleThinking: () -> Unit,
    tokensPerSecond: Float,
    trace: ai.ondevice.engine.ResourceTrace?,
) {
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        // Reading the prompt is the longest part of a turn on a phone and the
        // only part with nothing to show for it. Said out loud, because a
        // minute of an empty screen is indistinguishable from a hang.
        if (streaming.content.isBlank() && streaming.thinking.isBlank()) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(7.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                NDot(color = NocturneColors.Accent)
                Text(
                    streaming.promptTokens
                        ?.let { "Read $it tokens · answering…" }
                        ?: "Reading the conversation…",
                    style = NocturneType.Meta.copy(fontSize = NocturneType.Row.fontSize),
                    color = NocturneColors.TextMuted,
                )
            }
        }
        if (streaming.thinking.isNotBlank()) {
            ThinkingBlock(
                thinking = streaming.thinking,
                millis = streaming.thinkingMillis,
                tokens = streaming.thinkingTokens,
                expanded = expanded,
                onToggle = onToggleThinking,
                live = !streaming.thinkingComplete,
            )
        }
        if (streaming.content.isNotBlank()) {
            Row(verticalAlignment = Alignment.Bottom) {
                Text(streaming.content, style = NocturneType.Message, modifier = Modifier.weight(1f, fill = false))
                // The caret from the canvas: a solid accent block at the tail.
                Box(
                    Modifier
                        .padding(start = 2.dp, bottom = 3.dp)
                        .size(width = 7.dp, height = 15.dp)
                        .background(NocturneColors.Accent),
                )
            }
        }
        if (tokensPerSecond > 0) {
            Row(Modifier.fillMaxWidth().alpha(0.55f)) {
                Box(Modifier.weight(1f))
                Text(
                    Fmt.tokensPerSecond(tokensPerSecond),
                    style = NocturneType.MonoXs,
                    color = NocturneColors.Accent300,
                )
            }
        }
        // Live: expandable while it runs, because the moment a graph of what a model is doing to the device is worth watching is while it is doing it.
        trace?.let {
            var traceExpanded by rememberSaveable { mutableStateOf(false) }
            ResourceBlock(
                trace = it,
                expanded = traceExpanded,
                onToggle = { traceExpanded = !traceExpanded },
                live = true,
            )
        }
    }
}

/** The collapsed reasoning block. */
@Composable
private fun ThinkingBlock(
    thinking: String,
    millis: Long?,
    tokens: Int?,
    expanded: Boolean,
    onToggle: () -> Unit,
    live: Boolean = false,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(
            Modifier
                .fillMaxWidth()
                .background(NocturneColors.Neutral900, Radius.Md)
                .ring(NocturneColors.Divider, Radius.Md)
                .nClickableFlat(onClick = onToggle)
                .padding(horizontal = 10.dp, vertical = 7.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                NIcons.Think,
                contentDescription = null,
                tint = NocturneColors.Accent,
                modifier = Modifier.size(13.dp),
            )
            Text(
                when {
                    live -> "Thinking…"
                    millis != null && tokens != null ->
                        "Thought for ${String.format("%.1f", millis / 1000f)}s · $tokens tokens"
                    else -> "Thinking"
                },
                style = NocturneType.Meta.copy(fontSize = NocturneType.Row.fontSize),
                color = NocturneColors.Accent300,
                modifier = Modifier.weight(1f),
            )
            Text(
                if (expanded) "⌃" else "⌄",
                style = NocturneType.Row,
                color = NocturneColors.Accent300,
            )
        }
        if (expanded) {
            Text(
                thinking,
                style = NocturneType.MonoCode,
                color = NocturneColors.Text.copy(alpha = 0.72f),
                modifier = Modifier
                    .fillMaxWidth()
                    .background(NocturneColors.Neutral900, Radius.Md)
                    .padding(horizontal = 11.dp, vertical = 9.dp),
            )
        }
    }
}

@Composable
private fun MessageActions(
    tokensPerSecond: Float?,
    onRegenerate: () -> Unit,
    onInspect: () -> Unit,
) {
    Row(
        Modifier.fillMaxWidth().alpha(0.55f),
        horizontalArrangement = Arrangement.spacedBy(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(NIcons.Copy, "Copy", Modifier.size(15.dp), NocturneColors.Text)
        Icon(
            NIcons.Rotate,
            "Regenerate",
            Modifier.size(15.dp).nClickableFlat(onClick = onRegenerate),
            NocturneColors.Text,
        )
        Icon(
            NIcons.Activity,
            "Inspect prompt",
            Modifier.size(15.dp).nClickableFlat(onClick = onInspect),
            NocturneColors.Text,
        )
        Icon(NIcons.Speaker, "Read aloud", Modifier.size(15.dp), NocturneColors.Text)
        Box(Modifier.weight(1f))
        tokensPerSecond?.takeIf { it > 0 }?.let {
            Text(Fmt.tokensPerSecond(it), style = NocturneType.MonoXs, color = NocturneColors.Accent300)
        }
    }
}

/** The attached files, before sending. */
@OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
private fun AttachmentStrip(
    attachments: List<ai.ondevice.ui.vm.PendingAttachment>,
    onRemove: (String) -> Unit,
) {
    androidx.compose.foundation.layout.FlowRow(
        Modifier.fillMaxWidth().padding(bottom = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        attachments.forEach { attachment ->
            Row(
                Modifier
                    .background(NocturneColors.Surface, Radius.Sm)
                    .ring(NocturneColors.Divider, Radius.Sm)
                    .padding(4.dp),
                horizontalArrangement = Arrangement.spacedBy(7.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (attachment.kind == ai.ondevice.data.AttachmentKind.IMAGE) {
                    coil3.compose.AsyncImage(
                        model = attachment.path,
                        contentDescription = attachment.name,
                        contentScale = androidx.compose.ui.layout.ContentScale.Crop,
                        modifier = Modifier
                            .size(30.dp)
                            .clip(Radius.Sm)
                            .background(NocturneColors.Neutral900),
                    )
                } else {
                    Icon(
                        NIcons.File,
                        contentDescription = null,
                        tint = NocturneColors.Accent300,
                        modifier = Modifier.size(16.dp).padding(start = 4.dp),
                    )
                }
                Column {
                    Text(
                        attachment.name,
                        style = NocturneType.Meta,
                        maxLines = 1,
                        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                        modifier = Modifier.widthIn(max = 140.dp),
                    )
                    Text(
                        "${Fmt.grouped(attachment.tokenCost)} tok",
                        style = NocturneType.Mono2Xs,
                        color = NocturneColors.TextMuted,
                    )
                }
                Text(
                    "×",
                    style = NocturneType.Row,
                    color = NocturneColors.TextMuted,
                    modifier = Modifier
                        .nClickableFlat { onRemove(attachment.path) }
                        .padding(horizontal = 6.dp),
                )
            }
        }
    }
}

@Composable
private fun ChatComposer(
    state: ChatState,
    onInputChange: (String) -> Unit,
    onSend: () -> Unit,
    onStop: () -> Unit,
    onAttach: () -> Unit,
    onRemoveAttachment: (String) -> Unit,
) {
    Column(
        Modifier
            .fillMaxWidth()
            .ruleAbove()
            .padding(start = 14.dp, end = 14.dp, top = 8.dp, bottom = 12.dp),
    ) {
        if (state.pendingAttachments.isNotEmpty()) {
            AttachmentStrip(state.pendingAttachments, onRemoveAttachment)
            // §4.5 — the price is stated before the send, not discovered after.
            NHelp(
                "${state.pendingAttachments.size} attached · " +
                    "${Fmt.grouped(state.pendingAttachments.sumOf { it.tokenCost })} tokens before you send",
                Modifier.padding(bottom = 6.dp),
            )
        }
        state.runningTool?.let { tool ->
            NHelp("Running $tool…", Modifier.padding(bottom = 6.dp), color = NocturneColors.Accent300)
        }
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.Bottom,
        ) {
            Icon(
                NIcons.PlusThin,
                contentDescription = "Attach",
                tint = NocturneColors.Text.copy(alpha = 0.6f),
                modifier = Modifier
                    .padding(bottom = 11.dp)
                    .size(22.dp)
                    .nClickableFlat(onClick = onAttach),
            )
            Box(Modifier.weight(1f)) {
                ai.ondevice.ui.components.NInput(
                    value = state.input,
                    onValueChange = onInputChange,
                    placeholder = state.model?.let { "Message ${it.label}…" } ?: "No model loaded",
                    singleLine = false,
                    minHeight = 44.dp,
                    textStyle = NocturneType.Message,
                )
            }
            // A load is not "generating", and for the minute or two a 9B model
            // takes to come off storage this was the one control on the screen
            // with nothing to do — it showed Send, and a press queued a second
            // message behind the load rather than abandoning it.
            //
            // The same phase machine the other tabs use. Chat has no stopping
            // flag of its own to feed it: llama's stop sets a flag the sampler
            // reads between tokens rather than waiting on a native call, so
            // there is no interval here where the run is neither going nor
            // stopped. That is a property of this runtime, not an omission.
            val busy = ai.ondevice.core.runPhaseOf(
                loading = state.loadingModel,
                running = state.generating,
            ).busy
            NCircleButton(
                icon = if (busy) NIcons.Stop else NIcons.Send,
                contentDescription = if (busy) "Stop" else "Send",
                onClick = if (busy) onStop else onSend,
            )
        }
    }
}
