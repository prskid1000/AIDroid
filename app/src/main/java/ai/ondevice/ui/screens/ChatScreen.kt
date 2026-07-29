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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
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
import ai.ondevice.ui.components.PhoneScaffold
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

/**
 * **S6 — Chat**, with **S7** as its settings sheet.
 *
 * Everything the canvas annotates is real here: the thinking block is
 * collapsible and shows its own token count and elapsed time, the footer
 * carries live tok/s, images declare their token cost before they are sent, and
 * the send button becomes a stop that actually cancels.
 */
@Composable
fun ChatScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onOpenParameters: (ai.ondevice.core.Tier) -> Unit,
    onOpenPromptInspector: () -> Unit,
    onOpenModels: () -> Unit,
    // Activity-scoped so the prompt inspector inspects *this* conversation.
    viewModel: ChatViewModel = activityChatViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val listState = rememberLazyListState()
    var sheetOpen by rememberSaveable { mutableStateOf(false) }

    // One picker for all three kinds. `OpenDocument` rather than the photo
    // picker because the composer accepts documents too, and making the user
    // guess which of two "attach" buttons handles their file is worse than one
    // that takes everything and says what it did with it.
    val context = LocalContext.current
    val attachLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri ->
        uri?.let {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    it,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION,
                )
            }
            viewModel.attach(it)
        }
    }
    val pickAttachment = {
        attachLauncher.launch(arrayOf("image/*", "text/*", "application/pdf", "application/json", "audio/*"))
    }

    LaunchedEffect(state.messages.size, state.streaming?.content) {
        val target = state.messages.size
        if (target > 0) listState.animateScrollToItem(target)
    }

    Box(Modifier.fillMaxWidth()) {
        PhoneScaffold(
            toolbar = { ChatToolbar(state, onOpenMenu = { sheetOpen = true }, onOpenSettings = { sheetOpen = true }) },
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
                    )
                }
                state.streaming?.let { streaming ->
                    item(key = "streaming") {
                        StreamingBubble(
                            streaming = streaming,
                            expanded = streaming.id in state.expandedThinking,
                            onToggleThinking = { viewModel.toggleThinking(streaming.id) },
                            tokensPerSecond = state.tokensPerSecond,
                        )
                    }
                }
            }
        }

        if (sheetOpen) {
            ChatSettingsSheet(
                state = state,
                onDismiss = { sheetOpen = false },
                onSelectModel = viewModel::setModel,
                onSelectPreset = viewModel::setPreset,
                onSelectPersona = viewModel::setPersona,
                onSystemPromptChange = viewModel::setSystemPrompt,
                onLiveParam = viewModel::setLiveParam,
                onOpenParametersAtTier = { tier ->
                    sheetOpen = false
                    onOpenParameters(tier)
                },
            )
        }
    }
}

/** Model name, preset, and the live backend/context readout. */
@Composable
private fun ChatToolbar(state: ChatState, onOpenMenu: () -> Unit, onOpenSettings: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.spacedBy(9.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            NIcons.Menu,
            contentDescription = "Conversations",
            tint = NocturneColors.Text,
            modifier = Modifier.size(20.dp).nClickableFlat(onClick = onOpenMenu),
        )
        Column(Modifier.weight(1f)) {
            Text(
                listOfNotNull(state.model?.displayName, state.presetName).joinToString(" · ")
                    .ifBlank { "No model" },
                style = NocturneType.CardTitle,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.alpha(0.6f),
            ) {
                NDot(color = if (state.generating) NocturneColors.Accent else NocturneColors.Neutral500)
                Text(
                    buildString {
                        append(state.model?.backendOverride?.label ?: "OpenCL")
                        append(" · ${Fmt.grouped(state.contextUsed)} / ${Fmt.grouped(state.contextLimit)} ctx")
                    },
                    style = NocturneType.NavLabel,
                )
            }
        }
        Icon(
            NIcons.Settings,
            contentDescription = "Chat settings",
            tint = NocturneColors.Text,
            modifier = Modifier.size(19.dp).nClickableFlat(onClick = onOpenSettings),
        )
    }
}

@Composable
private fun EmptyChat(state: ChatState, onOpenModels: () -> Unit) {
    NCard(Modifier.padding(top = 12.dp)) {
        Text(
            if (state.model == null) "No model loaded" else "Ready",
            style = NocturneType.CardTitleSm,
        )
        Text(
            if (state.model == null) {
                "Nothing is loaded yet. Add a model and the chat runs entirely on this device — after " +
                    "download there is no network at all."
            } else {
                "${state.model.displayName} is installed. Send a message and it loads on first use; " +
                    "the KV cache is reused across turns so follow-ups don't reprocess the prompt."
            },
            style = NocturneType.CardBody,
            color = NocturneColors.Text.copy(alpha = 0.8f),
        )
        if (state.model == null) {
            NButton("Add a model", onOpenModels, style = NButtonStyle.Primary, block = true)
        }
    }
}

/** A user bubble, or an assistant reply with its thinking block and actions. */
@Composable
private fun MessageBubble(
    message: MessageEntity,
    expanded: Boolean,
    onToggleThinking: () -> Unit,
    onRegenerate: () -> Unit,
    onInspect: () -> Unit,
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
            val images = SparseParams.parse(message.imagePathsJson).stringList("images").orEmpty()
            images.forEach { path ->
                Box(
                    Modifier
                        .fillMaxWidth(0.8f)
                        .height(96.dp)
                        .background(NocturneColors.Neutral900, Radius.Md)
                        .ring(NocturneColors.Divider, Radius.Md),
                    contentAlignment = Alignment.Center,
                ) {
                    // Per-image token cost, shown before it is sent — images
                    // consume context fast (SPEC §4.5).
                    Text(
                        "${path.substringAfterLast('/')} · ${Fmt.grouped(message.imageTokenCount ?: 0)} img tokens",
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMuted,
                    )
                }
            }
        }

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
            Text(message.content, style = NocturneType.Message)
            MessageActions(
                tokensPerSecond = message.tokensPerSecond,
                onRegenerate = onRegenerate,
                onInspect = onInspect,
            )
        }
    }
}

@Composable
private fun StreamingBubble(
    streaming: StreamingMessage,
    expanded: Boolean,
    onToggleThinking: () -> Unit,
    tokensPerSecond: Float,
) {
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
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
    }
}

/**
 * The collapsed reasoning block. SPEC §4.4 — thinking is parsed out of the
 * configured tag pair and rendered collapsed, with its cost stated.
 */
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

/**
 * The attached files, before sending.
 *
 * An image shows itself; a document shows its name and how much of the context
 * it will occupy. Both are removable, because an attachment you cannot take
 * back is a trap — especially a 40 000-token one.
 */
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
                    placeholder = state.model?.let { "Message ${it.displayName}…" } ?: "No model loaded",
                    singleLine = false,
                    minHeight = 44.dp,
                    textStyle = NocturneType.Message,
                )
            }
            NCircleButton(
                icon = if (state.generating) NIcons.Stop else NIcons.Send,
                contentDescription = if (state.generating) "Stop" else "Send",
                onClick = if (state.generating) onStop else onSend,
            )
        }
    }
}
