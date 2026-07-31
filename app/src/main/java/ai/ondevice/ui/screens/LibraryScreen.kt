package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.core.PredictionKind
import ai.ondevice.data.db.ConversationEntity
import ai.ondevice.data.db.GeneratedImageEntity
import ai.ondevice.data.db.SynthesisEntity
import ai.ondevice.data.db.TranscriptEntity
import ai.ondevice.ui.BottomDestinations
import ai.ondevice.ui.components.NBottomBar
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NIconButton
import ai.ondevice.ui.components.NPills
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.RootToolbar
import ai.ondevice.ui.components.ToolbarToggle
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.ConversationSummary
import ai.ondevice.ui.vm.LibrarySection
import ai.ondevice.ui.vm.LibraryViewModel

/**
 * **Library** — everything this device has produced.
 *
 * Three of the four kinds were already being written to the database and only
 * one of them could be read back. Conversations were reachable only as
 * "whichever was open last", transcripts accumulated in a table no screen
 * queried, and a rendered WAV survived exactly as long as the process that made
 * it. Generation without a history is a feature that quietly loses the user's
 * work, so this is the one place that lists all of it.
 */
@Composable
fun LibraryScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onOpenItem: (PredictionKind, String) -> Unit,
    viewModel: LibraryViewModel = hiltViewModel(),
) {
    val section by viewModel.section.collectAsStateWithLifecycle()
    val conversations by viewModel.conversations.collectAsStateWithLifecycle()
    val images by viewModel.images.collectAsStateWithLifecycle()
    val syntheses by viewModel.syntheses.collectAsStateWithLifecycle()
    val transcripts by viewModel.transcripts.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = {
            RootToolbar("Library") {
                // The three sections, as icons beside the count. They repeat the
                // bottom bar's glyphs, which is a real risk — but they mean
                // something different here (things you made, not places to go)
                // and the selected plate says which one you are reading.
                LibrarySection.entries.forEach { entry ->
                    ToolbarToggle(
                        when (entry) {
                            LibrarySection.CHATS -> NIcons.Chat
                            LibrarySection.IMAGES -> NIcons.Image
                            LibrarySection.VOICE -> NIcons.Voice
                        },
                        entry.label,
                        selected = section == entry,
                        onClick = { viewModel.show(entry) },
                    )
                }
                Text(
                    when (section) {
                        LibrarySection.CHATS -> "${conversations.size}"
                        LibrarySection.IMAGES -> "${images.size}"
                        LibrarySection.VOICE -> "${syntheses.size + transcripts.size}"
                    },
                    style = NocturneType.MonoValue,
                    color = NocturneColors.TextMuted,
                )
            }
        },
        bottomBar = { NBottomBar(BottomDestinations, currentRoute) { onNavigate(it.route) } },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        // Every row opens the same detail screen. Deleting is still here, so a
        // clear-out does not cost one push per item — but *reading* one is now
        // a push rather than four different behaviours per section.
        when (section) {
            LibrarySection.CHATS -> ChatsSection(
                conversations = conversations,
                onOpen = { id -> onOpenItem(PredictionKind.CHAT, id) },
                onDelete = viewModel::deleteConversation,
            )

            LibrarySection.IMAGES -> ImagesSection(
                images = images,
                onOpen = { id -> onOpenItem(PredictionKind.IMAGE, id) },
            )

            LibrarySection.VOICE -> VoiceSection(
                syntheses = syntheses,
                transcripts = transcripts,
                onOpen = onOpenItem,
                onDeleteSynthesis = viewModel::deleteSynthesis,
                onDeleteTranscript = viewModel::deleteTranscript,
            )
        }
    }
}

// — chats —

@Composable
private fun ChatsSection(
    conversations: List<ConversationSummary>,
    onOpen: (String) -> Unit,
    onDelete: (String) -> Unit,
) {
    if (conversations.isEmpty()) {
        NHelp("No conversations yet. Every thread is kept until you delete it.")
        return
    }
    LazyColumn(
        Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        contentPadding = PaddingValues(bottom = 12.dp),
    ) {
        items(conversations, key = { it.conversation.id }) { summary ->
            LibraryRow(
                icon = NIcons.Chat,
                title = summary.conversation.displayTitle(summary.preview),
                subtitle = "${summary.messageCount} message${if (summary.messageCount == 1) "" else "s"} · " +
                    Fmt.relative(summary.conversation.updatedAt),
                onClick = { onOpen(summary.conversation.id) },
                onDelete = { onDelete(summary.conversation.id) },
            )
        }
    }
}

/**
 * A thread's own title if it has earned one, otherwise its opening line.
 *
 * Nothing renames a conversation, so every row would otherwise read "New
 * conversation" and the list would be unusable at exactly the size where a list
 * starts to matter.
 */
private fun ConversationEntity.displayTitle(preview: String): String =
    title.takeIf { it.isNotBlank() && it != "New conversation" }
        ?: preview.takeIf { it.isNotBlank() }
        ?: "Empty conversation"

// — images —

@Composable
private fun ImagesSection(
    images: List<GeneratedImageEntity>,
    onOpen: (String) -> Unit,
) {
    if (images.isEmpty()) {
        NHelp(
            "Nothing generated yet. Every image is written with its full parameter set embedded " +
                "in the PNG, so any result can be reproduced exactly.",
        )
        return
    }
    Column(Modifier.fillMaxWidth()) {
        LazyVerticalGrid(
            columns = GridCells.Fixed(3),
            modifier = Modifier.fillMaxWidth().weight(1f),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
            contentPadding = PaddingValues(bottom = 10.dp),
        ) {
            items(images, key = { it.id }) { image ->
                Box(
                    Modifier
                        .aspectRatio(1f)
                        .clip(Radius.Sm)
                        .background(NocturneColors.Neutral900)
                        .nClickableFlat(onClick = { onOpen(image.id) }),
                    contentAlignment = Alignment.BottomStart,
                ) {
                    // The file on disk. These tiles used to draw a gradient
                    // derived from the seed — a placeholder from before sd.cpp
                    // could produce pixels, which outlived the reason for it and
                    // turned the gallery into a wall of things that looked
                    // generated and were not.
                    coil3.compose.AsyncImage(
                        model = image.path,
                        contentDescription = image.prompt,
                        contentScale = androidx.compose.ui.layout.ContentScale.Crop,
                        modifier = Modifier.matchParentSize(),
                    )
                    Text(
                        image.seed.toString(),
                        style = NocturneType.MonoXs,
                        color = NocturneColors.Accent100.copy(alpha = 0.75f),
                        modifier = Modifier.padding(4.dp),
                    )
                }
            }
        }
        NHelp("Tap any image for its full parameter set, what it cost to make, and one-tap reuse.")
    }
}

// — voice —

@Composable
private fun VoiceSection(
    syntheses: List<SynthesisEntity>,
    transcripts: List<TranscriptEntity>,
    onOpen: (PredictionKind, String) -> Unit,
    onDeleteSynthesis: (SynthesisEntity) -> Unit,
    onDeleteTranscript: (TranscriptEntity) -> Unit,
) {
    if (syntheses.isEmpty() && transcripts.isEmpty()) {
        NHelp(
            "Nothing here yet. Rendering a script to WAV on the Voice screen files it here, and " +
                "so does every transcription.",
        )
        return
    }
    LazyColumn(
        Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        contentPadding = PaddingValues(bottom = 12.dp),
    ) {
        items(syntheses, key = { "synth-${it.id}" }) { synthesis ->
            LibraryRow(
                icon = NIcons.Speaker,
                title = synthesis.text.lineSequence().firstOrNull()?.take(90).orEmpty()
                    .ifBlank { "Untitled" },
                subtitle = listOfNotNull(
                    synthesis.voice ?: synthesis.engineId,
                    Fmt.duration(synthesis.durationMillis),
                    Fmt.relative(synthesis.createdAt),
                ).joinToString(" · "),
                onClick = { onOpen(PredictionKind.SPEECH, synthesis.id) },
                onDelete = { onDeleteSynthesis(synthesis) },
            )
        }
        items(transcripts, key = { "transcript-${it.id}" }) { transcript ->
            LibraryRow(
                icon = NIcons.Waveform,
                title = transcript.title,
                subtitle = listOf(
                    Fmt.duration(transcript.durationMillis),
                    Fmt.relative(transcript.createdAt),
                ).joinToString(" · "),
                // Tapping this did nothing at all until now — the only row in
                // the library that listed something and then refused to open it.
                onClick = { onOpen(PredictionKind.TRANSCRIBE, transcript.id) },
                onDelete = { onDeleteTranscript(transcript) },
            )
        }
    }
}

// — the one row shape all three sections use —

@Composable
private fun LibraryRow(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    subtitle: String,
    onClick: (() -> Unit)?,
    onDelete: () -> Unit,
) {
    Row(
        Modifier
            .fillMaxWidth()
            .background(NocturneColors.Surface, Radius.Md)
            .ring(NocturneColors.Divider, Radius.Md)
            .then(if (onClick != null) Modifier.nClickableFlat(onClick = onClick) else Modifier)
            .padding(horizontal = 11.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            icon,
            contentDescription = null,
            tint = NocturneColors.Accent300,
            modifier = Modifier.size(16.dp),
        )
        Column(Modifier.weight(1f)) {
            Text(
                title,
                style = NocturneType.CardTitleSm,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                subtitle,
                style = NocturneType.Help,
                color = NocturneColors.TextMuted,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        NIconButton(NIcons.Trash, "Delete", onClick = onDelete, size = 34.dp, iconSize = 15.dp)
    }
}
