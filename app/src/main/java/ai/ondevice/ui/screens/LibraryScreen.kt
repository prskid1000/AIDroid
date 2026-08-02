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
import androidx.compose.runtime.setValue
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
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NDialog
import ai.ondevice.ui.components.NDialogActions
import ai.ondevice.ui.components.NDialogBody
import ai.ondevice.ui.components.NDialogTitle
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

/** **Library** — everything this device has produced. */
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
    val clips by viewModel.clips.collectAsStateWithLifecycle()
    val syntheses by viewModel.syntheses.collectAsStateWithLifecycle()
    val transcripts by viewModel.transcripts.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = {
            RootToolbar("Library") {
                // The three sections, as icons beside the count.
                LibrarySection.entries.forEach { entry ->
                    ToolbarToggle(
                        when (entry) {
                            LibrarySection.CHATS -> NIcons.Chat
                            LibrarySection.IMAGES -> NIcons.Image
                            LibrarySection.CLIPS -> NIcons.Video
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
                        LibrarySection.CLIPS -> "${clips.size}"
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
        // Every row opens the same detail screen.
        when (section) {
            LibrarySection.CHATS -> ChatsSection(
                conversations = conversations,
                onOpen = { id -> onOpenItem(PredictionKind.CHAT, id) },
                onDelete = viewModel::deleteConversation,
            )

            LibrarySection.IMAGES -> ImagesSection(
                images = images,
                onOpen = { id -> onOpenItem(PredictionKind.IMAGE, id) },
                onDelete = viewModel::deleteImage,
            )

            LibrarySection.CLIPS -> ClipsSection(
                clips = clips,
                onOpen = { id -> onOpenItem(PredictionKind.VIDEO, id) },
                onDelete = viewModel::deleteClip,
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
                kind = "conversation",
                deleteDetail = "All ${summary.messageCount} message" +
                    (if (summary.messageCount == 1) "" else "s") +
                    " go with it, including anything attached to them.",
            )
        }
    }
}

/** A thread's own title if it has earned one, otherwise its opening line. */
private fun ConversationEntity.displayTitle(preview: String): String =
    title.takeIf { it.isNotBlank() && it != "New conversation" }
        ?: preview.takeIf { it.isNotBlank() }
        ?: "Empty conversation"

// — images —

@Composable
private fun ImagesSection(
    images: List<GeneratedImageEntity>,
    onOpen: (String) -> Unit,
    onDelete: (GeneratedImageEntity) -> Unit,
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
                    // The file on disk.
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
                    TileActions(
                        onDelete = { onDelete(image) },
                        kind = "image",
                        deleteDetail = "Seed ${image.seed}, ${image.width}x${image.height}. " +
                            "The PNG and its embedded parameter set both go, so this exact " +
                            "result cannot be reused or reproduced from the library again.",
                        modifier = Modifier.align(Alignment.TopEnd),
                    )
                }
            }
        }
        NHelp("Open any image for its full parameter set, what it cost to make, and one-tap reuse.")
    }
}

// — clips —

/**
 * One tile per clip, showing its first frame.
 *
 * The first frame rather than a thumbnail file: the frames are already PNGs on
 * disk, so a separate thumbnail would be a second copy of something that is
 * already there and able to go stale against it.
 */
@Composable
private fun ClipsSection(
    clips: List<ai.ondevice.data.db.GeneratedClipEntity>,
    onOpen: (String) -> Unit,
    onDelete: (ai.ondevice.data.db.GeneratedClipEntity) -> Unit,
) {
    if (clips.isEmpty()) {
        NHelp(
            "No clips yet. Image → the film icon generates them, on the same models and the " +
                "same runtime as stills.",
        )
        return
    }
    Column(Modifier.fillMaxWidth()) {
        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            modifier = Modifier.fillMaxWidth().weight(1f),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
            contentPadding = PaddingValues(bottom = 10.dp),
        ) {
            items(clips, key = { it.id }) { clip ->
                Box(
                    Modifier
                        .aspectRatio(1f)
                        .clip(Radius.Sm)
                        .background(NocturneColors.Neutral900)
                        .nClickableFlat(onClick = { onOpen(clip.id) }),
                    contentAlignment = Alignment.BottomStart,
                ) {
                    coil3.compose.AsyncImage(
                        model = "${clip.directory}/frame_0000.png",
                        contentDescription = clip.prompt,
                        contentScale = androidx.compose.ui.layout.ContentScale.Crop,
                        modifier = Modifier.matchParentSize(),
                    )
                    Text(
                        "${clip.frameCount} frames · ${clip.fps} fps" +
                            if (clip.audioPath != null) " · sound" else "",
                        style = NocturneType.MonoXs,
                        color = NocturneColors.Accent100.copy(alpha = 0.85f),
                        modifier = Modifier.padding(4.dp),
                    )
                    TileActions(
                        onDelete = { onDelete(clip) },
                        kind = "clip",
                        deleteDetail = "${clip.frameCount} frames at ${clip.fps} fps" +
                            (if (clip.audioPath != null) " and its audio" else "") +
                            ". The whole folder of frames is removed.",
                        modifier = Modifier.align(Alignment.TopEnd),
                    )
                }
            }
        }
        NHelp("Tap a clip to play it through. Deleting one removes its whole folder of frames.")
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
                kind = "reading",
                deleteDetail = "${Fmt.duration(synthesis.durationMillis)} of audio is removed. " +
                    "The text it was read from is not kept anywhere else.",
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
                kind = "transcript",
                deleteDetail = "${transcript.title} — every segment and its timings. " +
                    "The recording it was made from is not deleted.",
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
    /** What this row is, for the confirm: "conversation", "recording", "reading". */
    kind: String,
    /** What deleting it actually removes, in the confirm's own words. */
    deleteDetail: String,
) {
    var confirming by androidx.compose.runtime.saveable.rememberSaveable(title) {
        androidx.compose.runtime.mutableStateOf(false)
    }
    if (confirming) {
        ConfirmDelete(
            what = kind,
            detail = deleteDetail,
            onDismiss = { confirming = false },
            onConfirm = onDelete,
        )
    }
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
        NIconButton(
            NIcons.Trash,
            "Delete",
            onClick = { confirming = true },
            size = 34.dp,
            iconSize = 15.dp,
        )
    }
}

/**
 * The confirm every delete in this screen goes through.
 *
 * A row's Trash and a tile's Trash are both a single tap on a list of things
 * the user made, several of which took minutes of the phone's life to produce
 * and none of which can be recovered. Asking first is not friction here; the
 * question just has to be answerable, so the body says what specifically goes
 * rather than "are you sure?".
 */
@Composable
private fun ConfirmDelete(
    what: String,
    detail: String,
    onDismiss: () -> Unit,
    onConfirm: () -> Unit,
) {
    NDialog(onDismissRequest = onDismiss) {
        NDialogTitle("Delete this $what?")
        NDialogBody(detail)
        NDialogActions {
            NButton("Cancel", onClick = onDismiss, style = NButtonStyle.Secondary)
            NButton(
                "Delete",
                onClick = { onDismiss(); onConfirm() },
                style = NButtonStyle.Primary,
            )
        }
    }
}

/**
 * Delete, over the corner of a thumbnail.
 *
 * Opening is what tapping the tile already does, so the only action that needs
 * a control of its own is the destructive one. Images had no delete at all and
 * the clips grid was handed an `onDelete` it never called — throwing either
 * away meant opening it first, which is a detour to get rid of something.
 *
 * It sits *on* the picture rather than beside it, over a scrim, because a tile
 * has no margin to put it in and a frame can be any brightness.
 */
@Composable
private fun TileActions(
    onDelete: () -> Unit,
    kind: String,
    deleteDetail: String,
    modifier: Modifier = Modifier,
) {
    var confirming by androidx.compose.runtime.saveable.rememberSaveable(deleteDetail) {
        androidx.compose.runtime.mutableStateOf(false)
    }
    if (confirming) {
        ConfirmDelete(
            what = kind,
            detail = deleteDetail,
            onDismiss = { confirming = false },
            onConfirm = onDelete,
        )
    }
    Box(
        modifier
            .padding(4.dp)
            .background(NocturneColors.Neutral900.copy(alpha = 0.72f), Radius.Sm),
    ) {
        NIconButton(
            NIcons.Trash,
            "Delete",
            onClick = { confirming = true },
            size = 28.dp,
            iconSize = 13.dp,
        )
    }
}
