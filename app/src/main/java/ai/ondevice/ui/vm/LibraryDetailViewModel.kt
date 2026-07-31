package ai.ondevice.ui.vm

import ai.ondevice.core.PredictionKind
import ai.ondevice.core.SparseParams
import ai.ondevice.data.db.ConversationEntity
import ai.ondevice.data.db.GeneratedImageEntity
import ai.ondevice.data.db.MessageEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.PredictionRunEntity
import ai.ondevice.data.db.SynthesisEntity
import ai.ondevice.data.db.TranscriptEntity
import ai.ondevice.engine.ResourceTrace
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * One artifact, in full.
 *
 * The library could list four kinds of thing and open none of them: an image
 * tile went to a grid, a synthesis fired a share sheet, and a transcript row was
 * inert. Everything needed to show what a stored artifact *is* was already in
 * the database — the prompt, the parameter set, the model, and now the run that
 * produced it — and nothing read it back.
 *
 * One view model for all four kinds rather than four. They differ in what they
 * display and agree on everything else: load by id, find the runs keyed to that
 * id, delete the row and its file together. Splitting them would have meant
 * writing that agreement out four times.
 */
@HiltViewModel
class LibraryDetailViewModel @Inject constructor(
    private val db: OnDeviceDatabase,
    private val archive: ai.ondevice.data.ConversationArchive,
    private val storage: ai.ondevice.data.ModelStorage,
    private val exports: ai.ondevice.data.ExportStore,
    savedState: SavedStateHandle,
) : ViewModel() {

    private val kind: PredictionKind = savedState.get<String>("kind")
        ?.let { runCatching { PredictionKind.valueOf(it) }.getOrNull() }
        ?: PredictionKind.CHAT
    private val id: String = savedState.get<String>("id").orEmpty()

    private val _state = MutableStateFlow(LibraryDetailState(kind = kind))
    val state: StateFlow<LibraryDetailState> = _state.asStateFlow()

    init {
        viewModelScope.launch { load() }
    }

    private suspend fun load() {
        when (kind) {
            PredictionKind.CHAT -> {
                val conversation = db.conversations().get(id)
                val messages = db.messages().getFor(id)
                _state.value = _state.value.copy(
                    conversation = conversation,
                    messages = messages,
                    // A conversation is not one run. Every assistant turn in it
                    // is, so the runs are gathered across the messages rather
                    // than looked up under the conversation's own id — which no
                    // run has ever been keyed to.
                    runs = messages.flatMap { db.predictionRuns().getFor(it.id) },
                    // The same rule the list uses. Nothing renames a thread, so
                    // the stored title is "New conversation" for almost all of
                    // them — a detail screen headed that way names nothing, and
                    // would disagree with the row that was just tapped.
                    title = conversation?.title
                        ?.takeIf { it.isNotBlank() && it != "New conversation" }
                        ?: messages.firstOrNull { it.content.isNotBlank() }
                            ?.content?.lineSequence()?.firstOrNull()?.take(60)
                        ?: "Empty conversation",
                    loaded = true,
                )
            }
            PredictionKind.IMAGE -> {
                val image = db.images().get(id)
                _state.value = _state.value.copy(
                    image = image,
                    runs = db.predictionRuns().getFor(id),
                    title = image?.prompt?.lineSequence()?.firstOrNull()?.take(60).orEmpty(),
                    loaded = true,
                )
            }
            PredictionKind.SPEECH -> {
                val synthesis = db.syntheses().get(id)
                _state.value = _state.value.copy(
                    synthesis = synthesis,
                    runs = db.predictionRuns().getFor(id),
                    title = synthesis?.text?.lineSequence()?.firstOrNull()?.take(60).orEmpty(),
                    loaded = true,
                )
            }
            PredictionKind.TRANSCRIBE -> {
                val transcript = db.transcripts().get(id)
                _state.value = _state.value.copy(
                    transcript = transcript,
                    runs = db.predictionRuns().getFor(id),
                    title = transcript?.title.orEmpty(),
                    loaded = true,
                )
            }
        }
    }

    /** Which format this kind can be written as. One entry means no choice to offer. */
    val formats: List<String>
        get() = when (kind) {
            PredictionKind.CHAT -> listOf("md", "zip")
            PredictionKind.IMAGE -> listOf("png")
            PredictionKind.SPEECH -> listOf("wav")
            PredictionKind.TRANSCRIBE ->
                ai.ondevice.core.TranscriptFormat.entries.map { it.extension }
        }

    fun selectFormat(format: String) {
        _state.value = _state.value.copy(format = format)
    }

    /**
     * Produce the file, in whatever the chosen format is.
     *
     * An image and a synthesis already exist on disk as the thing you would
     * export, so those are handed over as they are rather than copied into a
     * staging area first. A conversation and a transcript are rendered on
     * demand, because neither is stored in the shape anyone wants to read.
     */
    private suspend fun build(): ai.ondevice.core.Export? {
        val state = _state.value
        val stem = state.title
            .lowercase()
            .replace(Regex("[^a-z0-9]+"), "-")
            .trim('-')
            .ifBlank { kind.name.lowercase() }
        return when (kind) {
            PredictionKind.CHAT -> {
                val conversation = state.conversation ?: return null
                val file = if (state.format == "zip") {
                    archive.exportArchive(
                        conversationIds = listOf(conversation.id),
                        destination = java.io.File(storage.exportsDir(), "$stem.zip"),
                    )
                } else {
                    archive.exportMarkdown(
                        conversationId = conversation.id,
                        destination = java.io.File(storage.exportsDir(), "$stem.md"),
                    )
                }
                ai.ondevice.core.Export(file, file.name, ai.ondevice.core.Export.mimeFor(file.name))
            }
            PredictionKind.IMAGE -> state.image?.let {
                ai.ondevice.core.Export(
                    java.io.File(it.path),
                    "$stem.png",
                    ai.ondevice.core.Export.MIME_PNG,
                )
            }
            PredictionKind.SPEECH -> state.synthesis?.let {
                ai.ondevice.core.Export(
                    java.io.File(it.path),
                    "$stem.wav",
                    ai.ondevice.core.Export.MIME_WAV,
                )
            }
            PredictionKind.TRANSCRIBE -> state.transcript?.let { transcript ->
                val format = ai.ondevice.core.TranscriptFormat.entries
                    .firstOrNull { it.extension == state.format }
                    ?: ai.ondevice.core.TranscriptFormat.TXT
                val file = java.io.File(storage.exportsDir(), "$stem.${format.extension}")
                file.writeText(
                    ai.ondevice.core.TranscriptExport.render(
                        format = format,
                        segments = ai.ondevice.core.TranscriptSegments.parse(transcript.segmentsJson),
                        title = transcript.title,
                        modelId = transcript.modelId,
                    ),
                )
                ai.ondevice.core.Export(file, file.name, format.mime)
            }
        }
    }

    /** Build and hand back for a share sheet. Nothing is written to the user's storage. */
    fun share(onReady: (ai.ondevice.core.Export) -> Unit) {
        viewModelScope.launch {
            val export = runCatching { build() }.getOrNull()
            if (export == null) {
                _state.value = _state.value.copy(exportMessage = "There was nothing to export.")
                return@launch
            }
            onReady(export)
        }
    }

    /**
     * Write it to the folder the user chose, asking for one the first time.
     *
     * [onNeedFolder] is called rather than the picker being opened here, because
     * only a composable can launch one — and the save is retried by
     * [folderPicked] once the answer comes back, so the user's tap is not
     * silently swallowed by the permission round trip.
     */
    fun save(onNeedFolder: () -> Unit) {
        viewModelScope.launch {
            val folder = exports.folder()
            if (folder == null) {
                onNeedFolder()
                return@launch
            }
            writeTo(folder)
        }
    }

    fun folderPicked(uri: android.net.Uri) {
        viewModelScope.launch {
            exports.remember(uri)
            writeTo(uri)
        }
    }

    private suspend fun writeTo(folder: android.net.Uri) {
        val export = runCatching { build() }.getOrNull()
        if (export == null) {
            _state.value = _state.value.copy(exportMessage = "There was nothing to export.")
            return
        }
        exports.save(listOf(export), folder).fold(
            onSuccess = { saved ->
                _state.value = _state.value.copy(
                    exportMessage = "Saved ${export.suggestedName} to ${saved.displayPath}.",
                )
            },
            onFailure = {
                _state.value = _state.value.copy(
                    exportMessage = "That could not be saved: ${it.message}",
                )
            },
        )
    }

    fun clearExportMessage() {
        _state.value = _state.value.copy(exportMessage = null)
    }

    /**
     * Remove the artifact, its file and its runs.
     *
     * All three or none. A row without its file is a broken thumbnail, a file
     * without its row is storage nothing can reach, and a run pointing at a
     * deleted artifact is a record of work on something that no longer exists.
     */
    fun delete(onDone: () -> Unit) {
        viewModelScope.launch {
            val state = _state.value
            when (kind) {
                PredictionKind.CHAT -> {
                    state.messages.forEach { db.predictionRuns().deleteForArtifact(it.id) }
                    db.messages().clearFor(id)
                    db.conversations().deleteById(id)
                }
                PredictionKind.IMAGE -> {
                    state.image?.let { runCatching { java.io.File(it.path).delete() } }
                    db.predictionRuns().deleteForArtifact(id)
                    db.images().deleteById(id)
                }
                PredictionKind.SPEECH -> {
                    state.synthesis?.let { runCatching { java.io.File(it.path).delete() } }
                    db.predictionRuns().deleteForArtifact(id)
                    db.syntheses().deleteById(id)
                }
                PredictionKind.TRANSCRIBE -> {
                    // The source audio is the user's own file, chosen by them.
                    // Only the record goes.
                    db.predictionRuns().deleteForArtifact(id)
                    db.transcripts().deleteById(id)
                }
            }
            onDone()
        }
    }
}

data class LibraryDetailState(
    val kind: PredictionKind,
    val loaded: Boolean = false,
    val title: String = "",
    val conversation: ConversationEntity? = null,
    val messages: List<MessageEntity> = emptyList(),
    val image: GeneratedImageEntity? = null,
    val synthesis: SynthesisEntity? = null,
    val transcript: TranscriptEntity? = null,
    val runs: List<PredictionRunEntity> = emptyList(),
    /** The extension the next Save or Share will produce. */
    val format: String = when (kind) {
        PredictionKind.CHAT -> "md"
        PredictionKind.IMAGE -> "png"
        PredictionKind.SPEECH -> "wav"
        PredictionKind.TRANSCRIBE -> "txt"
    },
    /** Where the last save went, or why it did not. Cleared when acknowledged. */
    val exportMessage: String? = null,
) {
    /** True once loading finished and found nothing — deleted from elsewhere. */
    val missing: Boolean
        get() = loaded && conversation == null && image == null && synthesis == null && transcript == null

    val traces: List<ResourceTrace> get() = runs.mapNotNull { ResourceTrace.parse(it.traceJson) }

    /**
     * The parameter set this artifact was produced under, whichever kind it is.
     *
     * A conversation has no parameters of its own — they are recorded per
     * message, because they can change mid-thread — so it reports the ones the
     * last assistant turn actually ran with rather than a blank table.
     */
    val params: SparseParams
        get() = when (kind) {
            PredictionKind.CHAT -> SparseParams.parse(
                messages.lastOrNull { it.role == ai.ondevice.core.MessageRole.ASSISTANT }
                    ?.generationParamsJson,
            )
            PredictionKind.IMAGE -> SparseParams.parse(image?.paramsJson)
            PredictionKind.SPEECH -> SparseParams.parse(synthesis?.paramsJson)
            PredictionKind.TRANSCRIBE -> SparseParams.parse(transcript?.paramsJson)
        }

    /**
     * Which model made this.
     *
     * A conversation records one only when the user picks it explicitly, so most
     * threads carry null and the row read "—" while the reply above it had
     * plainly been generated by something. The run knows — it is written from
     * whatever was loaded at the time — so it is the better answer where the
     * conversation has none.
     */
    val modelId: String?
        get() = when (kind) {
            PredictionKind.CHAT -> conversation?.modelId
                ?: runs.lastOrNull { it.modelId != null }?.modelId
            PredictionKind.IMAGE -> image?.modelId
            PredictionKind.SPEECH -> synthesis?.modelId
            PredictionKind.TRANSCRIBE -> transcript?.modelId
        }

    val createdAt: Long
        get() = when (kind) {
            PredictionKind.CHAT -> conversation?.createdAt ?: 0
            PredictionKind.IMAGE -> image?.createdAt ?: 0
            PredictionKind.SPEECH -> synthesis?.createdAt ?: 0
            PredictionKind.TRANSCRIBE -> transcript?.createdAt ?: 0
        }
}
