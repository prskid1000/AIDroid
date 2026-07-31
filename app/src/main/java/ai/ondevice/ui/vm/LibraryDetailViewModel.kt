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
