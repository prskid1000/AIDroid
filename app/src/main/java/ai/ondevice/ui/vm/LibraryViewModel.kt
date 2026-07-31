package ai.ondevice.ui.vm

import ai.ondevice.data.db.ConversationEntity
import ai.ondevice.data.db.GeneratedImageEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.SynthesisEntity
import ai.ondevice.data.db.TranscriptEntity
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

/** Which kind of output the library is showing. */
enum class LibrarySection(val label: String) {
    CHATS("Chats"),
    IMAGES("Images"),
    VOICE("Voice"),
}

/** A conversation plus the two facts a list row needs and the table does not hold. */
data class ConversationSummary(
    val conversation: ConversationEntity,
    val messageCount: Int,
    val preview: String,
)

/**
 * Everything this device has produced.
 *
 * Three of the four kinds were already being written and only one could be read
 * back: images had a gallery, conversations were reachable only as "whichever
 * one was open last", transcripts accumulated in a table nothing queried, and
 * rendered speech was not recorded at all. The work was never the storage — it
 * was that nothing enumerated it.
 */
@HiltViewModel
class LibraryViewModel @Inject constructor(
    private val db: OnDeviceDatabase,
) : ViewModel() {

    private val _section = MutableStateFlow(LibrarySection.CHATS)
    val section: StateFlow<LibrarySection> = _section.asStateFlow()

    /**
     * Counts and previews come from the messages table rather than the
     * conversation row, because the row has no idea what is in it — a thread
     * listed by title alone is unrecognisable once the titles are all "New
     * conversation".
     */
    val conversations: StateFlow<List<ConversationSummary>> =
        combine(
            db.conversations().observeAll(),
            db.messages().observeAllOrdered(),
        ) { threads, messages ->
            val byThread = messages.groupBy { it.conversationId }
            threads.mapNotNull { thread ->
                val theirs = byThread[thread.id].orEmpty()
                // A conversation row exists from the moment Chat opens, before
                // anything has been said. That is not history, it is the empty
                // page you are standing on — listing it put "Empty conversation
                // · 0 messages" at the top of the library every time.
                if (theirs.isEmpty()) return@mapNotNull null
                ConversationSummary(
                    conversation = thread,
                    messageCount = theirs.size,
                    preview = theirs.firstOrNull { it.content.isNotBlank() }?.content
                        ?.lineSequence()?.firstOrNull()?.take(120).orEmpty(),
                )
            }
        }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    val images: StateFlow<List<GeneratedImageEntity>> = db.images().observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    val syntheses: StateFlow<List<SynthesisEntity>> = db.syntheses().observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    val transcripts: StateFlow<List<TranscriptEntity>> = db.transcripts().observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    fun show(section: LibrarySection) {
        _section.value = section
    }

    /**
     * Deletes take the file with them.
     *
     * Leaving the WAV or the PNG behind would turn every removal into an orphan
     * the storage meter still counts and no screen can reach — the exact state
     * the orphan report exists to clean up after.
     */
    fun deleteImage(image: GeneratedImageEntity) {
        viewModelScope.launch {
            runCatching { java.io.File(image.path).delete() }
            db.images().deleteById(image.id)
        }
    }

    fun deleteSynthesis(synthesis: SynthesisEntity) {
        viewModelScope.launch {
            runCatching { java.io.File(synthesis.path).delete() }
            db.syntheses().deleteById(synthesis.id)
        }
    }

    /**
     * A transcript's source is the user's own recording or a file they chose, so
     * only the record goes.
     */
    fun deleteTranscript(transcript: TranscriptEntity) {
        viewModelScope.launch { db.transcripts().deleteById(transcript.id) }
    }
}
