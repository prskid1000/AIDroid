package ai.ondevice.ui.vm

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * A conversation and the turn it is generating, outliving any screen.
 *
 * The shortest of the four runs and the one most often interrupted, because
 * chat is what people leave the app *from*: a nine-billion-parameter answer is
 * minutes on this device, and switching away to read something else used to
 * end it — losing the tokens already generated and the prompt cache built to
 * produce them, so coming back cost the whole context again.
 *
 * See [VideoSession] for why the scope belongs here rather than in the view
 * model.
 */
@Singleton
class ChatSession @Inject constructor() {

    val state = MutableStateFlow(ChatState())

    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    var generationJob: Job? = null

    private var observing = false

    /** True the first time only, so the caller can attach its observers once. */
    fun claimObservers(): Boolean {
        if (observing) return false
        observing = true
        return true
    }
}
