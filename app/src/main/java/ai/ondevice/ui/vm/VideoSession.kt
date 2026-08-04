package ai.ondevice.ui.vm

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * A clip being made, and everything known about it, outliving any screen.
 *
 * The run used to belong to the view model: its state was the view model's
 * field and its coroutine was `viewModelScope`, so anything that cleared the
 * view model cancelled the generation. Making the view model activity-scoped
 * fixed the first way that happened — the Stills toggle popping the Video
 * entry — and gave the impression the problem was solved. It was not; it was
 * narrowed. Every other route to a destroyed activity took the run with it:
 * tapping the notification that reported the run built a second activity and
 * destroyed the first, and swiping the app out of recents still does.
 *
 * The scope is the app's, so a run ends when it finishes, fails, or somebody
 * cancels it — the three reasons a generation should end, and no others. What
 * a screen does is watch: two screens or none, opened and closed and opened
 * again, and the run neither notices nor stops.
 *
 * It also closes a quieter fault. Cancelling the collector did not reliably
 * reach the native call, which is a blocking one — so a view model dying
 * mid-generation could leave sd.cpp denoising in a process nothing was
 * listening to, holding its weights, with no way to stop it but to kill the
 * app. Nothing cancels this scope implicitly, so that path is gone.
 */
@Singleton
class VideoSession @Inject constructor() {

    /**
     * Held here rather than in the view model, which is the whole point: the
     * screen renders this, and a screen built after the run started sees a run
     * in progress rather than an idle form.
     */
    val state = MutableStateFlow(VideoState())

    /**
     * Main-immediate so state lands without a frame's delay, as it did on
     * `viewModelScope`, and supervised so one failed child cannot take the
     * session down with it.
     */
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    var generationJob: Job? = null

    var playbackJob: Job? = null

    /**
     * Whether the long-lived observers have been attached.
     *
     * They watch the model library and the download queue for the life of the
     * process, and a view model is built each time the screen is opened — so
     * without this each visit added another copy of every collector.
     */
    private var observing = false

    /** True the first time only, so the caller can attach its observers once. */
    fun claimObservers(): Boolean {
        if (observing) return false
        observing = true
        return true
    }
}
