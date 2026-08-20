package ai.ondevice.engine

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * What a run left behind, announced by whatever made it.
 *
 * The finished-run notification was edge-triggered off the three in-app
 * sessions — `ImageSession`, `VideoSession`, `ChatSession` — which works for
 * anything started by tapping a screen and cannot work for anything else. A
 * proxy request touches none of them, so a picture generated over HTTP, a clip
 * that took three minutes and an answer written while the app was in the
 * background all completed in silence. The one class of run most likely to
 * finish while nobody is looking was the one class that said nothing when it
 * did.
 *
 * A flow rather than more edge detection, because there is nothing to detect an
 * edge in: the producer knows the moment it is done and knows what it produced,
 * and every other arrangement is that fact inferred from something else.
 *
 * `MutableSharedFlow` with no replay: a result is an event. Replaying it would
 * re-announce the last picture every time the service restarted, which is the
 * notification equivalent of shouting an old headline.
 */
@Singleton
class RunResults @Inject constructor() {

    private val _produced = MutableSharedFlow<RunResultNotifier.Result>(
        replay = 0,
        extraBufferCapacity = 8,
    )
    val produced: SharedFlow<RunResultNotifier.Result> = _produced.asSharedFlow()

    fun picture(path: String, caption: String) =
        emit(RunResultNotifier.Result.Picture("Picture ready", path, caption))

    fun clip(firstFrame: String?, caption: String) =
        emit(RunResultNotifier.Result.Clip("Clip ready", firstFrame, caption))

    fun words(title: String, body: String, caption: String) =
        emit(RunResultNotifier.Result.Words(title, body, caption))

    fun sound(path: String, caption: String) =
        emit(RunResultNotifier.Result.Sound("Speech ready", path, caption))

    /**
     * Non-blocking, and dropped rather than queued if nobody is listening.
     *
     * The producer is inside a generation and must not be made to wait on a
     * notification. A result nobody collected is a result that went unshown,
     * which is a smaller failure than a run that stalled to report itself.
     */
    private fun emit(result: RunResultNotifier.Result) {
        _produced.tryEmit(result)
    }
}
