package ai.ondevice.ui.vm

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * A picture being made, outliving any screen — the still half of [VideoSession].
 *
 * Same reasoning, and the same fault it fixes: a run whose state was a view
 * model's field and whose coroutine was `viewModelScope` ended whenever the
 * view model did, for reasons that had nothing to do with the run. A still is
 * quicker than a clip and the window is narrower, but a four-step FLUX at 512
 * square is minutes on this hardware, and the upscaler is longer than that.
 *
 * The two are separate sessions rather than one because they are two runs with
 * two forms, and the diffusion engine already serialises what matters: they
 * share a context and a load lock, so only one can be sampling at a time.
 */
@Singleton
class ImageSession @Inject constructor() {

    val state = MutableStateFlow(ImageState())

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
