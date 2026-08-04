package ai.ondevice.ui.vm

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Speaking and transcribing, outliving any screen — see [VideoSession].
 *
 * Transcription is the one that most needs it. A clip is minutes and a spoken
 * line is seconds, but an hour of recorded audio is an hour of CPU, and it was
 * running in a scope that ended when the screen did.
 *
 * Recording is deliberately *not* moved here. It owns the microphone, and a
 * microphone that keeps recording after its screen has gone is a different
 * kind of surprise from a generation that keeps generating.
 */
@Singleton
class VoiceSession @Inject constructor() {

    val state = MutableStateFlow(VoiceState())

    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    var speakJob: Job? = null

    var transcribeJob: Job? = null

    private var observing = false

    /** True the first time only, so the caller can attach its observers once. */
    fun claimObservers(): Boolean {
        if (observing) return false
        observing = true
        return true
    }
}
