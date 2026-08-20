package ai.ondevice.proxy

import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * A clip being made for somebody who is not holding the connection.
 *
 * Video is the one modality that cannot be answered inline. A Wan run measured
 * on this hardware is three quarters of an hour at 384 square; no client's idle
 * timeout survives that, and neither does a phone's Wi-Fi as it walks out of
 * range of one access point and into another. So the route answers immediately
 * with an id, and the work carries on in the app's own scope.
 *
 * The shape mirrors OpenAI's Videos API — create, poll, fetch content — because
 * a client that already speaks it needs no special case, and because that shape
 * exists for exactly this reason.
 */
@Singleton
class VideoJobs @Inject constructor() {

    enum class State { QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED }

    data class Job(
        val id: String = "video_" + UUID.randomUUID().toString().replace("-", "").take(20),
        val model: String,
        val prompt: String,
        val state: State = State.QUEUED,
        val createdAt: Long = System.currentTimeMillis(),
        val completedAt: Long? = null,
        val step: Int = 0,
        val steps: Int = 0,
        val phase: String = "",
        val secondsPerStep: Float = 0f,
        /** Where the frames landed, once there are any. */
        val directory: String? = null,
        val frames: List<String> = emptyList(),
        val fps: Int = 0,
        val audioPath: String? = null,
        val error: String? = null,
        val suggestion: String? = null,
        /** Jobs waiting behind this one, for the notification's last clause. */
        val queuedBehind: Int = 0,
    ) {
        val terminal: Boolean get() = state == State.COMPLETED || state == State.FAILED || state == State.CANCELLED

        /**
         * Rough, and honest about it.
         *
         * Steps are the only thing a diffusion run counts, and the phases
         * either side of sampling — loading weights, decoding the latents —
         * report none. A clip measured on this device spent more than half its
         * wall clock in decode, so a bar driven by steps alone reads as stuck
         * at 100% for minutes. Reported as a fraction of sampling, and the
         * phase name is carried beside it so the number is never the only thing
         * on screen.
         */
        val progress: Float
            get() = when {
                state == State.COMPLETED -> 1f
                steps > 0 -> (step.toFloat() / steps).coerceIn(0f, 1f)
                else -> 0f
            }
    }

    private val _jobs = MutableStateFlow<List<Job>>(emptyList())
    val jobs: StateFlow<List<Job>> = _jobs.asStateFlow()

    /**
     * The one job worth naming, or null when none is running.
     *
     * Its own flow because a clip is the only run with no request behind it:
     * the caller was answered within the second and hung up, so nothing in the
     * request log knows this is happening and the notification had nothing to
     * say. Measured — three minutes of "Working" with an empty second line.
     *
     * The oldest still running, because the engine gate admits one; anything
     * else is queued behind it, and [Job.queuedBehind] is what says so.
     */
    private val _current = MutableStateFlow<Job?>(null)
    val current: StateFlow<Job?> = _current.asStateFlow()

    private val cancels = mutableMapOf<String, () -> Unit>()

    fun create(model: String, prompt: String): Job {
        val job = Job(model = model, prompt = prompt)
        _jobs.update { (listOf(job) + it).take(CAPACITY) }
        recompute()
        return job
    }

    fun get(id: String): Job? = _jobs.value.firstOrNull { it.id == id }

    fun update(id: String, transform: (Job) -> Job) {
        _jobs.update { existing -> existing.map { if (it.id == id) transform(it) else it } }
        recompute()
    }

    private fun recompute() {
        val live = _jobs.value.filter { !it.terminal }
        val next = live.minByOrNull { it.createdAt }
            ?.copy(queuedBehind = (live.size - 1).coerceAtLeast(0))
        if (next != _current.value) _current.value = next
    }

    fun attachCancel(id: String, cancel: () -> Unit) {
        synchronized(cancels) { cancels[id] = cancel }
    }

    /**
     * Stop a job, whether it has started or not.
     *
     * A queued job is simply marked; a running one has to have its native call
     * reached, because cancelling the coroutine is not enough — a JNI call is
     * not interruptible and sd.cpp would carry on denoising in a process
     * nothing was listening to.
     */
    fun cancel(id: String): Boolean {
        val job = get(id) ?: return false
        if (job.terminal) return false
        synchronized(cancels) { cancels.remove(id) }?.invoke()
        update(id) { it.copy(state = State.CANCELLED, completedAt = System.currentTimeMillis()) }
        return true
    }

    fun finished(id: String) {
        synchronized(cancels) { cancels.remove(id) }
    }

    /** Jobs still worth showing — anything running, plus recent finished ones. */
    val active: Int get() = _jobs.value.count { !it.terminal }

    private companion object {
        const val CAPACITY = 50
    }
}
