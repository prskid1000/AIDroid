package ai.ondevice.proxy

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/** One thing the proxy did inside a request, worth showing on its own line. */
data class InterceptRecord(
    val kind: Kind,
    val name: String,
    val detail: String,
) {
    enum class Kind { TOOL_SEARCH, RAN_TOOL, AUTO_LOADED, BLOCKED, UNKNOWN_TOOL, REFUSED }
}

/**
 * What the proxy is doing right now, in the shape a status bar can show.
 *
 * Derived rather than stored: the request records already carry all of it, and
 * a second copy updated alongside them is a second thing that can be wrong.
 *
 * Null when nothing is in flight — which is what tells the notification to fall
 * back to saying the port is open rather than claiming work that has finished.
 */
data class ProxyActivity(
    val inFlight: Int,
    /** "Answering", "Making a picture" — what the person would call it. */
    val phase: String,
    /** The profile name, or the User-Agent. Who is asking. */
    val client: String,
    val model: String,
    val tokensPerSecond: Float = 0f,
    val step: Int = 0,
    val steps: Int = 0,
    val rounds: Int = 0,
)

data class RequestRecord(
    val id: String = UUID.randomUUID().toString(),
    val startedAt: Long = System.currentTimeMillis(),
    val method: String,
    val path: String,
    /**
     * What this request is doing, said the way the screens say it.
     *
     * Set by the route rather than derived from the path, because the path is
     * the protocol's word for it and this is the person's: `/v1/audio/speech`
     * is "Speaking", and nobody watching a notification wants the former.
     */
    val phase: String = "Working",
    /** Diffusion progress, for the routes that count steps. */
    val step: Int = 0,
    val steps: Int = 0,
    /** Whatever identified the caller: a profile name, else the User-Agent. */
    val client: String,
    val protocol: Protocol,
    /** What the client asked for, before the alias table. */
    val requestedModel: String,
    /** What actually ran. Differs whenever an alias or a default was involved. */
    val resolvedModel: String = "",
    val streaming: Boolean = false,
    val rounds: Int = 0,
    val promptTokens: Int = 0,
    val generatedTokens: Int = 0,
    val tokensPerSecond: Float = 0f,
    val intercepts: List<InterceptRecord> = emptyList(),
    val status: Int = 0,
    val error: String? = null,
    val finishedAt: Long? = null,
) {
    val durationMillis: Long get() = (finishedAt ?: System.currentTimeMillis()) - startedAt
    val inFlight: Boolean get() = finishedAt == null
}

/**
 * The last few hundred requests, in memory, and nowhere else.
 *
 * Nowhere else on purpose. This is the only place the answer to "why was that
 * one slow" exists — which round searched for what, which tool ran, what was
 * blocked — and it is also a record of every prompt's shape. A ring in RAM
 * gives the first without keeping the second past the life of the process.
 * Telecode's equivalent clears its disk dumps on startup for the same reason;
 * this one never writes them unless `proxy.debug` says to.
 */
@Singleton
class RequestLog @Inject constructor() {

    private val _records = MutableStateFlow<List<RequestRecord>>(emptyList())
    val records: StateFlow<List<RequestRecord>> = _records.asStateFlow()

    /**
     * The one in-flight request worth naming, or null.
     *
     * Its own flow rather than something the notification derives from
     * [records], because that list changes on every intercept and every token
     * and the notification must not be rebuilt that often. This only changes
     * when what it says would change.
     */
    private val _activity = MutableStateFlow<ProxyActivity?>(null)
    val activity: StateFlow<ProxyActivity?> = _activity.asStateFlow()

    /** Live request count, which is what the status card counts as "clients". */
    val inFlight: Int get() = _records.value.count { it.inFlight }

    fun begin(record: RequestRecord): String {
        _records.update { existing -> (listOf(record) + existing).take(CAPACITY) }
        recomputeActivity()
        return record.id
    }

    fun update(id: String, transform: (RequestRecord) -> RequestRecord) {
        _records.update { existing ->
            existing.map { if (it.id == id) transform(it) else it }
        }
        recomputeActivity()
    }

    /**
     * The oldest thing still running, because one engine runs at a time.
     *
     * Anything else in flight is queued behind it, so the oldest is the one
     * actually consuming the device — and the count beside it is what says the
     * others exist.
     */
    private fun recomputeActivity() {
        val live = _records.value.filter { it.inFlight }
        val current = live.minByOrNull { it.startedAt }
        val next = current?.let {
            ProxyActivity(
                inFlight = live.size,
                phase = it.phase,
                client = it.client,
                model = it.resolvedModel.ifBlank { it.requestedModel },
                tokensPerSecond = it.tokensPerSecond,
                step = it.step,
                steps = it.steps,
                rounds = it.rounds,
            )
        }
        if (next != _activity.value) _activity.value = next
    }

    fun intercept(id: String, record: InterceptRecord) {
        update(id) { it.copy(intercepts = it.intercepts + record) }
    }

    fun finish(id: String, status: Int, error: String? = null) {
        update(id) {
            it.copy(status = status, error = error, finishedAt = System.currentTimeMillis())
        }
    }

    /** Set once, by the route, before any work starts. */
    fun phase(id: String, phase: String) = update(id) { it.copy(phase = phase) }

    fun record(id: String): RequestRecord? = _records.value.firstOrNull { it.id == id }

    fun clear() {
        _records.value = emptyList()
        recomputeActivity()
    }

    private companion object {
        /**
         * Enough to cover a working session, small enough that the intercept
         * traces inside it cannot grow without bound — each record holds a list
         * of its own, and a long agentic run can produce fifteen per request.
         */
        const val CAPACITY = 200
    }
}
