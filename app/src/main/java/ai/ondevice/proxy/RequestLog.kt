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

data class RequestRecord(
    val id: String = UUID.randomUUID().toString(),
    val startedAt: Long = System.currentTimeMillis(),
    val method: String,
    val path: String,
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

    /** Live request count, which is what the status card counts as "clients". */
    val inFlight: Int get() = _records.value.count { it.inFlight }

    fun begin(record: RequestRecord): String {
        _records.update { existing -> (listOf(record) + existing).take(CAPACITY) }
        return record.id
    }

    fun update(id: String, transform: (RequestRecord) -> RequestRecord) {
        _records.update { existing ->
            existing.map { if (it.id == id) transform(it) else it }
        }
    }

    fun intercept(id: String, record: InterceptRecord) {
        update(id) { it.copy(intercepts = it.intercepts + record) }
    }

    fun finish(id: String, status: Int, error: String? = null) {
        update(id) {
            it.copy(status = status, error = error, finishedAt = System.currentTimeMillis())
        }
    }

    fun record(id: String): RequestRecord? = _records.value.firstOrNull { it.id == id }

    fun clear() {
        _records.value = emptyList()
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
