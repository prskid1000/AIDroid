package ai.ondevice.proxy

import ai.ondevice.data.log.LogFile
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/** One thing the proxy did inside a request, worth showing on its own line. */
@Serializable
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

@Serializable
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
    /**
     * What the client actually sent, and what went back.
     *
     * Kept because the intercept list says *what* the proxy did and these say
     * *why* — a tool the model would not call, a system prompt that was not
     * what you thought, a history the client re-sent with something extra in
     * it. Telecode writes the same pair to disk under `proxy_full_*.json`; these
     * go to disk too, once the request has finished — see [RequestLog].
     *
     * Redacted and capped on the way in: see [RequestLog.forDisplay].
     */
    val requestBody: String = "",
    val responseBody: String = "",
    /** SSE frames written, for a streamed answer that has no single body. */
    val frames: Int = 0,
) {
    val durationMillis: Long get() = (finishedAt ?: System.currentTimeMillis()) - startedAt
    val inFlight: Boolean get() = finishedAt == null
}

/**
 * The last few hundred requests, and the file they are written to.
 *
 * This is the only place the answer to "why was that one slow" exists — which
 * round searched for what, which tool ran, what was blocked — and it is also a
 * record of every prompt's shape. It was a ring in RAM and nothing else, which
 * kept the second of those from outliving the process; it also meant the
 * process being killed, which is the thing most worth understanding, wiped the
 * evidence of what led up to it. So it is now written down, deliberately and
 * with the bodies, because a row without its request body answers "what ran"
 * and not "why did that come back".
 *
 * **A finished request is what gets written.** One line per request, appended
 * once, at the end: a record is touched on every intercept and every token, and
 * re-encoding twenty kilobytes of body per token would cost more than the
 * answer. It also means an in-flight request does not survive a kill, which is
 * correct — it did not finish, and a restored row claiming otherwise would be
 * the silent lie in §1.2.
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
        record(id)?.let { finished -> write(finished) }
    }

    /** Set once, by the route, before any work starts. */
    fun phase(id: String, phase: String) = update(id) { it.copy(phase = phase) }

    fun request(id: String, body: String) =
        update(id) { it.copy(requestBody = forDisplay(body)) }

    fun response(id: String, body: String, frames: Int = 0) =
        update(id) { it.copy(responseBody = forDisplay(body), frames = frames) }

    companion object {
        /**
         * A body, made safe to keep and possible to read.
         *
         * Two things happen here. Base64 payloads are replaced by a note of
         * their size: a single screenshot on a chat turn is four megabytes of
         * text, twenty of those is the whole ring buffer, and none of it is
         * legible anyway. Then the result is capped, because a long
         * conversation re-sends its entire history every turn and the tail is
         * the part nobody is reading.
         */
        fun forDisplay(body: String): String {
            if (body.isEmpty()) return body
            val stripped = BASE64.replace(body) { match ->
                val bytes = match.groupValues[1].length * 3 / 4
                "\"<${bytes / 1024} kB of base64, not kept>\""
            }
            return if (stripped.length <= MAX_BODY_CHARS) {
                stripped
            } else {
                stripped.take(MAX_BODY_CHARS) +
                    "\n\n… ${stripped.length - MAX_BODY_CHARS} more characters not kept"
            }
        }

        /** Long runs of base64, whether bare or inside a data URI. */
        private val BASE64 = Regex("\"(?:data:[^;\"]*;base64,)?([A-Za-z0-9+/=]{512,})\"")

        private const val MAX_BODY_CHARS = 20_000

        /**
         * Enough to cover a working session, small enough that what the records
         * hold cannot grow without bound — each carries an intercept list and
         * two request bodies, and a long agentic run produces fifteen
         * intercepts per request.
         */
        const val CAPACITY = 200
    }

    fun record(id: String): RequestRecord? = _records.value.firstOrNull { it.id == id }

    fun clear() {
        _records.value = emptyList()
        store?.clear()
        recomputeActivity()
    }

    // ── the file behind the ring ────────────────────────────────────────

    /**
     * Read what the last session did, and write this one's down.
     *
     * Called once, from the application. Restored records are merged behind
     * whatever this process has already served rather than replacing it, and
     * de-duplicated by id: a request that somehow finished twice would
     * otherwise appear twice, and the second line is the more complete one.
     */
    fun persistTo(file: File, scope: CoroutineScope) {
        val log = LogFile(file, CAPACITY)
        store = log
        writer = scope
        scope.launch(Dispatchers.IO) {
            val restored = log.read().mapNotNull { line ->
                runCatching { JSON.decodeFromString(RequestRecord.serializer(), line) }.getOrNull()
            }
            if (restored.isEmpty()) return@launch
            _records.update { live ->
                val seen = live.mapTo(mutableSetOf()) { it.id }
                // Newest first, which is the order this list is in and the
                // order the screen reads it in.
                (live + restored.reversed().filter { seen.add(it.id) }).take(CAPACITY)
            }
        }
    }

    private fun write(record: RequestRecord) {
        val log = store ?: return
        val line = runCatching { JSON.encodeToString(RequestRecord.serializer(), record) }
            .getOrNull() ?: return
        // Off whatever thread finished the request — that one is a Ktor worker
        // holding a connection, and a file append is not something to make a
        // client wait for.
        writer?.launch(Dispatchers.IO) { log.append(listOf(line)) }
    }

    @Volatile
    private var store: LogFile? = null

    @Volatile
    private var writer: CoroutineScope? = null

    private val JSON = Json { ignoreUnknownKeys = true; encodeDefaults = true }
}
