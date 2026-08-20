package ai.ondevice.proxy

import ai.ondevice.core.Modality
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.engine.ModelRunner
import ai.ondevice.tools.ToolProviderFactory
import ai.ondevice.tools.ToolRegistry
import ai.ondevice.tools.Workspace
import android.content.Context
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.request.header
import io.ktor.server.request.receiveText
import io.ktor.server.response.respondBytes
import io.ktor.server.response.respondText
import io.ktor.server.response.respondTextWriter
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject

/**
 * One request, with everything it needs.
 *
 * A class rather than eight parameters on nine signatures, and the reason the
 * route implementations in the sibling files read as the thing they are for
 * rather than as plumbing.
 */
class ProxyCall(
    val call: ApplicationCall,
    val protocol: Protocol,
    val config: ProxyConfig,
    val requestId: String,
    val runner: ModelRunner,
    val log: RequestLog,
    val jobs: VideoJobs,
    val media: MediaSink,
    val toolProviders: ToolProviderFactory,
    val prefs: AppPrefs,
    val scope: CoroutineScope,
    val context: Context,
    private val allowedOrigins: List<String>,
) {

    /**
     * Name what this request is doing, for the notification.
     *
     * The route's word rather than the path's: `/v1/audio/speech` is
     * "Speaking", and the notification is read by somebody who did not send the
     * request and should not have to know the protocol to understand it.
     */
    fun phase(name: String) = log.phase(requestId, name)

    /** The body, parsed once, and kept for the log on the way past. */
    suspend fun body(): JsonObject {
        val raw = call.receiveText()
        log.request(requestId, raw)
        return runCatching { ProxyJson.parseToJsonElement(raw) as? JsonObject }.getOrNull()
            ?: throw ProxyRefusal.badRequest("The body is not a JSON object.")
    }

    /**
     * Resolve the model a request named, through the alias table.
     *
     * The alias table is the whole reason an unmodified Claude Code can be
     * pointed at this phone: it sends `claude-sonnet-4-6` because that is what
     * it was configured with, and the mapping turns that into whatever GGUF is
     * actually installed. Refusing by name, with the list, is the only useful
     * failure — "model not found" without saying what *was* found sends people
     * to the logs.
     */
    suspend fun resolveModel(requested: String, expect: Set<Modality>, route: String): ModelEntity {
        val target = config.resolveAlias(requested)
        val installed = runner.installed()

        val model = runner.model(target)
            ?: installed.firstOrNull { it.id.equals(target, ignoreCase = true) }
            ?: byLabel(target, installed)
            ?: throw ProxyRefusal.notFound(
                "No model called `$requested` is installed on this device.",
                "GET /v1/models lists what is — by id, by the name you gave it on the model's " +
                    "own screen, and by any alias configured here.",
            )
        ChatPipeline.requireModality(model, expect, route)
        log.update(requestId) { it.copy(requestedModel = requested, resolvedModel = model.id) }
        return model
    }

    /**
     * The name a person gave this model, or failing that its repo name.
     *
     * A model id here is `owner/repo:quant` — `unsloth/Qwen3.5-9B-GGUF:Q4_K_M` —
     * which is precise, unambiguous and no fun at all to type into a client's
     * config file. The app already has the answer: the model's own screen has a
     * rename field, and [ModelEntity.label] is that name or the repo name when
     * there is none. Matching it means renaming a model to `local` makes
     * `"model": "local"` work, with no alias to keep in step.
     *
     * Two rows can share a label — two quants of one repo, most obviously — and
     * picking one would be picking silently. Both are named instead, which is
     * something the caller can act on; an alias or the full id settles it.
     */
    private fun byLabel(target: String, installed: List<ModelEntity>): ModelEntity? {
        val matches = installed.filter { it.label.equals(target, ignoreCase = true) }
        if (matches.size <= 1) return matches.firstOrNull()
        throw ProxyRefusal.conflict(
            "`$target` names ${matches.size} installed models: " +
                matches.joinToString(", ") { "`${it.id}`" } + ".",
            "Ask for one by its full id, rename one of them on its own screen, or add an " +
                "alias under Settings → Proxy.",
        )
    }

    /** The model for a modality when a request named none. */
    suspend fun defaultModel(modality: Modality, route: String): ModelEntity =
        runner.defaultFor(modality)
            ?: throw ProxyRefusal.notFound(
                "No ${modality.label.lowercase()} model is installed on this device, " +
                    "so $route cannot be served.",
                "Install one from the Models screen in the app.",
            )

    /**
     * The tools this device runs itself, if tools are switched on at all.
     *
     * Off by default at the app level, and that gate is honoured here: a model
     * that is never told tools exist cannot call one, which is the safe answer
     * when some of them reach a server the person does not control — and more
     * so when the caller is on the other end of a socket.
     */
    suspend fun localTools(offerVideo: Boolean = false): ToolRegistry? {
        if (!prefs.toolsEnabled.first()) return null
        return toolProviders.registry(
            enabled = prefs.enabledToolProviders.first(),
            fileScope = if (prefs.fileScopeDevice.first()) Workspace.Scope.DEVICE else Workspace.Scope.SANDBOX,
            tuning = ai.ondevice.core.SparseParams.parse(prefs.toolParams.first()),
            offerVideo = offerVideo,
        )
    }

    // ── responding ──────────────────────────────────────────────────────

    suspend fun json(body: String) {
        log.response(requestId, body)
        cors()
        call.respondText(body, ContentType.Application.Json)
    }

    suspend fun bytes(body: ByteArray, contentType: ContentType) {
        cors()
        call.respondBytes(body, contentType)
    }

    /**
     * Server-sent events, flushed after every write.
     *
     * The flush is explicit and unconditional, and it is the single most
     * important line in this file. Without it a client shows nothing for two
     * minutes and then everything at once — which reads as a hung server rather
     * than a slow model, and is the symptom telecode spent the most effort on.
     */
    suspend fun stream(body: suspend (Emit) -> Unit) {
        cors()
        // The frames are kept as they are written, capped, so a streamed answer
        // is as inspectable afterwards as a buffered one. A stream has no
        // single response body, and "streamed to client — not captured" is what
        // telecode's log says here; it is the one thing in its viewer that
        // cannot answer the question you opened it to ask.
        val transcript = StringBuilder()
        var frames = 0
        call.respondTextWriter(ContentType.Text.EventStream, HttpStatusCode.OK) {
            val writer = this
            body(
                object : Emit {
                    override suspend fun invoke(chunk: String) {
                        if (chunk.isEmpty()) return
                        frames++
                        if (transcript.length < TRANSCRIPT_CEILING) transcript.append(chunk)
                        withContext(Dispatchers.IO) {
                            writer.write(chunk)
                            writer.flush()
                        }
                    }
                },
            )
        }
        log.response(requestId, transcript.toString(), frames)
    }

    private companion object {
        /** Enough to hold a whole answer's frames; the log caps again after. */
        const val TRANSCRIPT_CEILING = 60_000
    }

    /** One frame out. */
    fun interface Emit {
        suspend operator fun invoke(chunk: String)
    }

    private fun cors() {
        call.applyCors(call.request.header(HttpHeaders.Origin), allowedOrigins)
    }
}
