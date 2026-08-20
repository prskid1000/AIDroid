package ai.ondevice.proxy

import ai.ondevice.core.Modality
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.data.secure.TokenStore
import ai.ondevice.engine.ModelRunner
import ai.ondevice.tools.ToolProviderFactory
import android.content.Context
import android.os.BatteryManager
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.cio.CIO
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.embeddedServer
import io.ktor.server.plugins.origin
import io.ktor.server.request.header
import io.ktor.server.request.httpMethod
import io.ktor.server.request.path
import io.ktor.server.response.header
import io.ktor.server.response.respondText
import io.ktor.server.routing.Routing
import io.ktor.server.routing.RoutingContext
import io.ktor.server.routing.get
import io.ktor.server.routing.options
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.security.SecureRandom
import javax.inject.Inject
import javax.inject.Singleton

/**
 * The HTTP surface: an Anthropic and an OpenAI API in front of this device's
 * own engines.
 *
 * Lives inside `InferenceService` rather than in a service of its own, because
 * that service already declares the `specialUse` foreground type and already
 * exists to stop the system reclaiming a process holding gigabytes. A second
 * foreground service holding a socket beside it would be two answers to one
 * question.
 *
 * Nothing here touches the main thread. `WorkflowSession.scope` is
 * `Main.immediate` because it holds state; nothing in this file holds state a
 * frame cares about, and a blocked main thread does not merely ANR — it also
 * freezes the activity lifecycle callbacks, which is how a backgrounded app
 * once came to believe it was still on screen.
 */
@Singleton
class ProxyServer @Inject constructor(
    @dagger.hilt.android.qualifiers.ApplicationContext private val context: Context,
    private val prefs: AppPrefs,
    private val tokens: TokenStore,
    private val runner: ModelRunner,
    private val toolProviders: ToolProviderFactory,
    private val log: RequestLog,
    private val videoJobs: VideoJobs,
    @ai.ondevice.di.ApplicationScope private val scope: CoroutineScope,
) {

    data class Status(
        /**
         * Whether the proxy is switched on, as distinct from listening.
         *
         * The two come apart for a second at a time and something depends on
         * the difference: `sync()` closes the socket before it opens the new
         * one, so `listening` is briefly false during every rebind — and the
         * service was reading that as "nothing to do here" and stopping itself
         * mid-restart. What survived was a frozen background process still
         * holding the port, which accepts a connection and then never answers.
         */
        val enabled: Boolean = false,
        val listening: Boolean = false,
        val address: String? = null,
        val port: Int = 0,
        /** The MagicDNS name, when a reverse lookup found one. */
        val hostname: String? = null,
        val onTailnet: Boolean = false,
        /** Why it is not listening, when it was asked to be and is not. */
        val refusal: String? = null,
    ) {
        val url: String? get() = address?.let { "http://${hostname ?: it}:$port" }
    }

    private val _status = MutableStateFlow(Status())
    val status: StateFlow<Status> = _status.asStateFlow()

    /**
     * What the proxy is working on, for the notification.
     *
     * Forwarded from the request log rather than tracked again here: the log
     * already knows, and a second copy is a second thing that can be stale.
     */
    val activity: StateFlow<ProxyActivity?> get() = log.activity

    private var server: EmbeddedServer<*, *>? = null

    /**
     * What is allowed to wait for the engine.
     *
     * The engine gate itself is in [ModelRunner] and admits one. This is the
     * queue in front of it, and its whole job is to make "everything is waiting"
     * answerable with a 429 rather than with a connection that never returns.
     */
    private var admission = Semaphore(DEFAULT_QUEUE_DEPTH)
    private var admissionDepth = DEFAULT_QUEUE_DEPTH

    private val media = FileMediaSink { runner.scratchDir("inbound") }

    /** Written only when `proxy.debug` is on; see [RequestDump]. */
    private val dump = RequestDump { runner.scratchDir("logs") }

    /** Registered once, so a tailnet that comes back brings the server with it. */
    private var networkWatch: android.net.ConnectivityManager.NetworkCallback? = null

    private suspend fun document(): ProxyDocument =
        ProxyDocument.parse(prefs.proxyDocument.first())

    // ── lifecycle ───────────────────────────────────────────────────────

    /**
     * Start, stop or restart to match the stored configuration.
     *
     * Idempotent, so the service's start command, the settings screen and a
     * network change can all call it without any of them knowing what the
     * others did.
     */
    suspend fun sync() {
        val config = ProxyConfig(document())
        if (!config.enabled) {
            stop()
            unwatchNetwork()
            _status.value = Status()
            return
        }
        watchNetwork()

        when (val bind = Reachability.resolveBindAddress(config.bind)) {
            is Reachability.BindResult.NoTailnet -> {
                // A refusal, not a fallback. Quietly binding 0.0.0.0 because a
                // VPN happened to be down would put a generation server on
                // whatever Wi-Fi this phone is on, which is the one outcome the
                // tailnet default exists to prevent.
                stop()
                _status.value = Status(
                    enabled = true,
                    refusal = "Tailscale is not connected, so there is no 100.x address to " +
                        "bind to. Open the Tailscale app, or change Listen on.",
                )
            }
            is Reachability.BindResult.Ok -> start(bind.address, config)
        }
    }

    private suspend fun start(address: String, config: ProxyConfig) {
        val port = config.port
        val current = _status.value
        if (server != null && current.listening && current.address == address && current.port == port) {
            return
        }

        stop()
        if (config.requireAuth && tokens.proxyToken == null) regenerateToken()

        // Last session's dumps go when this one starts. They are worth exactly
        // as long as the session that produced them, and a folder that only
        // grows is one somebody eventually finds full of old prompts.
        if (config.debug) {
            val removed = dump.clearPrevious()
            if (removed > 0) {
                ai.ondevice.engine.EngineLog.i(
                    "ProxyServer",
                    "cleared $removed request dump(s) from the previous session",
                )
            }
        }

        admissionDepth = config.queueDepth
        admission = Semaphore(admissionDepth)

        runCatching {
            embeddedServer(CIO, host = address, port = port) {
                routing { install() }
            }.also { it.start(wait = false) }
        }.onSuccess { engine ->
            server = engine
            val onTailnet = Reachability.isTailscale(address)
            _status.value = Status(
                enabled = true,
                listening = true,
                address = address,
                port = port,
                onTailnet = onTailnet,
            )
            // The card goes live on the address and gains a name afterwards if
            // there is one: a reverse lookup has to leave the device and often
            // returns nothing, and a status that waits for it looks broken.
            if (onTailnet) {
                scope.launch {
                    Reachability.magicDnsName(address)?.let { name ->
                        _status.update { it.copy(hostname = name) }
                    }
                }
            }
        }.onFailure { failure ->
            _status.value = Status(
                enabled = true,
                refusal = "Could not listen on $address:$port — ${failure.message}",
            )
        }
    }

    fun stop() {
        runCatching { server?.stop(GRACE_MILLIS, TIMEOUT_MILLIS) }
        server = null
        _status.update { it.copy(listening = false, hostname = null) }
    }

    /**
     * Re-sync whenever the set of networks changes.
     *
     * Without this the tailnet address is read exactly once, at whatever moment
     * the service happened to start, and a VPN that was reconnecting at that
     * instant left the proxy refusing forever — switched on, saying "Tailscale
     * is not connected", and never asking again. Tailscale on Android is a VPN
     * that drops on a network change, a doze, or its own reconnect, so "once"
     * was never going to be enough.
     *
     * Every callback, not only the VPN one: losing Wi-Fi tears down the tunnel
     * over it, and the tunnel coming back is a separate event from the network
     * under it coming back. Both end in the same idempotent [sync].
     */
    private fun watchNetwork() {
        if (networkWatch != null) return
        val manager = context.getSystemService(android.net.ConnectivityManager::class.java)
            ?: return
        val callback = object : android.net.ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: android.net.Network) = resync()
            override fun onLost(network: android.net.Network) = resync()
            override fun onCapabilitiesChanged(
                network: android.net.Network,
                capabilities: android.net.NetworkCapabilities,
            ) = resync()
        }
        networkWatch = callback
        runCatching {
            manager.registerNetworkCallback(
                android.net.NetworkRequest.Builder()
                    // VPN transports are not in the default request, and the
                    // tailnet is a VPN — asking for the default network only
                    // would miss the one interface this cares about.
                    .removeCapability(android.net.NetworkCapabilities.NET_CAPABILITY_NOT_VPN)
                    .build(),
                callback,
            )
        }.onFailure { networkWatch = null }
    }

    private fun unwatchNetwork() {
        val callback = networkWatch ?: return
        networkWatch = null
        runCatching {
            context.getSystemService(android.net.ConnectivityManager::class.java)
                ?.unregisterNetworkCallback(callback)
        }
    }

    /**
     * Debounced, because a single reconnect produces a burst of callbacks.
     *
     * Re-binding a socket per callback would mean tearing the server down and
     * up several times in a second, and each teardown is a window where a
     * client in the middle of a request is dropped.
     */
    private fun resync() {
        resyncJob?.cancel()
        resyncJob = scope.launch {
            kotlinx.coroutines.delay(RESYNC_DEBOUNCE_MILLIS)
            runCatching { sync() }
        }
    }

    private var resyncJob: kotlinx.coroutines.Job? = null

    val listening: Boolean get() = server != null && _status.value.listening

    /** A fresh token, replacing any previous one. */
    fun regenerateToken(): String {
        val bytes = ByteArray(TOKEN_BYTES)
        SecureRandom().nextBytes(bytes)
        val token = "ond-" + bytes.joinToString("") { "%02x".format(it) }
        tokens.proxyToken = token
        return token
    }

    // ── routing ─────────────────────────────────────────────────────────

    private fun Routing.install() {
        // Pre-flight, hand-rolled rather than the CORS plugin because the
        // allowed list is edited at runtime and a plugin is configured once.
        options("/{...}") { call.preflight() }

        get("/health") { call.health() }
        get("/v1/models") { call.models() }
        get("/v1/models/{id}") { call.singleModel() }

        post("/v1/messages") { serve(Protocol.ANTHROPIC) { it.chat() } }
        post("/v1/messages/count_tokens") { serve(Protocol.ANTHROPIC) { it.countTokens() } }
        post("/v1/chat/completions") { serve(Protocol.OPENAI) { it.chat() } }

        post("/v1/images/generations") { serve(Protocol.OPENAI) { it.images(ImageMode.GENERATE) } }
        post("/v1/images/edits") { serve(Protocol.OPENAI) { it.images(ImageMode.EDIT) } }
        post("/v1/images/upscales") { serve(Protocol.OPENAI) { it.images(ImageMode.UPSCALE) } }

        post("/v1/audio/speech") { serve(Protocol.OPENAI) { it.speech() } }
        post("/v1/audio/transcriptions") { serve(Protocol.OPENAI) { it.transcription(translate = false) } }
        post("/v1/audio/translations") { serve(Protocol.OPENAI) { it.transcription(translate = true) } }

        post("/v1/videos") { serve(Protocol.OPENAI) { it.createVideo() } }
        get("/v1/videos/{id}") { call.videoStatus() }
        get("/v1/videos/{id}/content") { call.videoContent() }
        post("/v1/videos/{id}/cancel") { call.videoCancel() }

        // Named rather than silently 404: a client asking for embeddings has a
        // specific need, and the difference between "you typed it wrong" and
        // "this device cannot do that yet" is the difference between retrying
        // and going elsewhere. See docs/proxy-plan.md 2.3.
        post("/v1/embeddings") {
            call.refuse(
                Protocol.OPENAI,
                ProxyRefusal.notImplemented(
                    "This device does not compute embeddings.",
                    "Nothing in this app has ever asked for a vector, so there is no " +
                        "embeddings path through the JNI boundary. Adding one is a native " +
                        "change and a runtime contract bump, not a setting.",
                ),
            )
        }
    }

    /**
     * The wrapper every generating route shares.
     *
     * Auth, profile matching, admission, power, logging and refusal translation
     * in one place — so a route below is only ever the thing it is for.
     */
    private suspend fun RoutingContext.serve(
        protocol: Protocol,
        block: suspend (ProxyCall) -> Unit,
    ) {
        val document = document()
        val bearer = call.bearer()
        val profile = ProxyConfig.match(document, { call.request.header(it) }, bearer)
        val config = ProxyConfig(document, profile)

        val id = log.begin(
            RequestRecord(
                method = call.request.httpMethod.value,
                path = call.request.path(),
                client = profile?.name
                    ?: call.request.header(HttpHeaders.UserAgent)
                    ?: call.request.origin.remoteHost,
                protocol = protocol,
                requestedModel = "",
            ),
        )

        try {
            surfaceEnabled(config, protocol, call.request.path())
            authorize(config, bearer)
            guardPower(config)

            // Two refusals, and the difference matters to whoever is asking.
            // A full queue means "come back later"; a wait that ran out means
            // this device is stuck on something long, and only the second is
            // worth investigating.
            if (!admission.tryAcquire()) {
                throw ProxyRefusal.busy(
                    "This device is already working and $admissionDepth requests are waiting.",
                    retryAfter = RETRY_AFTER_SECONDS,
                )
            }
            try {
                // The engine gate is inside ModelRunner and admits one at a
                // time, so waiting is the normal case. Waiting forever is not,
                // and a connection that never returns is the hardest kind of
                // failure to diagnose from the far end.
                withTimeoutOrNull(config.queueTimeoutSeconds.toLong() * 1000L) {
                    // Held for the whole request, exactly as every in-app
                    // generation path holds it.
                    //
                    // The foreground service keeps the *process*; this keeps the
                    // *CPU*. Without it a request arriving while the screen is
                    // off runs against a core the kernel is free to idle, and
                    // the symptom is a generation that takes six times as long
                    // for no reason anyone can see from either end. It also
                    // counts the run, which is what stops the service deciding
                    // it has nothing to do halfway through one.
                    ai.ondevice.engine.InferenceService.holdingWakeLock(context) {
                        block(
                            ProxyCall(
                                call = call,
                                protocol = protocol,
                                config = config,
                                requestId = id,
                                runner = runner,
                                log = log,
                                jobs = videoJobs,
                                media = media,
                                toolProviders = toolProviders,
                                prefs = prefs,
                                scope = scope,
                                context = context,
                                allowedOrigins = document.corsOrigins,
                            ),
                        )
                    }
                } ?: throw ProxyRefusal.unavailable(
                    "Gave up after ${config.queueTimeoutSeconds}s waiting for this device's engine.",
                    "Raise `proxy.queue_timeout_sec`, or wait for whatever is running to finish.",
                    retryAfter = RETRY_AFTER_SECONDS,
                )
                log.finish(id, call.response.status()?.value ?: HttpStatusCode.OK.value)
            } finally {
                admission.release()
            }
        } catch (refusal: ProxyRefusal) {
            log.finish(id, refusal.status, refusal.message)
            call.refuse(protocol, refusal)
        } catch (failure: Throwable) {
            val refusal = ChatPipeline.refusalFor(failure)
            log.finish(id, refusal.status, refusal.message)
            call.refuse(protocol, refusal)
        } finally {
            // After `finish`, so what is written is the whole record rather
            // than one missing its status and its duration. A refusal is worth
            // keeping too — it is the one somebody is most likely to be
            // looking for.
            if (config.debug) log.record(id)?.let { dump.write(it) }
        }
    }

    /** A protocol or a modality that has been switched off answers as switched off. */
    private fun surfaceEnabled(config: ProxyConfig, protocol: Protocol, path: String) {
        val on = when {
            protocol == Protocol.ANTHROPIC -> config.anthropicEnabled
            path.startsWith("/v1/images") -> config.openAiEnabled && config.servesImages
            path.startsWith("/v1/audio") -> config.openAiEnabled && config.servesAudio
            path.startsWith("/v1/videos") -> config.openAiEnabled && config.servesVideo
            else -> config.openAiEnabled
        }
        if (!on) {
            throw ProxyRefusal.notFound(
                "That surface is switched off on this device.",
                "Turn it on under Settings → Proxy.",
            )
        }
    }

    private fun authorize(config: ProxyConfig, bearer: String?) {
        if (!config.requireAuth) return
        val expected = tokens.proxyToken
            ?: throw ProxyRefusal.unauthorized(
                "This server requires a token and none has been generated.",
            )
        // Constant-time, because this runs for every request from anything that
        // can reach the port and a short-circuit compare leaks the prefix to
        // anything patient.
        if (bearer == null || !constantTimeEquals(bearer, expected)) {
            throw ProxyRefusal.unauthorized(
                "Missing or incorrect token. Send it as `Authorization: Bearer …` or `x-api-key`.",
            )
        }
    }

    /**
     * Refuse rather than get quietly worse.
     *
     * A phone that keeps accepting work as it drops below a usable charge, or
     * as it throttles, is the failure this codebase's first principle is
     * written against: the requests still succeed, they just take four times as
     * long, and nothing anywhere says why.
     */
    private fun guardPower(config: ProxyConfig) {
        val manager = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager ?: return
        val level = manager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        val charging = manager.isCharging

        if (config.chargingOnly && !charging) {
            throw ProxyRefusal.unavailable(
                "This device serves only while charging, and it is not charging.",
                "Turn off `proxy.charging_only`, or plug it in.",
            )
        }
        if (level in 0..config.batteryFloor && !charging) {
            throw ProxyRefusal.unavailable(
                "Battery is at $level%, at or below the floor of ${config.batteryFloor}%.",
                "Plug it in, or lower `proxy.battery_floor`.",
            )
        }
    }

    // ── the routes that neither generate nor need a body ────────────────

    private suspend fun ApplicationCall.health() {
        cors()
        val status = _status.value
        respondText(
            encode(
                buildJsonObject {
                    put("status", "ok")
                    put("listening", status.listening)
                    put("tailnet", status.onTailnet)
                    put("busy", runner.busy)
                    put("resident", runner.residentRuntime ?: "")
                    put("in_flight", log.inFlight)
                    put("video_jobs", videoJobs.active)
                },
            ),
            ContentType.Application.Json,
        )
    }

    /**
     * What this device can be asked for.
     *
     * Aliases are listed beside the real ids, because a client pointed here by
     * an alias has to see that name in the list or it refuses to use it. The
     * shape is chosen by header sniff, the way telecode does it: one route
     * serves two protocols and the path says which in neither.
     *
     * `modality` is in neither protocol's schema and is included anyway. A
     * client pointed at a phone cannot otherwise tell which of these ids draws
     * pictures and which of them talks, and guessing from the name would be the
     * `when (modelName)` this codebase does not allow.
     */
    private suspend fun ApplicationCall.models() {
        cors()
        val document = document()
        val installed = runner.installed()
        val anthropic = looksAnthropic()

        val rows = buildList {
            document.aliases.forEach { (alias, target) -> add(alias to target) }
            installed.forEach { add(it.id to it.id) }
        }.distinctBy { it.first }

        val data = rows.map { (name, target) ->
            val model = installed.firstOrNull { it.id == target }
            buildJsonObject {
                put("id", name)
                if (anthropic) {
                    put("type", "model")
                    put("display_name", model?.label ?: name)
                    put(
                        "created_at",
                        java.time.Instant.ofEpochMilli(model?.installedAt ?: 0L).toString(),
                    )
                } else {
                    put("object", "model")
                    put("created", (model?.installedAt ?: 0L) / 1000)
                    put("owned_by", "on-device")
                    // Present on both shapes, because it is the second name
                    // this row answers to and a client cannot discover it
                    // otherwise. Anthropic's schema has the field already.
                    put("display_name", model?.label ?: name)
                }
                put("modality", (model?.modality ?: Modality.UNKNOWN).name.lowercase())
                model?.contextLength?.let { put("context_length", it) }
                if (name != target) put("resolves_to", target)
            }
        }

        respondText(
            encode(
                buildJsonObject {
                    if (!anthropic) put("object", "list")
                    put("data", JsonArray(data))
                    if (anthropic) put("has_more", false)
                },
            ),
            ContentType.Application.Json,
        )
    }

    private suspend fun ApplicationCall.singleModel() {
        cors()
        val protocol = if (looksAnthropic()) Protocol.ANTHROPIC else Protocol.OPENAI
        val id = parameters["id"].orEmpty()
        val target = document().aliases[id] ?: id
        val model = runner.model(target)
            ?: return refuse(
                protocol,
                ProxyRefusal.notFound(
                    "No model called `$id` is installed on this device.",
                    "GET /v1/models lists what is.",
                ),
            )
        respondText(
            encode(
                buildJsonObject {
                    put("id", id)
                    put("object", "model")
                    put("display_name", model.label)
                    put("modality", model.modality.name.lowercase())
                    put("owned_by", "on-device")
                    model.contextLength?.let { put("context_length", it) }
                    put("created", model.installedAt / 1000)
                },
            ),
            ContentType.Application.Json,
        )
    }

    private suspend fun ApplicationCall.videoStatus() {
        cors()
        val job = videoJobs.get(parameters["id"].orEmpty())
            ?: return refuse(Protocol.OPENAI, ProxyRefusal.notFound("No such video job."))
        respondText(videoJson(job), ContentType.Application.Json)
    }

    private suspend fun ApplicationCall.videoContent() {
        cors()
        val job = videoJobs.get(parameters["id"].orEmpty())
            ?: return refuse(Protocol.OPENAI, ProxyRefusal.notFound("No such video job."))
        if (job.state != VideoJobs.State.COMPLETED) {
            return refuse(
                Protocol.OPENAI,
                ProxyRefusal.conflict(
                    "That job is ${job.state.name.lowercase()}, so it has no content yet.",
                    "Poll GET /v1/videos/${job.id} until it reports completed.",
                ),
            )
        }
        // The frames, as paths. A clip on this device is a directory of PNGs;
        // there is no muxer anywhere in this app, and adding one at the HTTP
        // boundary would be a second video pipeline nobody asked for.
        respondText(videoJson(job), ContentType.Application.Json)
    }

    private suspend fun ApplicationCall.videoCancel() {
        cors()
        val id = parameters["id"].orEmpty()
        if (!videoJobs.cancel(id)) {
            return refuse(Protocol.OPENAI, ProxyRefusal.conflict("That job is not running."))
        }
        respondText(videoJson(videoJobs.get(id)!!), ContentType.Application.Json)
    }

    // ── CORS, headers, small helpers ────────────────────────────────────

    private suspend fun ApplicationCall.preflight() {
        cors()
        response.header(HttpHeaders.AccessControlAllowMethods, "GET, POST, OPTIONS")
        response.header(
            HttpHeaders.AccessControlAllowHeaders,
            request.header(HttpHeaders.AccessControlRequestHeaders) ?: "*",
        )
        response.header(HttpHeaders.AccessControlMaxAge, "86400")
        respondText("", status = HttpStatusCode.NoContent)
    }

    private suspend fun ApplicationCall.cors() {
        applyCors(request.header(HttpHeaders.Origin), document().corsOrigins)
    }

    /**
     * The token, from either header.
     *
     * Both, because the protocols disagree and a server speaking both has to
     * accept both: Anthropic clients send `x-api-key`, OpenAI clients send
     * `Authorization: Bearer`.
     */
    private fun ApplicationCall.bearer(): String? =
        request.header(HttpHeaders.Authorization)
            ?.removePrefix("Bearer ")
            ?.trim()
            ?.takeIf { it.isNotBlank() }
            ?: request.header("x-api-key")?.trim()?.takeIf { it.isNotBlank() }

    private fun ApplicationCall.looksAnthropic(): Boolean =
        request.header("anthropic-version") != null ||
            request.header("x-api-key") != null ||
            request.header(HttpHeaders.UserAgent)?.contains("claude", ignoreCase = true) == true

    private suspend fun ApplicationCall.refuse(protocol: Protocol, refusal: ProxyRefusal) {
        applyCors(request.header(HttpHeaders.Origin), document().corsOrigins)
        refusal.retryAfter?.let { response.header(HttpHeaders.RetryAfter, it.toString()) }
        respondText(
            refusal.body(protocol),
            ContentType.Application.Json,
            HttpStatusCode.fromValue(refusal.status),
        )
    }

    private fun constantTimeEquals(a: String, b: String): Boolean {
        if (a.length != b.length) return false
        var difference = 0
        for (i in a.indices) difference = difference or (a[i].code xor b[i].code)
        return difference == 0
    }

    private companion object {
        const val DEFAULT_QUEUE_DEPTH = 4
        const val RETRY_AFTER_SECONDS = 30
        const val GRACE_MILLIS = 500L
        const val TIMEOUT_MILLIS = 2_000L
        const val TOKEN_BYTES = 24

        /** Long enough to let a reconnect settle, short enough to feel immediate. */
        const val RESYNC_DEBOUNCE_MILLIS = 1_500L
    }
}

/** Which of the three image routes is being served. */
enum class ImageMode { GENERATE, EDIT, UPSCALE }

/**
 * Allow only origins that were typed in.
 *
 * No wildcard by default and no reflection of arbitrary origins: a page from
 * anywhere being able to drive this phone's models is not a default anybody
 * would choose knowingly. Shared by the server and the per-call helper so the
 * two cannot drift.
 */
internal fun ApplicationCall.applyCors(origin: String?, allowed: List<String>) {
    origin ?: return
    if (allowed.isEmpty()) return
    if ("*" !in allowed && origin !in allowed) return
    response.header(HttpHeaders.AccessControlAllowOrigin, origin)
    response.header("Access-Control-Allow-Private-Network", "true")
    response.header(HttpHeaders.Vary, HttpHeaders.Origin)
}

/** A video job, in the shape the three video routes all answer with. */
internal fun videoJson(job: VideoJobs.Job): String = encode(
    buildJsonObject {
        put("id", job.id)
        put("object", "video")
        put("model", job.model)
        put("status", job.state.name.lowercase())
        put("created_at", job.createdAt / 1000)
        put("progress", job.progress)
        if (job.phase.isNotBlank()) put("phase", job.phase)
        if (job.steps > 0) {
            put("step", job.step)
            put("steps", job.steps)
        }
        if (job.secondsPerStep > 0f) put("seconds_per_step", job.secondsPerStep)
        job.completedAt?.let { put("completed_at", it / 1000) }
        job.directory?.let { put("directory", it) }
        if (job.frames.isNotEmpty()) {
            put("frames", JsonArray(job.frames.map { JsonPrimitive(it) }))
            put("fps", job.fps)
        }
        job.audioPath?.let { put("audio", it) }
        job.error?.let { message ->
            put(
                "error",
                buildJsonObject {
                    put("message", message)
                    job.suggestion?.let { put("suggestion", it) }
                },
            )
        }
    },
)
