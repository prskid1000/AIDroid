package ai.ondevice.ui.vm

import ai.ondevice.core.SparseParams
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.data.secure.TokenStore
import ai.ondevice.engine.InferenceService
import ai.ondevice.core.Modality
import ai.ondevice.params.ParamSpec
import ai.ondevice.params.PathChoice
import kotlinx.serialization.json.JsonPrimitive
import ai.ondevice.params.ParamType
import ai.ondevice.proxy.ProxyConfig
import ai.ondevice.proxy.ProxyDocument
import ai.ondevice.proxy.ProxyProfile
import ai.ondevice.proxy.ProxyServer
import ai.ondevice.proxy.ProxySpecs
import ai.ondevice.proxy.Reachability
import ai.ondevice.proxy.RequestLog
import ai.ondevice.proxy.RequestRecord
import ai.ondevice.proxy.VideoJobs
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ProxyState(
    val document: ProxyDocument = ProxyDocument.EMPTY,
    val status: ProxyServer.Status = ProxyServer.Status(),
    val requests: List<RequestRecord> = emptyList(),
    val videoJobs: List<VideoJobs.Job> = emptyList(),
    /** Whether a token exists at all, and what it looks like once masked. */
    val maskedToken: String? = null,
    /**
     * The token in full, once, right after it was made.
     *
     * Held only in memory and cleared on the next edit. A credential that stays
     * on screen is a credential in a screenshot, which is the same reasoning the
     * Hugging Face block already follows.
     */
    val revealedToken: String? = null,
    /** Which client profile is expanded. One at a time, as on the Tools screen. */
    val expandedProfile: String? = null,
    val tailnetAddress: String? = null,
    /** What is installed, so a default can be chosen from it rather than typed. */
    val models: List<ai.ondevice.data.db.ModelEntity> = emptyList(),
    /** What the loaded voice model can actually say. */
    val voices: List<String> = emptyList(),
    /** The certificate on disk, when HTTPS has ever been switched on. */
    val certificate: Certificate? = null,
    /**
     * What this device will let the proxy do while nobody is looking.
     *
     * On the screen because it is the difference between a server and a server
     * that works, and because neither half of it is something this app can
     * grant itself. It was invisible, and what that looked like was an API that
     * went quiet overnight with no explanation anywhere on the phone.
     */
    val resilience: Resilience = Resilience(),
) {
    /**
     * The certificate as a screen needs it.
     *
     * Not [ProxyTls.Material] itself: that carries an `SSLContext`, which is a
     * live object with a key in it and has no business being in a state class
     * that Compose compares for equality.
     */
    data class Certificate(
        val fingerprint: String,
        val names: List<String>,
        val pem: String,
    )

    /**
     * The two permissions that decide whether the proxy survives being left alone.
     *
     * Neither is a setting this app owns, and that is exactly why they belong
     * on this screen: without them the failure is a server that answers for as
     * long as you are looking at it. Read fresh each time rather than held,
     * because both are changed in Settings — outside this process — and a
     * cached "granted" would be the most misleading value on the card.
     */
    data class Resilience(
        /** Whether an exact alarm may restart the proxy with nobody present. */
        val canRestartUnattended: Boolean = true,
        /** Whether the system has been told to stop economising on this app. */
        val exemptFromBattery: Boolean = true,
    ) {
        /** Nothing to say when the platform is already allowing both. */
        val settled: Boolean get() = canRestartUnattended && exemptFromBattery
    }

    val settings: SparseParams get() = document.settings

    val config: ProxyConfig get() = ProxyConfig(document)

    /** Rows for the screen. Nothing here decides what a row *is* — see [ProxySpecs]. */
    val specs: List<ParamSpec> get() = ProxySpecs.ALL

    fun spec(key: String): ParamSpec? = ProxySpecs.spec(key)

    /** What the dropdown's first entry says, and what clearing the key means. */
    private val NONE get() = "None"

    /**
     * A default-model row, rendered as the app's own model picker.
     *
     * `PATH` rather than `ENUM`, and the reason is that the two enum controls
     * are not interchangeable. An enum draws pills, which is right for
     * `tailnet | loopback | all` and wrong for ten entries spelled
     * `owner/repo:quant` — the current one is lost in the wall. `PATH` already
     * draws a dropdown and already puts "None" first; its own comment in the
     * renderer calls it "the same shape as the chat model picker: choose from
     * what is installed", which is exactly what this is.
     *
     * The alternative was a size threshold inside the enum branch. That works,
     * and it would also have turned five existing rows — whisper's `language`,
     * sd.cpp's two sampler lists and its `schedule`, Kokoro's `voice` — into
     * dropdowns on screens nobody asked to change.
     *
     * "None" is the cleared key, which means "whichever was used last": the
     * behaviour before this setting existed, still reachable after it.
     */
    fun picker(key: String): ParamSpec? =
        ProxySpecs.spec(key)?.copy(
            type = ParamType.PATH,
            // "None" rather than the empty string these are declared with.
            //
            // The row's reset link prints the default, and an empty default
            // printed "reset to" followed by nothing at all — a link to a value
            // with no name. "None" is also what the dropdown's own first entry
            // says, so the two now agree about what clearing this means.
            default = JsonPrimitive(NONE),
        )

    /**
     * What that row may be pointed at.
     *
     * Derived from the library rather than declared, so a model installed
     * tomorrow appears without an app change. Components are excluded: a VAE
     * and a text encoder carry the same modality as the checkpoint they belong
     * to, and offering one is offering a failure — sd.cpp answers "get sd
     * version from file failed" when handed a VAE, which reads as a broken
     * model rather than a wrong pick.
     */
    fun choices(key: String): List<PathChoice> = when (key) {
        ProxySpecs.DEFAULT_TEXT ->
            modelChoices { it == Modality.TEXT || it == Modality.VISION }
        ProxySpecs.DEFAULT_IMAGE, ProxySpecs.DEFAULT_VIDEO ->
            modelChoices { it == Modality.DIFFUSION }
        ProxySpecs.DEFAULT_VOICE -> modelChoices { it == Modality.TEXT_TO_SPEECH }
        ProxySpecs.DEFAULT_SPEECH -> modelChoices { it == Modality.SPEECH_TO_TEXT }
        ProxySpecs.TTS_VOICE -> voices.map { voice ->
            PathChoice(label = voice, detail = "", path = voice)
        }
        else -> emptyList()
    }

    /**
     * The installed models for one surface, as picker rows.
     *
     * The **short name is what gets stored**, and that is a layout decision as
     * much as anything. Every parameter row prints its current value right
     * aligned on the same line as its label; a full id is
     * `hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF:Q4_K_M`, which wraps onto a second line
     * and runs back under the label. The short name fits, and the id sits under
     * the dropdown where there is room for it.
     *
     * Storing a name rather than an id is safe because the resolver accepts
     * both — and where two rows share a name it falls back to the id for those
     * two, so the stored value is never ambiguous even though the list is
     * built from whatever happens to be installed.
     *
     * **No detail line.** The id used to sit under the dropdown, which made
     * three lines saying one thing: the name right-aligned on the label line,
     * `owner/repo:quant` under the control, and the name again inside it. The
     * dropdown is the one of the three you can act on, so it is the one that
     * stayed — and the renderer now omits a blank detail rather than drawing an
     * empty line where it was.
     */
    private fun modelChoices(wanted: (Modality) -> Boolean): List<PathChoice> {
        val offered = models.filter { it.attachmentRole == null && wanted(it.modality) }
        return offered.map { model ->
            val shared = offered.count { it.label == model.label } > 1
            PathChoice(
                label = model.label,
                detail = "",
                path = if (shared) model.id else model.label,
            )
        }
    }
}

/**
 * The Proxy screen's state, and the only place it is written.
 *
 * Every setting is a key in one sparse map, so this view model has one setter
 * for all of them rather than one per switch. That is the same shape the
 * parameter screens use and the reason the screen contains no `when` on a key
 * name: adding a setting is adding a line to [ProxySpecs].
 */
@HiltViewModel
class ProxyViewModel @Inject constructor(
    private val prefs: AppPrefs,
    private val tokens: TokenStore,
    private val server: ProxyServer,
    private val log: RequestLog,
    private val jobs: VideoJobs,
    private val db: ai.ondevice.data.db.OnDeviceDatabase,
    private val tls: ai.ondevice.proxy.ProxyTls,
    private val watchdog: ai.ondevice.proxy.ProxyWatchdog,
    /**
     * Asked what voices exist rather than told.
     *
     * A list of voice ids in this file would be the hardcoding SPEC 1.3 rules
     * out, and would go stale the moment a different Kokoro pack was installed.
     */
    private val synthesizer: ai.ondevice.speech.SpeechSynthesizer,
    @dagger.hilt.android.qualifiers.ApplicationContext private val context: android.content.Context,
) : ViewModel() {

    private val _state = MutableStateFlow(ProxyState())
    val state: StateFlow<ProxyState> = _state.asStateFlow()

    init {
        refreshResilience()
        viewModelScope.launch {
            prefs.proxyDocument.collect { raw ->
                _state.value = _state.value.copy(document = ProxyDocument.parse(raw))
            }
        }
        viewModelScope.launch {
            server.status.collect { _state.value = _state.value.copy(status = it) }
        }
        viewModelScope.launch {
            log.records.collect { _state.value = _state.value.copy(requests = it) }
        }
        viewModelScope.launch {
            jobs.jobs.collect { _state.value = _state.value.copy(videoJobs = it) }
        }
        viewModelScope.launch {
            db.models().observeInstalled().collect { installed ->
                _state.value = _state.value.copy(models = installed)
            }
        }
        viewModelScope.launch {
            // Asked of the synthesiser rather than listed here, so a voice pack
            // installed tomorrow appears without an app change — the same stance
            // SPEC 1.3 takes about models.
            val available = runCatching {
                (synthesizer.kokoroVoices() + synthesizer.omniVoiceVoices())
                    .filter { it.available }
                    .map { it.id }
            }.getOrDefault(emptyList())
            _state.value = _state.value.copy(voices = available)
        }
        _state.value = _state.value.copy(
            maskedToken = tokens.maskedProxyToken(),
            tailnetAddress = Reachability.tailnetAddress(),
        )
        refreshCertificate()
    }

    /**
     * Read the stored certificate, off the main thread.
     *
     * Loading a PKCS12 is a key derivation and a parse — small, and not small
     * enough to do while a screen is being composed.
     */
    private fun refreshCertificate() {
        viewModelScope.launch(kotlinx.coroutines.Dispatchers.IO) {
            val material = runCatching { tls.current() }.getOrNull()
            val certificate = material?.let {
                ProxyState.Certificate(
                    fingerprint = it.fingerprint,
                    names = it.names,
                    pem = it.pem,
                )
            }
            _state.value = _state.value.copy(certificate = certificate)
        }
    }

    /**
     * The certificate as a file, ready to be handed to another app.
     *
     * Staged in the cache rather than exported anywhere findable: this is not
     * an artifact, it is one step of pointing a client at this device, and the
     * step after it is Taildrop dropping it in a laptop's Downloads. Null when
     * there is no certificate yet, which the caller shows rather than hides.
     */
    fun stageCertificate(): java.io.File? {
        val pem = _state.value.certificate?.pem ?: return null
        val directory = java.io.File(context.cacheDir, "certificates")
        return runCatching {
            directory.mkdirs()
            java.io.File(directory, "ondevice.pem").apply { writeText(pem) }
        }.getOrNull()
    }

    /**
     * Throw the certificate away and make another on the next start.
     *
     * A deliberate act, because it is the one thing that breaks a client that
     * was told to trust the old one — which is also the reason to want it, when
     * the key may have been copied off the device.
     */
    fun regenerateCertificate() {
        viewModelScope.launch {
            tls.forget()
            runCatching { server.sync() }
            refreshCertificate()
        }
    }

    // ── settings ────────────────────────────────────────────────────────

    /**
     * One setter for every row.
     *
     * Null clears the key rather than storing a null, which is what makes the
     * stored map sparse — and sparse is what lets a default that moves in a
     * later release move for everyone who never touched that row.
     */
    fun set(key: String, value: Any?) = edit { it.withSetting(key, value) }

    fun setAlias(from: String, to: String, replacing: String? = null) = edit { document ->
        val aliases = document.aliases.toMutableMap()
        replacing?.let { aliases.remove(it) }
        if (from.isNotBlank()) aliases[from] = to
        document.copy(aliases = aliases)
    }

    fun removeAlias(from: String) = edit { it.copy(aliases = it.aliases - from) }

    fun setCorsOrigin(index: Int, value: String) = edit { document ->
        val origins = document.corsOrigins.toMutableList()
        if (index in origins.indices) origins[index] = value else origins += value
        document.copy(corsOrigins = origins.filter { it.isNotBlank() })
    }

    fun addCorsOrigin() = edit { it.copy(corsOrigins = it.corsOrigins + "") }

    fun removeCorsOrigin(index: Int) = edit { document ->
        document.copy(corsOrigins = document.corsOrigins.filterIndexed { i, _ -> i != index })
    }

    fun setCoreTools(names: List<String>) = edit { it.copy(coreTools = names.filter { n -> n.isNotBlank() }) }

    // ── client profiles ─────────────────────────────────────────────────

    fun addProfile() = edit { document ->
        document.copy(
            profiles = document.profiles + ProxyProfile(name = "client-${document.profiles.size + 1}"),
        )
    }

    fun updateProfile(index: Int, transform: (ProxyProfile) -> ProxyProfile) = edit { document ->
        document.copy(
            profiles = document.profiles.mapIndexed { i, profile ->
                if (i == index) transform(profile) else profile
            },
        )
    }

    fun removeProfile(index: Int) = edit { document ->
        document.copy(profiles = document.profiles.filterIndexed { i, _ -> i != index })
    }

    /** A profile's own answer to one setting. Null falls back to the global one. */
    fun setProfileOverride(index: Int, key: String, value: Any?) = updateProfile(index) { profile ->
        val current = profile.overrides
        val next = when (value) {
            null -> current.without(key)
            is Boolean -> current.with(key, value)
            is Int -> current.with(key, value)
            is Float -> current.with(key, value)
            is String -> current.with(key, value)
            else -> current.with(key, value.toString())
        }
        profile.copy(overridesJson = next.toJsonString())
    }

    fun expandProfile(name: String?) {
        _state.value = _state.value.copy(expandedProfile = name)
    }

    // ── the token ───────────────────────────────────────────────────────

    fun regenerateToken() {
        val token = server.regenerateToken()
        _state.value = _state.value.copy(
            revealedToken = token,
            maskedToken = tokens.maskedProxyToken(),
        )
    }

    fun dismissRevealedToken() {
        _state.value = _state.value.copy(revealedToken = null)
    }

    // ── lifecycle ───────────────────────────────────────────────────────

    /**
     * Apply the configuration to the running server.
     *
     * Started through the foreground service rather than directly, because the
     * socket has to be held by something the system will not reclaim — see
     * `InferenceService`, whose stop condition now counts a listening proxy as a
     * reason to stay alive.
     */
    fun apply() {
        viewModelScope.launch {
            val enabled = ProxyConfig(ProxyDocument.parse(prefs.proxyDocument.first())).enabled
            if (enabled) {
                runCatching {
                    context.startForegroundService(
                        android.content.Intent(context, InferenceService::class.java),
                    )
                }
            }
            runCatching { server.sync() }
            runCatching { watchdog.sync() }
            _state.value = _state.value.copy(tailnetAddress = Reachability.tailnetAddress())
            refreshCertificate()
        }
        refreshResilience()
    }

    fun refreshReachability() {
        viewModelScope.launch {
            _state.value = _state.value.copy(tailnetAddress = Reachability.tailnetAddress())
            runCatching { server.sync() }
        }
        refreshResilience()
    }

    /**
     * Re-read what the platform is currently allowing.
     *
     * Called on every entry to the screen and after every trip out to Settings,
     * because both answers are changed somewhere this process is not and the
     * screen would otherwise keep showing the state at the moment it was built
     * — telling somebody who has just granted the permission that they have not.
     */
    fun refreshResilience() {
        _state.value = _state.value.copy(
            resilience = ProxyState.Resilience(
                canRestartUnattended = watchdog.canRestartUnattended,
                exemptFromBattery = watchdog.exemptFromBatteryOptimisation,
            ),
        )
    }

    /**
     * Open whichever Settings page grants the thing that is missing.
     *
     * Returns whether there was anywhere to go, so the caller does not have to
     * ask the same question twice. `NEW_TASK` because the context here is the
     * application's, not the activity's.
     */
    fun openRestrictionSettings(exactAlarms: Boolean): Boolean {
        val intent = if (exactAlarms) watchdog.exactAlarmSettings() else watchdog.batterySettings()
        intent ?: return false
        return runCatching {
            context.startActivity(intent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK))
        }.isSuccess
    }

    fun clearLog() = log.clear()

    fun cancelVideoJob(id: String) {
        jobs.cancel(id)
    }

    private fun edit(transform: (ProxyDocument) -> ProxyDocument) {
        viewModelScope.launch {
            val current = ProxyDocument.parse(prefs.proxyDocument.first())
            val next = transform(current)
            prefs.setProxyDocument(next.encode())

            // The service first, then the server.
            //
            // `sync()` opens the socket in whatever process calls it, and this
            // one is the app's. That worked and was wrong: nothing was keeping
            // that process alive, so the port stayed open exactly as long as
            // the app happened to survive in the background — and the screen
            // said "Listening" throughout. The service is what the platform
            // will not reclaim, and starting it from here is allowed because a
            // screen is on and this is a tap.
            if (ProxyConfig(next).enabled) {
                runCatching {
                    context.startForegroundService(
                        android.content.Intent(context, InferenceService::class.java),
                    )
                }.onFailure {
                    android.util.Log.w("ProxyViewModel", "service would not start", it)
                }
            }

            // Applied as it is edited rather than behind a Save.
            //
            // Two of these keys carry `requiresReload` and the rest take effect
            // per request, so "saved but not in force" would be a state the
            // screen could not usefully draw. The server's own sync is
            // idempotent and cheap when nothing that matters has changed.
            runCatching { server.sync() }
            // Arm or disarm the restart check in the same breath. Switching the
            // proxy off has to take the fifteen-minute wake-up with it, or the
            // phone keeps waking up for a server nobody asked for.
            runCatching { watchdog.sync() }
            _state.value = _state.value.copy(revealedToken = null)
            // A start with HTTPS on may have made a certificate, or replaced one
            // that no longer named this address.
            refreshCertificate()
        }
    }
}
