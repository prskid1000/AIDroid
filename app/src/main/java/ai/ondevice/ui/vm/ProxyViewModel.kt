package ai.ondevice.ui.vm

import ai.ondevice.core.SparseParams
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.data.secure.TokenStore
import ai.ondevice.engine.InferenceService
import ai.ondevice.params.ParamSpec
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
) {
    val settings: SparseParams get() = document.settings

    val config: ProxyConfig get() = ProxyConfig(document)

    /** Rows for the screen. Nothing here decides what a row *is* — see [ProxySpecs]. */
    val specs: List<ParamSpec> get() = ProxySpecs.ALL

    fun spec(key: String): ParamSpec? = ProxySpecs.spec(key)
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
    @dagger.hilt.android.qualifiers.ApplicationContext private val context: android.content.Context,
) : ViewModel() {

    private val _state = MutableStateFlow(ProxyState())
    val state: StateFlow<ProxyState> = _state.asStateFlow()

    init {
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
        _state.value = _state.value.copy(
            maskedToken = tokens.maskedProxyToken(),
            tailnetAddress = Reachability.tailnetAddress(),
        )
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
            _state.value = _state.value.copy(tailnetAddress = Reachability.tailnetAddress())
        }
    }

    fun refreshReachability() {
        viewModelScope.launch {
            _state.value = _state.value.copy(tailnetAddress = Reachability.tailnetAddress())
            runCatching { server.sync() }
        }
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
            _state.value = _state.value.copy(revealedToken = null)
        }
    }
}
