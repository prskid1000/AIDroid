package ai.ondevice.ui.vm

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.ondevice.BuildConfig
import ai.ondevice.core.RuntimeState
import ai.ondevice.data.ModelStorage
import ai.ondevice.data.db.McpServerEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.RuntimeBundleEntity
import ai.ondevice.data.hf.DeviceCapabilities
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.data.secure.TokenStore
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

/** S15 and the Settings root. */
/** Tool use and the MCP server list. */
@HiltViewModel
class ToolsViewModel @Inject constructor(
    private val db: OnDeviceDatabase,
    private val prefs: AppPrefs,
    private val factory: ai.ondevice.tools.ToolProviderFactory,
    @dagger.hilt.android.qualifiers.ApplicationContext private val context: android.content.Context,
) : ViewModel() {

    private val _state = MutableStateFlow(ToolsState())
    val state: StateFlow<ToolsState> = _state.asStateFlow()

    init {
        _state.value = _state.value.copy(toolSettings = factory.allSettings(context))
        viewModelScope.launch {
            db.mcpServers().observeAll().collect { servers ->
                _state.value = _state.value.copy(servers = servers)
            }
        }
        viewModelScope.launch {
            prefs.toolsEnabled.collect { _state.value = _state.value.copy(enabled = it) }
        }
        viewModelScope.launch {
            prefs.enabledToolProviders.collect { _state.value = _state.value.copy(enabledProviders = it) }
        }
        viewModelScope.launch {
            prefs.fileScopeDevice.collect { _state.value = _state.value.copy(fileScopeDevice = it) }
        }
        viewModelScope.launch {
            ai.ondevice.tools.ShellLog.runs.collect { _state.value = _state.value.copy(shellRuns = it) }
        }
        viewModelScope.launch {
            prefs.toolParams.collect {
                _state.value = _state.value.copy(tuning = ai.ondevice.core.SparseParams.parse(it))
            }
        }
    }

    /** Which provider's settings are open; only one at a time, like the servers. */
    fun expandProvider(id: String?) {
        _state.value = _state.value.copy(expandedProviderId = id)
    }

    /**
     * Change one tool setting.
     *
     * A null value removes the key rather than storing one, so "back to the
     * default" stays the *current* default rather than freezing today's number
     * into the preferences.
     */
    fun setToolParam(key: String, value: Any?) {
        viewModelScope.launch {
            val current = ai.ondevice.core.SparseParams.parse(prefs.toolParams.first())
            val next = if (value == null) {
                ai.ondevice.core.SparseParams(current.values - key)
            } else {
                ai.ondevice.core.SparseParams(
                    current.values + (key to ai.ondevice.core.SparseParams.of(key to value).values.getValue(key)),
                )
            }
            prefs.setToolParams(next.toJsonString())
        }
    }

    fun setEnabled(value: Boolean) {
        viewModelScope.launch { prefs.setToolsEnabled(value) }
    }

    /** One of the app's own sets — built-in, files, shell. A server's switch is [pauseServer]. */
    fun toggleProvider(id: String) {
        viewModelScope.launch {
            val current = _state.value.enabledProviders
            prefs.setEnabledToolProviders(if (id in current) current - id else current + id)
        }
    }

    /**
     * Ask for the whole device rather than the app's own folders.
     *
     * Only records the intent. The permission itself is a system screen the
     * caller opens, and it can be taken away there afterwards, so what is held
     * is asked of the system at the moment the tools are built.
     */
    fun setFileScopeDevice(value: Boolean) {
        viewModelScope.launch { prefs.setFileScopeDevice(value) }
    }

    fun clearShellLog() = ai.ondevice.tools.ShellLog.clear()

    /** Pause without forgetting. */
    fun pauseServer(server: McpServerEntity, enabled: Boolean) {
        viewModelScope.launch { db.mcpServers().upsert(server.copy(enabled = enabled)) }
    }

    /** Switch one tool off on one server. */
    fun toggleTool(server: McpServerEntity, toolName: String) {
        viewModelScope.launch {
            val disabled = ai.ondevice.tools.McpTools.disabled(server)
            val next = if (toolName in disabled) disabled - toolName else disabled + toolName
            db.mcpServers().upsert(
                server.copy(disabledToolsJson = ai.ondevice.tools.McpTools.encodeDisabled(next)),
            )
        }
    }

    fun expandServer(id: String?) {
        _state.value = _state.value.copy(expandedServerId = id)
    }

    fun setDraft(name: String? = null, url: String? = null, auth: String? = null) {
        _state.value = _state.value.copy(
            draftName = name ?: _state.value.draftName,
            draftUrl = url ?: _state.value.draftUrl,
            draftAuth = auth ?: _state.value.draftAuth,
        )
    }

    /** Add a server only after it has answered. */
    fun addServer() {
        val url = _state.value.draftUrl.trim()
        if (url.isBlank()) {
            _state.value = _state.value.copy(draftError = "A server needs a URL.")
            return
        }
        if (!url.startsWith("https://") && !url.startsWith("http://")) {
            _state.value = _state.value.copy(draftError = "That is not an http(s) URL.")
            return
        }
        _state.value = _state.value.copy(testing = true, draftError = null)

        viewModelScope.launch {
            val candidate = McpServerEntity(
                id = java.util.UUID.randomUUID().toString(),
                name = _state.value.draftName.ifBlank { url.substringAfter("://").substringBefore('/') },
                url = url,
                authHeader = _state.value.draftAuth.takeIf { it.isNotBlank() },
                enabled = true,
                lastToolsJson = null,
                disabledToolsJson = "[]",
                lastCheckedAt = System.currentTimeMillis(),
                lastError = null,
                createdAt = System.currentTimeMillis(),
            )
            val probe = factory.provider(candidate).probe()
            if (!probe.ok) {
                _state.value = _state.value.copy(
                    testing = false,
                    draftError = "That server did not answer: ${probe.error}",
                )
                return@launch
            }
            db.mcpServers().upsert(
                candidate.copy(
                    name = _state.value.draftName.ifBlank { probe.serverName },
                    lastToolsJson = ai.ondevice.tools.McpTools.encode(probe.tools),
                ),
            )
            _state.value = _state.value.copy(
                testing = false,
                draftName = "",
                draftUrl = "",
                draftAuth = "",
                draftError = null,
                expandedServerId = candidate.id,
            )
        }
    }

    fun removeServer(id: String) {
        viewModelScope.launch { db.mcpServers().deleteById(id) }
    }

    /** Ask the server what it offers now. */
    fun refresh(server: McpServerEntity) {
        viewModelScope.launch {
            _state.value = _state.value.copy(testing = true)
            val probe = factory.provider(server).probe()
            db.mcpServers().upsert(
                server.copy(
                    lastCheckedAt = System.currentTimeMillis(),
                    // A failed probe leaves the last known list alone.
                    lastToolsJson = if (probe.ok) {
                        ai.ondevice.tools.McpTools.encode(probe.tools)
                    } else {
                        server.lastToolsJson
                    },
                    lastError = probe.error,
                ),
            )
            _state.value = _state.value.copy(testing = false)
        }
    }
}

data class ToolsState(
    val enabled: Boolean = false,
    val enabledProviders: Set<String> = emptySet(),
    val servers: List<McpServerEntity> = emptyList(),
    val builtInTools: List<String> =
        ai.ondevice.tools.BuiltInToolProvider.toolNames(webSearchAvailable = true),
    /** Only one server's tool list is open at a time; the rest stay one line. */
    val expandedServerId: String? = null,
    val draftName: String = "",
    val draftUrl: String = "",
    val draftAuth: String = "",
    val draftError: String? = null,
    val testing: Boolean = false,
    val fileTools: List<String> = ai.ondevice.tools.FileToolProvider.toolNames(),
    val shellTools: List<String> = ai.ondevice.tools.ShellToolProvider.toolNames(),
    /** Whether the whole device was asked for; the grant itself is checked live. */
    val fileScopeDevice: Boolean = false,
    val shellRuns: List<ai.ondevice.tools.ShellRun> = emptyList(),
    /** Every settings row the app's own tools expose, grouped by tool name. */
    val toolSettings: List<ai.ondevice.params.ParamSpec> = emptyList(),
    /** What has actually been changed; absent keys mean "still the default". */
    val tuning: ai.ondevice.core.SparseParams = ai.ondevice.core.SparseParams.EMPTY,
    val expandedProviderId: String? = null,
) {
    /** The settings belonging to one provider's tools, in declaration order. */
    fun settingsFor(providerId: String, tools: List<String>): List<ai.ondevice.params.ParamSpec> =
        toolSettings.filter { it.group in tools }

    val builtInEnabled: Boolean
        get() = ai.ondevice.tools.BuiltInToolProvider.ID in enabledProviders

    val filesEnabled: Boolean
        get() = ai.ondevice.tools.FileToolProvider.ID in enabledProviders

    val shellEnabled: Boolean
        get() = ai.ondevice.tools.ShellToolProvider.ID in enabledProviders

    /** How many tools are actually reaching the model. */
    val liveToolCount: Int
        get() = (if (builtInEnabled) builtInTools.size else 0) +
            (if (filesEnabled) fileTools.size else 0) +
            (if (shellEnabled) shellTools.size else 0) +
            servers.filter { it.enabled }.sumOf { server ->
                val disabled = ai.ondevice.tools.McpTools.disabled(server)
                ai.ondevice.tools.McpTools.parse(server.lastToolsJson).count { it.name !in disabled }
            }
}

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val prefs: AppPrefs,
    private val tokens: TokenStore,
    private val db: OnDeviceDatabase,
    private val capabilities: DeviceCapabilities,
    private val storage: ModelStorage,
) : ViewModel() {

    /**
     * The stored token, as something that emits.
     *
     * The keystore is a plain getter and setter with nothing to observe, and
     * the state below used to read it inside the map over `wifiOnly`. That made
     * the token visible only when the Wi-Fi switch happened to move: saving one
     * and removing one both changed the store and left the screen showing what
     * it had shown before, which reads as neither button working.
     */
    private val tokenRevision = MutableStateFlow(tokens.hfToken)

    val settings: StateFlow<SettingsState> = combine(
        prefs.wifiOnly,
        tokenRevision,
    ) { wifiOnly, _ ->
        SettingsState(
            wifiOnly = wifiOnly,
            hasToken = tokens.hasToken,
            maskedToken = tokens.maskedToken(),
            batteryPercent = capabilities.batteryPercent,
            totalRamBytes = capabilities.totalRamBytes,
            freeStorageBytes = capabilities.freeStorageBytes,
            performanceCores = capabilities.performanceCores,
            totalCores = capabilities.totalCores,
            soc = capabilities.socModel,
            storageUsedBytes = storage.usedBytes(),
            canSelfUpdateRuntimes = BuildConfig.CAN_SELF_UPDATE_RUNTIMES,
            updateChannel = BuildConfig.UPDATE_CHANNEL,
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), SettingsState())

    val runtimes: StateFlow<List<RuntimeBundleEntity>> = db.runtimes().observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    fun setWifiOnly(value: Boolean) = viewModelScope.launch { prefs.setWifiOnly(value) }

    fun setToken(value: String?) {
        tokens.hfToken = value
        // Read back rather than echoing the argument: the store treats blank
        // as absent, so what it kept is the only honest thing to publish.
        tokenRevision.value = tokens.hfToken
    }
}

data class SettingsState(
    val wifiOnly: Boolean = true,
    val hasToken: Boolean = false,
    val maskedToken: String? = null,
    val batteryPercent: Int = 100,
    val totalRamBytes: Long = 0,
    val freeStorageBytes: Long = 0,
    val performanceCores: Int = 0,
    val totalCores: Int = 0,
    val soc: String = "",
    val storageUsedBytes: Long = 0,
    val canSelfUpdateRuntimes: Boolean = true,
    val updateChannel: String = "SIDELOAD",
)
