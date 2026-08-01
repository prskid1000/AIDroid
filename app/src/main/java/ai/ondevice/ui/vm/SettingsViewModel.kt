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
import ai.ondevice.params.ParamRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
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
) : ViewModel() {

    private val _state = MutableStateFlow(ToolsState())
    val state: StateFlow<ToolsState> = _state.asStateFlow()

    init {
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
    }

    fun setEnabled(value: Boolean) {
        viewModelScope.launch { prefs.setToolsEnabled(value) }
    }

    /** The built-in set only. A server's switch is [pauseServer]. */
    fun toggleBuiltIn() {
        viewModelScope.launch {
            val current = _state.value.enabledProviders
            val id = ai.ondevice.tools.BuiltInToolProvider.ID
            prefs.setEnabledToolProviders(if (id in current) current - id else current + id)
        }
    }

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
    val builtInTools: List<String> = listOf("get_current_time", "calculate", "device_status"),
    /** Only one server's tool list is open at a time; the rest stay one line. */
    val expandedServerId: String? = null,
    val draftName: String = "",
    val draftUrl: String = "",
    val draftAuth: String = "",
    val draftError: String? = null,
    val testing: Boolean = false,
) {
    val builtInEnabled: Boolean
        get() = ai.ondevice.tools.BuiltInToolProvider.ID in enabledProviders

    /** How many tools are actually reaching the model. */
    val liveToolCount: Int
        get() = (if (builtInEnabled) builtInTools.size else 0) +
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
    private val paramRepository: ParamRepository,
    private val registry: ai.ondevice.engine.RuntimeRegistry,
) : ViewModel() {

    val settings: StateFlow<SettingsState> = combine(
        prefs.backendMode,
        prefs.wifiOnly,
    ) { backend, wifiOnly ->
        SettingsState(
            backendMode = backend,
            availableBackends = registry.backendsFor(ai.ondevice.engine.RuntimeRegistry.LLAMA),
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

    private val _manifest = MutableStateFlow(ManifestState())
    val manifest: StateFlow<ManifestState> = _manifest.asStateFlow()

    val runtimes: StateFlow<List<RuntimeBundleEntity>> = db.runtimes().observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    init {
        viewModelScope.launch {
            _manifest.value = ManifestState(version = paramRepository.manifest().manifestVersion)
        }
    }

    fun setBackendMode(mode: String) = viewModelScope.launch { prefs.setBackendMode(mode) }
    fun setWifiOnly(value: Boolean) = viewModelScope.launch { prefs.setWifiOnly(value) }

    fun setToken(value: String?) {
        tokens.hfToken = value
    }

    /** SPEC §17.4 — engines are separately installable. */
    fun installRuntime(engine: String) {
        viewModelScope.launch {
            val bundle = db.runtimes().get(engine) ?: return@launch
            db.runtimes().upsert(
                bundle.copy(
                    state = RuntimeState.INSTALLED,
                    installedAt = System.currentTimeMillis(),
                ),
            )
        }
    }

    fun updateRuntime(engine: String) {
        viewModelScope.launch {
            val bundle = db.runtimes().get(engine) ?: return@launch
            val target = bundle.availableBuildTag ?: return@launch
            // The previous bundle is kept so a runtime that fails to init twice
            // can revert on its own (SPEC §17.8).
            db.runtimes().upsert(
                bundle.copy(
                    previousBuildTag = bundle.buildTag,
                    buildTag = target,
                    availableBuildTag = null,
                    availableNotes = null,
                    state = RuntimeState.INSTALLED,
                    installedAt = System.currentTimeMillis(),
                ),
            )
        }
    }
}

data class SettingsState(
    val backendMode: String = ai.ondevice.core.BackendId.CPU.name,
    /** What ggml registered here — the rest are shown, dimmed, and inert. */
    val availableBackends: List<ai.ondevice.core.BackendId> = emptyList(),
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

/** Just the version. */
data class ManifestState(val version: Int = 0)
