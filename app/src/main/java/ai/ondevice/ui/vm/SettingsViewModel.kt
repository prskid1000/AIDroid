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
/**
 * Tool use and the MCP server list.
 *
 * Separate from the main settings state because it is the one screen in the app
 * where the user hands something outward: an MCP server is a third party, and
 * the decision to enable one deserves its own surface with its own evidence —
 * what it is called, what tools it offers, and when it was last reachable.
 */
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

    fun toggleProvider(id: String) {
        viewModelScope.launch {
            val current = _state.value.enabledProviders
            prefs.setEnabledToolProviders(if (id in current) current - id else current + id)
        }
    }

    fun setDraft(name: String? = null, url: String? = null, auth: String? = null) {
        _state.value = _state.value.copy(
            draftName = name ?: _state.value.draftName,
            draftUrl = url ?: _state.value.draftUrl,
            draftAuth = auth ?: _state.value.draftAuth,
        )
    }

    /**
     * Add a server only after it has answered. A URL that was never reachable
     * would sit in the list looking installed and fail silently at the moment
     * the model tries to use it — mid-conversation, which is the worst time.
     */
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
                    lastToolsJson = probe.toolNames.joinToString(","),
                ),
            )
            prefs.setEnabledToolProviders(
                _state.value.enabledProviders + "${ai.ondevice.tools.McpToolProvider.ID_PREFIX}${candidate.id}",
            )
            _state.value = _state.value.copy(
                testing = false,
                draftName = "",
                draftUrl = "",
                draftAuth = "",
                draftError = null,
            )
        }
    }

    fun removeServer(id: String) {
        viewModelScope.launch { db.mcpServers().deleteById(id) }
    }

    fun refresh(server: McpServerEntity) {
        viewModelScope.launch {
            _state.value = _state.value.copy(testing = true)
            val probe = factory.provider(server).probe()
            db.mcpServers().upsert(
                server.copy(
                    lastCheckedAt = System.currentTimeMillis(),
                    lastToolsJson = probe.toolNames.joinToString(","),
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
    val draftName: String = "",
    val draftUrl: String = "",
    val draftAuth: String = "",
    val draftError: String? = null,
    val testing: Boolean = false,
)

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val prefs: AppPrefs,
    private val tokens: TokenStore,
    private val db: OnDeviceDatabase,
    private val capabilities: DeviceCapabilities,
    private val storage: ModelStorage,
    private val paramRepository: ParamRepository,
) : ViewModel() {

    val settings: StateFlow<SettingsState> = combine(
        prefs.backendMode,
        prefs.wifiOnly,
    ) { backend, wifiOnly ->
        SettingsState(
            backendMode = backend,
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

    /**
     * SPEC §17.4 — engines are separately installable. Installing one is a
     * package-manager operation on the sideload channel and a Play Feature
     * Delivery module on the Play channel; the updater is channel-aware, which
     * is why the button's behaviour is decided by a build flag rather than an
     * `#ifdef` scattered through the UI.
     */
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
    val backendMode: String = AppPrefs.BACKEND_AUTO,
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

/**
 * Just the version. It used to carry a bundled-versus-stored comparison, a
 * signature verdict and a last-checked time — all three describing an update
 * mechanism the app does not have.
 */
data class ManifestState(val version: Int = 0)
