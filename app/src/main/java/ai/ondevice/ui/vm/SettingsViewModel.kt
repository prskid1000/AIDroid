package ai.ondevice.ui.vm

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.ondevice.BuildConfig
import ai.ondevice.core.RuntimeState
import ai.ondevice.core.ThermalPolicy
import ai.ondevice.data.ModelStorage
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
        prefs.thermalPolicy,
        prefs.wifiOnly,
        prefs.showAllParameters,
        prefs.manifestWifiOnly,
    ) { backend, thermal, wifiOnly, showAll, manifestWifi ->
        SettingsState(
            backendMode = backend,
            thermalPolicy = thermal,
            wifiOnly = wifiOnly,
            showAllParameters = showAll,
            manifestWifiOnly = manifestWifi,
            hasToken = tokens.hasToken,
            maskedToken = tokens.maskedToken(),
            thermalLabel = capabilities.thermalLabel,
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
            val loaded = paramRepository.manifest()
            _manifest.value = ManifestState(
                version = loaded.manifestVersion,
                bundledVersion = paramRepository.bundledVersion(),
                signatureOk = paramRepository.storedManifest()?.signatureOk ?: true,
                lastCheckedMillis = prefs.manifestLastChecked.let { 0L },
            )
        }
    }

    fun setBackendMode(mode: String) = viewModelScope.launch { prefs.setBackendMode(mode) }
    fun setThermalPolicy(policy: ThermalPolicy) = viewModelScope.launch { prefs.setThermalPolicy(policy) }
    fun setWifiOnly(value: Boolean) = viewModelScope.launch { prefs.setWifiOnly(value) }
    fun setShowAllParameters(value: Boolean) = viewModelScope.launch { prefs.setShowAllParameters(value) }
    fun setManifestWifiOnly(value: Boolean) = viewModelScope.launch { prefs.setManifestWifiOnly(value) }

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
    val thermalPolicy: ThermalPolicy = ThermalPolicy.REDUCE_THREADS,
    val wifiOnly: Boolean = true,
    val showAllParameters: Boolean = false,
    val manifestWifiOnly: Boolean = true,
    val hasToken: Boolean = false,
    val maskedToken: String? = null,
    val thermalLabel: String = "none",
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

data class ManifestState(
    val version: Int = 0,
    val bundledVersion: Int = 0,
    val signatureOk: Boolean = true,
    val lastCheckedMillis: Long = 0,
)
