package ai.ondevice.data.prefs

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import ai.ondevice.core.BackendId
import ai.ondevice.core.ThermalPolicy
import ai.ondevice.core.Tier
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "ondevice_prefs")

/**
 * Everything on the Settings screen that isn't a row in the database.
 *
 * There is deliberately no analytics opt-in and no account here: SPEC §13 —
 * no telemetry, no login. The HF token is the one secret and it lives in the
 * Keystore, not in this store.
 */
class AppPrefs(private val context: Context) {

    private object Keys {
        val backendMode = stringPreferencesKey("backend_mode")
        val threadCount = intPreferencesKey("thread_count")
        val thermalPolicy = stringPreferencesKey("thermal_policy")
        val batteryGuardPercent = intPreferencesKey("battery_guard_percent")
        val wifiOnly = booleanPreferencesKey("wifi_only")
        val parallelConnections = intPreferencesKey("parallel_connections")
        val maxRetries = intPreferencesKey("max_retries")
        val storageRoot = stringPreferencesKey("storage_root")
        val storageReserveMb = intPreferencesKey("storage_reserve_mb")
        val showAllParameters = booleanPreferencesKey("show_all_parameters")
        val defaultTier = stringPreferencesKey("default_tier")
        val manifestAutoCheck = booleanPreferencesKey("manifest_auto_check")
        val manifestWifiOnly = booleanPreferencesKey("manifest_wifi_only")
        val manifestLastChecked = androidx.datastore.preferences.core.longPreferencesKey("manifest_last_checked")
        val preloadPinned = booleanPreferencesKey("preload_pinned")
        val blockPickle = booleanPreferencesKey("block_pickle")
        val lastConversationId = stringPreferencesKey("last_conversation_id")
        val toolsEnabled = booleanPreferencesKey("tools_enabled")
        val enabledToolProviders = androidx.datastore.preferences.core.stringSetPreferencesKey("enabled_tool_providers")
    }

    /** "auto" means benchmark-driven selection (SPEC §8.1). */
    val backendMode: Flow<String> = context.dataStore.data.map { it[Keys.backendMode] ?: BACKEND_AUTO }

    /** Default to performance-core count, not total cores (SPEC §8.1). */
    val threadCount: Flow<Int> = context.dataStore.data.map { it[Keys.threadCount] ?: 0 }

    val thermalPolicy: Flow<ThermalPolicy> = context.dataStore.data.map {
        it[Keys.thermalPolicy]?.let { v -> runCatching { ThermalPolicy.valueOf(v) }.getOrNull() }
            ?: ThermalPolicy.REDUCE_THREADS
    }

    val batteryGuardPercent: Flow<Int> = context.dataStore.data.map { it[Keys.batteryGuardPercent] ?: 20 }
    val wifiOnly: Flow<Boolean> = context.dataStore.data.map { it[Keys.wifiOnly] ?: true }
    val parallelConnections: Flow<Int> = context.dataStore.data.map { it[Keys.parallelConnections] ?: 4 }
    val maxRetries: Flow<Int> = context.dataStore.data.map { it[Keys.maxRetries] ?: 5 }
    val storageRoot: Flow<String?> = context.dataStore.data.map { it[Keys.storageRoot] }
    val storageReserveMb: Flow<Int> = context.dataStore.data.map { it[Keys.storageReserveMb] ?: 1024 }

    /** The global switch that collapses the three tiers (SPEC §9). */
    val showAllParameters: Flow<Boolean> = context.dataStore.data.map { it[Keys.showAllParameters] ?: false }

    val defaultTier: Flow<Tier> = context.dataStore.data.map {
        it[Keys.defaultTier]?.let { v -> runCatching { Tier.valueOf(v) }.getOrNull() } ?: Tier.BASIC
    }

    val manifestAutoCheck: Flow<Boolean> = context.dataStore.data.map { it[Keys.manifestAutoCheck] ?: true }
    val manifestWifiOnly: Flow<Boolean> = context.dataStore.data.map { it[Keys.manifestWifiOnly] ?: true }
    val manifestLastChecked: Flow<Long> = context.dataStore.data.map { it[Keys.manifestLastChecked] ?: 0L }
    val preloadPinned: Flow<Boolean> = context.dataStore.data.map { it[Keys.preloadPinned] ?: false }

    /** Pickle files are blocked by default; expert override (SPEC §3.2). */
    val blockPickle: Flow<Boolean> = context.dataStore.data.map { it[Keys.blockPickle] ?: true }

    val lastConversationId: Flow<String?> = context.dataStore.data.map { it[Keys.lastConversationId] }

    /**
     * Tool use is off until asked for. A model that is never told tools exist
     * cannot call one, and that is the safe default when some of them reach a
     * third-party server.
     */
    val toolsEnabled: Flow<Boolean> = context.dataStore.data.map { it[Keys.toolsEnabled] ?: false }

    /**
     * Which providers are offered. The built-in set is pre-selected because it
     * touches nothing but this device; an MCP server has to be ticked by hand.
     */
    val enabledToolProviders: Flow<Set<String>> = context.dataStore.data.map {
        it[Keys.enabledToolProviders] ?: setOf(ai.ondevice.tools.BuiltInToolProvider.ID)
    }

    suspend fun setBackendMode(v: String) = edit { it[Keys.backendMode] = v }
    suspend fun setThreadCount(v: Int) = edit { it[Keys.threadCount] = v }
    suspend fun setThermalPolicy(v: ThermalPolicy) = edit { it[Keys.thermalPolicy] = v.name }
    suspend fun setBatteryGuardPercent(v: Int) = edit { it[Keys.batteryGuardPercent] = v }
    suspend fun setWifiOnly(v: Boolean) = edit { it[Keys.wifiOnly] = v }
    suspend fun setParallelConnections(v: Int) = edit { it[Keys.parallelConnections] = v }
    suspend fun setMaxRetries(v: Int) = edit { it[Keys.maxRetries] = v }
    suspend fun setStorageRoot(v: String?) = edit { p -> if (v == null) p.remove(Keys.storageRoot) else p[Keys.storageRoot] = v }
    suspend fun setStorageReserveMb(v: Int) = edit { it[Keys.storageReserveMb] = v }
    suspend fun setShowAllParameters(v: Boolean) = edit { it[Keys.showAllParameters] = v }
    suspend fun setDefaultTier(v: Tier) = edit { it[Keys.defaultTier] = v.name }
    suspend fun setManifestAutoCheck(v: Boolean) = edit { it[Keys.manifestAutoCheck] = v }
    suspend fun setManifestWifiOnly(v: Boolean) = edit { it[Keys.manifestWifiOnly] = v }
    suspend fun setManifestLastChecked(v: Long) = edit { it[Keys.manifestLastChecked] = v }
    suspend fun setPreloadPinned(v: Boolean) = edit { it[Keys.preloadPinned] = v }
    suspend fun setBlockPickle(v: Boolean) = edit { it[Keys.blockPickle] = v }
    suspend fun setToolsEnabled(v: Boolean) = edit { it[Keys.toolsEnabled] = v }
    suspend fun setEnabledToolProviders(v: Set<String>) = edit { it[Keys.enabledToolProviders] = v }
    suspend fun setLastConversationId(v: String?) =
        edit { p -> if (v == null) p.remove(Keys.lastConversationId) else p[Keys.lastConversationId] = v }

    private suspend fun edit(block: (androidx.datastore.preferences.core.MutablePreferences) -> Unit) {
        context.dataStore.edit(block)
    }

    companion object {
        const val BACKEND_AUTO = "auto"

        fun backendModeLabel(mode: String): String = when (mode) {
            BACKEND_AUTO -> "Auto (benchmark-driven)"
            else -> runCatching { BackendId.valueOf(mode).label }.getOrDefault(mode)
        }
    }
}
