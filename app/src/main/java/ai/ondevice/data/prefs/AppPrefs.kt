package ai.ondevice.data.prefs

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import ai.ondevice.core.BackendId
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
        val batteryGuardPercent = intPreferencesKey("battery_guard_percent")
        val wifiOnly = booleanPreferencesKey("wifi_only")
        val parallelConnections = intPreferencesKey("parallel_connections")
        val maxRetries = intPreferencesKey("max_retries")
        val storageRoot = stringPreferencesKey("storage_root")
        val storageReserveMb = intPreferencesKey("storage_reserve_mb")
        val defaultTier = stringPreferencesKey("default_tier")
        val blockPickle = booleanPreferencesKey("block_pickle")
        val lastConversationId = stringPreferencesKey("last_conversation_id")
        val exportFolder = stringPreferencesKey("export_folder")
        val toolsEnabled = booleanPreferencesKey("tools_enabled")
        val enabledToolProviders = androidx.datastore.preferences.core.stringSetPreferencesKey("enabled_tool_providers")
    }

    /**
     * Which piece of silicon to run on: one of [BackendId], by name.
     *
     * There is no "auto" any more, and its history is the argument against it.
     * It first meant benchmark-driven selection, which sounded better than it
     * was — llama.cpp registered one backend, so the benchmark compared CPU
     * with itself. It then meant "the first backend registered", which is not a
     * choice, it is an ordering. A device with three real options deserves to
     * be asked, and every answer is clamped to what actually registered, so a
     * wrong one degrades rather than fails.
     *
     * CPU is the default because it is the one that always works. Anything
     * stored by an older build — including the literal string "auto" — reads
     * back as CPU.
     */
    val backendMode: Flow<String> = context.dataStore.data.map { stored ->
        val raw = stored[Keys.backendMode]
        BackendId.entries.firstOrNull { it.name == raw }?.name ?: BackendId.CPU.name
    }

    /** Default to performance-core count, not total cores (SPEC §8.1). */
    val threadCount: Flow<Int> = context.dataStore.data.map { it[Keys.threadCount] ?: 0 }

    val batteryGuardPercent: Flow<Int> = context.dataStore.data.map { it[Keys.batteryGuardPercent] ?: 20 }
    val wifiOnly: Flow<Boolean> = context.dataStore.data.map { it[Keys.wifiOnly] ?: true }
    val parallelConnections: Flow<Int> = context.dataStore.data.map { it[Keys.parallelConnections] ?: 4 }
    val maxRetries: Flow<Int> = context.dataStore.data.map { it[Keys.maxRetries] ?: 5 }
    val storageRoot: Flow<String?> = context.dataStore.data.map { it[Keys.storageRoot] }
    val storageReserveMb: Flow<Int> = context.dataStore.data.map { it[Keys.storageReserveMb] ?: 1024 }


    val defaultTier: Flow<Tier> = context.dataStore.data.map {
        it[Keys.defaultTier]?.let { v -> runCatching { Tier.valueOf(v) }.getOrNull() } ?: Tier.BASIC
    }

    /** Pickle files are blocked by default; expert override (SPEC §3.2). */
    val blockPickle: Flow<Boolean> = context.dataStore.data.map { it[Keys.blockPickle] ?: true }

    val lastConversationId: Flow<String?> = context.dataStore.data.map { it[Keys.lastConversationId] }

    /**
     * The SAF tree the user picked for exports, as a URI string.
     *
     * Stored rather than asked for every time, because a picker on every
     * save is the reason people stop using export. [ai.ondevice.data.ExportStore]
     * re-checks the grant before trusting this.
     */
    val exportFolder: Flow<String?> = context.dataStore.data.map { it[Keys.exportFolder] }

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
    suspend fun setBatteryGuardPercent(v: Int) = edit { it[Keys.batteryGuardPercent] = v }
    suspend fun setWifiOnly(v: Boolean) = edit { it[Keys.wifiOnly] = v }
    suspend fun setParallelConnections(v: Int) = edit { it[Keys.parallelConnections] = v }
    suspend fun setMaxRetries(v: Int) = edit { it[Keys.maxRetries] = v }
    suspend fun setStorageRoot(v: String?) = edit { p -> if (v == null) p.remove(Keys.storageRoot) else p[Keys.storageRoot] = v }
    suspend fun setStorageReserveMb(v: Int) = edit { it[Keys.storageReserveMb] = v }
    suspend fun setDefaultTier(v: Tier) = edit { it[Keys.defaultTier] = v.name }
    suspend fun setBlockPickle(v: Boolean) = edit { it[Keys.blockPickle] = v }
    suspend fun setToolsEnabled(v: Boolean) = edit { it[Keys.toolsEnabled] = v }
    suspend fun setEnabledToolProviders(v: Set<String>) = edit { it[Keys.enabledToolProviders] = v }
    suspend fun setExportFolder(v: String?) =
        edit { p -> if (v == null) p.remove(Keys.exportFolder) else p[Keys.exportFolder] = v }

    suspend fun setLastConversationId(v: String?) =
        edit { p -> if (v == null) p.remove(Keys.lastConversationId) else p[Keys.lastConversationId] = v }

    private suspend fun edit(block: (androidx.datastore.preferences.core.MutablePreferences) -> Unit) {
        context.dataStore.edit(block)
    }

    companion object {
        fun backendModeLabel(mode: String): String =
            runCatching { BackendId.valueOf(mode).label }.getOrDefault(mode)
    }
}
