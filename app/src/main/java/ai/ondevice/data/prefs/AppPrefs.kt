package ai.ondevice.data.prefs

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "ondevice_prefs")

/** Everything on the Settings screen that isn't a row in the database. */
class AppPrefs(private val context: Context) {

    private object Keys {
        val threadCount = intPreferencesKey("thread_count")
        val batteryGuardPercent = intPreferencesKey("battery_guard_percent")
        val wifiOnly = booleanPreferencesKey("wifi_only")
        val parallelConnections = intPreferencesKey("parallel_connections")
        val maxRetries = intPreferencesKey("max_retries")
        val storageRoot = stringPreferencesKey("storage_root")
        val storageReserveMb = intPreferencesKey("storage_reserve_mb")
        val blockPickle = booleanPreferencesKey("block_pickle")
        val lastConversationId = stringPreferencesKey("last_conversation_id")

        /**
         * The tab last looked at, so closing the app and opening it again does
         * not always land on Chat.
         *
         * Video is a pushed destination rather than a tab of its own, so this
         * is the route and it distinguishes the two visual modes on its own.
         */
        val lastRoute = stringPreferencesKey("last_route")
        val exportFolder = stringPreferencesKey("export_folder")
        val toolsEnabled = booleanPreferencesKey("tools_enabled")
        val enabledToolProviders = androidx.datastore.preferences.core.stringSetPreferencesKey("enabled_tool_providers")
        val fileScopeDevice = booleanPreferencesKey("file_scope_device")
        val toolParams = stringPreferencesKey("tool_params")
    }

    /** Default to performance-core count, not total cores (SPEC §8.1). */
    val threadCount: Flow<Int> = context.dataStore.data.map { it[Keys.threadCount] ?: 0 }

    val batteryGuardPercent: Flow<Int> = context.dataStore.data.map { it[Keys.batteryGuardPercent] ?: 20 }
    val wifiOnly: Flow<Boolean> = context.dataStore.data.map { it[Keys.wifiOnly] ?: true }
    val parallelConnections: Flow<Int> = context.dataStore.data.map { it[Keys.parallelConnections] ?: 4 }
    val maxRetries: Flow<Int> = context.dataStore.data.map { it[Keys.maxRetries] ?: 5 }
    val storageRoot: Flow<String?> = context.dataStore.data.map { it[Keys.storageRoot] }
    val storageReserveMb: Flow<Int> = context.dataStore.data.map { it[Keys.storageReserveMb] ?: 1024 }



    /** Pickle files are blocked by default; expert override (SPEC §3.2). */
    val blockPickle: Flow<Boolean> = context.dataStore.data.map { it[Keys.blockPickle] ?: true }

    val lastConversationId: Flow<String?> = context.dataStore.data.map { it[Keys.lastConversationId] }
    val lastRoute: Flow<String?> = context.dataStore.data.map { it[Keys.lastRoute] }

    suspend fun setLastRoute(v: String) {
        edit { p -> p[Keys.lastRoute] = v }
    }

    /** The SAF tree the user picked for exports, as a URI string. */
    val exportFolder: Flow<String?> = context.dataStore.data.map { it[Keys.exportFolder] }

    /** Tool use is off until asked for. */
    val toolsEnabled: Flow<Boolean> = context.dataStore.data.map { it[Keys.toolsEnabled] ?: false }

    /** Which providers are offered. */
    val enabledToolProviders: Flow<Set<String>> = context.dataStore.data.map {
        it[Keys.enabledToolProviders] ?: setOf(ai.ondevice.tools.BuiltInToolProvider.ID)
    }

    /**
     * Whether the file tools and the shell may leave the app's own folders.
     *
     * Off by default, and deliberately not remembered as "granted": the
     * all-files permission behind it is revocable from system Settings at any
     * time, so this records only what was *asked for* and [Workspace] asks the
     * system what is actually held.
     */
    val fileScopeDevice: Flow<Boolean> = context.dataStore.data.map { it[Keys.fileScopeDevice] ?: false }

    suspend fun setFileScopeDevice(v: Boolean) = edit { it[Keys.fileScopeDevice] = v }

    /**
     * What the tools have been tuned to, as one sparse JSON object.
     *
     * Sparse on purpose, and keyed `<tool>.<setting>`: only what was actually
     * changed is stored, so a default that moves in a later release moves for
     * everyone who never touched it. The same reasoning as the model
     * parameters, and the same storage shape.
     */
    val toolParams: Flow<String> = context.dataStore.data.map { it[Keys.toolParams] ?: "{}" }

    suspend fun setToolParams(v: String) = edit { it[Keys.toolParams] = v }

    suspend fun setThreadCount(v: Int) = edit { it[Keys.threadCount] = v }
    suspend fun setBatteryGuardPercent(v: Int) = edit { it[Keys.batteryGuardPercent] = v }
    suspend fun setWifiOnly(v: Boolean) = edit { it[Keys.wifiOnly] = v }
    suspend fun setParallelConnections(v: Int) = edit { it[Keys.parallelConnections] = v }
    suspend fun setMaxRetries(v: Int) = edit { it[Keys.maxRetries] = v }
    suspend fun setStorageRoot(v: String?) = edit { p -> if (v == null) p.remove(Keys.storageRoot) else p[Keys.storageRoot] = v }
    suspend fun setStorageReserveMb(v: Int) = edit { it[Keys.storageReserveMb] = v }
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
}
