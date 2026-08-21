package ai.ondevice.data.secure

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/** Every credential the app holds, behind the Android Keystore. */
class TokenStore(context: Context) {

    private val prefs: SharedPreferences by lazy {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        EncryptedSharedPreferences.create(
            context,
            FILE_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    }

    var hfToken: String?
        get() = prefs.getString(KEY_HF_TOKEN, null)?.takeIf { it.isNotBlank() }
        set(value) {
            prefs.edit().apply {
                if (value.isNullOrBlank()) remove(KEY_HF_TOKEN) else putString(KEY_HF_TOKEN, value)
            }.apply()
        }

    val hasToken: Boolean get() = hfToken != null

    fun maskedToken(): String? = hfToken?.let {
        if (it.length <= 11) "•".repeat(it.length) else "${it.take(6)}…${it.takeLast(4)}"
    }

    // — MCP OAuth —
    //
    // Here rather than in the database for the same reason the Hugging Face
    // token is: a bearer token is a credential, the database is a plain file
    // inside the app's directory, and a backup or an extraction would carry it
    // off intact. The server row keeps the discovered endpoints and the client
    // id, which are not secrets; only these three are.

    fun oauthTokens(serverId: String): Triple<String, String?, Long?>? {
        val access = prefs.getString("$KEY_MCP_ACCESS$serverId", null)?.takeIf { it.isNotBlank() }
            ?: return null
        val refresh = prefs.getString("$KEY_MCP_REFRESH$serverId", null)?.takeIf { it.isNotBlank() }
        val expiry = prefs.getLong("$KEY_MCP_EXPIRY$serverId", 0L).takeIf { it > 0L }
        return Triple(access, refresh, expiry)
    }

    fun setOauthTokens(serverId: String, access: String, refresh: String?, expiresAt: Long?) {
        prefs.edit().apply {
            putString("$KEY_MCP_ACCESS$serverId", access)
            if (refresh.isNullOrBlank()) {
                remove("$KEY_MCP_REFRESH$serverId")
            } else {
                putString("$KEY_MCP_REFRESH$serverId", refresh)
            }
            if (expiresAt == null) {
                remove("$KEY_MCP_EXPIRY$serverId")
            } else {
                putLong("$KEY_MCP_EXPIRY$serverId", expiresAt)
            }
        }.apply()
    }

    // — the proxy's own bearer token —
    //
    // Here for the same reason the other two credentials are: this one lets
    // anything holding it generate on this device, read whatever the file tools
    // can read and spend the battery, so it is worth at least as much as a
    // Hugging Face token and the database is a plain file inside the app's
    // directory.

    var proxyToken: String?
        get() = prefs.getString(KEY_PROXY_TOKEN, null)?.takeIf { it.isNotBlank() }
        set(value) {
            prefs.edit().apply {
                if (value.isNullOrBlank()) remove(KEY_PROXY_TOKEN) else putString(KEY_PROXY_TOKEN, value)
            }.apply()
        }

    /**
     * The token as the screen shows it after the first time.
     *
     * Shown in full once, on generation, and masked from then on — the shape
     * the Hugging Face block already uses, and the reason is the same: a
     * credential on screen is a credential in a screenshot.
     */
    fun maskedProxyToken(): String? = proxyToken?.let {
        if (it.length <= 11) "•".repeat(it.length) else "${it.take(6)}…${it.takeLast(4)}"
    }

    // — the password on the proxy's TLS keystore —
    //
    // A PKCS12 file cannot be written without one, so this exists because the
    // format insists rather than because it is protecting much: the keystore
    // sits inside the app's own directory beside everything else. It is here
    // and not in the file's name or a constant for the one case that does
    // matter — a backup or an extraction that carries the private key off, and
    // with it the ability to be this server to anything that trusted it.

    var proxyKeystorePassword: String?
        get() = prefs.getString(KEY_PROXY_KEYSTORE, null)?.takeIf { it.isNotBlank() }
        set(value) {
            prefs.edit().apply {
                if (value.isNullOrBlank()) {
                    remove(KEY_PROXY_KEYSTORE)
                } else {
                    putString(KEY_PROXY_KEYSTORE, value)
                }
            }.apply()
        }

    /** Called when a server is forgotten, and when its authorisation is revoked. */
    fun clearOauthTokens(serverId: String) {
        prefs.edit()
            .remove("$KEY_MCP_ACCESS$serverId")
            .remove("$KEY_MCP_REFRESH$serverId")
            .remove("$KEY_MCP_EXPIRY$serverId")
            .apply()
    }

    private companion object {
        const val FILE_NAME = "hf_token"
        const val KEY_HF_TOKEN = "hf_token"
        const val KEY_MCP_ACCESS = "mcp_access:"
        const val KEY_MCP_REFRESH = "mcp_refresh:"
        const val KEY_MCP_EXPIRY = "mcp_expiry:"
        const val KEY_PROXY_TOKEN = "proxy_token"
        const val KEY_PROXY_KEYSTORE = "proxy_keystore"
    }
}
