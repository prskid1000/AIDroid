package ai.ondevice.data.secure

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/**
 * The Hugging Face token, and nothing else.
 *
 * SPEC §13: there is no account of any kind. The token is optional, exists only
 * so gated repos can be fetched, and — as the refusal card on S5 puts it — "is
 * stored in the Android Keystore and used for nothing else".
 *
 * Backed by EncryptedSharedPreferences so the key material stays in the
 * Keystore rather than in the file.
 */
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

    /** For display: `hf_abc…wxyz`, never the whole thing. */
    fun maskedToken(): String? = hfToken?.let {
        if (it.length <= 11) "•".repeat(it.length) else "${it.take(6)}…${it.takeLast(4)}"
    }

    private companion object {
        const val FILE_NAME = "hf_token"
        const val KEY_HF_TOKEN = "hf_token"
    }
}
