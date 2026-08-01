package ai.ondevice.data.secure

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/** The Hugging Face token, and nothing else. */
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

    private companion object {
        const val FILE_NAME = "hf_token"
        const val KEY_HF_TOKEN = "hf_token"
    }
}
