package ai.ondevice.speech

import ai.ondevice.engine.EngineLog

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/** Text to IPA, for Kokoro. */
class Phonemizer(private val context: Context) {

    private val mutex = Mutex()

    @Volatile
    private var unpacked: File? = null

    @Volatile
    private var currentVoice: String? = null

    val available: Boolean get() = PhonemizerBridge.available

    val unavailableReason: String?
        get() = if (available) null else {
            PhonemizerBridge.loadError ?: "The espeak-ng phonemiser is not installed in this build."
        }

    /** The espeak voice for a Kokoro voice id. */
    fun espeakVoiceFor(kokoroVoiceId: String): String? = when (kokoroVoiceId.firstOrNull()) {
        'a' -> "en-us"
        'b' -> "en-gb"
        'e' -> "es"
        'f' -> "fr-fr"
        'h' -> "hi"
        'i' -> "it"
        'p' -> "pt-br"
        else -> null
    }

    /** [languageOverride] is the `lang_code` parameter: an espeak voice to use instead of the one the Kokoro voice id implies. */
    /** A `lang_code` value to an espeak voice, or null if this build has no data staged for it. */
    private fun espeakVoiceForCode(code: String): String? = when (code.lowercase()) {
        "en-us" -> "en-us"
        "en-gb" -> "en-gb"
        "es" -> "es"
        "fr" -> "fr-fr"
        "hi" -> "hi"
        "it" -> "it"
        "pt-br" -> "pt-br"
        else -> null
    }

    suspend fun phonemize(
        text: String,
        kokoroVoiceId: String,
        languageOverride: String? = null,
    ): Result<String> =
        withContext(Dispatchers.Default) {
            runCatching {
                check(available) { unavailableReason!! }
                val derived = espeakVoiceFor(kokoroVoiceId)
                val voice = languageOverride?.takeIf { it.isNotBlank() && it != "auto" }
                    ?.let { requested ->
                        espeakVoiceForCode(requested) ?: error(
                            "This build has no espeak data for \"$requested\". " +
                                "Language must be auto, en-us, en-gb, es, fr, hi, it or pt-br.",
                        )
                    }
                    ?: derived
                    ?: error(
                        "This build cannot pronounce ${KokoroVoices.languageOf(kokoroVoiceId)} text. " +
                            "Kokoro uses a different front end for it than espeak-ng.",
                    )

                mutex.withLock {
                    val dataParent = unpacked ?: unpackAssets().also { unpacked = it }
                    if (currentVoice != voice) {
                        PhonemizerBridge.nativeInit(dataParent.absolutePath, voice)
                        currentVoice = voice
                    }
                    PhonemizerBridge.nativePhonemize(text.trim()).also { ipa ->
                        // espeak returning nothing is not an error it reports — it hands back an empty string, which tokenises to no ids, which the model turns into silence.
                        if (ipa.isBlank()) {
                            EngineLog.w(
                                TAG,
                                "espeak returned nothing for voice=$voice, ${text.length} chars",
                            )
                        }
                    }
                }
            }.onFailure { EngineLog.e(TAG, "phonemize failed", it) }
        }

    /** For the runtimes screen: what is actually installed. */
    suspend fun version(): String? = withContext(Dispatchers.Default) {
        runCatching {
            check(available)
            mutex.withLock {
                val dataParent = unpacked ?: unpackAssets().also { unpacked = it }
                if (currentVoice == null) {
                    PhonemizerBridge.nativeInit(dataParent.absolutePath, "en-us")
                    currentVoice = "en-us"
                }
                PhonemizerBridge.nativeVersion()
            }
        }.getOrNull()
    }

    private fun unpackAssets(): File {
        val parent = File(context.filesDir, "espeak")
        val stamp = File(parent, ".unpacked")
        if (stamp.isFile && stamp.readText().trim() == STAMP) return parent

        parent.deleteRecursively()
        parent.mkdirs()
        copyAssetDirectory(ASSET_ROOT, File(parent, ASSET_ROOT))
        stamp.writeText(STAMP)
        return parent
    }

    private fun copyAssetDirectory(assetPath: String, destination: File) {
        val children = context.assets.list(assetPath).orEmpty()
        if (children.isEmpty()) {
            // A leaf. `list` returns empty for files as well as empty
            // directories, so the distinction is made by trying to open it.
            destination.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                destination.outputStream().use(input::copyTo)
            }
            return
        }
        destination.mkdirs()
        children.forEach { child ->
            copyAssetDirectory("$assetPath/$child", File(destination, child))
        }
    }

    private companion object {
        const val TAG = "Phonemizer"
        const val ASSET_ROOT = "espeak-ng-data"

        /** Bumped whenever the staged tables change. */
        const val STAMP = "espeak-ng 1.52.0 phondata 0x014801"
    }
}
