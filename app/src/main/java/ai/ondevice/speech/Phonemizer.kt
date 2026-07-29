package ai.ondevice.speech

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Text to IPA, for Kokoro.
 *
 * espeak reads its phoneme tables and dictionaries with stdio, and an entry in
 * an APK's asset archive is not a file — there is no path to hand it. So the
 * tables are unpacked once into app storage and espeak is pointed there.
 *
 * The unpack is guarded by a stamp file holding the tables' format version
 * rather than by "does the directory exist". A half-finished copy from a
 * process killed mid-unpack would otherwise look complete forever, and the
 * failure it produces — espeak refusing a truncated dictionary — reads like a
 * corrupt install rather than what it is.
 */
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

    /**
     * The espeak voice for a Kokoro voice id.
     *
     * Kokoro names a voice `<language><gender>_<name>`, and the leading letter
     * is the language. Japanese and Mandarin are absent on purpose: Kokoro's
     * own pipeline phonemises those with misaki, not espeak, and feeding it
     * espeak's output for them would produce confident nonsense.
     */
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

    /**
     * [languageOverride] is the `lang_code` parameter: an espeak voice to use
     * instead of the one the Kokoro voice id implies. It is a real thing to
     * want — an American-trained voice reading British spellings pronounces
     * "schedule" the American way, and forcing en-gb fixes it — but it is also
     * a way to hand the model phonemes it was never trained on, so it stays an
     * Advanced control with the default doing the sensible thing.
     */
    /**
     * A `lang_code` value to an espeak voice, or null if this build has no data
     * staged for it. `ja` and `zh` are in the manifest's list because Kokoro
     * has voices for them; they resolve to null here because
     * tools/stage-espeak-data.py deliberately does not ship their tables.
     */
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
                    PhonemizerBridge.nativePhonemize(text.trim())
                }
            }
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
        const val ASSET_ROOT = "espeak-ng-data"

        /**
         * Bumped whenever the staged tables change. It is the phondata format
         * version from tools/stage-espeak-data.py, so a runtime bump that
         * changes the tables forces a re-unpack instead of leaving a stale
         * copy espeak will reject.
         */
        const val STAMP = "espeak-ng 1.52.0 phondata 0x014801"
    }
}
