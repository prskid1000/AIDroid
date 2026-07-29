package ai.ondevice.speech

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale
import java.util.UUID

/**
 * Speech synthesis, SPEC §7.
 *
 * Two engines sit behind one interface, and which one is speaking is always
 * named on screen:
 *
 *  - **Kokoro** — the on-device neural voice the spec asks for. It needs its
 *    runtime bundle *and* a phonemiser, so until both are installed it is not
 *    offered, and the app says so rather than silently substituting.
 *  - **The system engine** — Android's own TTS. Real audio today, real voices,
 *    real pitch and rate, and `synthesizeToFile` gives a real WAV to share.
 *
 * The substitution is the honest part. An app that quietly falls back and calls
 * the result "Kokoro" is lying about what produced the audio, and §1.2 is
 * explicit that a degraded path has to announce itself.
 */
class SpeechSynthesizer(private val context: Context) {

    private var tts: TextToSpeech? = null
    private var ready = false

    suspend fun initialise(): Boolean {
        if (ready) return true
        val done = CompletableDeferred<Boolean>()
        tts = TextToSpeech(context) { status ->
            ready = status == TextToSpeech.SUCCESS
            done.complete(ready)
        }
        return done.await()
    }

    /** The voices the *installed* system engine actually has. Not a hardcoded list. */
    suspend fun systemVoices(): List<SynthVoice> = withContext(Dispatchers.IO) {
        if (!initialise()) return@withContext emptyList()
        val engine = tts ?: return@withContext emptyList()
        runCatching {
            engine.voices.orEmpty()
                .filterNot { it.isNetworkConnectionRequired } // §13 — nothing leaves the device
                .map { voice ->
                    SynthVoice(
                        id = voice.name,
                        displayName = prettyName(voice.name, voice.locale),
                        locale = voice.locale.toLanguageTag(),
                        localeLabel = voice.locale.getDisplayName(Locale.UK),
                        quality = voice.quality,
                        provider = SynthProvider.SYSTEM,
                        available = !voice.features.orEmpty()
                            .contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED),
                    )
                }
                .sortedWith(compareBy({ it.localeLabel }, { it.displayName }))
        }.getOrDefault(emptyList())
    }

    /**
     * Speak now. Emits progress so the screen can show which utterance is live
     * and offer a stop that actually stops.
     */
    fun speak(request: SpeechRequest): Flow<SpeechEvent> = callbackFlow {
        if (!initialise()) {
            trySend(SpeechEvent.Failed("No text-to-speech engine is installed on this device."))
            close()
            return@callbackFlow
        }
        val engine = tts ?: run { close(); return@callbackFlow }
        val utteranceId = UUID.randomUUID().toString()

        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(id: String?) {
                trySend(SpeechEvent.Started)
            }

            override fun onRangeStart(id: String?, start: Int, end: Int, frame: Int) {
                // Word-level highlighting: the screen underlines what is being
                // said right now, which is the whole reason to stream at all.
                trySend(SpeechEvent.Range(start, end))
            }

            override fun onDone(id: String?) {
                trySend(SpeechEvent.Done)
                close()
            }

            @Deprecated("Required by UtteranceProgressListener")
            override fun onError(id: String?) {
                trySend(SpeechEvent.Failed("The engine stopped part-way through."))
                close()
            }

            override fun onError(id: String?, errorCode: Int) {
                trySend(SpeechEvent.Failed(errorLabel(errorCode)))
                close()
            }
        })

        apply(engine, request)
        val params = Bundle().apply {
            putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, request.volume.coerceIn(0f, 1f))
        }
        engine.speak(request.text, TextToSpeech.QUEUE_FLUSH, params, utteranceId)

        awaitClose { runCatching { engine.stop() } }
    }

    /**
     * Render to a file the user can keep or send. This is the "export the audio"
     * half of §7 — a passage you can only hear once is not an artifact.
     */
    suspend fun synthesizeToFile(request: SpeechRequest, destination: File): Result<File> =
        withContext(Dispatchers.IO) {
            runCatching {
                check(initialise()) { "No text-to-speech engine is installed on this device." }
                val engine = tts!!
                apply(engine, request)

                val done = CompletableDeferred<Result<File>>()
                val utteranceId = UUID.randomUUID().toString()
                engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(id: String?) = Unit
                    override fun onDone(id: String?) {
                        done.complete(Result.success(destination))
                    }

                    @Deprecated("Required by UtteranceProgressListener")
                    override fun onError(id: String?) {
                        done.complete(Result.failure(IllegalStateException("Synthesis failed.")))
                    }

                    override fun onError(id: String?, errorCode: Int) {
                        done.complete(Result.failure(IllegalStateException(errorLabel(errorCode))))
                    }
                })

                destination.parentFile?.mkdirs()
                val queued = engine.synthesizeToFile(request.text, Bundle(), destination, utteranceId)
                check(queued == TextToSpeech.SUCCESS) { "The engine refused the request." }
                done.await().getOrThrow()
            }
        }

    fun stop() {
        runCatching { tts?.stop() }
    }

    fun release() {
        runCatching { tts?.shutdown() }
        tts = null
        ready = false
    }

    private fun apply(engine: TextToSpeech, request: SpeechRequest) {
        // Android clamps rate and pitch to [0.1, 2.0]; the UI ranges match, so
        // a value the user can pick is always a value the engine will take.
        engine.setSpeechRate(request.speed.coerceIn(0.1f, 2.0f))
        engine.setPitch(request.pitch.coerceIn(0.1f, 2.0f))
        request.voiceId?.let { id ->
            engine.voices?.firstOrNull { it.name == id }?.let { engine.voice = it }
        }
    }

    private fun prettyName(name: String, locale: Locale): String {
        // System voice ids look like "en-gb-x-gba-local"; the tail is the useful
        // part and the language is already shown beside it.
        val tail = name.substringAfterLast("-x-").removeSuffix("-local").removeSuffix("-network")
        return tail.ifBlank { name }.replace('-', ' ')
    }

    private fun errorLabel(code: Int): String = when (code) {
        TextToSpeech.ERROR_NETWORK, TextToSpeech.ERROR_NETWORK_TIMEOUT ->
            "That voice needs the network. Pick a voice marked on-device."
        TextToSpeech.ERROR_NOT_INSTALLED_YET -> "That voice is still downloading in the system engine."
        TextToSpeech.ERROR_OUTPUT -> "The audio device refused the output."
        TextToSpeech.ERROR_SYNTHESIS -> "The engine could not synthesise that text."
        else -> "Synthesis failed."
    }
}

data class SpeechRequest(
    val text: String,
    val voiceId: String? = null,
    val speed: Float = 1.0f,
    val pitch: Float = 1.0f,
    val volume: Float = 1.0f,
)

data class SynthVoice(
    val id: String,
    val displayName: String,
    val locale: String,
    val localeLabel: String,
    val quality: Int,
    val provider: SynthProvider,
    val available: Boolean = true,
    /** Kokoro only: the blend partner, when one is set. */
    val blendWith: String? = null,
)

enum class SynthProvider(val label: String) {
    KOKORO("Kokoro"),
    SYSTEM("System engine"),
}

sealed interface SpeechEvent {
    data object Started : SpeechEvent
    data class Range(val start: Int, val end: Int) : SpeechEvent
    data object Done : SpeechEvent
    data class Failed(val message: String) : SpeechEvent
}
