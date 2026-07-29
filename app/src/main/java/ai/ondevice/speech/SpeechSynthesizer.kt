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
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.onCompletion
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
 *  - **Kokoro** — the on-device neural voice the spec asks for: espeak-ng turns
 *    the text into IPA, the ONNX graph turns that into a 24 kHz waveform, and
 *    an AudioTrack plays it. Available once its weights are installed.
 *  - **The system engine** — Android's own TTS, for languages this build has no
 *    phonemiser for and for devices where Kokoro's weights are not present.
 *
 * Routing is by the *voice the user picked*, never by availability. Asking for
 * a Kokoro voice and getting the system engine is the silent substitution §1.2
 * forbids, so a Kokoro voice whose model is missing is an error with a reason,
 * not a quiet downgrade.
 */
class SpeechSynthesizer(
    private val context: Context,
    private val kokoro: KokoroEngine,
) {

    private var tts: TextToSpeech? = null
    private var ready = false

    @Volatile
    private var kokoroDirectory: File? = null

    @Volatile
    private var player: android.media.AudioTrack? = null

    /**
     * Point the synthesiser at an installed Kokoro model, or null if none is.
     * The caller owns the library, so it is the one that knows.
     */
    fun useKokoroModel(directory: File?) {
        kokoroDirectory = directory
    }

    val kokoroReady: Boolean get() = kokoro.runtimeAvailable && kokoroDirectory != null

    /** Kokoro's voices, marked available only if this build can speak them. */
    fun kokoroVoices(): List<SynthVoice> = KokoroVoices.catalogue(available = kokoroReady)

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
    fun speak(request: SpeechRequest): Flow<SpeechEvent> =
        if (request.isKokoro) speakWithKokoro(request) else speakWithSystem(request)

    /**
     * Kokoro: synthesise the whole passage, then play it.
     *
     * Not streamed. The graph produces a sentence at a time and the seams
     * matter, so playing chunk one while chunk two is still on the CPU would
     * mean a gap whose length depends on the load — audibly worse than a
     * slightly later start. The screen shows the wait for what it is.
     */
    private fun speakWithKokoro(request: SpeechRequest): Flow<SpeechEvent> = flow {
        val audio = renderKokoro(request).getOrElse { failure ->
            emit(SpeechEvent.Failed(failure.message ?: "Kokoro could not speak that."))
            return@flow
        }
        emit(SpeechEvent.Started)
        play(audio)
        emit(SpeechEvent.Done)
    }.flowOn(Dispatchers.IO).onCompletion { stopPlayback() }

    private fun speakWithSystem(request: SpeechRequest): Flow<SpeechEvent> = callbackFlow {
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
            if (request.isKokoro) {
                return@withContext renderKokoro(request).map { audio ->
                    WavFile.write(destination, audio.samples, audio.sampleRate)
                }
            }
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
        stopPlayback()
    }

    fun release() {
        runCatching { tts?.shutdown() }
        tts = null
        ready = false
        stopPlayback()
        runCatching { kotlinx.coroutines.runBlocking { kokoro.unload() } }
    }

    // — Kokoro —

    /**
     * Load if needed, then synthesise. The load is deferred to first use
     * because an ONNX session costs both time and a couple of hundred megabytes
     * of mapped weights, and most sessions of this app never speak.
     */
    private suspend fun renderKokoro(request: SpeechRequest): Result<KokoroAudio> {
        val directory = kokoroDirectory ?: return Result.failure(
            IllegalStateException(
                "No Kokoro model is installed. Models → Add a model, and search for Kokoro.",
            ),
        )
        val voiceId = request.voiceId ?: return Result.failure(
            IllegalStateException("No Kokoro voice was selected."),
        )
        if (!KokoroVoices.hasPhonemiser(voiceId)) {
            return Result.failure(
                IllegalStateException(
                    "This build has no ${KokoroVoices.languageOf(voiceId)} phonemiser. " +
                        "Kokoro uses a different front end for it than espeak-ng.",
                ),
            )
        }

        val pack = voicePack(directory, voiceId) ?: return Result.failure(
            IllegalStateException(
                "The voice pack ${KokoroVoices.packName(voiceId)} is not in ${directory.name}. " +
                    "Re-download the model to fetch its voices.",
            ),
        )

        kokoro.load(directory).onFailure { return Result.failure(it) }

        return kokoro.synthesize(
            KokoroRequest(
                text = request.text,
                voiceId = voiceId,
                voicePack = pack,
                // Kokoro takes speed as a multiplier on duration prediction,
                // which is the same sense as the system engine's rate.
                speed = request.speed.coerceIn(0.5f, 2.0f),
                blendPack = request.blendVoiceId?.let { voicePack(directory, it) },
                blendRatio = request.blendRatio,
            ),
        )
    }

    private fun voicePack(directory: File, voiceId: String): File? {
        val name = KokoroVoices.packName(voiceId)
        return directory.walkTopDown().firstOrNull { it.isFile && it.name == name }
    }

    /**
     * Play 24 kHz mono floats, blocking until the audio has actually finished.
     *
     * `AudioTrack.write` returns once the data is queued, not once it is heard,
     * so a naive version reports "done" while the last second is still in the
     * buffer — and [stop] would then cut off audio the user was told had
     * finished. Hence the drain on the playback head.
     */
    private fun play(audio: KokoroAudio) {
        val minimum = android.media.AudioTrack.getMinBufferSize(
            audio.sampleRate,
            android.media.AudioFormat.CHANNEL_OUT_MONO,
            android.media.AudioFormat.ENCODING_PCM_FLOAT,
        ).coerceAtLeast(audio.sampleRate * 4)

        val track = android.media.AudioTrack.Builder()
            .setAudioAttributes(
                android.media.AudioAttributes.Builder()
                    .setUsage(android.media.AudioAttributes.USAGE_MEDIA)
                    .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(
                android.media.AudioFormat.Builder()
                    .setEncoding(android.media.AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(audio.sampleRate)
                    .setChannelMask(android.media.AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            )
            .setBufferSizeInBytes(minimum)
            .setTransferMode(android.media.AudioTrack.MODE_STREAM)
            .build()

        player = track
        track.play()

        var offset = 0
        while (offset < audio.samples.size && player === track) {
            val written = track.write(
                audio.samples,
                offset,
                audio.samples.size - offset,
                android.media.AudioTrack.WRITE_BLOCKING,
            )
            if (written <= 0) break
            offset += written
        }

        while (player === track && track.playbackHeadPosition < audio.samples.size) {
            Thread.sleep(20)
        }
        stopPlayback()
    }

    private fun stopPlayback() {
        val track = player ?: return
        player = null
        runCatching { track.pause() }
        runCatching { track.flush() }
        runCatching { track.release() }
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
    /** Which engine the user asked for — routing never infers this. */
    val provider: SynthProvider = SynthProvider.SYSTEM,
    /** Kokoro only: a second voice to interpolate towards. */
    val blendVoiceId: String? = null,
    val blendRatio: Float = 0f,
) {
    val isKokoro: Boolean get() = provider == SynthProvider.KOKORO
}

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
