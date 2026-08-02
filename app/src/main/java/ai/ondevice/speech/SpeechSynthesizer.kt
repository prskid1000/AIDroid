package ai.ondevice.speech

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale
import java.util.UUID

/** Speech synthesis, SPEC §7. */
class SpeechSynthesizer(
    private val context: Context,
    private val kokoro: KokoroEngine,
    private val omniVoice: OmniVoiceEngine,
    private val capabilities: ai.ondevice.data.hf.DeviceCapabilities,
) {

    private var tts: TextToSpeech? = null
    private var ready = false

    @Volatile
    private var kokoroDirectory: File? = null

    @Volatile
    private var omniVoiceDirectory: File? = null

    @Volatile
    private var player: android.media.AudioTrack? = null

    /** Point the synthesiser at an installed Kokoro model, or null if none is. */
    fun useKokoroModel(directory: File?) {
        kokoroDirectory = directory
    }

    /** Point the synthesiser at an installed OmniVoice model, or null if none is. */
    fun useOmniVoiceModel(directory: File?) {
        omniVoiceDirectory = directory?.takeIf { omniVoice.looksInstalled(it) }
    }

    /** So the caller can sort several installed models into the right engine. */
    fun omniVoiceLooksInstalled(directory: File): Boolean = omniVoice.looksInstalled(directory)

    fun kokoroLooksInstalled(directory: File): Boolean = kokoro.looksInstalled(directory)

    /**
     * Whether Kokoro's graph is on the device and its speaker vectors are not.
     *
     * The distinction the error message needs: `looksInstalled` answers "can
     * this run", and both halves being required makes a half-installed model
     * indistinguishable from an absent one. Telling someone to download a model
     * they already have is the least useful thing the screen can say.
     */
    fun kokoroGraphWithoutVoices(directories: List<File>): Boolean =
        directories.any { kokoro.graphOnly(it) }

    /**
     * Which engine the model in [directory] *belongs to*, complete or not.
     *
     * Deliberately a weaker question than "can this run". It used to be the
     * stronger one, and the consequence was that a Kokoro folder missing its
     * voice packs belonged to no engine at all — so it vanished from the
     * engine's model list, and switching engines appeared to do nothing to the
     * picker. A half-installed model is still that engine's model; whether it
     * is ready is [kokoroReady] and [omniVoiceReady]'s business, and the error
     * message's.
     */
    fun providerFor(directory: File): SynthProvider? = when {
        omniVoice.looksInstalled(directory) -> SynthProvider.OMNIVOICE
        kokoro.looksInstalled(directory) -> SynthProvider.KOKORO
        // Half-installed Kokoro: a graph and none of its voices.
        //
        // Guarded against claiming a half-installed *OmniVoice*, which
        // `graphOnly` alone would — it asks only for an `.onnx` and no
        // 522,240-byte packs, and an OmniVoice folder missing one of its four
        // graphs answers yes to both halves of that. The result was the two
        // engines' models appearing under each other.
        kokoro.graphOnly(directory) && !omniVoice.partlyInstalled(directory) ->
            SynthProvider.KOKORO
        else -> null
    }

    val kokoroReady: Boolean get() = kokoro.runtimeAvailable && kokoroDirectory != null

    val omniVoiceReady: Boolean get() = omniVoice.runtimeAvailable && omniVoiceDirectory != null

    /** Whether the installed OmniVoice can copy a voice from a recording. */
    val omniVoiceCloningReady: Boolean
        get() = omniVoiceReady && omniVoiceDirectory?.let { omniVoice.cloningLooksInstalled(it) } == true

    /** Kokoro's voices, marked available only if this build can speak them. */
    fun kokoroVoices(): List<SynthVoice> = KokoroVoices.catalogue(available = kokoroReady)

    /** OmniVoice's single entry. */
    fun omniVoiceVoices(): List<SynthVoice> = listOf(
        SynthVoice(
            id = OMNIVOICE_VOICE_ID,
            displayName = "OmniVoice",
            locale = "mul",
            localeLabel = "Any language · emotion tags",
            quality = 400,
            provider = SynthProvider.OMNIVOICE,
            available = omniVoiceReady,
        ),
    )

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

    /** Speak now. */
    fun speak(request: SpeechRequest): Flow<SpeechEvent> = when (request.provider) {
        SynthProvider.KOKORO -> speakWithKokoro(request)
        SynthProvider.OMNIVOICE -> speakWithNeural(request) { renderOmniVoice(it) }
        SynthProvider.SYSTEM -> speakWithSystem(request)
    }

    /** Kokoro: synthesise the whole passage, then play it. */
    private fun speakWithKokoro(request: SpeechRequest): Flow<SpeechEvent> =
        speakWithNeural(request) { renderKokoro(it) }

    /** Both neural engines share this: synthesise the whole passage, then play it. */
    private fun speakWithNeural(
        request: SpeechRequest,
        render: suspend (SpeechRequest) -> Result<KokoroAudio>,
    ): Flow<SpeechEvent> = flow {
        val audio = render(request).getOrElse { failure ->
            emit(SpeechEvent.Failed(failure.message ?: "That could not be spoken."))
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

    /** Render to a file the user can keep or send. */
    suspend fun synthesizeToFile(request: SpeechRequest, destination: File): Result<File> =
        withContext(Dispatchers.IO) {
            val neural = when (request.provider) {
                SynthProvider.KOKORO -> renderKokoro(request)
                SynthProvider.OMNIVOICE -> renderOmniVoice(request)
                SynthProvider.SYSTEM -> null
            }
            if (neural != null) {
                neural.onFailure { android.util.Log.e(TAG, "render failed", it) }
                return@withContext neural.mapCatching { audio ->
                    android.util.Log.i(
                        TAG,
                        "writing ${audio.samples.size} samples at ${audio.sampleRate} Hz " +
                            "to ${destination.name}",
                    )
                    // A waveform with no samples is not a file worth claiming to have saved.
                    check(audio.samples.isNotEmpty()) {
                        "The engine returned no audio for that script, so there was nothing to save."
                    }
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
                val volume = Bundle().apply {
                    putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, request.volume.coerceIn(0f, 1f))
                }
                val queued = engine.synthesizeToFile(request.text, volume, destination, utteranceId)
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
        runCatching {
            kotlinx.coroutines.runBlocking {
                kokoro.unload()
                omniVoice.unload()
            }
        }
    }

    companion object {
        private const val TAG = "SpeechSynthesizer"

        // The thread policy lives on DeviceCapabilities, so all four engines
        // answer this question the same way — see `inferenceThreads` there.

        const val OMNIVOICE_VOICE_ID = "omnivoice"

        /** How long past the audio's own duration playback is given to drain. */
        private const val PLAYBACK_GRACE_MS = 750L
        private const val PLAYBACK_POLL_MS = 20L
    }

    // — Kokoro —

    /** Load if needed, then synthesise. */
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

        kokoro.load(directory, threads = capabilities.inferenceThreads)
            .onFailure { return Result.failure(it) }

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
                splitPattern = request.splitPattern,
                trimSilence = request.trimSilence,
                // Kokoro has no volume input, so gain is applied to the waveform.
                volume = request.volume,
                languageOverride = request.languageCode,
            ),
        )
    }

    private suspend fun renderOmniVoice(request: SpeechRequest): Result<KokoroAudio> {
        val directory = omniVoiceDirectory ?: return Result.failure(
            IllegalStateException(
                "No OmniVoice model is installed. Models → Add a model, then " +
                    ai.ondevice.core.StarterModels.OMNIVOICE_REPO + ".",
            ),
        )
        omniVoice.load(directory, threads = capabilities.inferenceThreads)
            .onFailure { return Result.failure(it) }
        return omniVoice.synthesize(
            OmniVoiceRequest(
                text = request.text,
                speed = request.speed,
                trimSilence = request.trimSilence,
                language = request.languageCode?.takeIf { it.isNotBlank() && it != "auto" },
                instruction = request.voiceDesign?.takeIf { it.isNotBlank() },
                steps = request.steps ?: OmniVoiceEngine.DEFAULT_STEPS,
                frames = request.frames?.takeIf { it > 0 },
                guidance = request.guidance ?: OmniVoiceEngine.DEFAULT_GUIDANCE,
                timestepShift = request.timestepShift ?: OmniVoiceEngine.DEFAULT_T_SHIFT,
                layerPenalty = request.layerPenalty ?: OmniVoiceEngine.DEFAULT_LAYER_PENALTY,
                positionTemperature = request.positionTemperature
                    ?: OmniVoiceEngine.DEFAULT_POSITION_TEMPERATURE,
                classTemperature = request.classTemperature
                    ?: OmniVoiceEngine.DEFAULT_CLASS_TEMPERATURE,
                seed = request.seed,
                reference = request.voiceReference,
            ),
        ).map { audio ->
            // OmniVoice has no gain input either, so volume is applied here for
            // the same reason it is for Kokoro.
            if (request.volume == 1f) {
                audio
            } else {
                val gain = request.volume.coerceIn(0f, 2f)
                audio.copy(samples = FloatArray(audio.samples.size) {
                    (audio.samples[it] * gain).coerceIn(-1f, 1f)
                })
            }
        }
    }

    private fun voicePack(directory: File, voiceId: String): File? {
        val name = KokoroVoices.packName(voiceId)
        return directory.walkTopDown().firstOrNull { it.isFile && it.name == name }
    }

    /** Play 24 kHz mono floats, blocking until the audio has actually finished. */
    private suspend fun play(audio: KokoroAudio) {
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

        // Wait on what was actually queued, not on what we hoped to queue.
        val queued = offset
        val deadline = System.currentTimeMillis() +
            (queued * 1000L / audio.sampleRate.coerceAtLeast(1)) + PLAYBACK_GRACE_MS
        while (player === track &&
            track.playbackHeadPosition < queued &&
            System.currentTimeMillis() < deadline
        ) {
            delay(PLAYBACK_POLL_MS)
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
    /** System engine only. */
    val pitch: Float = 1.0f,
    val volume: Float = 1.0f,
    /** Which engine the user asked for — routing never infers this. */
    val provider: SynthProvider = SynthProvider.SYSTEM,
    /** Kokoro only: a second voice to interpolate towards. */
    val blendVoiceId: String? = null,
    val blendRatio: Float = 0f,
    /** Kokoro only, from the Advanced parameters. */
    val splitPattern: String = KokoroRequest.DEFAULT_SPLIT_PATTERN,
    val trimSilence: Boolean = true,
    /** Which language to speak in. */
    val languageCode: String? = null,
    /** OmniVoice's voice design: the speaker described in words rather than chosen from a list. */
    val voiceDesign: String? = null,
    /** OmniVoice only: how many iterative unmasking passes to run. */
    val steps: Int? = null,
    /** OmniVoice only: grid length in 40 ms frames. Null estimates from the text. */
    val frames: Int? = null,
    /** OmniVoice only: the rest of upstream's generation config. */
    val guidance: Float? = null,
    val timestepShift: Float? = null,
    val layerPenalty: Float? = null,
    val positionTemperature: Float? = null,
    val classTemperature: Float? = null,
    /** 0 picks a fresh seed per run; anything else makes the run repeatable. */
    val seed: Long = 0L,
    /** OmniVoice only: a recording to copy the voice from, with what it said. */
    val voiceReference: ai.ondevice.speech.VoiceReference? = null,
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

    OMNIVOICE("OmniVoice"),
    SYSTEM("System engine"),
}

sealed interface SpeechEvent {
    data object Started : SpeechEvent
    data class Range(val start: Int, val end: Int) : SpeechEvent
    data object Done : SpeechEvent
    data class Failed(val message: String) : SpeechEvent
}
