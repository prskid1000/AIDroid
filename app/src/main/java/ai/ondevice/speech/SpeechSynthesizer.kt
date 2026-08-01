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
    private val omniVoice: OmniVoiceEngine,
    private val capabilities: ai.ondevice.data.hf.DeviceCapabilities,
    private val computeDevice: ai.ondevice.engine.ComputeDevice,
) {

    private var tts: TextToSpeech? = null
    private var ready = false

    @Volatile
    private var kokoroDirectory: File? = null

    @Volatile
    private var omniVoiceDirectory: File? = null

    @Volatile
    private var player: android.media.AudioTrack? = null

    /**
     * Point the synthesiser at an installed Kokoro model, or null if none is.
     * The caller owns the library, so it is the one that knows.
     */
    fun useKokoroModel(directory: File?) {
        kokoroDirectory = directory
    }

    /**
     * Point the synthesiser at an installed OmniVoice model, or null if none is.
     * Checked by contents rather than by name — the install is four graphs and a
     * tokenizer, and any directory holding them will do.
     */
    fun useOmniVoiceModel(directory: File?) {
        omniVoiceDirectory = directory?.takeIf { omniVoice.looksInstalled(it) }
    }

    /** So the caller can sort several installed models into the right engine. */
    fun omniVoiceLooksInstalled(directory: File): Boolean = omniVoice.looksInstalled(directory)

    fun kokoroLooksInstalled(directory: File): Boolean = kokoro.looksInstalled(directory)

    /**
     * Which engine, if any, can run the model in [directory].
     *
     * The library holds "text-to-speech models" as one modality, but the engines
     * are not interchangeable, so a picker that lists every voice model under
     * whichever engine is selected is offering models that engine cannot load.
     * Each engine answers for itself, by file shape — OmniVoice first because
     * its four-graph layout is the more specific claim.
     */
    fun providerFor(directory: File): SynthProvider? = when {
        omniVoice.looksInstalled(directory) -> SynthProvider.OMNIVOICE
        kokoro.looksInstalled(directory) -> SynthProvider.KOKORO
        else -> null
    }

    val kokoroReady: Boolean get() = kokoro.runtimeAvailable && kokoroDirectory != null

    val omniVoiceReady: Boolean get() = omniVoice.runtimeAvailable && omniVoiceDirectory != null

    /** Whether the installed OmniVoice can copy a voice from a recording. */
    val omniVoiceCloningReady: Boolean
        get() = omniVoiceReady && omniVoiceDirectory?.let { omniVoice.cloningLooksInstalled(it) } == true

    /** Kokoro's voices, marked available only if this build can speak them. */
    fun kokoroVoices(): List<SynthVoice> = KokoroVoices.catalogue(available = kokoroReady)

    /**
     * OmniVoice's single entry.
     *
     * It has no voice *catalogue* — the voice comes from a written description
     * or a reference clip, not from a fixed list — so offering one row is the
     * honest shape rather than inventing names for it.
     */
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

    /**
     * Speak now. Emits progress so the screen can show which utterance is live
     * and offer a stop that actually stops.
     */
    fun speak(request: SpeechRequest): Flow<SpeechEvent> = when (request.provider) {
        SynthProvider.KOKORO -> speakWithKokoro(request)
        SynthProvider.OMNIVOICE -> speakWithNeural(request) { renderOmniVoice(it) }
        SynthProvider.SYSTEM -> speakWithSystem(request)
    }

    /**
     * Kokoro: synthesise the whole passage, then play it.
     *
     * Not streamed. The graph produces a sentence at a time and the seams
     * matter, so playing chunk one while chunk two is still on the CPU would
     * mean a gap whose length depends on the load — audibly worse than a
     * slightly later start. The screen shows the wait for what it is.
     */
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

    /**
     * Render to a file the user can keep or send. This is the "export the audio"
     * half of §7 — a passage you can only hear once is not an artifact.
     */
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
                    // A waveform with no samples is not a file worth claiming to
                    // have saved. WavFile.write happily emits the 44-byte header
                    // on its own, and the screen then reports the name of a file
                    // that plays nothing — which is how a broken engine passed
                    // for a working one.
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

        kokoro.load(
            directory,
            threads = capabilities.inferenceThreads,
            backend = computeDevice.chosen(ai.ondevice.engine.RuntimeRegistry.KOKORO),
        ).onFailure { return Result.failure(it) }

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
                // Kokoro has no volume input, so gain is applied to the
                // waveform. The system engine takes it as a playback parameter
                // instead — same slider, two honest implementations.
                volume = request.volume,
                languageOverride = request.languageCode,
            ),
        )
    }

    /**
     * OmniVoice needs no phonemiser and has no voice packs, so there is far less
     * to refuse over than with Kokoro — either the four graphs are installed or
     * they are not.
     */
    private suspend fun renderOmniVoice(request: SpeechRequest): Result<KokoroAudio> {
        val directory = omniVoiceDirectory ?: return Result.failure(
            IllegalStateException(
                "No OmniVoice model is installed. Models → Add a model, then " +
                    ai.ondevice.core.StarterModels.OMNIVOICE_REPO + ".",
            ),
        )
        omniVoice.load(
            directory,
            threads = capabilities.inferenceThreads,
            backend = computeDevice.chosen(ai.ondevice.engine.RuntimeRegistry.OMNIVOICE),
        ).onFailure { return Result.failure(it) }
        return omniVoice.synthesize(
            OmniVoiceRequest(
                text = request.text,
                speed = request.speed,
                trimSilence = request.trimSilence,
                // These three were dropped here, which quietly reduced OmniVoice
                // to a slower Kokoro: the engine has taken a language, a voice
                // design and a step count all along, and nothing upstream of
                // this call could reach them. Voice design in particular is the
                // model's whole answer to "which speaker" — it has no voice list.
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

    /**
     * Play 24 kHz mono floats, blocking until the audio has actually finished.
     *
     * `AudioTrack.write` returns once the data is queued, not once it is heard,
     * so a naive version reports "done" while the last second is still in the
     * buffer — and [stop] would then cut off audio the user was told had
     * finished. Hence the drain on the playback head.
     */
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
        // `write` can return short — the loop above breaks when it does — and
        // this used to wait for a playback head that had reached
        // `audio.samples.size`, a position it could then never reach. The wait
        // never ended, so `play` never returned, so the flow never completed
        // and Speak sat reading "Stop" for a passage that had already finished.
        //
        // The deadline covers the other direction: the head position only
        // advances while the track really is playing, and an underrun or a
        // silent audio HAL leaves it short of even the honest count. Waiting a
        // little past the audio's own duration is enough for any real playback.
        //
        // `delay`, not Thread.sleep, so pressing Stop actually interrupts this —
        // coroutine cancellation does not interrupt a sleeping thread.
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
    /**
     * System engine only. Kokoro has no pitch input — the graph takes token
     * ids, a style vector and a speed, and nothing else — so a pitch control
     * for it would either do nothing or resample, and resampling is the
     * chipmunk effect the screen promises not to produce.
     */
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
    /**
     * Which language to speak in.
     *
     * The two engines mean different things by it and both are honoured. Kokoro
     * needs an espeak voice, because its front end is a phonemiser and the
     * phonemes differ per language. OmniVoice needs no phonemiser at all — it
     * covers 600+ languages and takes the name straight into
     * `<|lang_start|>…<|lang_end|>` — so for it this is a plain language name
     * rather than a code from a fixed list.
     */
    val languageCode: String? = null,
    /**
     * OmniVoice's voice design: the speaker described in words rather than
     * chosen from a list.
     *
     * Upstream calls this out as a headline feature — "control voices via
     * assigned speaker attributes (gender, age, pitch, dialect/accent, whisper,
     * etc.)" — and it is why OmniVoice ships no voice list to pick from. It goes
     * into `<|instruct_start|>…<|instruct_end|>`, where the literal "None" is
     * what upstream writes when nothing is asked for.
     */
    val voiceDesign: String? = null,
    /**
     * OmniVoice only: how many iterative unmasking passes to run.
     *
     * It is a diffusion language model over a masked grid of audio tokens, so
     * this is the main quality-for-time dial. Null takes the engine's default.
     */
    val steps: Int? = null,
    /** OmniVoice only: grid length in 40 ms frames. Null estimates from the text. */
    val frames: Int? = null,
    /**
     * OmniVoice only: the rest of upstream's generation config.
     *
     * These arrived with the ported unmasking loop. The engine holds the
     * defaults so there is one source for them; null here means "whatever the
     * engine says", rather than a second copy that can drift.
     */
    val guidance: Float? = null,
    val timestepShift: Float? = null,
    val layerPenalty: Float? = null,
    val positionTemperature: Float? = null,
    val classTemperature: Float? = null,
    /** 0 picks a fresh seed per run; anything else makes the run repeatable. */
    val seed: Long = 0L,
    /**
     * OmniVoice only: a recording to copy the voice from, with what it said.
     *
     * Kokoro ignores it because it cannot do anything with it — its voice comes
     * from a fixed pack, and silently accepting a reference it will not use
     * would be the substitution this class exists to avoid.
     */
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

    /**
     * Slower than Kokoro by more than an order of magnitude, and worth it only
     * for what Kokoro cannot do at all — emotion tags, cloning, and languages
     * this build has no phonemiser for. Never selected automatically.
     */
    OMNIVOICE("OmniVoice"),
    SYSTEM("System engine"),
}

sealed interface SpeechEvent {
    data object Started : SpeechEvent
    data class Range(val start: Int, val end: Int) : SpeechEvent
    data object Done : SpeechEvent
    data class Failed(val message: String) : SpeechEvent
}
