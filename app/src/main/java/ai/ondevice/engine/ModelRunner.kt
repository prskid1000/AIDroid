package ai.ondevice.engine

import ai.ondevice.core.Modality
import ai.ondevice.core.SparseParams
import ai.ondevice.core.TranscriptSegment
import ai.ondevice.data.ModelStorage
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.speech.SpeechRequest
import ai.ondevice.speech.SpeechSynthesizer
import ai.ondevice.speech.SynthProvider
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Running a model, with no screen behind it.
 *
 * Everything that touches an engine goes through here: the workflow runner, and
 * the HTTP proxy. It exists because until it did, the orchestration for four of
 * the five modalities lived in view models — `ImageViewModel.generate`,
 * `VoiceViewModel.speak`, `VideoViewModel` — reachable only from a Compose
 * screen. A socket has no screen, and the alternative to extracting this was a
 * second copy of "load the model, apply the parameters, collect the flow, write
 * the file" for every modality, which is four more places for the next
 * parameter to be forgotten.
 *
 * It also owns the one piece of state no engine owns: **which runtime is
 * holding weights**. Each engine guards itself — the diffusion engine unloads
 * inside the lock it loads under, the llama manager warm-swaps — but none of
 * them knows the others exist, because until a workflow crossed runtimes none
 * of them had to. A chat model and a diffusion model resident together is
 * roughly fourteen gigabytes on a phone with fifteen, and that is a kill rather
 * than an error.
 */
@Singleton
class ModelRunner @Inject constructor(
    private val db: OnDeviceDatabase,
    private val engines: EngineManager,
    private val diffusion: DiffusionEngine,
    private val transcriber: Transcriber,
    private val synthesizer: SpeechSynthesizer,
    private val storage: ModelStorage,
) {

    /**
     * One run at a time, across every engine.
     *
     * Not a tunable. The diffusion engine holds a load lock and llama warm-swaps
     * on a mutex, so a second concurrent run cannot execute however many are
     * admitted — and a concurrency setting the engines ignore is a setting that
     * quietly does nothing, which is worse than not offering one.
     */
    private val gate = Mutex()

    /** How often to re-ask for the gate while waiting. */
    private val GATE_POLL_MILLIS = 100L

    /** Which runtime currently holds weights, so we know what to let go of. */
    @Volatile
    var residentRuntime: String? = null
        private set

    /** True while something holds the gate — what the queue and the screen ask. */
    val busy: Boolean get() = gate.isLocked

    /** Reaches the native call of whatever is running. Set only inside [exclusive]. */
    @Volatile
    var activeCancel: (() -> Unit)? = null
        private set

    /**
     * Marks a coroutine as already holding the gate.
     *
     * A context element rather than a thread-local, because the holder and the
     * re-entrant caller are the same *coroutine* and not necessarily the same
     * thread — every engine call here hops to `Dispatchers.Default` or `IO` on
     * the way in.
     */
    private class Held(val runtimeId: String) :
        kotlin.coroutines.AbstractCoroutineContextElement(Held) {
        companion object Key : kotlin.coroutines.CoroutineContext.Key<Held>
    }

    /**
     * Hold the engines for the length of [block], with room made first.
     *
     * The bracket rather than two calls, because the residency has to be
     * recorded on every way out — finished, failed or cancelled — and a run
     * that threw after loading used to leave `resident` naming the runtime
     * before it, so the next step made room for nothing and loaded on top.
     *
     * **Re-entrant, and it has to be.** A media tool is a [ToolProvider] like
     * any other, so a chat turn that calls `generate_image` reaches this from
     * inside a call that is already holding it. A plain mutex is not re-entrant
     * and that is a deadlock: the whole app stops, with a request in flight and
     * nothing to say why. Re-entering makes room if the runtime differs, which
     * is the honest answer on a device that can hold one — and it does mean the
     * outer caller's model is gone when the tool returns, which is why
     * `ChatPipeline` re-loads before every round rather than once at the start.
     */
    suspend fun <T> exclusive(
        runtimeId: String,
        onUnload: (String) -> Unit = {},
        /**
         * How long to wait for the gate before giving up, or null to wait.
         *
         * Bounds the **wait**, never the run. Those are different quantities
         * and conflating them is a bug this had: the proxy wrapped its whole
         * request in the same timeout, so a 120-second budget meant to stop a
         * caller queueing forever also killed every request that legitimately
         * took longer than two minutes. Measured: an image generation refused
         * at 120s while the checkpoint was still loading, having waited for
         * nothing at all — the engine was free the entire time.
         */
        waitMillis: Long? = null,
        block: suspend () -> T,
    ): T {
        val held = kotlinx.coroutines.currentCoroutineContext()[Held]
        if (held != null) {
            if (held.runtimeId != runtimeId) makeRoomFor(runtimeId, onUnload)
            return try {
                kotlinx.coroutines.withContext(Held(runtimeId)) { block() }
            } finally {
                residentRuntime = runtimeId
            }
        }

        acquire(waitMillis)
        try {
            makeRoomFor(runtimeId, onUnload)
            return kotlinx.coroutines.withContext(Held(runtimeId)) { block() }
        } finally {
            residentRuntime = runtimeId
            activeCancel = null
            gate.unlock()
        }
    }

    /**
     * Take the gate, waiting at most [waitMillis].
     *
     * Polled rather than `withTimeoutOrNull { gate.lock() }`, because that
     * races: the timeout can fire after `lock()` has succeeded and before the
     * result is returned, and the lock is then held by nobody and released by
     * nothing. `tryLock` cannot be interrupted between deciding and taking.
     */
    private suspend fun acquire(waitMillis: Long?) {
        if (waitMillis == null) {
            gate.lock()
            return
        }
        val deadline = System.currentTimeMillis() + waitMillis
        while (!gate.tryLock()) {
            if (System.currentTimeMillis() >= deadline) {
                throw EngineBusy(waitMillis / 1000)
            }
            kotlinx.coroutines.delay(GATE_POLL_MILLIS)
        }
    }

    /**
     * Let the other engine go before this one loads.
     *
     * The line that exists nowhere else in this app, and the reason a run can
     * change runtime at all.
     */
    private suspend fun makeRoomFor(runtimeId: String, onUnload: (String) -> Unit) {
        val holding = residentRuntime
        if (holding == null || holding == runtimeId) return

        val because = "$runtimeId needs the memory $holding is holding"
        onUnload(because)
        when (holding) {
            RuntimeRegistry.STABLE_DIFFUSION -> diffusion.unload(because)
            RuntimeRegistry.LLAMA -> engines.unload()
            RuntimeRegistry.WHISPER -> transcriber.unload()
            RuntimeRegistry.KOKORO, RuntimeRegistry.OMNIVOICE -> {
                synthesizer.unload(SynthProvider.KOKORO)
                synthesizer.unload(SynthProvider.OMNIVOICE)
            }
        }
        residentRuntime = null
    }

    /** Drop everything. Used when a request policy says to evict rather than queue. */
    suspend fun evict(because: String) = gate.withLock {
        when (residentRuntime) {
            RuntimeRegistry.STABLE_DIFFUSION -> diffusion.unload(because)
            RuntimeRegistry.LLAMA -> engines.unload()
            RuntimeRegistry.WHISPER -> transcriber.unload()
            RuntimeRegistry.KOKORO, RuntimeRegistry.OMNIVOICE -> {
                synthesizer.unload(SynthProvider.KOKORO)
                synthesizer.unload(SynthProvider.OMNIVOICE)
            }
            else -> Unit
        }
        residentRuntime = null
    }

    /** Which engine runs a row — from its own modality, never from its name. */
    fun runtimeFor(model: ModelEntity): String = when (model.modality) {
        Modality.DIFFUSION -> RuntimeRegistry.STABLE_DIFFUSION
        Modality.SPEECH_TO_TEXT -> RuntimeRegistry.WHISPER
        Modality.TEXT_TO_SPEECH -> RuntimeRegistry.KOKORO
        else -> RuntimeRegistry.LLAMA
    }

    /** The model's own stored overrides, with the caller's layered on top. */
    fun paramsFor(model: ModelEntity, overrides: SparseParams): SparseParams =
        SparseParams.parse(model.paramOverridesJson).overlaidWith(overrides)

    // ── text ────────────────────────────────────────────────────────────

    /**
     * Load a text model and hand back the engine, ready to generate.
     *
     * Two steps rather than one call that returns a Flow, because the caller
     * needs the engine itself: the proxy's tool loop runs several generations
     * against one loaded model, and reloading between rounds would throw away
     * the prompt cache that makes the second round cheap.
     */
    suspend fun loadText(model: ModelEntity, params: SparseParams): InferenceEngine {
        engines.load(model, SparseParams.parse(model.paramOverridesJson)).getOrThrow()
        val engine = engines.llama
            ?: throw IllegalStateException("The llama.cpp runtime is not installed in this build.")
        engine.applyParams(params)
        activeCancel = { (engine as? LlamaEngine)?.cancel() }
        return engine
    }

    fun text(engine: InferenceEngine, request: GenerateRequest): Flow<GenerationEvent> =
        engine.generate(request)

    // ── diffusion ───────────────────────────────────────────────────────

    suspend fun loadDiffusion(model: ModelEntity, params: SparseParams) {
        diffusion.load(
            model.id,
            model.localPath,
            emptyList(),
            params = SparseParams.parse(model.paramOverridesJson),
        ).getOrThrow()
        diffusion.applyParams(params)
        activeCancel = { diffusion.cancel() }
    }

    /**
     * One picture, written to [destination].
     *
     * A path out rather than pixels, because that is the convention every value
     * on a workflow edge already follows and because a 1024-square image is
     * four megabytes of `IntArray` that nothing downstream wants in memory.
     */
    fun image(
        request: DiffusionRequest,
        destination: File,
        onProgress: (DiffusionEvent.Progress) -> Unit = {},
    ): Flow<DiffusionOutcome> = kotlinx.coroutines.flow.flow {
        var written: String? = null
        diffusion.generate(request).collect { event ->
            when (event) {
                is DiffusionEvent.Progress -> {
                    onProgress(event)
                    emit(DiffusionOutcome.Progress(event))
                }
                is DiffusionEvent.Preview -> emit(DiffusionOutcome.Preview(event.image))
                is DiffusionEvent.Completed -> {
                    destination.parentFile?.mkdirs()
                    // The parameters go into a tEXt chunk, so the file alone is
                    // reproducible without this app's database — SPEC 5.4, and
                    // the reason a picture pulled off the device over HTTP is
                    // still worth something a month later.
                    destination.writeBytes(
                        event.image.toPng(request.params.toJsonString()),
                    )
                    written = destination.absolutePath
                }
                is DiffusionEvent.Failed ->
                    throw ModelRunFailure(event.message, event.suggestion)
                else -> Unit
            }
        }
        emit(
            DiffusionOutcome.Image(
                written ?: throw ModelRunFailure(
                    "The run produced no picture.",
                    "Check the model's parameters — a step count of zero produces nothing.",
                ),
            ),
        )
    }

    /** One clip, as the frames it is made of. */
    fun clip(
        request: VideoRequest,
        onProgress: (DiffusionEvent.Progress) -> Unit = {},
    ): Flow<DiffusionOutcome> = kotlinx.coroutines.flow.flow {
        var clip: DiffusionClip? = null
        diffusion.generateVideo(request).collect { event ->
            when (event) {
                is DiffusionEvent.Progress -> {
                    onProgress(event)
                    emit(DiffusionOutcome.Progress(event))
                }
                is DiffusionEvent.ClipCompleted -> clip = event.clip
                is DiffusionEvent.Failed ->
                    throw ModelRunFailure(event.message, event.suggestion)
                else -> Unit
            }
        }
        emit(
            DiffusionOutcome.Clip(
                clip ?: throw ModelRunFailure(
                    "The run produced no clip.",
                    "Check `video_frames` — a frame count of zero produces nothing.",
                ),
            ),
        )
    }

    /**
     * Enlarge a picture with an attached ESRGAN graph.
     *
     * The upscaler is its own context and shares nothing with the denoiser, so
     * the engine drops the denoiser first — holding both is what the kernel
     * kills the process for, with no exception and nothing in the crash buffer.
     */
    suspend fun upscale(image: DiffusionImage, esrganPath: String, factor: Int): DiffusionImage =
        diffusion.upscale(image, esrganPath, factor).getOrThrow()

    // ── speech ──────────────────────────────────────────────────────────

    suspend fun transcribe(
        model: ModelEntity,
        params: SparseParams,
        audio: File,
    ): List<TranscriptSegment> {
        if (!transcriber.isCurrent(model.id)) {
            transcriber.load(model.id, model.localPath, params).getOrThrow()
        } else {
            transcriber.applyParams(params)
        }
        activeCancel = { transcriber.cancel() }
        return transcriber.transcribeFile(audio).getOrThrow()
    }

    /**
     * Speak, to a file.
     *
     * The provider is asked of the folder rather than assumed, which is the same
     * structural test the Voice tab uses and needs no model names: an explicit
     * `provider` parameter wins, and otherwise what is actually installed beside
     * the weights decides. This defaulted to Kokoro once, so pointing a step at
     * OmniVoice ran Kokoro instead — quietly, since both produce a WAV.
     */
    suspend fun speak(
        model: ModelEntity,
        params: SparseParams,
        text: String,
        destination: File,
    ): File {
        val directory = File(model.localPath).let { if (it.isDirectory) it else it.parentFile }
        synthesizer.useKokoroModel(directory)
        synthesizer.useOmniVoiceModel(directory)

        val provider = when {
            params.string("provider").equals(SynthProvider.OMNIVOICE.name, ignoreCase = true) ->
                SynthProvider.OMNIVOICE
            params.string("provider").equals(SynthProvider.KOKORO.name, ignoreCase = true) ->
                SynthProvider.KOKORO
            directory != null && synthesizer.omniVoiceLooksInstalled(directory) ->
                SynthProvider.OMNIVOICE
            else -> SynthProvider.KOKORO
        }

        destination.parentFile?.mkdirs()
        return synthesizer.synthesizeToFile(
            SpeechRequest(
                text = text,
                voiceId = params.string("voice")?.takeIf { it.isNotBlank() }
                    ?: defaultVoice(provider),
                speed = params.float("speed") ?: 1.0f,
                provider = provider,
                voiceDesign = params.string("voice_design"),
                languageCode = params.string("lang_code"),
                seed = params.int("seed")?.toLong() ?: 0L,
                steps = params.int("steps"),
                trimSilence = params.bool("trim_silence") ?: true,
                volume = params.float("volume") ?: 1.0f,
            ),
            destination,
        ).getOrThrow()
    }

    /**
     * A voice, when the caller named none.
     *
     * Kokoro refuses outright without one — "No Kokoro voice was selected" —
     * and a request over HTTP has no screen to have picked one on. The Voice
     * tab chooses the first available entry when it opens; this is the same
     * choice made in the same way, by asking the synthesiser what it has rather
     * than by naming a voice here. A voice id in this file would be exactly the
     * hardcoding SPEC 1.3 rules out, and would break the moment somebody
     * installed a Kokoro pack with a different set.
     *
     * Null when nothing is available, so the engine's own refusal still reaches
     * the caller rather than being replaced by a guess that fails later.
     */
    private fun defaultVoice(provider: SynthProvider): String? = when (provider) {
        SynthProvider.OMNIVOICE -> synthesizer.omniVoiceVoices().firstOrNull { it.available }?.id
        else -> synthesizer.kokoroVoices().firstOrNull { it.available }?.id
    }

    // ── model lookup ────────────────────────────────────────────────────

    suspend fun installed(): List<ModelEntity> = db.models().getInstalled()

    suspend fun model(id: String): ModelEntity? = db.models().get(id)

    /**
     * Whether this row is a part rather than a thing that runs.
     *
     * Asked of the row's own `attachmentRole`, which the resolver sets from the
     * file — SPEC 1.3, and the reason a checkpoint released tomorrow needs no
     * app update to be told apart from an encoder.
     */
    fun isComponent(model: ModelEntity): Boolean = model.attachmentRole != null

    /**
     * The model to use for a modality when the request named none.
     *
     * Most recently used, which is what `getInstalledByModality` already orders
     * by — the same rule the tabs use when they open, so the answer a client
     * gets with no `model` field is the one the person was last working with
     * rather than an arbitrary row.
     *
     * **Components are not models.** A VAE, a text encoder and a LoRA are all
     * stored as rows of the same modality as the checkpoint they belong to, so
     * without this filter the most recently touched one wins — and a request
     * for a picture picked `flux2-vae.safetensors`, on which sd.cpp answered
     * "get sd version from file failed": true of the file, and useless about
     * the mistake. The Image tab has always filtered these out; this had not.
     */
    suspend fun defaultFor(modality: Modality): ModelEntity? =
        db.models().getInstalledByModality(modality)
            .firstOrNull { it.attachmentRole == null }

    fun scratchDir(name: String): File =
        File(storage.root(), "proxy/$name").apply { mkdirs() }

    /**
     * Mark a model as just used.
     *
     * The row's `lastUsedAt` is what orders every default pick in the app, so a
     * model reached over HTTP has to move it exactly as one reached from a tab
     * does — otherwise the phone's own screens keep offering whatever was used
     * before the server started answering.
     */
    suspend fun touch(modelId: String) {
        runCatching { db.models().touch(modelId, System.currentTimeMillis()) }
    }
}

/** The gate was held by something else for longer than the caller would wait. */
class EngineBusy(val waitedSeconds: Long) : Exception(
    "Gave up after ${waitedSeconds}s waiting for this device's engine.",
)

/** What a run produced, as it produced it. */
sealed interface DiffusionOutcome {
    data class Progress(val event: DiffusionEvent.Progress) : DiffusionOutcome
    data class Preview(val image: DiffusionImage) : DiffusionOutcome
    data class Image(val path: String) : DiffusionOutcome
    data class Clip(val clip: DiffusionClip) : DiffusionOutcome
}

/**
 * A run that failed with something worth repeating to the caller.
 *
 * Carries the suggestion as well as the message, because the engines already
 * produce both — "not enough memory … lower the context size, pick a smaller
 * quant, or set cache_type_k and cache_type_v to q8_0" — and dropping the
 * second half at a protocol boundary is exactly the silent failure SPEC 1.2
 * forbids.
 */
class ModelRunFailure(
    override val message: String,
    val suggestion: String?,
) : Exception(message)
