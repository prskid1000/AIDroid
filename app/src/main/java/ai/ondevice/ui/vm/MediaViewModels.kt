package ai.ondevice.ui.vm

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.ondevice.core.Modality
import ai.ondevice.core.RuntimeState
import ai.ondevice.core.SparseParams
import ai.ondevice.core.TranscriptExport
import ai.ondevice.core.TranscriptFormat
import ai.ondevice.core.TranscriptSegment
import ai.ondevice.data.ModelStorage
import ai.ondevice.data.db.GeneratedImageEntity
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.TranscriptEntity
import ai.ondevice.data.hf.DeviceCapabilities
import ai.ondevice.engine.CaptureEvent
import ai.ondevice.engine.record
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject
import kotlin.random.Random

/** S11/S12/S13 — image generation, the mask editor and the gallery. */
@HiltViewModel
class ImageViewModel @Inject constructor(
    private val db: OnDeviceDatabase,
    private val storage: ModelStorage,
    private val capabilities: DeviceCapabilities,
    private val diffusion: ai.ondevice.engine.DiffusionEngine,
    private val recorder: ai.ondevice.engine.ResourceRecorder,
) : ViewModel() {

    private val _state = MutableStateFlow(ImageState())
    val state: StateFlow<ImageState> = _state.asStateFlow()

    private var generationJob: Job? = null

    init {
        viewModelScope.launch {
            // "No runtime" and "no model" are different problems with different
            // fixes, and SPEC §1.2 says a refusal has to name which one it is.
            val runtimeInstalled = db.runtimes().get(RUNTIME_ID)?.state != RuntimeState.NOT_INSTALLED
            val model = baseModelsOnly(
                db.models().observeInstalledByModality(Modality.DIFFUSION).first(),
            ).firstOrNull()
            _state.value = _state.value.copy(
                model = model,
                runtimeInstalled = runtimeInstalled,
            )
            refreshAttachmentLibrary()
        }

        // Live, so a model that finishes downloading while this screen is open
        // appears — the same mistake the chat picker used to make.
        viewModelScope.launch {
            db.models().observeInstalledByModality(Modality.DIFFUSION).collect { all ->
                val models = baseModelsOnly(all)
                _state.value = _state.value.copy(
                    availableModels = models,
                    model = _state.value.model?.let { current ->
                        models.firstOrNull { it.id == current.id }
                    } ?: models.firstOrNull(),
                )
            }
        }
    }

    /** The diffusion entries that can be the *base* model, which is not the same set as the diffusion entries. */
    private fun baseModelsOnly(models: List<ModelEntity>): List<ModelEntity> =
        models.filter { it.attachmentRole == null }

    /** With more than one diffusion model installed, which one runs is the user's choice — not whichever the database happened to return first. */
    fun selectModel(model: ModelEntity) {
        if (_state.value.model?.id == model.id) return
        // The loaded context belongs to the old model; keep them in step.
        diffusion.unload()
        _state.value = _state.value.copy(model = model, error = null, errorHint = null, previewBitmap = null)
        viewModelScope.launch { db.models().touch(model.id, System.currentTimeMillis()) }
    }

    fun setMode(mode: ImageMode) {
        _state.value = _state.value.copy(mode = mode)
    }

    fun setPrompt(value: String) = update { copy(prompt = value) }
    fun setNegativePrompt(value: String) = update { copy(negativePrompt = value) }
    fun setSteps(value: Int) = update { copy(steps = value) }
    fun setCfg(value: Float) = update { copy(cfgScale = value) }
    fun setSize(value: Int) = update { copy(width = value, height = value) }
    fun setSeed(value: Long) = update { copy(seed = value) }
    fun setStrength(value: Float) = update { copy(strength = value) }
    fun setVaeTiling(value: Boolean) = update { copy(vaeTiling = value) }

    fun setSourceImage(uri: String?) = update { copy(sourceImageUri = uri) }

    fun setControlImage(uri: String?) = update { copy(controlImageUri = uri) }

    fun setControlStrength(value: Float) = update { copy(controlStrength = value) }

    /** Outpaint margins, in pixels, per edge. */
    fun setExtend(left: Int? = null, top: Int? = null, right: Int? = null, bottom: Int? = null) = update {
        copy(
            extendLeft = left ?: extendLeft,
            extendTop = top ?: extendTop,
            extendRight = right ?: extendRight,
            extendBottom = bottom ?: extendBottom,
        )
    }

    fun setExtendAll(pixels: Int) = update {
        copy(extendLeft = pixels, extendTop = pixels, extendRight = pixels, extendBottom = pixels)
    }

    /** Pull back anything the Advanced screen changed. */
    fun refreshFromOverrides() {
        viewModelScope.launch {
            val model = db.models().observeInstalledByModality(Modality.DIFFUSION).first().firstOrNull()
            val runtimeInstalled = db.runtimes().get(RUNTIME_ID)?.state != RuntimeState.NOT_INSTALLED
            val p = SparseParams.parse(model?.paramOverridesJson)
            _state.value = _state.value.copy(
                model = model,
                runtimeInstalled = runtimeInstalled,
                steps = p.int("steps") ?: _state.value.steps,
                cfgScale = p.float("cfg_scale") ?: _state.value.cfgScale,
                width = p.int("width") ?: _state.value.width,
                height = p.int("height") ?: _state.value.height,
                strength = p.float("strength") ?: _state.value.strength,
                samplingMethod = p.string("sampling_method") ?: _state.value.samplingMethod,
                schedule = p.string("schedule") ?: _state.value.schedule,
                clipSkip = p.int("clip_skip") ?: _state.value.clipSkip,
                vaeTiling = p.bool("vae_tiling") ?: _state.value.vaeTiling,
            )
            checkEnvelope()
        }
    }

    private fun update(block: ImageState.() -> ImageState) {
        _state.value = _state.value.block()
        checkEnvelope()
    }

    /** SPEC §5.4 — warn when width × height × batch exceeds a measured-safe envelope, and suggest `vae_tiling` rather than letting it OOM. */
    private fun checkEnvelope() {
        val s = _state.value
        val pixels = s.width.toLong() * s.height * s.batchCount
        val exceeded = pixels > SAFE_PIXEL_ENVELOPE && !s.vaeTiling
        _state.value = s.copy(exceedsEnvelope = exceeded)
    }

    fun generate() {
        val model = _state.value.model ?: return
        val seed = if (_state.value.seed < 0) Random.nextLong(0, Int.MAX_VALUE.toLong()) else _state.value.seed
        _state.value = _state.value.copy(
            generating = true,
            step = 0,
            usedSeed = seed,
            error = null,
            previewBitmap = null,
        )

        generationJob = viewModelScope.launch {
            val started = System.currentTimeMillis()
            val recording = recorder.start(viewModelScope)
            val liveJob = viewModelScope.launch {
                recording.live.collect { trace ->
                    _state.value = _state.value.copy(liveTrace = trace)
                }
            }
            db.models().touch(model.id, started)
            try {
                if (!diffusion.isCurrent(model.id)) {
                    _state.value = _state.value.copy(loadingModel = true)
                    val loaded = diffusion.load(
                        model.id,
                        model.localPath,
                        _state.value.attachments,
                        params = SparseParams.parse(model.paramOverridesJson),
                    )
                    _state.value = _state.value.copy(loadingModel = false)
                    if (loaded.isFailure) {
                        _state.value = _state.value.copy(
                            generating = false,
                            error = loaded.exceptionOrNull()?.message ?: "The diffusion model could not be loaded.",
                            errorHint = "Some GGUF converters emit tensor names longer than ggml's 64-character " +
                                "limit. Try another quantisation of the same model, or a repo published for " +
                                "stable-diffusion.cpp.",
                        )
                        return@launch
                    }
                }

                val params = currentParams(seed)
                diffusion.generate(
                    ai.ondevice.engine.DiffusionRequest(
                        params = params,
                        initImageUri = _state.value.sourceImageUri,
                        controlImageUri = _state.value.controlImageUri,
                        maskPngPath = _state.value.maskPath,
                        attachments = _state.value.attachments,
                    ),
                ).collect { event ->
                    when (event) {
                        is ai.ondevice.engine.DiffusionEvent.Progress -> {
                            // `steps` is the user's setting and must never be written from here.
                            val remaining = (event.steps - event.step).coerceAtLeast(0)
                            _state.value = _state.value.copy(
                                step = event.step,
                                progressSteps = event.steps,
                                phase = event.phase,
                                secondsPerStep = event.secondsPerStep,
                                etaSeconds = if (event.secondsPerStep > 0f) {
                                    (remaining * event.secondsPerStep).toLong()
                                } else {
                                    0L
                                },
                            )
                        }
                        is ai.ondevice.engine.DiffusionEvent.Preview -> {
                            // The actual denoising state, decoded by TAESD —
                            // §5.4's "intermediate latents, not a spinner".
                            _state.value = _state.value.copy(previewBitmap = event.image.toBitmap())
                        }
                        is ai.ondevice.engine.DiffusionEvent.Completed -> {
                            val file = withContext(Dispatchers.IO) {
                                val target = java.io.File(storage.galleryDir(), "$seed.png")
                                target.writeBytes(event.image.toPng(params.toJsonString()))
                                target
                            }
                            val image = GeneratedImageEntity(
                                id = UUID.randomUUID().toString(),
                                path = file.absolutePath,
                                prompt = _state.value.prompt,
                                negativePrompt = _state.value.negativePrompt.takeIf { it.isNotBlank() },
                                paramsJson = params.toJsonString(),
                                modelId = model.id,
                                seed = seed,
                                width = event.image.width,
                                height = event.image.height,
                                createdAt = System.currentTimeMillis(),
                            )
                            db.images().upsert(image)
                            val elapsed = System.currentTimeMillis() - started
                            val trace = recording.stop()
                            db.predictionRuns().record(
                                kind = ai.ondevice.core.PredictionKind.IMAGE,
                                artifactId = image.id,
                                modelId = model.id,
                                startedAt = started,
                                trace = trace,
                                stats = SparseParams.of(
                                    "steps" to _state.value.steps,
                                    "seconds_per_step" to _state.value.secondsPerStep,
                                ),
                            )
                            _state.value = _state.value.copy(
                                lastImage = image,
                                previewBitmap = event.image.toBitmap(),
                                elapsedMillis = elapsed,
                                lastTrace = trace,
                            )
                        }
                        is ai.ondevice.engine.DiffusionEvent.Failed -> {
                            _state.value = _state.value.copy(
                                error = event.message,
                                errorHint = event.suggestion,
                            )
                        }
                    }
                }
            } finally {
                // Cancellation must reach the native loop, not merely stop the
                // flow — otherwise sd.cpp keeps denoising and keeps its buffers.
                diffusion.cancel()
                liveJob.cancel()
                // Idempotent: a completed run already stopped it, and a cancelled one never reached that point.
                recording.stop()
                _state.value = _state.value.copy(
                    generating = false,
                    step = 0,
                    loadingModel = false,
                    liveTrace = null,
                )
            }
        }
    }

    fun cancel() {
        diffusion.cancel()
        generationJob?.cancel()
        generationJob = null
    }

    /** Clear the composition and start again. */
    fun reset() {
        cancel()
        update {
            copy(
                prompt = "",
                negativePrompt = "",
                sourceImageUri = null,
                controlImageUri = null,
                maskPath = null,
                lastImage = null,
                usedSeed = null,
                error = null,
                errorHint = null,
                step = 0,
                progressSteps = 0,
                previewBitmap = null,
            )
        }
    }

    // — attachments (SPEC §5, generically) —

    /** Everything installed that can hang off a diffusion run. */
    private suspend fun refreshAttachmentLibrary() {
        val installed = db.models().getInstalled()
        val available = installed.mapNotNull { entity ->
            val role = entity.attachmentRole ?: return@mapNotNull null
            ai.ondevice.core.ModelAttachment(
                modelId = entity.id,
                role = role,
                path = entity.localPath,
                displayName = entity.displayName,
                enabled = false,
            )
        }
        _state.value = _state.value.copy(availableAttachments = available)
    }

    fun toggleAttachment(modelId: String) {
        val updated = _state.value.availableAttachments.map {
            if (it.modelId == modelId) it.copy(enabled = !it.enabled) else it
        }
        _state.value = _state.value.copy(availableAttachments = updated)
    }

    fun setAttachmentWeight(modelId: String, weight: Float) {
        val updated = _state.value.availableAttachments.map {
            if (it.modelId == modelId) it.copy(weight = weight) else it
        }
        _state.value = _state.value.copy(availableAttachments = updated)
    }

    fun setMaskPath(path: String?) = update { copy(maskPath = path) }

    /** Upscale the picture just produced. */
    fun upscale(factor: Int = 0) {
        val source = _state.value.lastImage ?: return
        val model = _state.value.model
        val esrgan = _state.value.availableAttachments
            .firstOrNull { it.role == ai.ondevice.core.AttachmentRole.UPSCALER }?.path
            ?: SparseParams.parse(model?.paramOverridesJson).string("upscale_model")
        if (esrgan.isNullOrBlank()) {
            _state.value = _state.value.copy(
                error = "No upscaler is installed.",
                errorHint = "Add an ESRGAN model — its filename gives away the role, so it appears " +
                    "under Attachments once downloaded.",
            )
            return
        }

        generationJob = viewModelScope.launch {
            _state.value = _state.value.copy(generating = true, error = null, errorHint = null)
            try {
                val decoded = withContext(Dispatchers.IO) {
                    android.graphics.BitmapFactory.decodeFile(source.path)
                } ?: error("The image file could not be read.")
                val pixels = IntArray(decoded.width * decoded.height)
                decoded.getPixels(pixels, 0, decoded.width, 0, 0, decoded.width, decoded.height)

                val result = diffusion.upscale(
                    image = ai.ondevice.engine.DiffusionImage(decoded.width, decoded.height, pixels),
                    esrganPath = esrgan,
                    factor = factor,
                )
                result.fold(
                    onSuccess = { bigger ->
                        val entity = withContext(Dispatchers.IO) {
                            val file = java.io.File(
                                storage.galleryDir(),
                                "${source.seed}-x${bigger.width / decoded.width}.png",
                            )
                            file.writeBytes(bigger.toPng(source.paramsJson))
                            GeneratedImageEntity(
                                id = UUID.randomUUID().toString(),
                                path = file.absolutePath,
                                prompt = source.prompt,
                                negativePrompt = source.negativePrompt,
                                paramsJson = source.paramsJson,
                                modelId = source.modelId,
                                seed = source.seed,
                                width = bigger.width,
                                height = bigger.height,
                                createdAt = System.currentTimeMillis(),
                            )
                        }
                        db.images().upsert(entity)
                        _state.value = _state.value.copy(
                            lastImage = entity,
                            previewBitmap = bigger.toBitmap(),
                        )
                    },
                    onFailure = {
                        _state.value = _state.value.copy(
                            error = it.message ?: "Upscaling failed.",
                            errorHint = "Lower the factor, or try a smaller source image.",
                        )
                    },
                )
            } catch (cancelled: kotlinx.coroutines.CancellationException) {
                // Cancelling is not failing.
                throw cancelled
            } catch (failure: Throwable) {
                _state.value = _state.value.copy(error = failure.message ?: "Upscaling failed.")
            } finally {
                _state.value = _state.value.copy(generating = false)
            }
        }
    }

    private fun currentParams(seed: Long): SparseParams {
        val s = _state.value
        return SparseParams.of(
            "prompt" to s.prompt,
            "negative_prompt" to s.negativePrompt,
            "steps" to s.steps,
            "cfg_scale" to s.cfgScale,
            "width" to s.width,
            "height" to s.height,
            "seed" to seed,
            "sampling_method" to s.samplingMethod,
            "schedule" to s.schedule,
            "clip_skip" to s.clipSkip,
            "vae_tiling" to s.vaeTiling,
            "mode" to s.mode.name.lowercase(),
            // Strength only means something when there is a source to denoise,
            // and that is now a property of the picture rather than the mode.
            "strength" to s.strength.takeIf { s.sourceImageUri != null },
            "init_img" to s.sourceImageUri,
            "control_image" to s.controlImageUri,
            "control_strength" to s.controlStrength.takeIf { s.controlImageUri != null },
            "extend" to listOf(s.extendLeft, s.extendTop, s.extendRight, s.extendBottom)
                .takeIf { s.mode == ImageMode.OUTPAINT && it.any { px -> px > 0 } },
        )
    }

    /** SPEC §5.4 — "reuse parameters" repopulates the whole form. */
    fun reuseParameters(image: GeneratedImageEntity) {
        val p = SparseParams.parse(image.paramsJson)
        _state.value = _state.value.copy(
            prompt = p.string("prompt") ?: image.prompt,
            negativePrompt = p.string("negative_prompt") ?: image.negativePrompt.orEmpty(),
            steps = p.int("steps") ?: _state.value.steps,
            cfgScale = p.float("cfg_scale") ?: _state.value.cfgScale,
            width = p.int("width") ?: image.width,
            height = p.int("height") ?: image.height,
            seed = image.seed,
            samplingMethod = p.string("sampling_method") ?: _state.value.samplingMethod,
            schedule = p.string("schedule") ?: _state.value.schedule,
            clipSkip = p.int("clip_skip") ?: _state.value.clipSkip,
            vaeTiling = p.bool("vae_tiling") ?: _state.value.vaeTiling,
        )
    }

    private companion object {
        const val STEP_MILLIS = 3100L // the canvas' 3.1 s/it on CPU
        const val SAFE_PIXEL_ENVELOPE = 768L * 768L
        const val RUNTIME_ID = "stable-diffusion.cpp"
    }
}

/** What the Image screen's primary action should say and do right now. */
enum class ImageAction { INSTALL_RUNTIME, ADD_MODEL, PICK_SOURCE, GENERATE, CANCEL }

enum class ImageMode(val label: String) {
    /** Prompt in, picture out — with an *optional* source image. */
    GENERATE("Generate"),
    INPAINT("Inpaint"),

    OUTPAINT("Extend"),
}

data class ImageState(
    val mode: ImageMode = ImageMode.GENERATE,
    val model: ModelEntity? = null,
    val prompt: String = "low-key studio portrait of a lynx, black backdrop, rim light <lora:filmgrain:0.6>",
    val negativePrompt: String = "blurry, oversaturated",
    val steps: Int = 28,
    val cfgScale: Float = 7.0f,
    val width: Int = 512,
    val height: Int = 512,
    val batchCount: Int = 1,
    val seed: Long = 812934177,
    val usedSeed: Long? = null,
    val strength: Float = 0.75f,
    /** The one init image. */
    val sourceImageUri: String? = null,
    /** ControlNet's structural reference: a pose, depth or edge map that steers composition without contributing pixels. */
    val controlImageUri: String? = null,
    val controlStrength: Float = 0.9f,
    /** Outpaint margins in pixels, per edge. */
    val extendLeft: Int = 0,
    val extendTop: Int = 0,
    val extendRight: Int = 0,
    val extendBottom: Int = 0,
    val samplingMethod: String = "dpm++2m",
    val schedule: String = "karras",
    val clipSkip: Int = 2,
    val vaeTiling: Boolean = true,
    val generating: Boolean = false,
    val step: Int = 0,
    /** What the engine reports as its total — *not* the user's step setting. */
    val progressSteps: Int = 0,
    val phase: ai.ondevice.engine.DiffusionPhase = ai.ondevice.engine.DiffusionPhase.PREPARING,
    val secondsPerStep: Float = 0f,
    val etaSeconds: Long = 0,
    val exceedsEnvelope: Boolean = false,
    val lastImage: GeneratedImageEntity? = null,
    val runtimeInstalled: Boolean = false,
    val loadingModel: Boolean = false,
    val error: String? = null,
    val errorHint: String? = null,
    val elapsedMillis: Long = 0,
    /** The decoded intermediate latent, or the finished image. */
    val previewBitmap: android.graphics.Bitmap? = null,
    val maskPath: String? = null,
    /** Sampled while generating; null once the run ends. */
    val liveTrace: ai.ondevice.engine.ResourceTrace? = null,
    /** What the finished run cost, kept so the graph survives the generation. */
    val lastTrace: ai.ondevice.engine.ResourceTrace? = null,
    val availableAttachments: List<ai.ondevice.core.ModelAttachment> = emptyList(),
    val availableModels: List<ModelEntity> = emptyList(),
) {
    /** Only the ones actually ticked go to the runtime. */
    val attachments: List<ai.ondevice.core.ModelAttachment>
        get() = availableAttachments.filter { it.enabled }

    /** Combinations that will not work, said before Generate rather than after. */
    val missingComponents: List<ai.ondevice.core.MissingComponent>
        get() = ai.ondevice.core.ComponentCheck.forDiffusion(availableAttachments)
    val progress: Float
        get() = if (progressSteps > 0) (step.toFloat() / progressSteps).coerceIn(0f, 1f) else 0f
    /** The denoise dial appears when, and only when, there is a source. */
    val showStrength: Boolean get() = sourceImageUri != null

    /** Inpaint and Extend cannot proceed without one; Generate can. */
    val requiresSource: Boolean get() = mode != ImageMode.GENERATE

    val outputWidth: Int get() = width + extendLeft + extendRight
    val outputHeight: Int get() = height + extendTop + extendBottom

    val action: ImageAction
        get() = when {
            generating -> ImageAction.CANCEL
            !runtimeInstalled -> ImageAction.INSTALL_RUNTIME
            model == null -> ImageAction.ADD_MODEL
            requiresSource && sourceImageUri == null -> ImageAction.PICK_SOURCE
            else -> ImageAction.GENERATE
        }

    val actionLabel: String
        get() = when (action) {
            ImageAction.CANCEL -> "Cancel — frees native memory"
            ImageAction.INSTALL_RUNTIME -> "Install stable-diffusion.cpp first"
            ImageAction.ADD_MODEL -> "Add a diffusion model"
            ImageAction.PICK_SOURCE -> "Choose a source image"
            ImageAction.GENERATE -> "Generate"
        }

    /** The runtime is installed but there is nothing for it to load. */
    val actionHint: String?
        get() = when (action) {
            ImageAction.INSTALL_RUNTIME ->
                "Diffusion is optional and ships separately. Settings → Runtimes installs it."
            ImageAction.ADD_MODEL ->
                "stable-diffusion.cpp is installed, but no diffusion model is. Paste an SD or " +
                    "SDXL repo on the Add model screen."
            ImageAction.PICK_SOURCE ->
                "${mode.label} starts from an image. Denoise strength has nothing to act on until " +
                    "you pick one."
            else -> null
        }
}

// GalleryViewModel lived here.

/** S14 — live and file transcription, plus the Kokoro read-aloud panel. */
@HiltViewModel
class VoiceViewModel @Inject constructor(
    private val db: OnDeviceDatabase,
    private val storage: ModelStorage,
    private val synthesizer: ai.ondevice.speech.SpeechSynthesizer,
    private val attachments: ai.ondevice.data.AttachmentStore,
    private val transcriber: ai.ondevice.engine.Transcriber,
    private val recorder: ai.ondevice.engine.ResourceRecorder,
) : ViewModel() {

    private val _state = MutableStateFlow(VoiceState())
    val state: StateFlow<VoiceState> = _state.asStateFlow()

    private var recordingJob: Job? = null
    private var speakJob: Job? = null

    /** Where the take in progress is being written; null when not recording. */
    private var captureFile: java.io.File? = null

    private companion object {
        const val TAG = "VoiceViewModel"
    }

    init {
        viewModelScope.launch {
            // Read first, write second — and never `copy(x = <a suspending call>)`.
            val transcripts = db.transcripts().observeAll().first()
            _state.value = _state.value.copy(transcripts = transcripts)
        }

        // Both model lists are collected, not sampled once with `first()`.
        viewModelScope.launch {
            db.models().observeInstalledByModality(Modality.SPEECH_TO_TEXT)
                .catch { failure ->
                    android.util.Log.e(TAG, "speech model list failed", failure)
                }
                .collect { models ->
                _state.value = _state.value.copy(
                    sttModels = models,
                    // Keep the current choice if it survives; otherwise fall back.
                    sttModel = _state.value.sttModel
                        ?.let { current -> models.firstOrNull { it.id == current.id } }
                        ?: models.firstOrNull(),
                )
            }
        }

        viewModelScope.launch {
            db.models().observeInstalledByModality(Modality.TEXT_TO_SPEECH).collect {
                loadVoices(preferred = _state.value.ttsModel)
            }
        }
    }

    // ——— SPEAK (SPEC §7) ———

    /** The voice list is read from what is *installed*, not declared. */
    /** Which installed voice model the Speak tab uses. */
    fun selectTtsModel(model: ModelEntity) {
        if (_state.value.ttsModel?.id == model.id) return
        if (_state.value.speaking) stopSpeaking()
        _state.value = _state.value.copy(ttsModel = model, speakError = null)
        viewModelScope.launch {
            db.models().touch(model.id, System.currentTimeMillis())
            loadVoices(preferred = model)
        }
    }

    private suspend fun loadVoices(preferred: ModelEntity? = null) {
        val ttsModels = db.models().observeInstalledByModality(Modality.TEXT_TO_SPEECH).first()
        // A chosen model is offered to the engines first, so an explicit choice
        // beats the scan order when two are installed.
        val ordered = preferred?.let { listOf(it) + ttsModels.filterNot { m -> m.id == it.id } }
            ?: ttsModels
        val directories = ordered.mapNotNull { model ->
            java.io.File(model.localPath).let { if (it.isDirectory) it else it.parentFile }
        }
        synthesizer.useKokoroModel(directories.firstOrNull { synthesizer.kokoroLooksInstalled(it) })
        synthesizer.useOmniVoiceModel(directories.firstOrNull { synthesizer.omniVoiceLooksInstalled(it) })

        // Which engine each installed model belongs to, so the picker can offer an engine only the models it can actually load.
        val providers = ordered.mapNotNull { model ->
            val directory = java.io.File(model.localPath)
                .let { if (it.isDirectory) it else it.parentFile }
                ?: return@mapNotNull null
            synthesizer.providerFor(directory)?.let { model.id to it }
        }.toMap()

        val system = synthesizer.systemVoices()
        val kokoro = synthesizer.kokoroVoices()
        val omni = synthesizer.omniVoiceVoices()
        val all = kokoro + omni + system
        _state.value = _state.value.copy(
            voices = all,
            ttsModels = ttsModels,
            ttsModelProviders = providers,
            ttsModel = preferred ?: _state.value.ttsModel ?: ttsModels.firstOrNull(),
            systemEngineAvailable = system.isNotEmpty(),
            kokoroAvailable = synthesizer.kokoroReady,
            omniVoiceAvailable = synthesizer.omniVoiceReady,
            cloningAvailable = synthesizer.omniVoiceCloningReady,
            // Default to something that can actually speak right now.
            voice = _state.value.voice.takeIf { id -> all.any { it.id == id && it.available } }
                ?: all.firstOrNull { it.available }?.id
                ?: _state.value.voice,
        )
    }

    /** Switch engine explicitly. */
    fun selectProvider(provider: ai.ondevice.speech.SynthProvider) {
        val first = _state.value.voices.firstOrNull { it.provider == provider && it.available }
        if (first == null) {
            _state.value = _state.value.copy(
                speakError = when (provider) {
                    ai.ondevice.speech.SynthProvider.OMNIVOICE ->
                        "OmniVoice is not installed. Models → Add a model, then " +
                            (ai.ondevice.core.StarterModels.installHint(
                                ai.ondevice.core.StarterModels.OMNIVOICE_REPO,
                            ) ?: ai.ondevice.core.StarterModels.OMNIVOICE_REPO) + "."
                    ai.ondevice.speech.SynthProvider.KOKORO ->
                        "Kokoro is not installed. Models → Add a model, and search for Kokoro."
                    else -> "This device has no system speech engine."
                },
            )
            return
        }
        _state.value = _state.value.copy(voice = first.id, speakError = null)
    }

    fun setScript(value: String) {
        _state.value = _state.value.copy(script = value, scriptSource = null)
    }

    /** Load a script from a text file the user picked. */
    fun loadScript(uri: android.net.Uri) {
        viewModelScope.launch {
            val attachment = attachments.copyIn(uri) ?: run {
                _state.value = _state.value.copy(speakError = "That file could not be read.")
                return@launch
            }
            val extraction = attachments.extractText(attachment)
            if (extraction.text.isBlank()) {
                _state.value = _state.value.copy(
                    speakError = extraction.error ?: "Nothing readable in ${attachment.displayName}.",
                )
                return@launch
            }
            _state.value = _state.value.copy(
                script = extraction.text,
                scriptSource = attachment.displayName,
                speakError = extraction.error,
            )
        }
    }

    fun refreshFromOverrides() {
        viewModelScope.launch {
            val model = db.models().observeInstalledByModality(Modality.TEXT_TO_SPEECH).first().firstOrNull()
            val tts = SparseParams.parse(model?.paramOverridesJson)
            val stt = SparseParams.parse(
                db.models().observeInstalledByModality(Modality.SPEECH_TO_TEXT).first().firstOrNull()
                    ?.paramOverridesJson,
            )
            _state.value = _state.value.copy(
                ttsModel = model ?: _state.value.ttsModel,
                speed = tts.float("speed") ?: _state.value.speed,
                volume = tts.float("volume") ?: _state.value.volume,
                splitPattern = tts.string("split_pattern") ?: _state.value.splitPattern,
                trimSilence = tts.bool("trim_silence") ?: _state.value.trimSilence,
                languageCode = tts.string("lang_code") ?: _state.value.languageCode,
                // OmniVoice's own three.
                voiceDesign = tts.string("voice_design") ?: _state.value.voiceDesign,
                omniSteps = tts.int("steps") ?: _state.value.omniSteps,
                omniFrames = tts.int("frames") ?: _state.value.omniFrames,
                omniGuidance = tts.float("guidance_scale") ?: _state.value.omniGuidance,
                omniTimestepShift = tts.float("t_shift") ?: _state.value.omniTimestepShift,
                omniLayerPenalty = tts.float("layer_penalty") ?: _state.value.omniLayerPenalty,
                omniPositionTemperature =
                    tts.float("position_temperature") ?: _state.value.omniPositionTemperature,
                omniClassTemperature =
                    tts.float("class_temperature") ?: _state.value.omniClassTemperature,
                omniSeed = tts.int("seed") ?: _state.value.omniSeed,
            )
            tts.string("voice_blend")?.let(::applyBlendSpec)
        }
    }

    private fun applyBlendSpec(spec: String) {
        val parts = spec.split(':')
        if (parts.size < 2 || parts.any { it.isBlank() }) {
            if (spec.isBlank()) _state.value = _state.value.copy(blendVoice = null)
            return
        }
        _state.value = _state.value.copy(
            voice = parts[0].trim(),
            blendVoice = parts[1].trim(),
            blendRatio = parts.getOrNull(2)?.trim()?.toFloatOrNull()?.coerceIn(0f, 1f)
                ?: _state.value.blendRatio,
        )
    }

    /** Which speech model transcribes. */
    fun selectSttModel(model: ModelEntity) {
        if (_state.value.sttModel?.id == model.id) return
        if (_state.value.recording) stopRecording()
        transcriber.unload()
        _state.value = _state.value.copy(
            sttModel = model,
            segments = emptyList(),
            error = null,
            errorHint = null,
        )
        viewModelScope.launch { db.models().touch(model.id, System.currentTimeMillis()) }
    }

    fun setPitch(value: Float) = update { copy(pitch = value) }
    fun setVolume(value: Float) = update { copy(volume = value) }
    fun setVoiceQuery(value: String) = update { copy(voiceQuery = value) }
    fun setBlendVoice(id: String?) = update { copy(blendVoice = id) }
    fun setBlendRatio(value: Float) = update { copy(blendRatio = value) }

    private fun update(block: VoiceState.() -> VoiceState) {
        _state.value = _state.value.block()
    }

    fun selectVoice(id: String) {
        val voice = _state.value.voices.firstOrNull { it.id == id } ?: return
        if (!voice.available) {
            _state.value = _state.value.copy(
                speakError = when {
                    !ai.ondevice.speech.KokoroVoices.hasPhonemiser(voice.id) ->
                        "${voice.displayName} speaks ${ai.ondevice.speech.KokoroVoices.languageOf(voice.id)}, " +
                            "and this build has no phonemiser for it — Kokoro uses a different front end " +
                            "for that language than espeak-ng."
                    else ->
                        "${voice.displayName} needs Kokoro's weights installed. " +
                            "Models → Add a model, and search for Kokoro."
                },
            )
            return
        }
        _state.value = _state.value.copy(voice = id, speakError = null)
    }

    /** Read the script aloud — by rendering it, storing it, and playing the file. */
    fun speak() {
        speakJob?.cancel()
        speakJob = viewModelScope.launch { render(autoPlay = true) }
    }

    /** Render, store, and record the run. The only path that makes audio. */
    private suspend fun render(autoPlay: Boolean) {
        val text = _state.value.script.trim()
        if (text.isEmpty()) {
            _state.value = _state.value.copy(speakError = "There is no script to render.")
            return
        }
        _state.value = _state.value.copy(rendering = true, speakError = null)
        kotlinx.coroutines.coroutineScope {
            val stem = (_state.value.scriptSource ?: "read-aloud")
                .substringBeforeLast('.')
                .lowercase()
                .replace(Regex("[^a-z0-9]+"), "-")
                .trim('-')
                .ifBlank { "read-aloud" }
            // Uniquified.
            val destination = java.io.File(
                storage.speechDir(),
                "$stem-${System.currentTimeMillis()}.wav",
            )
            val request = currentRequest(text)
            val startedAt = System.currentTimeMillis()
            _state.value.ttsModel?.let { db.models().touch(it.id, startedAt) }
            val recording = recorder.start(viewModelScope)
            val liveJob = launch {
                recording.live.collect { trace ->
                    _state.value = _state.value.copy(liveTrace = trace)
                }
            }
            val result = synthesizer.synthesizeToFile(request, destination)
            liveJob.cancel()
            val trace = recording.stop()
            result.getOrNull()?.let { file ->
                val info = ai.ondevice.speech.WavFile.describe(file)
                val synthesisId = java.util.UUID.randomUUID().toString()
                db.syntheses().upsert(
                    ai.ondevice.data.db.SynthesisEntity(
                        id = synthesisId,
                        path = file.absolutePath,
                        text = text,
                        engineId = request.provider.name.lowercase(),
                        modelId = _state.value.ttsModel?.id,
                        voice = request.voiceId,
                        paramsJson = ai.ondevice.core.SparseParams.of(
                            "speed" to _state.value.speed,
                            "pitch" to _state.value.pitch,
                            "volume" to _state.value.volume,
                        ).toJsonString(),
                        durationMillis = info?.millis ?: 0L,
                        sampleRate = info?.sampleRate ?: 0,
                        createdAt = System.currentTimeMillis(),
                    ),
                )
                db.predictionRuns().record(
                    kind = ai.ondevice.core.PredictionKind.SPEECH,
                    artifactId = synthesisId,
                    modelId = _state.value.ttsModel?.id,
                    startedAt = startedAt,
                    trace = trace,
                    stats = ai.ondevice.core.SparseParams.of(
                        "audio_millis" to (info?.millis ?: 0L),
                        // Audio seconds produced per wall second — the same honest speed figure transcription reports, so the two halves of the voice screen can be compared.
                        "realtime_factor" to
                            if (trace.elapsedMillis > 0) {
                                (info?.millis ?: 0L).toFloat() / trace.elapsedMillis
                            } else {
                                0f
                            },
                    ),
                )
            }
            _state.value = _state.value.copy(
                rendering = false,
                speakError = result.exceptionOrNull()?.message,
                lastAudioPath = result.getOrNull()?.absolutePath,
                // Only a Speak plays itself.
                autoPlay = autoPlay && result.isSuccess,
                liveTrace = null,
                lastTrace = trace,
            )
        }
    }

    /** Stop, whichever half is running. */
    fun stopSpeaking() {
        speakJob?.cancel()
        speakJob = null
        synthesizer.stop()
        _state.value = _state.value.copy(rendering = false, speaking = false, autoPlay = false)
    }

    /** The player consumed the one-shot; do not start again on the next recomposition. */
    fun autoPlayHandled() {
        if (_state.value.autoPlay) _state.value = _state.value.copy(autoPlay = false)
    }

    /** Clear the work in progress, keep the setup. */
    fun reset() {
        stopSpeaking()
        if (_state.value.recording) stopRecording()
        _state.value = _state.value.copy(
            script = "",
            scriptSource = null,
            speakError = null,
            lastAudioPath = null,
            spokenRange = null,
            referenceSamples = null,
            referencePath = null,
            referenceTranscript = "",
            referenceSeconds = 0f,
            referenceName = "",
            transcribingReference = false,
            segments = emptyList(),
            title = "",
            sourcePath = null,
            sourceName = null,
            sourceIsRecording = false,
            error = null,
            errorHint = null,
            elapsedMillis = 0,
        )
    }

    /** The request carries the *provider the user chose*, so the synthesiser routes on intent rather than on what happens to be loadable. */
    private fun currentRequest(text: String): ai.ondevice.speech.SpeechRequest {
        val voice = _state.value.voices.firstOrNull { it.id == _state.value.voice }
        return ai.ondevice.speech.SpeechRequest(
            text = text,
            voiceId = voice?.id,
            speed = _state.value.speed,
            pitch = _state.value.pitch,
            volume = _state.value.volume,
            provider = voice?.provider ?: ai.ondevice.speech.SynthProvider.SYSTEM,
            blendVoiceId = _state.value.blendVoice
                ?.takeIf { voice?.provider == ai.ondevice.speech.SynthProvider.KOKORO },
            blendRatio = _state.value.blendRatio,
            splitPattern = _state.value.splitPattern,
            trimSilence = _state.value.trimSilence,
            languageCode = _state.value.languageCode,
            voiceDesign = _state.value.voiceDesign,
            steps = _state.value.omniSteps,
            frames = _state.value.omniFrames.takeIf { it > 0 },
            guidance = _state.value.omniGuidance,
            timestepShift = _state.value.omniTimestepShift,
            layerPenalty = _state.value.omniLayerPenalty,
            positionTemperature = _state.value.omniPositionTemperature,
            classTemperature = _state.value.omniClassTemperature,
            seed = _state.value.omniSeed.toLong(),
            voiceReference = _state.value.referenceSamples?.let { samples ->
                ai.ondevice.speech.VoiceReference(
                    samples = samples,
                    sampleRate = REFERENCE_SAMPLE_RATE,
                    transcript = _state.value.referenceTranscript.takeIf { it.isNotBlank() },
                )
            },
        )
    }

    /** Take a recording as the voice to copy. */
    fun useReferenceClip(uri: android.net.Uri) {
        viewModelScope.launch {
            _state.value = _state.value.copy(speakError = null)
            // Copied in the same way an attachment is: a content URI granted to the picker is not guaranteed to still be readable by the time Speak is pressed.
            val attachment = attachments.copyIn(uri)
            if (attachment == null) {
                _state.value = _state.value.copy(speakError = "That recording could not be read.")
                return@launch
            }
            val file = java.io.File(attachment.path)
            val decoded = transcriber.decodeAudio(file, REFERENCE_SAMPLE_RATE)
            decoded.onFailure {
                _state.value = _state.value.copy(
                    speakError = it.message ?: "That recording could not be decoded.",
                )
                return@launch
            }
            val samples = decoded.getOrThrow()
            val seconds = samples.size.toFloat() / REFERENCE_SAMPLE_RATE
            _state.value = _state.value.copy(
                referenceSamples = samples,
                // The copied file, kept alongside the samples so the clip can be heard.
                referencePath = attachment.path,
                referenceName = attachment.displayName,
                referenceSeconds = seconds,
                referenceTranscript = "",
            )
            transcribeReference(samples)
        }
    }

    /** Fill in what the reference says, using the speech model already installed. */
    private suspend fun transcribeReference(samples: FloatArray) {
        val speech = db.models().observeInstalledByModality(Modality.SPEECH_TO_TEXT).first()
            .firstOrNull() ?: return
        _state.value = _state.value.copy(transcribingReference = true)
        db.models().touch(speech.id, System.currentTimeMillis())
        val text = runCatching {
            if (!transcriber.isCurrent(speech.id)) {
                transcriber.load(
                    speech.id,
                    speech.localPath,
                    SparseParams.parse(speech.paramOverridesJson),
                ).getOrThrow()
            }
            // The engine wants 24 kHz and whisper wants 16, so this is decoded
            // twice rather than resampled between them.
            transcriber.transcribeSamples(samples, REFERENCE_SAMPLE_RATE)
                .getOrThrow().joinToString(" ") { it.text }.trim()
        }.getOrDefault("")
        _state.value = _state.value.copy(
            transcribingReference = false,
            referenceTranscript = text,
        )
    }

    fun setReferenceTranscript(text: String) {
        _state.value = _state.value.copy(referenceTranscript = text)
    }

    fun clearReferenceClip() {
        _state.value = _state.value.copy(
            referenceSamples = null,
            referencePath = null,
            referenceName = "",
            referenceSeconds = 0f,
            referenceTranscript = "",
            transcribingReference = false,
        )
    }

    fun setMode(mode: VoiceMode) {
        _state.value = _state.value.copy(mode = mode, error = null, errorHint = null)
    }

    /** SPEC §6.2 — record, then transcribe. */
    fun startRecording() {
        val model = _state.value.sttModel
        if (model == null) {
            _state.value = _state.value.copy(
                error = "No speech model is installed.",
                errorHint = "Models → Add, then paste ggerganov/whisper.cpp.",
            )
            return
        }
        _state.value = _state.value.copy(
            recording = true,
            paused = false,
            sourcePath = null,
            sourceName = null,
            sourceIsRecording = false,
            segments = emptyList(),
            elapsedMillis = 0,
            error = null,
            errorHint = null,
        )

        // The take is the whole point now: it is what gets transcribed, not
        // just what can be replayed afterwards.
        captureFile = java.io.File(
            storage.speechDir(),
            "recording-${System.currentTimeMillis()}.wav",
        )

        // No resource trace over the recording.
        recordingJob = viewModelScope.launch {
            val levels = ArrayDeque<Float>()
            transcriber.listen(captureTo = captureFile).collect { event ->
                when (event) {
                    is CaptureEvent.Level -> {
                        levels.addLast(meterLevel(event.peak))
                        while (levels.size > WAVEFORM_BARS) levels.removeFirst()
                        _state.value = _state.value.copy(
                            elapsedMillis = event.elapsedMillis,
                            waveform = List(WAVEFORM_BARS - levels.size) { 0f } + levels.toList(),
                        )
                    }
                    is CaptureEvent.Failed -> {
                        _state.value = _state.value.copy(recording = false, error = event.message)
                    }
                }
            }
        }
    }

    /** Hold the take without ending it. */
    fun pauseRecording() {
        transcriber.pause()
        _state.value = _state.value.copy(paused = true)
    }

    fun resumeRecording() {
        transcriber.resume()
        _state.value = _state.value.copy(paused = false)
    }

    fun stopRecording() {
        val capture = recordingJob
        recordingJob = null
        capture?.cancel()
        // The take is closed here, not left to the capture's teardown.
        transcriber.finishCapture()
        val take = captureFile?.takeIf { it.isFile && it.length() > 44 }
        _state.value = _state.value.copy(
            recording = false,
            paused = false,
            liveTrace = null,
            sourcePath = take?.absolutePath,
            sourceName = take?.name,
            sourceIsRecording = take != null,
        )
        // Stop stops.
    }

    /** SPEC §6.3 — take a picked file as the clip to work on. */
    fun chooseFile(uri: android.net.Uri) {
        if (_state.value.recording) stopRecording()
        viewModelScope.launch {
            val attachment = attachments.copyIn(uri)
            if (attachment == null) {
                _state.value = _state.value.copy(error = "That file could not be read.")
                return@launch
            }
            _state.value = _state.value.copy(
                sourcePath = attachment.path,
                sourceName = attachment.displayName,
                sourceIsRecording = false,
                segments = emptyList(),
                fileProgress = 0f,
                error = null,
                errorHint = null,
            )
        }
    }

    /** Put the clip down without leaving the panel. */
    fun clearSource() {
        _state.value = _state.value.copy(
            sourcePath = null,
            sourceName = null,
            sourceIsRecording = false,
            segments = emptyList(),
            fileProgress = 0f,
        )
    }

    /** Decode the clip properly, whichever way it arrived. */
    fun process() {
        val path = _state.value.sourcePath ?: return
        val model = _state.value.sttModel
        if (model == null) {
            _state.value = _state.value.copy(
                error = "No speech model is installed.",
                errorHint = "Models → Add, then paste ggerganov/whisper.cpp.",
            )
            return
        }
        val file = java.io.File(path)
        val name = _state.value.sourceName ?: file.name
        viewModelScope.launch {
            db.models().touch(model.id, System.currentTimeMillis())
            _state.value = _state.value.copy(loading = true, error = null, fileProgress = 0f)
            if (!transcriber.isCurrent(model.id)) {
                val loaded = transcriber.load(
                    model.id,
                    model.localPath,
                    SparseParams.parse(model.paramOverridesJson),
                )
                if (loaded.isFailure) {
                    _state.value = _state.value.copy(
                        loading = false,
                        error = loaded.exceptionOrNull()?.message ?: "The speech model could not be loaded.",
                    )
                    return@launch
                }
            }
            val started = System.currentTimeMillis()
            val recording = recorder.start(viewModelScope)
            val liveJob = launch {
                recording.live.collect { trace ->
                    _state.value = _state.value.copy(liveTrace = trace)
                }
            }
            val result = transcriber.transcribeFile(file)
            liveJob.cancel()
            val trace = recording.stop()
            val elapsed = System.currentTimeMillis() - started
            result.fold(
                onSuccess = { segments ->
                    val duration = segments.maxOfOrNull { it.endMillis } ?: 0L
                    val transcriptId = UUID.randomUUID().toString()
                    db.transcripts().upsert(
                        TranscriptEntity(
                            id = transcriptId,
                            sourcePath = path,
                            title = name,
                            segmentsJson = ai.ondevice.core.TranscriptSegments.encode(segments),
                            modelId = model.id,
                            paramsJson = model.paramOverridesJson,
                            durationMillis = duration,
                            createdAt = System.currentTimeMillis(),
                        ),
                    )
                    // The honest speed figure: audio seconds per wall second.
                    val realtimeFactor = if (elapsed > 0) duration.toFloat() / elapsed else 0f
                    db.predictionRuns().record(
                        kind = ai.ondevice.core.PredictionKind.TRANSCRIBE,
                        artifactId = transcriptId,
                        modelId = model.id,
                        startedAt = started,
                        trace = trace,
                        stats = SparseParams.of(
                            "realtime_factor" to realtimeFactor,
                            "audio_millis" to duration,
                        ),
                    )
                    _state.value = _state.value.copy(
                        loading = false,
                        segments = segments,
                        title = name,
                        fileProgress = 1f,
                        realtimeFactor = realtimeFactor,
                        liveTrace = null,
                        lastTrace = trace,
                        transcripts = db.transcripts().observeAll().first(),
                        // A successful decode that found no words is a result, and it used to look like a failure — the screen simply stayed as it was.
                        error = if (segments.isEmpty()) "No speech found in $name." else null,
                        errorHint = if (segments.isEmpty()) {
                            "The audio decoded fine — whisper just did not hear any words in it. " +
                                "Play the clip back to check the microphone picked you up."
                        } else {
                            null
                        },
                    )
                },
                onFailure = {
                    _state.value = _state.value.copy(
                        loading = false,
                        liveTrace = null,
                        error = it.message ?: "Transcription failed.",
                    )
                },
            )
        }
    }

    fun setVoice(voice: String) {
        _state.value = _state.value.copy(voice = voice)
    }

    fun setSpeed(speed: Float) {
        _state.value = _state.value.copy(speed = speed)
    }

    /** Reopen a stored synthesis as a working script. */
    fun loadSynthesis(synthesis: ai.ondevice.data.db.SynthesisEntity) {
        stopSpeaking()
        val params = SparseParams.parse(synthesis.paramsJson)
        _state.value = _state.value.copy(
            mode = VoiceMode.SPEAK,
            script = synthesis.text,
            scriptSource = null,
            voice = synthesis.voice ?: _state.value.voice,
            speed = params.float("speed") ?: _state.value.speed,
            pitch = params.float("pitch") ?: _state.value.pitch,
            volume = params.float("volume") ?: _state.value.volume,
            lastAudioPath = synthesis.path,
            speakError = null,
        )
    }

    /** Reopen a stored transcript in the Transcribe panel, exports and all. */
    fun loadTranscript(transcript: TranscriptEntity) {
        if (_state.value.recording) stopRecording()
        _state.value = _state.value.copy(
            mode = VoiceMode.TRANSCRIBE,
            title = transcript.title,
            sourcePath = transcript.sourcePath?.takeIf { java.io.File(it).isFile },
            sourceName = transcript.title,
            sourceIsRecording = false,
            segments = ai.ondevice.core.TranscriptSegments.parse(transcript.segmentsJson),
            fileProgress = 1f,
            error = null,
            errorHint = null,
        )
    }

    fun export(format: TranscriptFormat, onReady: (java.io.File) -> Unit) {
        viewModelScope.launch {
            val state = _state.value
            val file = withContext(Dispatchers.IO) {
                val name = state.title.lowercase().replace(Regex("[^a-z0-9]+"), "-").trim('-')
                java.io.File(storage.transcriptsDir(), "$name.${format.extension}").apply {
                    writeText(
                        TranscriptExport.render(
                            format = format,
                            segments = state.segments,
                            title = state.title,
                            modelId = state.sttModel?.id,
                        ),
                    )
                }
            }
            onReady(file)
        }
    }

    override fun onCleared() {
        transcriber.unload()
        synthesizer.release()
        super.onCleared()
    }
}

/** The two things this screen does. */
enum class VoiceMode(val label: String) {
    TRANSCRIBE("Transcribe"),

    /** SPEC §7 — read-aloud has a script, a voice, an expression and an output file. */
    SPEAK("Speak"),
}


/** One bar per read of the input buffer, sized to the canvas' waveform. */
private const val WAVEFORM_BARS = 40

/** Quietest level the meter draws at all, in dBFS. */
private const val METER_FLOOR_DB = -55f

/**
 * A raw amplitude as a bar height.
 *
 * The recorder reports peak amplitude, which is linear, and loudness is not:
 * ordinary speech into a phone peaks around -25 dBFS, which is 0.056 linear
 * and drew a bar 5% tall. Forty of those is the flat dashed line the meter
 * looked like however loudly anyone spoke. On a dB scale the same speech fills
 * a little over half the bar and a shout fills it, which is what a level meter
 * is for.
 */
private fun meterLevel(peak: Float): Float {
    if (peak <= 0f) return 0f
    val db = 20f * kotlin.math.log10(peak.coerceIn(1e-5f, 1f))
    return ((db - METER_FLOOR_DB) / -METER_FLOOR_DB).coerceIn(0f, 1f)
}

/** A reference clip is decoded at the rate OmniVoice works in, so nothing has to guess later what a bare FloatArray is. */
private const val REFERENCE_SAMPLE_RATE = ai.ondevice.speech.OmniVoiceEngine.SAMPLE_RATE


data class VoiceState(
    val mode: VoiceMode = VoiceMode.TRANSCRIBE,
    val sttModel: ModelEntity? = null,
    /** Every installed speech model, so the tab can offer a choice. */
    val sttModels: List<ModelEntity> = emptyList(),
    val ttsModel: ModelEntity? = null,
    /** Every installed voice model, so the Speak tab can offer a choice. */
    /** OmniVoice: the speaker described in words, since it ships no voice list. */
    val voiceDesign: String = "",
    /** OmniVoice: iterative unmasking passes. Named apart from whisper's steps. */
    val omniSteps: Int = ai.ondevice.speech.OmniVoiceEngine.DEFAULT_STEPS,
    /** OmniVoice: grid length in 40 ms frames; 0 means estimate from the text. */
    val omniFrames: Int = 0,
    val omniGuidance: Float = ai.ondevice.speech.OmniVoiceEngine.DEFAULT_GUIDANCE,
    val omniTimestepShift: Float = ai.ondevice.speech.OmniVoiceEngine.DEFAULT_T_SHIFT,
    val omniLayerPenalty: Float = ai.ondevice.speech.OmniVoiceEngine.DEFAULT_LAYER_PENALTY,
    val omniPositionTemperature: Float =
        ai.ondevice.speech.OmniVoiceEngine.DEFAULT_POSITION_TEMPERATURE,
    val omniClassTemperature: Float =
        ai.ondevice.speech.OmniVoiceEngine.DEFAULT_CLASS_TEMPERATURE,
    /** 0 picks a fresh seed per run; anything else makes the run repeatable. */
    val omniSeed: Int = 0,
    /** The reference clip a clone copies, once decoded, and what it said. */
    val referenceSamples: FloatArray? = null,
    /** The same clip as a file, so it can be played back before it is copied. */
    val referencePath: String? = null,
    val referenceName: String = "",
    val referenceSeconds: Float = 0f,
    val referenceTranscript: String = "",
    /** True while whisper is working out what the reference says. */
    val transcribingReference: Boolean = false,
    /** Whether this OmniVoice install has the encoders a clone needs. */
    val cloningAvailable: Boolean = false,
    val ttsModels: List<ModelEntity> = emptyList(),
    /** Which engine can run each voice model, keyed by model id. */
    val ttsModelProviders: Map<String, ai.ondevice.speech.SynthProvider> = emptyMap(),
    val recording: Boolean = false,
    val elapsedMillis: Long = 0,
    val waveform: List<Float> = List(40) { 0.15f },
    val transcripts: List<TranscriptEntity> = emptyList(),
    val title: String = "standup-recording",
    /** The decoded file transcript. */
    val segments: List<TranscriptSegment> = emptyList(),
    val fileProgress: Float = 0.74f,

    /** One pair for the whole screen, not one per mode. */
    val liveTrace: ai.ondevice.engine.ResourceTrace? = null,
    val lastTrace: ai.ondevice.engine.ResourceTrace? = null,

    // — Speak (§7) —
    val script: String = "",
    /** The filename a script was loaded from, when it was not typed. */
    val scriptSource: String? = null,
    val voices: List<ai.ondevice.speech.SynthVoice> = emptyList(),
    val voice: String = "af_heart",
    val voiceQuery: String = "",
    val blendVoice: String? = null,
    val blendRatio: Float = 0.5f,
    val speed: Float = 1.0f,
    val pitch: Float = 1.0f,
    val volume: Float = 1.0f,
    /** Expert parameters, mirrored from the TTS model's overrides. */
    val splitPattern: String = ai.ondevice.speech.KokoroRequest.DEFAULT_SPLIT_PATTERN,
    val trimSilence: Boolean = true,
    val languageCode: String? = null,
    val speaking: Boolean = false,
    val rendering: Boolean = false,
    /** Word being spoken right now, as character offsets into the script. */
    val spokenRange: Pair<Int, Int>? = null,
    val lastAudioPath: String? = null,
    val sourcePath: String? = null,
    val sourceName: String? = null,
    /** True when [sourcePath] came from the microphone rather than the picker. */
    val sourceIsRecording: Boolean = false,
    val paused: Boolean = false,
    /** True for exactly one recomposition after a Speak, so the player starts itself. */
    val autoPlay: Boolean = false,
    val speakError: String? = null,
    val kokoroAvailable: Boolean = false,
    val omniVoiceAvailable: Boolean = false,
    val systemEngineAvailable: Boolean = false,
    val loading: Boolean = false,
    val error: String? = null,
    val errorHint: String? = null,
    /** Audio seconds decoded per wall second — measured, not asserted. */
    val realtimeFactor: Float = 0f,
) {
    val selectedVoice: ai.ondevice.speech.SynthVoice?
        get() = voices.firstOrNull { it.id == voice }

    /** Voices you can actually use come first. */
    /** The engine currently chosen, which is whichever the selected voice belongs to. */
    val selectedProvider: ai.ondevice.speech.SynthProvider
        get() = selectedVoice?.provider ?: ai.ondevice.speech.SynthProvider.SYSTEM

    val filteredVoices: List<ai.ondevice.speech.SynthVoice>
        get() = voices
            // Only the chosen engine's voices.
            .filter { it.provider == selectedProvider }
            .filter {
                voiceQuery.isBlank() ||
                    it.displayName.contains(voiceQuery, true) ||
                    it.id.contains(voiceQuery, true) ||
                    it.localeLabel.contains(voiceQuery, true)
            }
            .sortedWith(compareByDescending<ai.ondevice.speech.SynthVoice> { it.available }
                .thenBy { it.localeLabel }
                .thenBy { it.displayName })

    /** The chosen engine has no voice it can actually use. */
    val missingVoiceComponent: ai.ondevice.core.MissingComponent?
        get() = ai.ondevice.core.ComponentCheck.forSpeech(
            requiresVoicePacks = ttsModel != null,
            voicePackCount = voices.count { it.provider == selectedProvider && it.available },
        )

    /** ~150 words a minute at 1×, which is a normal reading pace. */
    val estimatedSeconds: Int
        get() = if (script.isBlank()) 0 else
            (script.trim().split(Regex("\\s+")).size / (150f * speed) * 60).toInt()
}
