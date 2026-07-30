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
import ai.ondevice.data.db.PresetEntity
import ai.ondevice.data.db.TranscriptEntity
import ai.ondevice.data.hf.DeviceCapabilities
import ai.ondevice.engine.CaptureEvent
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject
import kotlin.random.Random

/**
 * S11/S12/S13 — image generation, the mask editor and the gallery.
 *
 * SPEC §5.4's obligations that live here: a live TAESD preview rather than a
 * spinner, cancellation that actually frees memory, the used seed surfaced for
 * one-tap reuse, and the full parameter set stored with the artifact so any
 * image is reproducible.
 */
@HiltViewModel
class ImageViewModel @Inject constructor(
    private val db: OnDeviceDatabase,
    private val storage: ModelStorage,
    private val capabilities: DeviceCapabilities,
    private val diffusion: ai.ondevice.engine.DiffusionEngine,
) : ViewModel() {

    private val _state = MutableStateFlow(ImageState())
    val state: StateFlow<ImageState> = _state.asStateFlow()

    private var generationJob: Job? = null

    init {
        viewModelScope.launch {
            // "No runtime" and "no model" are different problems with different
            // fixes, and SPEC §1.2 says a refusal has to name which one it is.
            val runtimeInstalled = db.runtimes().get(RUNTIME_ID)?.state != RuntimeState.NOT_INSTALLED
            _state.value = _state.value.copy(
                model = baseModelsOnly(db.models().observeInstalledByModality(Modality.DIFFUSION).first())
                    .firstOrNull(),
                presets = db.presets().observeFor(Modality.DIFFUSION).first(),
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

    /**
     * The diffusion entries that can be the *base* model, which is not the same
     * set as the diffusion entries.
     *
     * Modality.DIFFUSION covers everything that belongs to an image run — the
     * checkpoint and every add-on hanging off it. The picker read that set
     * straight, so a ControlNet, an IP-Adapter, a LoRA and an ESRGAN upscaler
     * all appeared as things you could generate *with*, and since the selection
     * falls back to `firstOrNull()`, whichever the database returned first
     * became the base model — a ControlNet loaded as a checkpoint.
     *
     * The line is drawn by the role the user gave the model on the Add screen:
     * an add-on has one, a checkpoint does not.
     */
    private fun baseModelsOnly(models: List<ModelEntity>): List<ModelEntity> =
        models.filter { it.attachmentRole == null }

    /**
     * With more than one diffusion model installed, which one runs is the
     * user's choice — not whichever the database happened to return first.
     */
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

    /**
     * img2img and inpaint need a source image, and until one is chosen there is
     * nothing for `strength` to act on — so the picker is part of the form, not
     * a hidden prerequisite discovered at generate time.
     */
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

    /**
     * Pull back anything the Advanced screen changed. Both screens edit the same
     * diffusion model row, so this is the round trip rather than a second copy
     * of the truth — the Image screen simply mirrors the handful of parameters
     * it puts on the surface.
     */
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

    /**
     * SPEC §5.4 — warn when width × height × batch exceeds a measured-safe
     * envelope, and suggest `vae_tiling` rather than letting it OOM.
     */
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
            try {
                if (diffusion.loadedModelId != model.id) {
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
                            // `steps` is the user's setting and must never be
                            // written from here. sd.cpp counts its own internal
                            // work — hundreds of graph nodes, not sampler steps
                            // — and feeding that back turned the Steps slider
                            // into "686" and destroyed the value the user set.
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
                            _state.value = _state.value.copy(
                                lastImage = image,
                                previewBitmap = event.image.toBitmap(),
                                elapsedMillis = System.currentTimeMillis() - started,
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
                _state.value = _state.value.copy(generating = false, step = 0, loadingModel = false)
            }
        }
    }

    fun cancel() {
        diffusion.cancel()
        generationJob?.cancel()
        generationJob = null
    }

    // — attachments (SPEC §5, generically) —

    /**
     * Everything installed that can hang off a diffusion run.
     *
     * Filed by the role the user gave it, never by a list of known model names,
     * so a LoRA for an architecture released tomorrow shows up without an app
     * update. Whether it actually *loads* is the runtime's answer, not ours.
     */
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

    /**
     * Upscale the picture just produced.
     *
     * A deliberate action on a finished image rather than a step in every run:
     * sd.cpp keeps the upscaler in its own context, it is slow, and ×4 on a
     * 512 ² image is four megapixels of output. The result is a new gallery
     * entry, not a replacement — the original is what the recorded seed and
     * parameters reproduce, and overwriting it would make the pair inconsistent.
     */
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
                // Cancelling is not failing. Caught separately and rethrown
                // because `Throwable` below would otherwise turn every Stop
                // into an "Upscaling failed" banner and break the coroutine's
                // cancellation contract on the way.
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
    /**
     * Prompt in, picture out — with an *optional* source image.
     *
     * Text-only and image+text used to be separate tabs, which made the user
     * declare up front something the app can see for itself: attaching a source
     * is what makes it img2img, and the denoise dial appears exactly when there
     * is something to denoise. One tab, one fewer decision.
     */
    GENERATE("Generate"),
    INPAINT("Inpaint"),

    /**
     * Outpainting is inpainting with a mask the app derives instead of one the
     * user paints: the canvas is enlarged, the original is composited into it,
     * and the *new* border becomes the region to fill. Same sd.cpp code path,
     * same denoise dial — which is why it belongs here as a mode rather than a
     * separate screen.
     */
    OUTPAINT("Extend"),
}

data class ImageState(
    val mode: ImageMode = ImageMode.GENERATE,
    val model: ModelEntity? = null,
    val presets: List<PresetEntity> = emptyList(),
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
    /**
     * The one init image. sd.cpp takes exactly one — everything else in the
     * pipeline is derived from it, so this is a slot, not a list.
     */
    val sourceImageUri: String? = null,
    /**
     * ControlNet's structural reference: a pose, depth or edge map that steers
     * composition without contributing pixels. A genuinely *different* input
     * from the init image, which is why it is a second slot rather than a
     * second entry in one.
     */
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
    val availableAttachments: List<ai.ondevice.core.ModelAttachment> = emptyList(),
    val availableModels: List<ModelEntity> = emptyList(),
) {
    /** Only the ones actually ticked go to the runtime. */
    val attachments: List<ai.ondevice.core.ModelAttachment>
        get() = availableAttachments.filter { it.enabled }
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

@HiltViewModel
class GalleryViewModel @Inject constructor(
    private val db: OnDeviceDatabase,
) : ViewModel() {

    val images: StateFlow<List<GeneratedImageEntity>> = db.images().observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    private val _selected = MutableStateFlow<GeneratedImageEntity?>(null)
    val selected: StateFlow<GeneratedImageEntity?> = _selected.asStateFlow()

    fun select(image: GeneratedImageEntity) {
        _selected.value = image
    }

    fun delete(image: GeneratedImageEntity) {
        viewModelScope.launch {
            runCatching { java.io.File(image.path).delete() }
            db.images().deleteById(image.id)
            if (_selected.value?.id == image.id) _selected.value = null
        }
    }
}

/** S14 — live and file transcription, plus the Kokoro read-aloud panel. */
@HiltViewModel
class VoiceViewModel @Inject constructor(
    private val db: OnDeviceDatabase,
    private val storage: ModelStorage,
    private val synthesizer: ai.ondevice.speech.SpeechSynthesizer,
    private val attachments: ai.ondevice.data.AttachmentStore,
    private val transcriber: ai.ondevice.engine.Transcriber,
) : ViewModel() {

    private val _state = MutableStateFlow(VoiceState())
    val state: StateFlow<VoiceState> = _state.asStateFlow()

    private var recordingJob: Job? = null
    private var speakJob: Job? = null

    init {
        viewModelScope.launch {
            val ttsModel = db.models().observeInstalledByModality(Modality.TEXT_TO_SPEECH).first().firstOrNull()
            _state.value = _state.value.copy(
                sttModel = db.models().observeInstalledByModality(Modality.SPEECH_TO_TEXT).first().firstOrNull(),
                sttModels = db.models().observeInstalledByModality(Modality.SPEECH_TO_TEXT).first(),
                ttsModel = ttsModel,
                transcripts = db.transcripts().observeAll().first(),
            )
            loadVoices()
        }
    }

    // ——— SPEAK (SPEC §7) ———

    /**
     * The voice list is read from what is *installed*, not declared.
     *
     * Kokoro's 54 voices are shown whether or not its runtime is present,
     * because knowing what you would get is half the reason to install it — but
     * an unavailable voice is marked and cannot be selected, rather than being
     * silently swapped for a system voice at speak time.
     */
    /**
     * Which installed voice model the Speak tab uses.
     *
     * Needed once there is more than one, and there can easily be two: Kokoro
     * and OmniVoice are both TEXT_TO_SPEECH, and nothing stops two builds of
     * either. Until now the engine was inferred by sniffing directory contents
     * and the *model* was never selectable at all — so with both installed the
     * app decided for you and gave you no way to see or change which.
     */
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
        // Point the synthesiser at the installed weights *before* asking what
        // it can do — `kokoroReady` is a statement about this device right now,
        // not a capability the build declares.
        //
        // Both neural engines are text-to-speech models, so the library can hold
        // either or both. Each directory is offered to each engine and the
        // engine decides: Kokoro wants a graph plus voice packs, OmniVoice wants
        // its four graphs and a tokenizer, and neither is identified by name.
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

        // Which engine each installed model belongs to, so the picker can offer
        // an engine only the models it can actually load. Without this the
        // OmniVoice tab listed — and preselected — Kokoro.
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
            // Default to something that can actually speak right now.
            voice = _state.value.voice.takeIf { id -> all.any { it.id == id && it.available } }
                ?: all.firstOrNull { it.available }?.id
                ?: _state.value.voice,
        )
    }

    /**
     * Switch engine explicitly.
     *
     * Kokoro and OmniVoice are not interchangeable and the app never picks
     * between them on the user's behalf: one is fast and fixed-voice, the other
     * is slow and can do things the first cannot. Choosing moves the selected
     * voice to that engine's first usable one, so the two controls never
     * disagree about which engine is about to speak.
     */
    fun selectProvider(provider: ai.ondevice.speech.SynthProvider) {
        val first = _state.value.voices.firstOrNull { it.provider == provider && it.available }
        if (first == null) {
            _state.value = _state.value.copy(
                speakError = when (provider) {
                    ai.ondevice.speech.SynthProvider.OMNIVOICE ->
                        "OmniVoice is not installed. Models → Add a model, then " +
                            "onnx-community/OmniVoice-Onnx — about 683 MB."
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

    /**
     * Pull back what the Advanced screen changed — the same round trip the
     * Image screen makes, and for the same reason: both screens edit one row,
     * so the surfaced handful has to reflect it rather than being a second copy
     * of the truth that silently disagrees.
     *
     * Without this, every Kokoro parameter below Basic was inert: the Advanced
     * screen wrote `split_pattern` and `trim_silence` into the model row and
     * nothing ever read them back out.
     */
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
                // The capture readout used to print these from hardcoded
                // defaults, so setting them in Advanced changed neither the
                // capture nor the line claiming to describe it.
                stepMs = stt.int("step_ms") ?: _state.value.stepMs,
                vadEnabled = stt.bool("vad") ?: _state.value.vadEnabled,
            )
            // A blend written as "af_heart:bm_george:0.4" in Advanced and one
            // set with the Blend control are the same setting, so they share
            // storage rather than each having their own.
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

    /**
     * Which speech model transcribes.
     *
     * There was no way to say: the screen took whichever row the database
     * returned first, so installing whisper base alongside small gave you one of
     * them and no means of preferring the other. Switching drops the loaded
     * context, because it belongs to the previous model.
     */
    fun selectSttModel(model: ModelEntity) {
        if (_state.value.sttModel?.id == model.id) return
        if (_state.value.recording) stopRecording()
        transcriber.unload()
        _state.value = _state.value.copy(
            sttModel = model,
            segments = emptyList(),
            partial = emptyList(),
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
            // Two different reasons, and conflating them sends the user to the
            // wrong screen: a missing model is fixable by downloading one, a
            // missing phonemiser is not fixable at all in this build.
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

    fun speak() {
        val text = _state.value.script.trim()
        if (text.isEmpty()) {
            _state.value = _state.value.copy(speakError = "There is no script to read.")
            return
        }
        speakJob?.cancel()
        _state.value = _state.value.copy(speaking = true, speakError = null, spokenRange = null)
        speakJob = viewModelScope.launch {
            try {
                synthesizer.speak(currentRequest(text)).collect { event ->
                    when (event) {
                        is ai.ondevice.speech.SpeechEvent.Started ->
                            _state.value = _state.value.copy(speaking = true)
                        is ai.ondevice.speech.SpeechEvent.Range ->
                            _state.value = _state.value.copy(spokenRange = event.start to event.end)
                        is ai.ondevice.speech.SpeechEvent.Done ->
                            _state.value = _state.value.copy(speaking = false, spokenRange = null)
                        is ai.ondevice.speech.SpeechEvent.Failed ->
                            _state.value = _state.value.copy(speaking = false, speakError = event.message)
                    }
                }
            } finally {
                // The flow *ending* is the terminal signal, not the Done event.
                // Clearing `speaking` only on Done meant any engine that
                // finished without emitting one left the button reading "Stop"
                // for good — audio played, the run was over, and the only way
                // out was to press Stop on something already stopped.
                //
                // Assignment to a StateFlow does not suspend, so unlike the
                // teardown in ChatViewModel this needs no NonCancellable; a
                // cancelled coroutine can still run it. speakError survives the
                // copy, so a failure keeps its message.
                _state.value = _state.value.copy(speaking = false, spokenRange = null)
            }
        }
    }

    fun stopSpeaking() {
        speakJob?.cancel()
        speakJob = null
        synthesizer.stop()
        _state.value = _state.value.copy(speaking = false, spokenRange = null)
    }

    /**
     * Render to a WAV the user can keep or send. §7 asks for export, and a
     * passage you can only hear once is not an artifact.
     */
    fun exportAudio(onReady: (java.io.File) -> Unit) {
        val text = _state.value.script.trim()
        if (text.isEmpty()) {
            _state.value = _state.value.copy(speakError = "There is no script to render.")
            return
        }
        _state.value = _state.value.copy(rendering = true, speakError = null)
        viewModelScope.launch {
            val name = (_state.value.scriptSource ?: "read-aloud")
                .substringBeforeLast('.')
                .lowercase()
                .replace(Regex("[^a-z0-9]+"), "-")
                .trim('-')
                .ifBlank { "read-aloud" }
            val destination = java.io.File(storage.transcriptsDir(), "$name.wav")
            val result = synthesizer.synthesizeToFile(currentRequest(text), destination)
            _state.value = _state.value.copy(
                rendering = false,
                speakError = result.exceptionOrNull()?.message,
                lastAudioPath = result.getOrNull()?.absolutePath,
            )
            result.getOrNull()?.let(onReady)
        }
    }

    /**
     * The request carries the *provider the user chose*, so the synthesiser
     * routes on intent rather than on what happens to be loadable. Picking a
     * Kokoro voice and hearing the system engine would be the substitution the
     * whole voice screen is built to avoid.
     */
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
        )
    }

    fun setMode(mode: VoiceMode) {
        _state.value = _state.value.copy(mode = mode, error = null, errorHint = null)
    }

    fun setSource(source: TranscribeSource) {
        // Switching source while the mic is open would leave it held.
        if (_state.value.recording) stopRecording()
        _state.value = _state.value.copy(source = source, error = null, errorHint = null)
    }

    fun setSpeakSource(source: SpeakSource) {
        if (_state.value.speaking) stopSpeaking()
        _state.value = _state.value.copy(speakSource = source, speakError = null)
    }

    /**
     * SPEC §6.2 — live transcription.
     *
     * Whisper has no incremental decode, so every partial is a fresh pass over
     * a sliding window. That is why earlier words keep changing and why the
     * screen fades them by the decoder's own confidence rather than presenting
     * them as settled.
     */
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
            elapsedMillis = 0,
            partial = emptyList(),
            error = null,
            errorHint = null,
        )

        recordingJob = viewModelScope.launch {
            if (transcriber.loadedModelId != model.id) {
                _state.value = _state.value.copy(loading = true)
                val loaded = transcriber.load(model.id, model.localPath, SparseParams.parse(model.paramOverridesJson))
                _state.value = _state.value.copy(loading = false)
                if (loaded.isFailure) {
                    _state.value = _state.value.copy(
                        recording = false,
                        error = loaded.exceptionOrNull()?.message ?: "The speech model could not be loaded.",
                    )
                    return@launch
                }
            }

            val levels = ArrayDeque<Float>()
            // From the model's overrides via refreshFromOverrides, so the
            // "step N ms" readout above the waveform describes the capture that
            // is actually running rather than a constant.
            val overrides = SparseParams.parse(model.paramOverridesJson)
            val stepMs = overrides.int("step_ms") ?: _state.value.stepMs
            _state.value = _state.value.copy(
                stepMs = stepMs,
                vadEnabled = overrides.bool("vad") ?: _state.value.vadEnabled,
            )
            transcriber.listen(stepMillis = stepMs).collect { event ->
                when (event) {
                    is CaptureEvent.Level -> {
                        levels.addLast(event.peak)
                        while (levels.size > WAVEFORM_BARS) levels.removeFirst()
                        _state.value = _state.value.copy(
                            elapsedMillis = event.elapsedMillis,
                            waveform = List(WAVEFORM_BARS - levels.size) { 0.05f } + levels.toList(),
                        )
                    }
                    is CaptureEvent.Partial -> {
                        _state.value = _state.value.copy(
                            elapsedMillis = event.elapsedMillis,
                            partial = event.segments.map { PartialSegment(it.text, it.confidence) },
                        )
                    }
                    is CaptureEvent.Failed -> {
                        _state.value = _state.value.copy(recording = false, error = event.message)
                    }
                }
            }
        }
    }

    fun stopRecording() {
        recordingJob?.cancel()
        recordingJob = null
        val segments = _state.value.partial
        viewModelScope.launch {
            if (segments.isNotEmpty()) {
                db.transcripts().upsert(
                    TranscriptEntity(
                        id = UUID.randomUUID().toString(),
                        sourcePath = null,
                        title = "Live capture",
                        segmentsJson = SparseParams.of("segments" to segments.map { it.text }).toJsonString(),
                        modelId = _state.value.sttModel?.id,
                        paramsJson = SparseParams.of(
                            "vad" to _state.value.vadEnabled,
                            "step_ms" to _state.value.stepMs,
                        ).toJsonString(),
                        durationMillis = _state.value.elapsedMillis,
                        createdAt = System.currentTimeMillis(),
                    ),
                )
            }
            _state.value = _state.value.copy(
                recording = false,
                transcripts = db.transcripts().observeAll().first(),
            )
        }
    }

    /** SPEC §6.3 — transcribe a file the user picked. */
    fun transcribeFile(uri: android.net.Uri) {
        val model = _state.value.sttModel
        if (model == null) {
            _state.value = _state.value.copy(
                error = "No speech model is installed.",
                errorHint = "Models → Add, then paste ggerganov/whisper.cpp.",
            )
            return
        }
        viewModelScope.launch {
            _state.value = _state.value.copy(loading = true, error = null, fileProgress = 0f)
            val attachment = attachments.copyIn(uri)
            if (attachment == null) {
                _state.value = _state.value.copy(loading = false, error = "That file could not be read.")
                return@launch
            }
            if (transcriber.loadedModelId != model.id) {
                val loaded = transcriber.load(model.id, model.localPath, SparseParams.parse(model.paramOverridesJson))
                if (loaded.isFailure) {
                    _state.value = _state.value.copy(
                        loading = false,
                        error = loaded.exceptionOrNull()?.message ?: "The speech model could not be loaded.",
                    )
                    return@launch
                }
            }
            val started = System.currentTimeMillis()
            val result = transcriber.transcribeFile(java.io.File(attachment.path))
            val elapsed = System.currentTimeMillis() - started
            result.fold(
                onSuccess = { segments ->
                    val duration = segments.maxOfOrNull { it.endMillis } ?: 0L
                    db.transcripts().upsert(
                        TranscriptEntity(
                            id = UUID.randomUUID().toString(),
                            sourcePath = attachment.path,
                            title = attachment.displayName,
                            segmentsJson = SparseParams.of("segments" to segments.map { it.text }).toJsonString(),
                            modelId = model.id,
                            paramsJson = model.paramOverridesJson,
                            durationMillis = duration,
                            createdAt = System.currentTimeMillis(),
                        ),
                    )
                    _state.value = _state.value.copy(
                        loading = false,
                        segments = segments,
                        title = attachment.displayName,
                        fileProgress = 1f,
                        // The honest speed figure: audio seconds per wall second.
                        realtimeFactor = if (elapsed > 0) duration.toFloat() / elapsed else 0f,
                        transcripts = db.transcripts().observeAll().first(),
                    )
                },
                onFailure = {
                    _state.value = _state.value.copy(
                        loading = false,
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

    /**
     * SPEC §6.5 — writes the transcript into the same user-browsable directory
     * the rest of the app's artifacts live in and hands the file back, so the
     * screen can share it rather than the app inventing a private export store.
     */
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

/**
 * The two things this screen does.
 *
 * Split by *what the feature is*, not by where its input comes from. An earlier
 * cut had Live / File / Speak at the top level, which put two halves of speech
 * -to-text beside its opposite and made the microphone look like a peer of
 * synthesis. Whether the audio arrives from the mic or a file is a source
 * choice inside [TRANSCRIBE]; it uses the same decoder and produces the same
 * segments either way.
 */
enum class VoiceMode(val label: String) {
    TRANSCRIBE("Transcribe"),

    /**
     * SPEC §7 — read-aloud has a script, a voice, an expression and an output
     * file. That is a workflow, not a setting, so it gets equal billing.
     */
    SPEAK("Speak"),
}

/**
 * Where each mode gets its input.
 *
 * The two tabs are deliberately symmetrical: each takes input either live from
 * the device or from a file, and each produces an artifact you can play back
 * *and* save. Transcribe turns audio into text; Speak turns text into audio.
 * Naming the sources the same way on both sides makes that inverse obvious
 * rather than something you have to work out.
 */
enum class TranscribeSource(val label: String) { MICROPHONE("Microphone"), FILE("File") }

enum class SpeakSource(val label: String) { TYPED("Type"), FILE("File") }

/** One bar per read of the input buffer, sized to the canvas' waveform. */
private const val WAVEFORM_BARS = 40

data class PartialSegment(val text: String, val confidence: Float)

data class VoiceState(
    val mode: VoiceMode = VoiceMode.TRANSCRIBE,
    val source: TranscribeSource = TranscribeSource.MICROPHONE,
    val speakSource: SpeakSource = SpeakSource.TYPED,
    val sttModel: ModelEntity? = null,
    /** Every installed speech model, so the tab can offer a choice. */
    val sttModels: List<ModelEntity> = emptyList(),
    val ttsModel: ModelEntity? = null,
    /** Every installed voice model, so the Speak tab can offer a choice. */
    val ttsModels: List<ModelEntity> = emptyList(),
    /**
     * Which engine can run each voice model, keyed by model id.
     *
     * The picker needs this to stay honest: "text-to-speech" is one modality but
     * two incompatible engines, so listing every voice model under whichever
     * engine is selected offers files that engine cannot load.
     */
    val ttsModelProviders: Map<String, ai.ondevice.speech.SynthProvider> = emptyMap(),
    val recording: Boolean = false,
    val elapsedMillis: Long = 0,
    val waveform: List<Float> = List(40) { 0.15f },
    val partial: List<PartialSegment> = emptyList(),
    val transcripts: List<TranscriptEntity> = emptyList(),
    val title: String = "standup-recording",
    /**
     * The decoded file transcript. Held as real segments with real timings
     * rather than pre-formatted strings, because the export has to produce SRT
     * and VTT cue times from them.
     */
    val segments: List<TranscriptSegment> = emptyList(),
    val fileProgress: Float = 0.74f,
    val vadEnabled: Boolean = true,
    val stepMs: Int = 3000,

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

    /**
     * Voices you can actually use come first.
     *
     * The Kokoro catalogue is longer than the system list and none of it works
     * until the runtime is installed, so listing it in declaration order fills
     * the whole visible area with greyed rows and makes the picker look broken.
     * The unavailable ones stay — knowing what installing Kokoro would get you
     * is the point — they just stop being the first thing you see.
     */
    val filteredVoices: List<ai.ondevice.speech.SynthVoice>
        get() = voices
            .filter {
                voiceQuery.isBlank() ||
                    it.displayName.contains(voiceQuery, true) ||
                    it.id.contains(voiceQuery, true) ||
                    it.localeLabel.contains(voiceQuery, true)
            }
            .sortedWith(compareByDescending<ai.ondevice.speech.SynthVoice> { it.available }
                .thenBy { it.localeLabel }
                .thenBy { it.displayName })

    /** ~150 words a minute at 1×, which is a normal reading pace. */
    val estimatedSeconds: Int
        get() = if (script.isBlank()) 0 else
            (script.trim().split(Regex("\\s+")).size / (150f * speed) * 60).toInt()
}
