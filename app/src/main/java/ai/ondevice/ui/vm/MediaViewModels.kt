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
                model = db.models().observeByModality(Modality.DIFFUSION).first().firstOrNull(),
                presets = db.presets().observeFor(Modality.DIFFUSION).first(),
                runtimeInstalled = runtimeInstalled,
            )
            refreshAttachmentLibrary()
        }

        // Live, so a model that finishes downloading while this screen is open
        // appears — the same mistake the chat picker used to make.
        viewModelScope.launch {
            db.models().observeByModality(Modality.DIFFUSION).collect { models ->
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
            val model = db.models().observeByModality(Modality.DIFFUSION).first().firstOrNull()
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
                    val loaded = diffusion.load(model.id, model.localPath, _state.value.attachments)
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
     * Classified by role from metadata, never from a list of known model names,
     * so a LoRA for an architecture released tomorrow shows up without an app
     * update. Whether it actually *loads* is the runtime's answer, not ours.
     */
    private suspend fun refreshAttachmentLibrary() {
        val installed = db.models().getAll()
        val available = installed.mapNotNull { entity ->
            val role = ai.ondevice.core.AttachmentRole.classify(entity.localPath)
                ?: return@mapNotNull null
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
            val ttsModel = db.models().observeByModality(Modality.TEXT_TO_SPEECH).first().firstOrNull()
            _state.value = _state.value.copy(
                sttModel = db.models().observeByModality(Modality.SPEECH_TO_TEXT).first().firstOrNull(),
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
    private suspend fun loadVoices() {
        val kokoroInstalled = db.runtimes().get(ai.ondevice.engine.RuntimeRegistry.KOKORO)?.state !=
            RuntimeState.NOT_INSTALLED &&
            _state.value.ttsModel != null
        val system = synthesizer.systemVoices()
        val kokoro = ai.ondevice.speech.KokoroVoices.catalogue(available = kokoroInstalled)
        _state.value = _state.value.copy(
            voices = kokoro + system,
            systemEngineAvailable = system.isNotEmpty(),
            kokoroAvailable = kokoroInstalled,
            // Default to something that can actually speak right now.
            voice = _state.value.voice.takeIf { id -> (kokoro + system).any { it.id == id && it.available } }
                ?: (kokoro + system).firstOrNull { it.available }?.id
                ?: _state.value.voice,
        )
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
                speakError = "${voice.displayName} needs the Kokoro runtime and a voice pack. " +
                    "Settings → Runtimes installs it.",
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

    private fun currentRequest(text: String) = ai.ondevice.speech.SpeechRequest(
        text = text,
        voiceId = _state.value.voices.firstOrNull { it.id == _state.value.voice }
            ?.takeIf { it.provider == ai.ondevice.speech.SynthProvider.SYSTEM }
            ?.id,
        speed = _state.value.speed,
        pitch = _state.value.pitch,
        volume = _state.value.volume,
    )

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
            transcriber.listen(stepMillis = _state.value.stepMs).collect { event ->
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
    val ttsModel: ModelEntity? = null,
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
    val speaking: Boolean = false,
    val rendering: Boolean = false,
    /** Word being spoken right now, as character offsets into the script. */
    val spokenRange: Pair<Int, Int>? = null,
    val lastAudioPath: String? = null,
    val speakError: String? = null,
    val kokoroAvailable: Boolean = false,
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
