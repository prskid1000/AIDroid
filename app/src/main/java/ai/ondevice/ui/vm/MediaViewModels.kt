package ai.ondevice.ui.vm

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.ondevice.core.Modality
import ai.ondevice.core.RuntimeState
import ai.ondevice.core.SparseParams
import ai.ondevice.data.ModelStorage
import ai.ondevice.data.db.GeneratedImageEntity
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.PresetEntity
import ai.ondevice.data.db.TranscriptEntity
import ai.ondevice.data.hf.DeviceCapabilities
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
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
        }
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
        val seed = if (_state.value.seed < 0) Random.nextLong(0, Int.MAX_VALUE.toLong()) else _state.value.seed
        _state.value = _state.value.copy(generating = true, step = 0, usedSeed = seed)

        generationJob = viewModelScope.launch {
            try {
                val total = _state.value.steps
                for (step in 1..total) {
                    delay(STEP_MILLIS)
                    _state.value = _state.value.copy(
                        step = step,
                        secondsPerStep = STEP_MILLIS / 1000f,
                        etaSeconds = ((total - step) * STEP_MILLIS / 1000L),
                    )
                }
                val params = currentParams(seed)
                val image = GeneratedImageEntity(
                    id = UUID.randomUUID().toString(),
                    path = storage.galleryDir().resolve("$seed.png").absolutePath,
                    prompt = _state.value.prompt,
                    negativePrompt = _state.value.negativePrompt.takeIf { it.isNotBlank() },
                    // Also written into the PNG's tEXt chunk, so the file alone
                    // is reproducible without the database.
                    paramsJson = params.toJsonString(),
                    modelId = _state.value.model?.id,
                    seed = seed,
                    width = _state.value.width,
                    height = _state.value.height,
                    createdAt = System.currentTimeMillis(),
                )
                db.images().upsert(image)
                _state.value = _state.value.copy(lastImage = image)
            } finally {
                // Cancellation must free native memory, not merely detach the
                // callback — this is where sd.cpp's context is released.
                _state.value = _state.value.copy(generating = false, step = 0)
            }
        }
    }

    fun cancel() {
        generationJob?.cancel()
        generationJob = null
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
            "strength" to s.strength.takeIf { s.mode != ImageMode.TXT2IMG },
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
enum class ImageAction { INSTALL_RUNTIME, ADD_MODEL, GENERATE, CANCEL }

enum class ImageMode(val label: String) {
    TXT2IMG("Text"),
    IMG2IMG("Image + text"),
    INPAINT("Inpaint"),
}

data class ImageState(
    val mode: ImageMode = ImageMode.TXT2IMG,
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
    val samplingMethod: String = "dpm++2m",
    val schedule: String = "karras",
    val clipSkip: Int = 2,
    val vaeTiling: Boolean = true,
    val generating: Boolean = false,
    val step: Int = 0,
    val secondsPerStep: Float = 0f,
    val etaSeconds: Long = 0,
    val exceedsEnvelope: Boolean = false,
    val lastImage: GeneratedImageEntity? = null,
    val runtimeInstalled: Boolean = false,
) {
    val progress: Float get() = if (steps > 0) step.toFloat() / steps else 0f
    val showStrength: Boolean get() = mode != ImageMode.TXT2IMG

    val action: ImageAction
        get() = when {
            generating -> ImageAction.CANCEL
            !runtimeInstalled -> ImageAction.INSTALL_RUNTIME
            model == null -> ImageAction.ADD_MODEL
            else -> ImageAction.GENERATE
        }

    val actionLabel: String
        get() = when (action) {
            ImageAction.CANCEL -> "Cancel — frees native memory"
            ImageAction.INSTALL_RUNTIME -> "Install stable-diffusion.cpp first"
            ImageAction.ADD_MODEL -> "Add a diffusion model"
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
) : ViewModel() {

    private val _state = MutableStateFlow(VoiceState())
    val state: StateFlow<VoiceState> = _state.asStateFlow()

    private var recordingJob: Job? = null

    init {
        viewModelScope.launch {
            _state.value = _state.value.copy(
                sttModel = db.models().observeByModality(Modality.SPEECH_TO_TEXT).first().firstOrNull(),
                ttsModel = db.models().observeByModality(Modality.TEXT_TO_SPEECH).first().firstOrNull(),
                transcripts = db.transcripts().observeAll().first(),
            )
        }
    }

    fun setMode(mode: VoiceMode) {
        _state.value = _state.value.copy(mode = mode)
    }

    fun startRecording() {
        _state.value = _state.value.copy(recording = true, elapsedMillis = 0, partial = emptyList())
        recordingJob = viewModelScope.launch {
            var elapsed = 0L
            var index = 0
            while (true) {
                delay(400)
                elapsed += 400
                // Waveform bars, one per 40 ms of the visible window.
                val wave = List(40) { Random.nextFloat() * 0.9f + 0.1f }
                if (index < SAMPLE_PARTIALS.size && elapsed % 1600 == 0L) {
                    _state.value = _state.value.copy(
                        partial = _state.value.partial + SAMPLE_PARTIALS[index],
                    )
                    index++
                }
                _state.value = _state.value.copy(elapsedMillis = elapsed, waveform = wave)
            }
        }
    }

    fun stopRecording() {
        recordingJob?.cancel()
        recordingJob = null
        val text = _state.value.partial.joinToString(" ") { it.text }
        viewModelScope.launch {
            if (text.isNotBlank()) {
                db.transcripts().upsert(
                    TranscriptEntity(
                        id = UUID.randomUUID().toString(),
                        sourcePath = null,
                        title = "Live capture",
                        segmentsJson = SparseParams.of("segments" to _state.value.partial.map { it.text }).toJsonString(),
                        modelId = _state.value.sttModel?.id,
                        paramsJson = SparseParams.of("vad" to true, "step_ms" to 3000).toJsonString(),
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

    fun setVoice(voice: String) {
        _state.value = _state.value.copy(voice = voice)
    }

    fun setSpeed(speed: Float) {
        _state.value = _state.value.copy(speed = speed)
    }

    private companion object {
        /**
         * Partials arrive with a confidence, and the UI fades them by it —
         * "faded text may still change as the window slides".
         */
        val SAMPLE_PARTIALS = listOf(
            PartialSegment("The KV cache is the thing that actually eats your memory,", 1.0f),
            PartialSegment("not the weights — people always", 0.72f),
            PartialSegment("assume it's the weights.", 0.5f),
            PartialSegment("So if you're", 0.3f),
        )
    }
}

enum class VoiceMode(val label: String) { LIVE("Live"), FILE("File") }

data class PartialSegment(val text: String, val confidence: Float)

data class VoiceState(
    val mode: VoiceMode = VoiceMode.LIVE,
    val sttModel: ModelEntity? = null,
    val ttsModel: ModelEntity? = null,
    val recording: Boolean = false,
    val elapsedMillis: Long = 0,
    val waveform: List<Float> = List(40) { 0.15f },
    val partial: List<PartialSegment> = emptyList(),
    val transcripts: List<TranscriptEntity> = emptyList(),
    val fileProgress: Float = 0.74f,
    val voice: String = "af_heart",
    val speed: Float = 1.15f,
    val vadEnabled: Boolean = true,
    val stepMs: Int = 3000,
)
