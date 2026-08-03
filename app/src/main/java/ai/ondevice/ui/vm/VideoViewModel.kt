package ai.ondevice.ui.vm

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.Modality
import ai.ondevice.core.ModelAttachment
import ai.ondevice.core.SparseParams
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.engine.DiffusionClip
import ai.ondevice.engine.DiffusionEngine
import ai.ondevice.engine.DiffusionEvent
import ai.ondevice.engine.DiffusionPhase
import ai.ondevice.engine.LoadCancelled
import ai.ondevice.engine.VideoRequest
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import javax.inject.Inject
import kotlin.random.Random

/**
 * Video generation, on the same runtime and the same models as stills.
 *
 * A separate screen rather than a mode of the image one, because the two ask
 * genuinely different questions of the runtime — `sd_vid_gen_params_t` has a
 * last frame and a frame count and no IP-Adapter at all — and a screen that
 * offered the union would be offering settings that reach nothing.
 *
 * What it shares is everything about *loading*: the same checkpoint, encoders,
 * decoder and LoRAs, held in one context. Generating here after generating
 * there does not reload, and switching model unloads for both.
 */
@HiltViewModel
class VideoViewModel @Inject constructor(
    private val db: OnDeviceDatabase,
    private val diffusion: DiffusionEngine,
) : ViewModel() {

    private val _state = MutableStateFlow(VideoState())
    val state: StateFlow<VideoState> = _state.asStateFlow()

    private var generationJob: Job? = null
    private var playbackJob: Job? = null

    init {
        viewModelScope.launch {
            val runtimeInstalled =
                ai.ondevice.engine.SdBridge.available
            _state.value = _state.value.copy(runtimeInstalled = runtimeInstalled)
        }
        viewModelScope.launch {
            db.models().observeInstalledByModality(Modality.DIFFUSION).collect { all ->
                // Stills and clips share a runtime and a modality, so the same
                // query answers for both and this list used to offer every
                // checkpoint on the device — a Flux that cannot make a frame
                // sat in the clip picker beside the Wan that can, and the only
                // way to find out which was which was to load one.
                //
                // `isVideo` is upstream's `sd_version_supports_video_generation`
                // by name. It answers null for a name the table does not know,
                // and null stays: hiding a model because nobody recognised its
                // name is hiding it for a reason the user cannot act on.
                //
                // The exception is SD 1.x, which is a still model until a
                // motion module is attached and a video model afterwards. So a
                // known-still checkpoint is kept when there is a motion module
                // installed for it to use.
                val hasMotionModule = all.any {
                    it.attachmentRole == ai.ondevice.core.AttachmentRole.MOTION_MODULE
                }
                val models = all.filter { it.attachmentRole == null }
                    .filter { model ->
                        ai.ondevice.core.DiffusionFamily.isVideo(
                            model.architecture ?: model.label,
                        ) != false || hasMotionModule
                    }
                val chosen = _state.value.model?.let { current ->
                    models.firstOrNull { it.id == current.id }
                } ?: models.firstOrNull()
                _state.value = _state.value.copy(models = models, model = chosen)
                refreshAttachments(all)
            }
        }

        // A part that is still arriving is not a part that is missing.
        //
        // Every list on this screen comes from `observeInstalledByModality`,
        // which is completed rows only — rightly, because half a file loads
        // into a crash. But the component warnings were computed from the same
        // list, so a UMT5 encoder nine per cent of the way here was reported as
        // "No T5-XXL for wan · add one from Models → Add", which is advice to
        // start the download that is already running.
        viewModelScope.launch {
            db.models().observeInstalling().collect { jobs ->
                _state.value = _state.value.copy(
                    installing = jobs.filter { it.modality == Modality.DIFFUSION },
                )
            }
        }
    }

    /**
     * The components installed for this model, as the loader will be told them.
     *
     * The same list the image screen builds, minus the two roles the video
     * struct has no field for. Offering an IP-Adapter here would be offering a
     * file that loads, costs its weights and its 2.5 GB vision encoder, and is
     * then never consulted — `sd_vid_gen_params_t` has nowhere to put the
     * picture it would read.
     */
    private fun refreshAttachments(all: List<ModelEntity>) {
        val attachments = all.mapNotNull { entity ->
            val role = entity.attachmentRole ?: return@mapNotNull null
            if (role in ROLES_VIDEO_IGNORES) return@mapNotNull null
            ModelAttachment(
                modelId = entity.id,
                role = role,
                path = entity.localPath,
                displayName = entity.label,
                enabled = _state.value.attachments.firstOrNull { it.modelId == entity.id }?.enabled
                    ?: (role in ROLES_ARMED_BY_DEFAULT),
            )
        }
        _state.value = _state.value.copy(availableAttachments = attachments)
    }

    fun selectModel(model: ModelEntity) {
        if (_state.value.model?.id == model.id) return
        diffusion.unload()
        _state.value = _state.value.copy(
            model = model,
            error = null,
            errorHint = null,
            clip = null,
            recognisedAs = null,
        )
        viewModelScope.launch { db.models().touch(model.id, System.currentTimeMillis()) }
    }

    /**
     * Start over, keeping the model and its settings.
     *
     * The prompt, the two end frames and the last clip go; the model, its
     * components and the sampling settings stay, because those are the setup
     * rather than the attempt. The clip's *files* are left alone — it is in the
     * library by now, and this button is not a delete.
     */
    fun reset() {
        cancel()
        stopPlayback()
        _state.value = _state.value.copy(
            prompt = "",
            negativePrompt = "",
            firstFrameUri = null,
            lastFrameUri = null,
            controlImageUri = null,
            clip = null,
            frameIndex = 0,
            usedSeed = null,
            error = null,
            errorHint = null,
            step = 0,
            progressSteps = 0,
            previewBitmap = null,
            loraOutcome = emptyList(),
        )
    }

    /** Give the weights back now — see the note on the image screen's copy. */
    fun unloadModel() {
        diffusion.unload("you asked for the memory back")
        _state.value = _state.value.copy(
            residentComponents = emptyList(),
            recognisedAs = null,
            supportsVideo = false,
        )
    }

    fun setPrompt(value: String) = update { copy(prompt = value) }
    fun setNegativePrompt(value: String) = update { copy(negativePrompt = value) }
    fun setFrames(value: Int) = update { copy(frames = value.coerceIn(1, 129)) }
    fun setFps(value: Int) = update { copy(fps = value.coerceIn(1, 60)) }
    fun setSteps(value: Int) = update { copy(steps = value.coerceIn(1, 60)) }
    fun setCfg(value: Float) = update { copy(cfgScale = value) }
    fun setSize(value: Int) = update { copy(width = value, height = value) }
    fun setSeed(value: Long) = update { copy(seed = value) }
    fun setVaeTiling(value: Boolean) = update { copy(vaeTiling = value) }
    fun setFirstFrame(uri: String?) = update { copy(firstFrameUri = uri) }
    fun setLastFrame(uri: String?) = update { copy(lastFrameUri = uri) }
    fun setControlImage(uri: String?) = update { copy(controlImageUri = uri) }

    fun toggleAttachment(modelId: String) = update {
        copy(
            availableAttachments = availableAttachments.map {
                if (it.modelId == modelId) it.copy(enabled = !it.enabled) else it
            },
        )
    }

    fun setAttachmentWeight(modelId: String, weight: Float) = update {
        copy(
            availableAttachments = availableAttachments.map {
                if (it.modelId == modelId) it.copy(weight = weight) else it
            },
        )
    }

    fun setControlStrength(value: Float) = update { copy(controlStrength = value) }

    private inline fun update(block: VideoState.() -> VideoState) {
        _state.value = _state.value.block()
    }

    /** What the run costs before it starts, so the arithmetic is not a surprise. */
    private fun estimatedBytes(): Long {
        val s = _state.value
        // Three copies is upstream's shape, not a guess: the decoder's output,
        // the conversion to RGB, and the buffer handed back.
        return s.width.toLong() * s.height * 3 * s.frames * 3
    }

    fun generate() {
        val model = _state.value.model ?: return
        val seed = if (_state.value.seed < 0) {
            Random.nextLong(0, Int.MAX_VALUE.toLong())
        } else {
            _state.value.seed
        }
        stopPlayback()
        _state.value = _state.value.copy(
            generating = true,
            step = 0,
            usedSeed = seed,
            error = null,
            errorHint = null,
            clip = null,
            frameIndex = 0,
        )

        generationJob = viewModelScope.launch {
            try {
                if (!diffusion.isCurrent(model.id)) {
                    val armed = _state.value.attachments
                    _state.value = _state.value.copy(
                        loadingModel = true,
                        loadingWhat = listOf(model.label) +
                            armed.map { "${it.role.label} · ${it.displayName}" },
                        loadingStage = null,
                    )
                    // A child of this job, not a sibling on viewModelScope.
                    //
                    // As a sibling it outlived the cancel that was meant to
                    // stop it: `stageJob.cancel()` below is never reached when
                    // the load is cancelled, because the cancellation surfaces
                    // out of `diffusion.load` and skips straight past it. The
                    // poller then ran for the life of the screen, writing a
                    // stage into a state that was no longer loading anything.
                    val stageJob = launch {
                        while (isActive) {
                            delay(LOAD_STAGE_POLL_MILLIS)
                            _state.value = _state.value.copy(loadingStage = diffusion.loadStage)
                        }
                    }
                    val loaded = try {
                        diffusion.load(
                            model.id,
                            model.localPath,
                            _state.value.availableAttachments,
                            params = SparseParams.parse(model.paramOverridesJson),
                        )
                    } finally {
                        stageJob.cancel()
                    }
                    _state.value = _state.value.copy(
                        loadingModel = false,
                        loadingWhat = emptyList(),
                        loadingStage = null,
                    )
                    if (loaded.isFailure) {
                        // Cancelling is not failing. The banner is for a load
                        // that went wrong, and being told "The load was
                        // cancelled" in red is the app reporting your own
                        // decision back to you as a fault.
                        val why = loaded.exceptionOrNull()
                        _state.value = _state.value.copy(
                            generating = false,
                            error = if (why is LoadCancelled) {
                                null
                            } else {
                                why?.message ?: "The model could not be loaded."
                            },
                        )
                        return@launch
                    }
                }

                _state.value = _state.value.copy(
                    recognisedAs = diffusion.detectedVersion,
                    supportsVideo = diffusion.supportsVideo,
                    residentComponents = listOfNotNull(diffusion.residentModel) +
                        diffusion.residentComponents.map {
                            "${it.role.label} · ${it.fileName} · " +
                                if (it.bytes >= 1_000_000_000L) {
                                    String.format("%.2f GB", it.bytes / 1_000_000_000.0)
                                } else {
                                    String.format("%.0f MB", it.bytes / 1_000_000.0)
                                }
                        },
                )

                // Asked of the loaded context rather than of the architecture's
                // name, because for SD 1.x the answer depends on whether a
                // motion module was attached — which is a fact about this load.
                if (!diffusion.supportsVideo) {
                    _state.value = _state.value.copy(
                        generating = false,
                        error = "${model.label} does not generate video.",
                        errorHint = if (diffusion.supportsImage) {
                            "An SD 1.x checkpoint can, once a motion module is attached under " +
                                "Components. Everything else here makes stills."
                        } else {
                            null
                        },
                    )
                    return@launch
                }

                diffusion.generateVideo(
                    VideoRequest(
                        params = currentParams(seed),
                        initImageUri = _state.value.firstFrameUri,
                        endImageUri = _state.value.lastFrameUri,
                        controlImageUri = _state.value.controlImageUri,
                        attachments = _state.value.attachments,
                    ),
                ).collect { event ->
                    when (event) {
                        is DiffusionEvent.Progress -> {
                            _state.value = _state.value.copy(
                                step = event.step,
                                progressSteps = event.steps,
                                phase = event.phase,
                                secondsPerStep = event.secondsPerStep,
                                runStage = event.stage,
                            )
                        }
                        is DiffusionEvent.Preview -> {
                            _state.value = _state.value.copy(
                                previewBitmap = event.image.toBitmap(),
                            )
                        }
                        is DiffusionEvent.ClipCompleted -> {
                            _state.value = _state.value.copy(
                                clip = event.clip,
                                frameIndex = 0,
                                loraOutcome = event.loras.filterNot { it.landed }.map {
                                    "${it.file} matched none of this model's tensors, so it " +
                                        "changed nothing about the clip."
                                },
                            )
                            // Recorded before playback starts: the frames are
                            // already on disk, and a clip that is not indexed is
                            // a folder nothing will ever find again.
                            db.clips().upsert(
                                ai.ondevice.data.db.GeneratedClipEntity(
                                    id = java.util.UUID.randomUUID().toString(),
                                    directory = event.clip.directory,
                                    frameCount = event.clip.frames.size,
                                    prompt = _state.value.prompt,
                                    negativePrompt = _state.value.negativePrompt
                                        .takeIf { it.isNotBlank() },
                                    paramsJson = currentParams(seed).toJsonString(),
                                    modelId = model.id,
                                    seed = seed,
                                    width = event.clip.width,
                                    height = event.clip.height,
                                    fps = event.clip.fps,
                                    audioPath = event.clip.audioPath,
                                    createdAt = System.currentTimeMillis(),
                                ),
                            )
                            play()
                        }
                        is DiffusionEvent.Completed ->
                            android.util.Log.w(
                                TAG,
                                "a still arrived on the video path; ignoring it",
                            )
                        is DiffusionEvent.Failed -> {
                            _state.value = _state.value.copy(
                                error = event.message,
                                errorHint = event.suggestion,
                            )
                        }
                    }
                }
            } finally {
                diffusion.cancel()
                _state.value = _state.value.copy(
                    generating = false,
                    cancelling = false,
                    loadingModel = false,
                    previewBitmap = null,
                )
            }
        }
    }

    fun cancel() {
        _state.value = _state.value.copy(cancelling = true)
        diffusion.cancel()
        generationJob?.cancel()
    }

    // — playback —
    //
    // Frame at a time off disk, because that is where they are. A clip is not
    // held decoded: 129 frames at 512² is 400 MB of Bitmap, and the screen
    // shows one.

    fun play() {
        val clip = _state.value.clip ?: return
        if (clip.frames.isEmpty()) return
        playbackJob?.cancel()
        _state.value = _state.value.copy(playing = true)
        playbackJob = viewModelScope.launch {
            val frameMillis = (1000L / clip.fps.coerceAtLeast(1)).coerceAtLeast(16L)
            while (isActive) {
                delay(frameMillis)
                val next = (_state.value.frameIndex + 1) % clip.frames.size
                _state.value = _state.value.copy(frameIndex = next)
            }
        }
    }

    fun pause() = stopPlayback()

    fun seekTo(index: Int) {
        stopPlayback()
        val clip = _state.value.clip ?: return
        _state.value = _state.value.copy(
            frameIndex = index.coerceIn(0, (clip.frames.size - 1).coerceAtLeast(0)),
        )
    }

    private fun stopPlayback() {
        playbackJob?.cancel()
        playbackJob = null
        _state.value = _state.value.copy(playing = false)
    }

    /** Delete a clip's directory — it is a run's worth of PNGs and nothing else refers to it. */
    fun discard() {
        val clip = _state.value.clip ?: return
        stopPlayback()
        viewModelScope.launch {
            runCatching { File(clip.directory).deleteRecursively() }
            _state.value = _state.value.copy(clip = null, frameIndex = 0)
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
            "vae_tiling" to s.vaeTiling,
            "video_frames" to s.frames,
            "fps" to s.fps,
            // Only where there is something for it to hold.
            "vace_strength" to s.controlStrength.takeIf { s.controlImageUri != null },
        )
    }

    override fun onCleared() {
        stopPlayback()
        super.onCleared()
    }

    private companion object {
        const val TAG = "VideoViewModel"
        const val RUNTIME_ID = "stable-diffusion.cpp"
        const val LOAD_STAGE_POLL_MILLIS = 300L

        /**
         * Roles `sd_vid_gen_params_t` has no field for.
         *
         * Not hidden as a matter of taste: there is no `ip_adapter_image` and
         * no `ip_adapter_strength` in the video struct at all, so an attached
         * one would cost its weights and a 2.5 GB vision encoder and never be
         * read. Same for the two identity adapters.
         *
         * A ControlNet joins them because `generate_video` hands the sampler an
         * empty control image and a strength of zero — a clip's control map
         * goes to VACE, whose blocks are part of the checkpoint. The slot was
         * offered, the file was loaded, and nothing ever asked it anything.
         *
         * The upscaler is the one that left: LTX-AV's hi-res stage is a latent
         * spatial upsampler in its own file, and refusing the role was refusing
         * the only way to supply it.
         */
        val ROLES_VIDEO_IGNORES = setOf(
            AttachmentRole.IP_ADAPTER,
            AttachmentRole.CLIP_VISION,
            AttachmentRole.PHOTO_MAKER,
            AttachmentRole.PULID,
            AttachmentRole.CONTROLNET,
        )

        /** The parts a model cannot run without, armed unless switched off. */
        val ROLES_ARMED_BY_DEFAULT = setOf(
            AttachmentRole.VAE,
            AttachmentRole.AUDIO_VAE,
            AttachmentRole.CLIP_L,
            AttachmentRole.CLIP_G,
            AttachmentRole.T5XXL,
            AttachmentRole.LLM_ENCODER,
            AttachmentRole.LLM_VISION,
            AttachmentRole.HIGH_NOISE_DIFFUSION,
            AttachmentRole.MOTION_MODULE,
        )
    }
}

/** S14 — one clip, and everything asked of the runtime to get it. */
data class VideoState(
    val models: List<ModelEntity> = emptyList(),
    val model: ModelEntity? = null,
    /** Downloads in flight, so an arriving part is not reported as an absent one. */
    val installing: List<ai.ondevice.data.db.InstallingModel> = emptyList(),
    val runtimeInstalled: Boolean = false,
    val prompt: String = "",
    val negativePrompt: String = "",
    /** Upstream's default: a second at 16 fps, which is the shortest clip worth watching. */
    val frames: Int = 16,
    val fps: Int = 16,
    val steps: Int = 20,
    val cfgScale: Float = 6.0f,
    val width: Int = 384,
    val height: Int = 384,
    val seed: Long = -1,
    val usedSeed: Long? = null,
    val vaeTiling: Boolean = true,
    /** The still a clip starts from, and the one it ends on. */
    val firstFrameUri: String? = null,
    val lastFrameUri: String? = null,
    val controlImageUri: String? = null,
    val controlStrength: Float = 1.0f,
    val availableAttachments: List<ModelAttachment> = emptyList(),
    val generating: Boolean = false,
    val cancelling: Boolean = false,
    val loadingModel: Boolean = false,
    val loadingWhat: List<String> = emptyList(),
    val loadingStage: String? = null,
    val residentComponents: List<String> = emptyList(),
    val step: Int = 0,
    val progressSteps: Int = 0,
    val secondsPerStep: Float = 0f,
    val phase: DiffusionPhase = DiffusionPhase.PREPARING,
    val runStage: String? = null,
    val previewBitmap: android.graphics.Bitmap? = null,
    /** What the loader decided the checkpoint is, once it has read the tensors. */
    val recognisedAs: String? = null,
    /** Whether the loaded context can make video at all — false until one is loaded. */
    val supportsVideo: Boolean = false,
    val clip: DiffusionClip? = null,
    val frameIndex: Int = 0,
    val playing: Boolean = false,
    val loraOutcome: List<String> = emptyList(),
    val error: String? = null,
    val errorHint: String? = null,
) {
    val attachments: List<ModelAttachment> get() = availableAttachments.filter { it.enabled }

    /** The clip's length in seconds, from what was asked for rather than what came back. */
    val requestedSeconds: Float get() = if (fps > 0) frames / fps.toFloat() else 0f

    /**
     * Roughly what the frames alone will occupy while the run finishes.
     *
     * Worth stating because it is the number that decides whether a clip is
     * possible, and it grows with three things at once. A 5 s 480p clip is
     * about 440 MB before anything is written.
     */
    val estimatedFrameMegabytes: Int
        get() = (width.toLong() * height * 3 * frames * 3 / 1_000_000).toInt()

    val currentFramePath: String? get() = clip?.frames?.getOrNull(frameIndex)

    /**
     * What this model needs and has not been given, asked of the same table the
     * image screen asks. Empty until a load has said what the checkpoint is —
     * before that there is nothing to be missing *from*.
     */
    val missingComponents: List<ai.ondevice.core.MissingComponent>
        get() = if (recognisedAs == null) {
            emptyList()
        } else {
            ai.ondevice.core.ComponentCheck.forDiffusion(
                available = availableAttachments,
                architecture = recognisedAs,
                arrivingRoles = arrivingRoles,
            )
        }

    /** The slots a download is on its way to filling. */
    val arrivingRoles: Set<ai.ondevice.core.AttachmentRole>
        get() = installing.mapNotNull { it.attachmentRole }.toSet()
}
