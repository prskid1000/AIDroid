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
import kotlinx.coroutines.isActive
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
    private val params: ai.ondevice.params.ParamRepository,
) : ViewModel() {

    private val _state = MutableStateFlow(ImageState())
    val state: StateFlow<ImageState> = _state.asStateFlow()

    private var generationJob: Job? = null

    /** What the components were called last time the sheet was built. */
    private var attachmentNames: String = ""

    /**
     * Role keys this screen filled in on the user's behalf, rather than being
     * told. Kept so that a load which disproves the assumption behind them can
     * take back its own guesses without touching anybody else's choices.
     */
    private var autoAdopted: Set<String> = emptySet()

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
                bareDenoiser = presumedBare(model),
            )
            refreshAttachmentLibrary()
        }

        // Live, so a model that finishes downloading while this screen is open
        // appears — the same mistake the chat picker used to make.
        //
        // It also carries the component choices. Those are written on the All
        // Parameters screen, into the base model's own row, so the row is where
        // this screen has to read them from: holding a snapshot taken when the
        // screen opened meant a component chosen over there never showed up
        // over here until the app restarted.
        viewModelScope.launch {
            db.models().observeInstalledByModality(Modality.DIFFUSION).collect { all ->
                val models = baseModelsOnly(all)
                val fresh = _state.value.model?.let { current ->
                    models.firstOrNull { it.id == current.id }
                } ?: models.firstOrNull()
                val chosenChanged =
                    fresh?.paramOverridesJson != _state.value.model?.paramOverridesJson
                // A rename changes no path, so watching the chosen paths alone
                // left the sheet showing the old name until something else
                // happened to rebuild it. The names are part of what this list
                // displays, so a change to one is a reason to rebuild it.
                val namesChanged = all.joinToString("\u0000") { "${it.id}=${it.label}" } != attachmentNames
                attachmentNames = all.joinToString("\u0000") { "${it.id}=${it.label}" }
                _state.value = _state.value.copy(availableModels = models, model = fresh)
                if (chosenChanged || namesChanged) refreshAttachmentLibrary()
            }
        }

        // A model that is still arriving is not a model that is missing, and
        // the tab said the second thing for both.
        viewModelScope.launch {
            db.models().observeInstalling().collect { jobs ->
                _state.value = _state.value.copy(
                    installing = jobs.filter { it.modality == Modality.DIFFUSION },
                )
            }
        }
    }

    /**
     * Whether this checkpoint is the denoiser alone.
     *
     * A load answers it outright, and the answer is kept on the row, so after
     * the first one this is a lookup rather than a guess.
     *
     * The guess only covers the gap before that first load, and it is a poor
     * one: the reasoning was that a quantised GGUF exists to ship the denoiser
     * on its own, which is true of most releases and wrong about the two the
     * app ships as starter cards. Being wrong here is expensive — a checkpoint
     * mistaken for bare has a stranger's CLIP adopted into it, which replaces
     * the encoder it was tuned with and costs about five-sixths of the local
     * detail in the picture — so the guess now only ever *warns*, and no longer
     * licenses adopting anything.
     */
    private fun presumedBare(model: ModelEntity?): Boolean? =
        model?.selfContained?.let { !it }
            ?: if (model?.format == ai.ondevice.core.ModelFormat.GGUF) true else null

    /** The diffusion entries that can be the *base* model, which is not the same set as the diffusion entries. */
    private fun baseModelsOnly(models: List<ModelEntity>): List<ModelEntity> =
        models.filter { it.attachmentRole == null }

    /** With more than one diffusion model installed, which one runs is the user's choice — not whichever the database happened to return first. */
    fun selectModel(model: ModelEntity) {
        if (_state.value.model?.id == model.id) return
        // The loaded context belongs to the old model; keep them in step.
        diffusion.unload()
        _state.value = _state.value.copy(
            model = model,
            error = null,
            errorHint = null,
            previewBitmap = null,
            recognisedAs = null,
            bareDenoiser = presumedBare(model),
        )
        viewModelScope.launch {
            db.models().touch(model.id, System.currentTimeMillis())
            // The components belong to the base model, so they change with it.
            refreshAttachmentLibrary()
        }
    }

    /**
     * Give the weights back now.
     *
     * The context is held after a run so the next one does not pay for the load
     * again; nothing drops it until another model is chosen. That is several
     * gigabytes resident while you go and do something else, and force-stopping
     * the app was the only way to reclaim it.
     */
    fun unloadModel() {
        diffusion.unload("you asked for the memory back")
        _state.value = _state.value.copy(
            residentComponents = emptyList(),
            residentSize = null,
            unloadReason = diffusion.lastUnloadReason,
            recognisedAs = null,
        )
    }

    fun setUse(use: ImageUse) {
        _state.value = _state.value.copy(use = use)
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
    fun setStyleImage(uri: String?) = update { copy(styleImageUri = uri) }

    fun setIdentityImage(uri: String?) = update { copy(identityImageUri = uri) }
    fun setStyleStrength(value: Float) = update { copy(styleStrength = value) }

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
            // The *base* model, not whichever diffusion row was touched last.
            // Components are rows in the same table with the same modality, and
            // since a run stamps every model it loads, the most recently used
            // one is often the VAE — whose overrides carry no steps or CFG, so
            // reading them reset the form to the built-in defaults.
            val model = baseModelsOnly(
                db.models().observeInstalledByModality(Modality.DIFFUSION).first(),
            ).firstOrNull()
            val runtimeInstalled = db.runtimes().get(RUNTIME_ID)?.state != RuntimeState.NOT_INSTALLED
            val p = SparseParams.parse(model?.paramOverridesJson)
            _state.value = _state.value.copy(
                model = model,
                runtimeInstalled = runtimeInstalled,
                bareDenoiser = _state.value.bareDenoiser ?: presumedBare(model),
                steps = p.int("steps") ?: _state.value.steps,
                cfgScale = p.float("cfg_scale") ?: _state.value.cfgScale,
                width = p.int("width") ?: _state.value.width,
                height = p.int("height") ?: _state.value.height,
                strength = p.float("strength") ?: _state.value.strength,
                samplingMethod = p.string("sampling_method") ?: _state.value.samplingMethod,
                schedule = p.string("schedule") ?: _state.value.schedule,
                clipSkip = p.int("clip_skip") ?: _state.value.clipSkip,
                vaeTiling = p.bool("vae_tiling") ?: _state.value.vaeTiling,
                // Both are editable on All Parameters as well, so the sheet has
                // to start from what that screen last wrote rather than from
                // the built-in default.
                controlStrength = p.float("control_strength") ?: _state.value.controlStrength,
                styleStrength = p.float("ip_adapter_strength") ?: _state.value.styleStrength,
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
            // Everything this run actually loads, not only the base model. A
            // VAE that decoded every picture you have made read "never used"
            // on the Models screen, because only the checkpoint was stamped.
            touchAll(model.id, _state.value.attachments.map { it.modelId }, started)
            try {
                // Whatever the last load left in memory, whether or not this run
                // is the one that put it there. A run against an already-warm
                // context skips the load entirely, and used to leave the card
                // saying nothing about a context that was holding four files.
                _state.value = _state.value.copy(
                    residentComponents = residentLines(),
                    residentSize = residentSize(),
                    unloadReason = diffusion.lastUnloadReason,
                    runStage = null,
                )
                if (!diffusion.isCurrent(model.id)) {
                    // What is about to go into memory, named before it does.
                    // A four-gigabyte checkpoint and three encoders is minutes
                    // of one opaque JNI call, and the screen could only say
                    // "loading" for all of it.
                    _state.value = _state.value.copy(
                        loadingModel = true,
                        loadingWhat = listOf(model.label) + _state.value.attachments
                            .map { "${it.role.label} · ${it.displayName}" },
                        loadingStage = null,
                    )
                    // A child of this job, not a sibling on viewModelScope —
                    // as a sibling it outlived the cancel meant to stop it,
                    // because `stageJob.cancel()` below is skipped when the
                    // load is cancelled out from under it.
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
                            // Everything the sheet lists, armed or not, so that a
                            // component switched off is one the loader is told to
                            // leave out rather than one it never hears about.
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
                        // What is *resident*, which is the loader's answer and
                        // not the sheet's intention. They differ whenever a
                        // component is switched off, and used to differ in
                        // silence.
                        residentComponents = residentLines(),
                        residentSize = residentSize(),
                        unloadReason = diffusion.lastUnloadReason,
                    )
                    // The loader has now read the tensors and said what this
                    // is. Nothing before this point could know — most GGUF
                    // releases carry no `general.architecture` — and the
                    // component warnings need it to say which encoders are
                    // missing. It changes no setting.
                    if (loaded.isSuccess) {
                        _state.value = _state.value.copy(bareDenoiser = diffusion.bareDiffusion)
                        rememberArchitecture(diffusion.detectedVersion)
                        rememberSelfContained(!diffusion.bareDiffusion)
                    }
                    if (loaded.isFailure) {
                        // The runtime's own message, and nothing invented on
                        // top of it.
                        //
                        // This used to append a diagnosis — "some converters
                        // emit tensor names past ggml's 64-character limit" —
                        // whatever had actually failed. It read as the
                        // runtime's finding, so a load that failed for an
                        // unrelated reason sent someone looking for a limit
                        // their file was nowhere near. A guess dressed as a
                        // cause is worse than no hint: it spends the reader's
                        // time and points away from the evidence. What follows
                        // is only what is known — which components were passed,
                        // because that is a fact about this run and narrows the
                        // search without claiming to end it.
                        val armed = _state.value.attachments
                        // Cancelling is not failing. A load the person stopped
                        // themselves does not get a red banner, and does not
                        // get the "switch components off one at a time" hint —
                        // there is nothing here to narrow down.
                        val why = loaded.exceptionOrNull()
                        val cancelled = why is ai.ondevice.engine.LoadCancelled
                        _state.value = _state.value.copy(
                            generating = false,
                            error = if (cancelled) {
                                null
                            } else {
                                why?.message ?: "The diffusion model could not be loaded."
                            },
                            errorHint = armed.takeIf { !cancelled && it.isNotEmpty() }?.let { components ->
                                "Passed " + components.joinToString(", ") { it.role.label } +
                                    ". Switching them off one at a time under Components narrows " +
                                    "it down when one of them is the problem."
                            },
                        )
                        return@launch
                    }
                }

                val params = currentParams(seed)
                diffusion.generate(
                    ai.ondevice.engine.DiffusionRequest(
                        params = params,
                        // The same picked file, down one of two roads: an
                        // edit model is shown it, everything else starts from
                        // it. Sending both would be asking for two things.
                        initImageUri = _state.value.sourceImageUri
                            .takeIf { _state.value.use != ImageUse.EDIT },
                        referenceImageUri = _state.value.sourceImageUri
                            .takeIf { _state.value.use == ImageUse.EDIT },
                        controlImageUri = _state.value.controlImageUri,
                        styleImageUri = _state.value.styleImageUri,
                        identityImageUri = _state.value.identityImageUri,
                        maskPngPath = _state.value.maskPath
                            ?.takeIf { _state.value.use == ImageUse.REPAINT },
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
                                runStage = event.stage,
                                secondsPerStep = event.secondsPerStep,
                                etaSeconds = if (event.secondsPerStep > 0f) {
                                    (remaining * event.secondsPerStep).toLong()
                                } else {
                                    0L
                                },
                            )
                        }
                        is ai.ondevice.engine.DiffusionEvent.Preview -> {
                            // The actual denoising state, projected out of the
                            // latent — §5.4's "intermediate latents, not a
                            // spinner".
                            _state.value = _state.value.copy(previewBitmap = event.image.toBitmap())
                        }
                        is ai.ondevice.engine.DiffusionEvent.Completed -> {
                            // A LoRA that matched nothing is the one outcome the
                            // picture cannot show you.
                            _state.value = _state.value.copy(
                                loraOutcome = event.loras.filterNot { it.landed }.map {
                                    "${it.file.substringBeforeLast('.')} changed nothing — " +
                                        "none of its ${it.total.takeIf { n -> n > 0 } ?: 0} " +
                                        "tensors match this model, so it was trained for a " +
                                        "different architecture"
                                },
                            )
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
                        // The image screen never asks for a clip, so this
                        // arriving would mean the engine answered a different
                        // question from the one asked. Not silently ignored:
                        // an unreachable branch that is reached is worth a line
                        // in the log rather than a shrug.
                        is ai.ondevice.engine.DiffusionEvent.ClipCompleted ->
                            android.util.Log.w(
                                "ImageViewModel",
                                "a video clip arrived on the image path; ignoring ${event.clip.frames.size} frames",
                            )
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
                    cancelling = false,
                    step = 0,
                    loadingModel = false,
                    liveTrace = null,
                )
            }
        }
    }

    fun cancel() {
        // Said out loud, because it is not always instant and pretending it is
        // makes a working Cancel look broken.
        //
        // Sampling and the VAE decode stop inside the current ggml graph —
        // measured at under four seconds against a step that takes nearly two
        // minutes. Encoding the prompt does not: FLUX.2 reads it with a 4B
        // language model, and abandoning *that* graph hands a null back to
        // sd.cpp's LLMEmbedder, which does not expect one and takes the process
        // down with it. So during that phase the press is recorded and lands at
        // the end of it.
        _state.value = _state.value.copy(cancelling = true)
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
                styleImageUri = null,
                identityImageUri = null,
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

    /**
     * The add-ons this run will actually pass, from what has been *chosen*.
     *
     * Not "everything installed with a role", which is what it used to be: with
     * FLUX.2 selected the sheet listed SD 3.5's three encoders, none of which a
     * DiT can read, and every one of them switched off — including the two
     * Klein cannot start without.
     *
     * Which file fills a role is a per-model decision and lives in the base
     * model's own parameters, under the key the runtime takes it by
     * ([AttachmentRole.paramKey]). All Parameters is where that is chosen; this
     * screen only arms and disarms what is already chosen, because arming is a
     * per-run thought and choosing is not.
     */
    private suspend fun refreshAttachmentLibrary() {
        val model = _state.value.model
        val installed = db.models().getInstalled()
        val chosen = model?.let { adoptObviousComponents(it, installed) }
            ?: SparseParams.parse(model?.paramOverridesJson)
        val previous = _state.value.availableAttachments.associateBy { it.modelId }

        val offered = applicableKeys(model)

        // One entry per *file*, not per role: a LoRA key holds a stack, and
        // reading it as a single path meant the second one silently vanished.
        val available = ai.ondevice.core.AttachmentRole.entries.flatMap { role ->
            ai.ondevice.core.WeightedPaths.parse(chosen[role.paramKey]).map { pick ->
                // A chosen path with no library row behind it is still chosen,
                // and the loader will still pass it. Dropping it here made the
                // sheet list five components while the model's own parameters
                // named seven — the two that had been renamed out from under it
                // were loaded on every run and mentioned nowhere.
                val entity = installed.firstOrNull { it.localPath == pick.path }
                val id = entity?.id ?: pick.path
                val before = previous[id]
                ai.ondevice.core.ModelAttachment(
                    modelId = id,
                    role = role,
                    path = pick.path,
                    displayName = entity?.label ?: java.io.File(pick.path).name,
                    // The sheet's dial wins until the stored strength itself
                    // changes, at which point the new one is the answer.
                    weight = before?.takeIf { it.chosenWeight == pick.weight }?.weight
                        ?: pick.weight,
                    // Chosen means armed. Turning one off is a "run without this
                    // one", so it survives a refresh but not a restart — and a
                    // role this model cannot use is never armed at all.
                    enabled = role.paramKey in offered && (before?.enabled ?: true),
                    chosenWeight = pick.weight,
                    applicable = role.paramKey in offered,
                )
            }
        }
        val claimed = available.map { it.role }.toSet()
        _state.value = _state.value.copy(
            availableAttachments = available,
            // Everything the library holds, chosen or not, so a warning can
            // tell "you have no CLIP-G" from "you have one and have not picked it".
            installedRoles = installed.mapNotNull { it.attachmentRole }.toSet(),
            // Installed, fits this model, and nobody has said which one — the
            // difference between "you have nothing" and "you have not picked".
            unchosenRoles = installed
                .mapNotNull { it.attachmentRole }
                .filter { it !in claimed && it.paramKey in offered }
                .distinct()
                .sortedBy { it.ordinal },
        )
    }

    /**
     * Fill the empty role slots this model cannot run without, when there is
     * only one candidate for each.
     *
     * A GGUF diffusion release is a bare denoiser: its VAE and its text
     * encoders are separate downloads, and until one is *chosen* the run has
     * no decoder and no way to read the prompt. Making the user open All
     * parameters and pick the only file that fits is a quiz with one answer.
     *
     * Only the families a picture cannot be made without are adopted. A LoRA,
     * a ControlNet or an IP-Adapter changes what comes out, so arming one on
     * the user's behalf would be choosing for them, and those stay unchosen
     * until somebody says otherwise.
     */
    private suspend fun adoptObviousComponents(
        model: ModelEntity,
        installed: List<ModelEntity>,
    ): SparseParams {
        val chosen = SparseParams.parse(model.paramOverridesJson)
        val offered = applicableKeys(model)

        // A checkpoint that carries its own encoders and decoder is filled.
        //
        // Adopting into it is not a convenience, it is a substitution: sd.cpp
        // takes an external `clip_l` in place of the one in the file, so a
        // generic CLIP replaces the one this checkpoint was tuned with, and the
        // conditioning it produces is not the conditioning the denoiser was
        // trained against. The picture still comes out — it comes out soft and
        // washed, which reads as a bad model rather than as a wrong part.
        //
        // Only the loader knows, and only after it has read the file, so this
        // stops adopting the moment it says so — and undoes what it adopted
        // while it did not know, in `rememberSelfContained`.
        if (_state.value.bareDenoiser == false) return chosen

        var next = chosen
        val adopted = autoAdopted.toMutableSet()
        for (role in ai.ondevice.core.AttachmentRole.entries) {
            if (role.family !in ADOPTABLE_FAMILIES) continue
            if (role.multiple) continue
            if (role.paramKey !in offered) continue
            // Presence of the key is the question having been answered, and an
            // empty answer is still an answer. Testing the *value* for blank
            // meant "I want no VAE" and "nobody has said" looked identical, so
            // a cleared slot was refilled on the next refresh and the choice
            // could not be made to stick.
            if (role.paramKey in chosen) continue
            val candidates = installed.filter {
                it.attachmentRole == role && java.io.File(it.localPath).isFile
            }
            val only = candidates.singleOrNull() ?: continue
            next = next.with(role.paramKey, only.localPath)
            adopted += role.paramKey
        }
        if (next == chosen) return chosen
        autoAdopted = adopted

        val json = next.toJsonString()
        db.models().upsert(model.copy(paramOverridesJson = json))
        // The row this screen holds is a copy; keep it in step or the next
        // refresh adopts the same files again.
        _state.value = _state.value.copy(model = model.copy(paramOverridesJson = json))
        return next
    }

    /**
     * Write down what the loader said this checkpoint is, and nothing else.
     *
     * The app used to go further and set steps, CFG, the sampler and the
     * schedule from a per-family table the moment it thought it had recognised
     * something. Those four are the user's, and settings that move on their own
     * are worse than settings that start wrong: a number nobody typed is a
     * number nobody can trust, and correcting it means first working out that
     * something else did.
     *
     * What survives is the name, because the component warnings are asked in
     * terms of it — which encoders this family reads its prompt with is a fact
     * about the file, not a preference — and storing it on the row means the
     * next session can answer before a load rather than after one.
     */
    private suspend fun rememberArchitecture(reported: String?) {
        val model = _state.value.model ?: return
        val name = reported ?: model.architecture
        _state.value = _state.value.copy(recognisedAs = name)
        // Only when the row has none. What the download recorded is spelled the
        // way the manifest's `appliesTo` gates are — `sdxl`, `flux2_klein` —
        // and the runtime prints its own spelling, "SDXL", "Flux.2 klein".
        // Overwriting one with the other would hide every parameter gated on it.
        if (reported.isNullOrBlank() || model.architecture != null) return

        val updated = model.copy(architecture = reported)
        db.models().upsert(updated)
        _state.value = _state.value.copy(model = updated)
    }

    /**
     * Write down whether this file carries its own encoders and decoder.
     *
     * The loader is the only thing that can answer this, and until it was
     * written down the answer died with the screen: every launch re-decided it
     * from the file extension, called a full checkpoint bare, and adopted
     * encoders into it that replaced the ones it shipped with. Storing it means
     * the second session knows what the first one found out.
     */
    private suspend fun rememberSelfContained(selfContained: Boolean) {
        val model = _state.value.model ?: return
        // Once, on the transition from not-knowing to knowing. Anything the
        // user attaches afterwards is a deliberate override and survives.
        if (model.selfContained == selfContained) return

        var overrides = SparseParams.parse(model.paramOverridesJson)
        if (selfContained) {
            // Undo the guess that was made while this was unknown.
            //
            // Before a load the app has to assume a quantised GGUF is a bare
            // denoiser, and on that assumption it adopts the one installed file
            // for each encoder and decoder slot. When the loader then says the
            // checkpoint had them all along, those adoptions stop being help:
            // sd.cpp takes an external `clip_l` *in place of* the one in the
            // file, so what was meant to fill a gap replaces a part instead.
            //
            // Only the automatic ones go. A file put here by hand is an
            // override somebody asked for, and overriding a built-in encoder is
            // a legitimate thing to want.
            ADOPTABLE_FAMILIES.forEach { family ->
                ai.ondevice.core.AttachmentRole.entries
                    .filter { it.family == family && !it.multiple && it.paramKey in autoAdopted }
                    .forEach { overrides = overrides.without(it.paramKey) }
            }
        }
        autoAdopted = emptySet()

        val json = overrides.toJsonString()
        val updated = model.copy(selfContained = selfContained, paramOverridesJson = json)
        db.models().upsert(updated)
        _state.value = _state.value.copy(model = updated)
        refreshAttachmentLibrary()
    }

    /** Stamp the base model and every component a run loads, so none of them reads "never used" after doing the work. */
    private suspend fun touchAll(baseModelId: String?, componentIds: List<String>, at: Long) {
        (listOfNotNull(baseModelId) + componentIds).distinct().forEach {
            db.models().touch(it, at)
        }
    }

    /** What the manifest offers this model, keyed the way roles are. */
    private suspend fun applicableKeys(model: ModelEntity?): Set<String> = params.applicableKeys(
        RUNTIME_ID,
        Modality.DIFFUSION.name.lowercase(),
        model?.architecture,
    )

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
            // The upscaler is the one model this run uses, so it is the one to stamp.
            touchAll(
                null,
                _state.value.availableAttachments
                    .filter { it.role == ai.ondevice.core.AttachmentRole.UPSCALER }
                    .map { it.modelId },
                System.currentTimeMillis(),
            )
            // Upscaling is a prediction like any other, and it was the one that
            // ran with no graph: 23 RRDB blocks over a 512-square frame is the
            // heaviest thing the tab does per second, and it was invisible.
            val recording = recorder.start(viewModelScope)
            val liveJob = viewModelScope.launch {
                recording.live.collect { trace ->
                    _state.value = _state.value.copy(liveTrace = trace)
                }
            }
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
                liveJob.cancel()
                _state.value = _state.value.copy(
                    generating = false,
                    liveTrace = null,
                    lastTrace = recording.stop(),
                )
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
            // Recorded so a picture in the library says what was done to it,
            // not only what was asked for.
            "use" to s.use.name.lowercase().takeIf { s.sourceImageUri != null },
            // Strength only means something when there is a source to denoise,
            // and that is now a property of the picture rather than the mode.
            "strength" to s.strength.takeIf {
                s.sourceImageUri != null && s.use != ImageUse.EDIT
            },
            "init_img" to s.sourceImageUri,
            "control_image" to s.controlImageUri,
            "control_strength" to s.controlStrength.takeIf { s.controlImageUri != null },
            "ip_adapter_strength" to s.styleStrength.takeIf { s.styleImageUri != null },
            // Each only where there is a face for it to hold.
            "style_strength" to s.photoMakerStrength.takeIf { s.identityImageUri != null },
            "id_weight" to s.pulidWeight.takeIf { s.identityImageUri != null },
            "extend" to listOf(s.extendLeft, s.extendTop, s.extendRight, s.extendBottom)
                .takeIf { s.use == ImageUse.EXTEND && it.any { px -> px > 0 } },
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

    /**
     * What is in memory, checkpoint first.
     *
     * The checkpoint leads because it is nearly all of the cost, and because
     * its absence is what made this card vanish for a self-contained model.
     */
    private fun residentLines(): List<String> = listOfNotNull(diffusion.residentModel) +
        diffusion.residentComponents.map {
            "${it.role.label} · ${it.fileName} · " +
                if (it.bytes >= 1_000_000_000L) {
                    String.format("%.2f GB", it.bytes / 1_000_000_000.0)
                } else {
                    String.format("%.0f MB", it.bytes / 1_000_000.0)
                }
        }

    private fun residentSize(): String? = diffusion.residentBytes
        .takeIf { it > 0L }
        ?.let { "≈${String.format("%.2f", it / 1_000_000_000.0)} GB of weights" }

    private companion object {
        const val STEP_MILLIS = 3100L // the canvas' 3.1 s/it on CPU
        const val SAFE_PIXEL_ENVELOPE = 768L * 768L
        const val RUNTIME_ID = "stable-diffusion.cpp"

        /** Often enough to look alive; a load's stages last seconds, not frames. */
        const val LOAD_STAGE_POLL_MILLIS = 300L

        /** The parts of a run that are plumbing, not authorship. */
        val ADOPTABLE_FAMILIES = setOf(
            ai.ondevice.core.RoleFamily.PROMPT_ENCODER,
            ai.ondevice.core.RoleFamily.DECODER,
            ai.ondevice.core.RoleFamily.POST,
        )
    }
}

/** What the Image screen's primary action should say and do right now. */
enum class ImageAction { INSTALL_RUNTIME, INSTALLING, ADD_MODEL, PICK_SOURCE, GENERATE, CANCEL }

/**
 * What the attached picture is *for*.
 *
 * These were four screens for one act: a prompt, optionally a picture, and a
 * picture out. With no picture attached none of this applies and the screen is
 * plain text-to-image, which is why it is a property of the attachment rather
 * than a mode of the screen.
 */
enum class ImageUse(val label: String) {
    /**
     * Show the model the picture and say what to change.
     *
     * The reference-image path — `-r` upstream — which an edit model reads
     * directly rather than travelling away from by a denoising strength.
     * sd.cpp picks the behaviour from the loaded model itself, so a model
     * without one simply answers from the prompt.
     */
    EDIT("Edit it"),

    /** img2img: begin at this picture and move away from it by `strength`. */
    START_FROM("Start from it"),

    /** Repaint inside a mask you paint. */
    REPAINT("Repaint part"),

    /** Grow the canvas and fill the new border. */
    EXTEND("Extend it"),
}

data class ImageState(
    val use: ImageUse = ImageUse.EDIT,
    val model: ModelEntity? = null,
    /**
     * Landscape, and no `<lora:…>` tag.
     *
     * A face is the hardest thing a small model draws and the first thing a
     * person spots as wrong, so a portrait made every early run look like a
     * broken app. The tag was worse: it named a LoRA nobody has installed, so
     * the default prompt asked for a file that was never going to be found.
     */
    val prompt: String = "a wide mountain valley at sunrise, mist over the pines, distant snow peaks",
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
    /** What an IP-Adapter takes its look from — see DiffusionRequest.styleImageUri. */
    val styleImageUri: String? = null,
    /** The face PhotoMaker and PuLID keep — see DiffusionRequest.identityImageUri. */
    val identityImageUri: String? = null,
    /** Upstream's own defaults for the two adapters that read that face. */
    val photoMakerStrength: Float = 20f,
    val pulidWeight: Float = 1f,
    val styleStrength: Float = 1.0f,
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
    /** Cancel pressed, and the run has not unwound yet. */
    val cancelling: Boolean = false,
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
    /** Roles with a file installed that fits, and no file chosen. */
    val unchosenRoles: List<ai.ondevice.core.AttachmentRole> = emptyList(),
    /** Every role the library can fill, whichever model it belongs to. */
    val installedRoles: Set<ai.ondevice.core.AttachmentRole> = emptySet(),
    /** Diffusion downloads still running, so "none" can be told from "not yet". */
    val installing: List<ai.ondevice.data.db.InstallingModel> = emptyList(),
    /** What stable-diffusion.cpp said this checkpoint is, once it has read it. */
    val recognisedAs: String? = null,
    /**
     * Whether the file is the denoiser alone, and so needs its encoders and its
     * decoder supplied — the loader's answer after a load, null before one.
     */
    val bareDenoiser: Boolean? = null,
    /** What is going into memory right now — the model first, then each component. */
    val loadingWhat: List<String> = emptyList(),
    /** Where the loader has got to, in its own words. */
    val loadingStage: String? = null,
    /**
     * What the loaded context is actually holding, once it is loaded — the
     * checkpoint first, then each component, each with its size on disk.
     *
     * The checkpoint used to be missing from this, so a self-contained model
     * with nothing attached produced an empty list and the card that reads it
     * never drew. The model most likely to be run was the one the screen said
     * nothing about.
     */
    val residentComponents: List<String> = emptyList(),
    /** Roughly what the above is costing, already formatted, or null when nothing is loaded. */
    val residentSize: String? = null,
    /** Why the app dropped the context on the user's behalf, when it did. */
    val unloadReason: String? = null,
    /** What the runtime says it is doing right now, mid-run. */
    val runStage: String? = null,
    /** LoRAs that were attached to the last run and did nothing to it. */
    val loraOutcome: List<String> = emptyList(),
) {
    /**
     * Whether a picture slot has anywhere to send a picture.
     *
     * Each of these is read by exactly one component and by nothing else, so
     * each appears with its component and not otherwise. An always-visible
     * field for a file nothing will open is the same mistake as a switch that
     * turns nothing on: the Control image slot invited a pose map for years
     * with no ControlNet installed to read one.
     */
    val usesStyleReference: Boolean
        get() = armed(ai.ondevice.core.AttachmentRole.IP_ADAPTER)

    val usesControlImage: Boolean
        get() = armed(ai.ondevice.core.AttachmentRole.CONTROLNET)

    /**
     * The face PhotoMaker keeps, which it never had.
     *
     * PhotoMaker only, and the distinction is easy to get wrong: the two
     * identity adapters do not take the same input. PhotoMaker reads
     * `pm_params.id_images` — photographs, encoded at generate time. PuLID
     * reads a *precomputed* embedding off disk, so a picture would do nothing
     * for it; its file is a path parameter instead.
     */
    val usesIdentityImage: Boolean
        get() = armed(ai.ondevice.core.AttachmentRole.PHOTO_MAKER)

    private fun armed(role: ai.ondevice.core.AttachmentRole) =
        availableAttachments.any { it.enabled && it.role == role }

    /** Only the ones actually ticked go to the runtime. */
    val attachments: List<ai.ondevice.core.ModelAttachment>
        get() = availableAttachments.filter { it.enabled }

    /** Combinations that will not work, said before Generate rather than after. */
    val missingComponents: List<ai.ondevice.core.MissingComponent>
        get() = ai.ondevice.core.ComponentCheck.forDiffusion(
            availableAttachments,
            recognisedAs ?: model?.architecture,
            installedRoles,
            bareDenoiser,
        )
    val progress: Float
        get() = if (progressSteps > 0) (step.toFloat() / progressSteps).coerceIn(0f, 1f) else 0f
    /** The denoise dial appears when, and only when, there is a source. */
    /** The denoise dial belongs to the two uses that denoise from a start. */
    val showStrength: Boolean
        get() = sourceImageUri != null && use != ImageUse.EDIT

    /** Inpaint and Extend cannot proceed without one; Generate can. */
    /** Every use of an attached picture needs one; nothing else does. */
    val requiresSource: Boolean get() = false

    val outputWidth: Int get() = width + extendLeft + extendRight
    val outputHeight: Int get() = height + extendTop + extendBottom

    val action: ImageAction
        get() = when {
            generating -> ImageAction.CANCEL
            !runtimeInstalled -> ImageAction.INSTALL_RUNTIME
            // A download in flight is not an absence, and "Add a diffusion
            // model" is the wrong instruction for someone already adding one.
            model == null && baseInstalling != null -> ImageAction.INSTALLING
            model == null -> ImageAction.ADD_MODEL
            requiresSource && sourceImageUri == null -> ImageAction.PICK_SOURCE
            else -> ImageAction.GENERATE
        }

    /** The base model on its way, if one is — add-ons do not unblock the tab. */
    val baseInstalling: ai.ondevice.data.db.InstallingModel?
        get() = installing.firstOrNull { it.attachmentRole == null }

    val actionLabel: String
        get() = when (action) {
            ImageAction.CANCEL -> if (cancelling) "Cancelling…" else "Cancel"
            ImageAction.INSTALL_RUNTIME -> "Install stable-diffusion.cpp first"
            ImageAction.INSTALLING -> baseInstalling?.label ?: "Downloading…"
            ImageAction.ADD_MODEL -> "Add a diffusion model"
            ImageAction.PICK_SOURCE -> "Choose a source image"
            ImageAction.GENERATE -> "Generate"
        }

    /** Whether pressing it does anything. A button that silently does nothing is worse than one that is plainly off. */
    val actionEnabled: Boolean
        get() = action != ImageAction.INSTALLING

    /** The runtime is installed but there is nothing for it to load. */
    val actionHint: String?
        get() = when (action) {
            ImageAction.INSTALL_RUNTIME ->
                "Diffusion is optional and ships separately. Settings → Runtimes installs it."
            ImageAction.INSTALLING ->
                "It becomes usable here the moment the last byte verifies. Models → Downloads " +
                    "shows the queue."
            ImageAction.ADD_MODEL ->
                "stable-diffusion.cpp is installed, but no diffusion model is. The Add model " +
                    "screen lists the ones this build runs."
            ImageAction.PICK_SOURCE ->
                "\"${use.label}\" needs a picture to act on."
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

        // Neither mode can start without its model, and both said "none
        // installed" while one was arriving.
        viewModelScope.launch {
            db.models().observeInstalling().collect { jobs ->
                _state.value = _state.value.copy(
                    installingStt = jobs.filter { it.modality == Modality.SPEECH_TO_TEXT },
                    installingTts = jobs.filter { it.modality == Modality.TEXT_TO_SPEECH },
                )
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
            // Computed here because this is where the scanned folders are, and
            // recorded because the error message is written elsewhere.
            kokoroVoicesMissing = synthesizer.kokoroGraphWithoutVoices(directories),
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
                    // Two different problems wore one sentence.
                    //
                    // Kokoro needs a graph *and* its speaker vectors, and
                    // `looksInstalled` is false when either is missing. Saying
                    // "not installed" for the second case is telling someone to
                    // download a folder they are looking at — the graph is
                    // there, the voices are not, and only one of those is worth
                    // acting on.
                    ai.ondevice.speech.SynthProvider.KOKORO ->
                        if (_state.value.kokoroVoicesMissing) {
                            "Kokoro's weights are installed but none of its voice packs are. " +
                                "Models → the Kokoro row → Components, and add the voices."
                        } else {
                            "Kokoro is not installed. Models → Add a model, and search for Kokoro."
                        }
                    else -> "This device has no system speech engine."
                },
            )
            return
        }
        _state.value = _state.value.copy(voice = first.id, speakError = null)
    }

    /**
     * Give back whichever voice engine's weights are resident.
     *
     * Per engine rather than all of them: this sits beside the engine picker,
     * and a button there that also dropped the other one would be doing
     * something the screen never said.
     */
    fun unloadVoiceModel() {
        val provider = _state.value.selectedVoice?.provider
            ?: ai.ondevice.speech.SynthProvider.SYSTEM
        viewModelScope.launch { synthesizer.unload(provider) }
    }

    /** The transcriber holds whisper between recordings for the same reason. */
    fun unloadTranscriber() {
        transcriber.unload()
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
    /** Speech models still downloading, so an empty list can say which empty it is. */
    val installingStt: List<ai.ondevice.data.db.InstallingModel> = emptyList(),
    val installingTts: List<ai.ondevice.data.db.InstallingModel> = emptyList(),
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
    /**
     * Kokoro's graph is installed and none of its speaker vectors are.
     *
     * Half-installed reads as absent to every other check, and the two want
     * opposite advice: one says download the model, the other says the model is
     * already here and its voices are not.
     */
    val kokoroVoicesMissing: Boolean = false,
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
