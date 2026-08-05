package ai.ondevice.ui.vm

import androidx.lifecycle.ViewModel
import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.Modality
import ai.ondevice.core.ModelAttachment
import ai.ondevice.core.SparseParams
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.fromSameRepoAs
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.engine.DiffusionClip
import ai.ondevice.engine.DiffusionEngine
import ai.ondevice.engine.DiffusionEvent
import ai.ondevice.engine.DiffusionPhase
import ai.ondevice.engine.LoadCancelled
import ai.ondevice.engine.record
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
    @dagger.hilt.android.qualifiers.ApplicationContext
    private val context: android.content.Context,
    // The run and its state live here, not in this class — see VideoSession.
    private val session: VideoSession,
    private val db: OnDeviceDatabase,
    private val diffusion: DiffusionEngine,
    private val recorder: ai.ondevice.engine.ResourceRecorder,
    private val params: ai.ondevice.params.ParamRepository,
) : ViewModel() {

    private val _state get() = session.state
    val state: StateFlow<VideoState> = session.state.asStateFlow()

    /**
     * The session's scope, under the name the body of this class already used.
     *
     * Every launch here was `viewModelScope`, which is exactly what made a run
     * die with the screen. Renaming rather than rewriting each call site keeps
     * the change to what it is: the same work, on a scope that outlives the
     * thing watching it.
     */
    private val runScope get() = session.scope

    private var generationJob: Job?
        get() = session.generationJob
        set(value) { session.generationJob = value }

    private var playbackJob: Job?
        get() = session.playbackJob
        set(value) { session.playbackJob = value }

    init {
        // Once per process, not once per visit.
        //
        // These watch the library and the download queue for as long as the
        // app lives. They used to hang off `viewModelScope` and so were built
        // and thrown away with the screen, which was wasteful and correct;
        // on a scope that outlives the screen it would merely be wasteful —
        // every reopening would add another copy of every collector, each
        // writing the same state.
        if (session.claimObservers()) attachObservers()
    }

    /** The long-lived watchers, attached once for the life of the process. */
    private fun attachObservers() {
        runScope.launch {
            val runtimeInstalled =
                ai.ondevice.engine.SdBridge.available
            _state.value = _state.value.copy(runtimeInstalled = runtimeInstalled)
        }
        runScope.launch {
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
                seedFrom(chosen)
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
        runScope.launch {
            db.models().observeInstalling().collect { jobs ->
                _state.value = _state.value.copy(
                    installing = jobs.filter { it.modality == Modality.DIFFUSION },
                )
            }
        }
    }

    /**
     * Where this model's form starts, before anyone has touched it.
     *
     * This screen had no such thing: every slider came up at the number
     * written into [VideoState] and stayed there, so a model with settings
     * saved against it showed twenty steps because twenty is what the Kotlin
     * default says — the stored value was on the model, visible on All
     * Parameters, and ignored here.
     *
     * Three sources, most specific first: what was stored against this model,
     * what the manifest says for its architecture, and the state's own
     * fallback. Only the first is somebody's decision; the second is what
     * keeps the app from asserting numbers it does not own.
     */
    /**
     * Seeding must not undo a choice somebody has already made.
     *
     * This runs whenever the model's stored parameters change, and returning
     * from All Parameters is one of those times — so setting a width here,
     * stepping into that screen to change something unrelated, and stepping
     * back put the width where the *model* said it was. Nothing announced it.
     * The next run then used a size nobody had asked for, which on a clip is
     * seven minutes to find out.
     *
     * So a value the person has touched this session is theirs until they
     * change it or a different model is chosen. A value they have not touched
     * still follows the model, which is what makes a checkpoint's own settings
     * appear when it is picked.
     */
    private fun <T> keep(key: String, seeded: T?, current: T): T =
        if (key in _state.value.touched) current else (seeded ?: current)

    private suspend fun seedFrom(model: ModelEntity?) {
        val p = SparseParams.parse(model?.paramOverridesJson)
        val d = params.defaultsFor(
            RUNTIME_ID,
            Modality.DIFFUSION.name.lowercase(),
            model?.architecture,
        )
        val s = _state.value
        _state.value = s.copy(
            steps = keep("steps", p.int("steps") ?: d.int("steps"), s.steps),
            cfgScale = keep("cfg_scale", p.float("cfg_scale") ?: d.float("cfg_scale"), s.cfgScale),
            width = keep("width", p.int("width") ?: d.int("width"), s.width),
            height = keep("height", p.int("height") ?: d.int("height"), s.height),
            frames = keep("video_frames", p.int("video_frames") ?: d.int("video_frames"), s.frames),
            fps = keep("fps", p.int("fps") ?: d.int("fps"), s.fps),
            // Tiling is not seeded from the runtime, deliberately.
            //
            // Upstream defaults it off, which is a reasonable assumption
            // about a desktop with VRAM to spare and the wrong one here:
            // the decoder's working buffer grows with area, and untiled at
            // 384 square the runtime reserved 7.47 GB for it on a phone
            // with 15.6 GB shared with the rest of Android. Whether to
            // decode in tiles is a fact about the device, which this app
            // knows and the model does not.
            vaeTiling = keep("vae_tiling", p.bool("vae_tiling"), s.vaeTiling),
            controlStrength = keep(
                "vace_strength",
                p.float("vace_strength") ?: d.float("vace_strength"),
                s.controlStrength,
            ),
        )
    }

    /**
     * The components chosen for this model, as the loader will be told them.
     *
     * Built from what has been *chosen* — the model's own parameters, plus the
     * obvious adoptions made on its behalf — and not from the library, which is
     * what it used to be and what made this screen read differently from the
     * still one for the same model.
     *
     * Sourcing it from the library put a row on screen for every add-on
     * installed anywhere on the device. A Wan clip listed FLUX.2's Qwen3 encoder
     * under Prompt encoder as "n/a · Not used by wan", which is the app
     * answering a question nobody asked: nothing had chosen that file for this
     * model, so there was nothing to disown. Meanwhile the still screen, reading
     * from the choices, listed the one encoder it had been given and no more.
     *
     * The distinction is not cosmetic. [ModelAttachment.applicable] exists so a
     * component someone *did* choose and this model cannot use is visible and
     * disarmed rather than dropped — because the loader falls back to the stored
     * path for any role the caller does not mention, and a dropped row is a file
     * quietly loaded anyway. A library-sourced row inverted that: the mere
     * presence of an unarmed LLM in the library shadowed a stored `llm` path, so
     * a choice made under All Parameters was silently discarded on this screen
     * and honoured on the other.
     *
     * What stays video-specific is which roles count at all — see
     * [ROLES_VIDEO_IGNORES] and [familyReads].
     */
    private suspend fun refreshAttachments(all: List<ModelEntity>) {
        val model = _state.value.model
        // The keys this architecture's parameter set actually offers, which is
        // what the image screen has always filtered on. Without it every role
        // read as usable here, so nothing was ever marked "n/a".
        val offered = params.applicableKeys(
            RUNTIME_ID,
            Modality.DIFFUSION.name.lowercase(),
            model?.architecture,
        )
        // What the loader said, or failing that what the file declares.
        val arch = _state.value.recognisedAs ?: model?.architecture
        val family = ai.ondevice.core.DiffusionFamily.forName(arch)

        // Three questions, asked in one place because they are one question:
        // does the video struct have a field for this, does the manifest offer
        // it for this architecture, and does this family read it.
        fun usable(role: AttachmentRole): Boolean =
            role !in ROLES_VIDEO_IGNORES &&
                role.paramKey in offered &&
                familyReads(family, role)

        val chosen = model
            ?.let { adoptObviousComponents(it, all, family, ::usable) }
            ?: SparseParams.EMPTY
        val previous = _state.value.availableAttachments.associateBy { it.modelId }

        // One entry per *file*, not per role: a LoRA key holds a stack.
        val attachments = AttachmentRole.entries.flatMap { role ->
            ai.ondevice.core.WeightedPaths.parse(chosen[role.paramKey]).map { pick ->
                // A chosen path with no library row behind it is still chosen,
                // and the loader will still pass it.
                val entity = all.firstOrNull { it.localPath == pick.path }
                val id = entity?.id ?: pick.path
                val before = previous[id]
                ModelAttachment(
                    modelId = id,
                    role = role,
                    path = pick.path,
                    displayName = entity?.label ?: File(pick.path).name,
                    // The sheet's dial wins until the stored strength itself
                    // changes, at which point the new one is the answer.
                    weight = before?.takeIf { it.chosenWeight == pick.weight }?.weight
                        ?: pick.weight,
                    chosenWeight = pick.weight,
                    // Chosen means armed, unless this run said otherwise.
                    enabled = usable(role) && (before?.enabled ?: true),
                    applicable = usable(role),
                    mismatch = mismatchNote(family, arch, role, entity),
                )
            }
        }

        val claimed = attachments.map { it.role }.toSet()
        _state.value = _state.value.copy(
            availableAttachments = attachments,
            installedRoles = all.mapNotNull { it.attachmentRole }.toSet(),
            // Installed, fits this model, and nobody has said which one — the
            // difference between "you have nothing" and "you have not picked".
            unchosenRoles = all
                .mapNotNull { it.attachmentRole }
                .filter { it !in claimed && usable(it) }
                .distinct()
                .sortedBy { it.ordinal },
        )
    }

    /**
     * Fill the empty slots this model cannot run without, where one file fits.
     *
     * The still screen's [ImageViewModel.adoptObviousComponents] by the same
     * rules, so a component adopted on one screen is the component the other one
     * shows. It writes the choice down rather than arming it in memory, which is
     * the difference: this screen used to decide afresh every refresh and tell
     * nobody, so All Parameters showed an empty slot for a file that was being
     * loaded on every run.
     *
     * Companion denoisers are adoptable here and are not on the still screen.
     * Wan 2.2's I2V splits its weights by timestep and both halves are the
     * model — leaving that unfilled is not restraint, it is a clip that cannot
     * be made — and this is the only screen that can run one.
     */
    private suspend fun adoptObviousComponents(
        model: ModelEntity,
        installed: List<ModelEntity>,
        family: ai.ondevice.core.DiffusionFamily?,
        usable: (AttachmentRole) -> Boolean,
    ): SparseParams {
        val chosen = SparseParams.parse(model.paramOverridesJson)
        // A checkpoint that carries its own encoders and decoder is filled, and
        // adopting into it substitutes rather than supplies. Only the loader
        // knows, and only after it has read the file.
        if (_state.value.bareDenoiser == false) return chosen

        var next = chosen
        for (role in AttachmentRole.entries) {
            if (role.family !in ADOPTABLE_FAMILIES) continue
            if (role.multiple) continue
            if (!usable(role)) continue
            // Presence of the key is the question having been answered, and an
            // empty answer is still an answer.
            if (role.paramKey in chosen) continue
            val candidates = installed
                .filter { it.attachmentRole == role && File(it.localPath).isFile }
                .filter { suits(family, role, it) }
            // Several for one slot is a question, unless one of them shipped in
            // the same repo as the checkpoint — see fromSameRepoAs.
            val only = candidates.fromSameRepoAs(model).singleOrNull() ?: continue
            next = next.with(role.paramKey, only.localPath)
        }
        if (next == chosen) return chosen

        val json = next.toJsonString()
        db.models().upsert(model.copy(paramOverridesJson = json))
        // The row this screen holds is a copy; keep it in step or the next
        // refresh adopts the same files again.
        _state.value = _state.value.copy(model = model.copy(paramOverridesJson = json))
        return next
    }

    /**
     * Why this file is the wrong one for the slot it is in, or null.
     *
     * The still screen says this and this one did not, which is the whole of it:
     * both T5-XXLs are called T5-XXL, and being told which one is in the slot is
     * the only thing on the row that can tell them apart.
     */
    private fun mismatchNote(
        family: ai.ondevice.core.DiffusionFamily?,
        arch: String?,
        role: AttachmentRole,
        candidate: ModelEntity?,
    ): String? {
        if (role != AttachmentRole.T5XXL) return null
        val wanted = family?.t5 ?: return null
        val vocab = candidate?.localPath?.let { ai.ondevice.data.hf.LocalGguf.vocabSize(it) }
        val kind = ai.ondevice.core.DiffusionFamily.T5Kind.of(vocab) ?: return null
        if (kind == wanted) return null
        return "This is ${kind.label}; ${arch ?: "this model"} reads ${wanted.label}. They share " +
            "the slot and not the vocabulary, so this loads and conditions on the wrong tokens."
    }

    /**
     * Whether this family has any use for this slot.
     *
     * [DiffusionFamily] describes two things and only two: how a family encodes
     * its prompt, and whether its decoder ships separately. So those are the
     * two questions it is asked. Everything else — a LoRA, an upscaler, the
     * second half of a split denoiser — it says nothing about, and silence is
     * not a refusal: the manifest's `appliesTo` already gates those per
     * architecture, which is how the still screen has always decided.
     *
     * Reading silence as "no" is what this used to do, by taking the family's
     * encoder keys as the whole set of usable roles. Nothing outside that set
     * could ever be armed, so a LoRA on a clip showed as "n/a · Not used by
     * wan" and was never passed to the loader — including the step-distilled
     * LoRAs that are the only thing making a clip on this hardware practical.
     * Wan 2.2's high-noise denoiser was refused the same way, which is half a
     * model dropped in silence.
     *
     * An unrecognised architecture says yes to everything the manifest offers,
     * because refusing what we cannot describe is worse than the guess.
     */
    private fun familyReads(
        family: ai.ondevice.core.DiffusionFamily?,
        role: AttachmentRole,
    ): Boolean {
        if (family == null) return true
        return when (role.family) {
            ai.ondevice.core.RoleFamily.PROMPT_ENCODER ->
                role.paramKey in family.encoders || role.paramKey in family.optionalEncoders
            // A checkpoint that carries its own decoder does not take one.
            // The audio decoder is a separate latent space and a separate
            // question, left to the manifest.
            ai.ondevice.core.RoleFamily.DECODER ->
                role != AttachmentRole.VAE || family.vaeSeparate
            else -> true
        }
    }

    /**
     * Whether this file is the right *kind* for the slot, where the role alone
     * does not say — which is T5 and only T5. See DiffusionFamily.T5Kind.
     */
    private fun suits(
        family: ai.ondevice.core.DiffusionFamily?,
        role: AttachmentRole,
        candidate: ModelEntity,
    ): Boolean {
        if (role != AttachmentRole.T5XXL) return true
        val wanted = family?.t5 ?: return true
        val vocab = ai.ondevice.data.hf.LocalGguf.vocabSize(candidate.localPath)
        val kind = ai.ondevice.core.DiffusionFamily.T5Kind.of(vocab) ?: return true
        return kind == wanted
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
            // A new model brings its own settings, so this session's edits stop
            // being an answer to anything.
            touched = emptySet(),
        )
        runScope.launch {
            // The new model's own settings, not the last one's left on screen.
            seedFrom(model)
            db.models().touch(model.id, System.currentTimeMillis())
        }
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
    /**
     * Free the weights, stopping the run if one is going.
     *
     * Off the main thread, because `nativeFree` is not a quick call. It cancels
     * the run, waits on its mutex until the native generate returns, and only
     * then deletes the context — which is the correct order, and the reason it
     * blocks for as long as the run takes to notice the cancel. Called from the
     * UI thread, as this was, that block is an ANR: input dispatch times out at
     * five seconds and the system kills the app. It reads as a crash, and the
     * system log calls it one --
     *
     *     ANR in ai.ondevice ... Waited 5000ms for MotionEvent
     *     Killing 27286:ai.ondevice ... user request after error
     *
     * -- but nothing had gone wrong in native code at all. The only fault was
     * where the wait happened.
     */
    fun unloadModel() {
        if (_state.value.unloading) return
        _state.value = _state.value.copy(unloading = true)
        runScope.launch {
            kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                diffusion.unload("you asked for the memory back")
            }
            _state.value = _state.value.copy(
                residentComponents = emptyList(),
                recognisedAs = null,
                supportsVideo = false,
                unloading = false,
            )
        }
    }

    fun setPrompt(value: String) = update { copy(prompt = value) }
    fun setNegativePrompt(value: String) = update { copy(negativePrompt = value) }
    fun setFrames(value: Int) =
        update { copy(frames = value.coerceIn(1, 129), touched = touched + "video_frames") }
    fun setFps(value: Int) = update { copy(fps = value.coerceIn(1, 60), touched = touched + "fps") }
    fun setSteps(value: Int) =
        update { copy(steps = value.coerceIn(1, 60), touched = touched + "steps") }
    fun setCfg(value: Float) = update { copy(cfgScale = value, touched = touched + "cfg_scale") }
    /**
     * Width and height apart, because a clip model has an aspect ratio.
     *
     * One square slider was the control, capped at 768, so the only shapes it
     * could ask for were squares — and Wan 2.2 TI2V is trained at 1280x704 and
     * 704x1280. Every clip this app has made has been off-distribution in both
     * dimensions at once, which is most of why they come out as texture rather
     * than as scenes.
     */
    fun setWidth(value: Int) = update { copy(width = value, touched = touched + "width") }

    fun setHeight(value: Int) = update { copy(height = value, touched = touched + "height") }

    /** Both at once, for the callers that want a square. */
    fun setSize(value: Int) =
        update { copy(width = value, height = value, touched = touched + "width" + "height") }
    fun setSeed(value: Long) = update { copy(seed = value) }
    fun setVaeTiling(value: Boolean) =
        update { copy(vaeTiling = value, touched = touched + "vae_tiling") }
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

        generationJob = runScope.launch {
            val started = System.currentTimeMillis()
            // What a clip costs, sampled while it runs.
            //
            // The image tab has had this since the upscaler ran with no graph;
            // video is the heavier of the two by a wide margin — a clip is the
            // same denoiser over N frames — and it was the one screen with no
            // way to see what the phone was doing.
            // Keep the process alive for the length of the run.
            //
            // A clip takes tens of minutes and this app holds gigabytes, which
            // makes it the first thing Android reclaims once it leaves the
            // screen — so switching to another app killed the run and there was
            // nothing in the shade to say so. Only the conversation ever held
            // this; every other kind of generation ran unprotected.
            ai.ondevice.engine.InferenceService.holdWakeLock(context)
            val recording = recorder.start(runScope)
            val liveJob = runScope.launch {
                recording.live.collect { trace ->
                    // The buffers come along with the trace because they arrive
                    // during the run, not at the end of the load: the decoder
                    // reserves nothing until the decode. Read at the sampling
                    // rate, which is slow enough to be free and often enough
                    // that the decode's figures appear while the decode is
                    // still the thing on screen.
                    _state.value = _state.value.copy(
                        liveTrace = trace,
                        runtimeBuffers = diffusion.buffers,
                    )
                }
            }
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
                    bareDenoiser = diffusion.bareDiffusion,
                    supportsVideo = diffusion.supportsVideo,
                    // Asked of the checkpoint that is loaded, because the Wan
                    // variants disagree and the one that cannot use a first
                    // frame discards it without saying so.
                    supportsStartFrame = diffusion.supportsStartFrame,
                    supportsEndFrame = diffusion.supportsEndFrame,
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
                        // Stored first, the sheet on top — see the still
                        // screen's copy. Flow shift is the one that bit here:
                        // saved against the model, shown as saved, and never
                        // sent, because this screen has no control for it.
                        params = SparseParams.parse(model.paramOverridesJson)
                            .overlaidWith(currentParams(seed)),
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
                                // The tiler's or the loader's count first,
                                // then the runtime's sentence: a phase
                                // with sub-progress leads with it.
                                runStage = listOfNotNull(event.detail, event.stage)
                                    .joinToString(" · ").takeIf { it.isNotBlank() },
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
                            val clipId = java.util.UUID.randomUUID().toString()
                            db.clips().upsert(
                                ai.ondevice.data.db.GeneratedClipEntity(
                                    id = clipId,
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
                            // The same record the image tab writes, so a clip's
                            // cost is on its library page rather than only on
                            // this screen until the next run clears it.
                            db.predictionRuns().record(
                                kind = ai.ondevice.core.PredictionKind.VIDEO,
                                artifactId = clipId,
                                modelId = model.id,
                                startedAt = started,
                                trace = recording.stop(),
                                stats = SparseParams.of(
                                    "steps" to _state.value.steps,
                                    "frames" to _state.value.frames,
                                    "seconds_per_step" to _state.value.secondsPerStep,
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
                ai.ondevice.engine.InferenceService.releaseWakeLock(context)
                diffusion.cancel()
                liveJob.cancel()
                // Idempotent: a completed run already stopped it, a cancelled
                // one never reached that point.
                val trace = recording.stop()
                _state.value = _state.value.copy(
                    generating = false,
                    cancelling = false,
                    loadingModel = false,
                    previewBitmap = null,
                    liveTrace = null,
                    // Read once more at the end: the decode's reservations are
                    // made last, and a run that finished quickly could stop
                    // before any live sample caught them.
                    runtimeBuffers = diffusion.buffers,
                    // Kept for a cancelled run too. What the phone was doing
                    // for the ninety seconds before you gave up is the most
                    // useful thing on the screen at that point.
                    lastTrace = trace.takeIf { !it.isEmpty },
                    elapsedMillis = System.currentTimeMillis() - started,
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
        playbackJob = runScope.launch {
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
        runScope.launch {
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

    /**
     * The screen going away is not a reason to stop anything.
     *
     * This used to stop playback, which was right when the state died with it
     * and is wrong now: the frames are still there, the run may still be going,
     * and the next screen should find both where they were left. A generation
     * ends when it finishes, fails, or somebody presses Cancel.
     */
    override fun onCleared() {
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

        /**
         * The parts of a run that are plumbing rather than authorship, and so
         * may be filled in on the user's behalf where exactly one file fits.
         *
         * The still screen's list plus companion denoisers. A LoRA changes what
         * comes out and stays unchosen until somebody says otherwise; Wan 2.2's
         * high-noise half changes nothing — it *is* the model, published in two
         * files — and this is the only screen that can run one.
         */
        val ADOPTABLE_FAMILIES = setOf(
            ai.ondevice.core.RoleFamily.PROMPT_ENCODER,
            ai.ondevice.core.RoleFamily.DECODER,
            ai.ondevice.core.RoleFamily.POST,
            ai.ondevice.core.RoleFamily.COMPANION_DENOISER,
        )
    }
}

/** S14 — one clip, and everything asked of the runtime to get it. */
data class VideoState(
    val models: List<ModelEntity> = emptyList(),
    val model: ModelEntity? = null,
    /** Downloads in flight, so an arriving part is not reported as an absent one. */
    val installing: List<ai.ondevice.data.db.InstallingModel> = emptyList(),
    /** Every role the library can fill, whichever model it belongs to. */
    val installedRoles: Set<AttachmentRole> = emptySet(),
    /** Roles with a file installed that fits, and no file chosen. */
    val unchosenRoles: List<AttachmentRole> = emptyList(),
    /** The loader's answer after a load, null before one — see the image screen. */
    val bareDenoiser: Boolean? = null,
    /** Sampled while the clip renders, and kept after it. */
    val liveTrace: ai.ondevice.engine.ResourceTrace? = null,
    val lastTrace: ai.ondevice.engine.ResourceTrace? = null,
    val elapsedMillis: Long = 0,
    val runtimeInstalled: Boolean = false,
    /**
     * Something to press Generate on, the way the still screen has one.
     *
     * A clip is described by its motion as much as its subject — the help
     * under this field says so and the screen then opened empty, leaving the
     * one instruction that matters unillustrated. So the default names a
     * subject, a movement and a camera, which is the shape a prompt for this
     * model wants.
     */
    val prompt: String = "a hot-air balloon rising through morning mist over a green valley, " +
        "camera tilting slowly up to follow it",
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
    /** The runtime's own working-memory reservations — see RuntimeBuffer. */
    val runtimeBuffers: List<ai.ondevice.engine.RuntimeBuffer> = emptyList(),
    /** See ImageState.unloading — a free is not instant and must say so. */
    val unloading: Boolean = false,
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
    /**
     * Whether the loaded checkpoint reads a supplied first — or last — frame.
     *
     * False until something is loaded, so neither control is offered before
     * there is a model whose answer is known. Offering a picker that the run
     * will ignore is worse than not offering it: the clip comes back looking
     * merely disappointing, with nothing to say the picture was dropped.
     */
    val supportsStartFrame: Boolean = true,
    val supportsEndFrame: Boolean = true,
    /**
     * Settings the person has changed by hand, by their parameter key.
     *
     * A re-seed from the model's stored parameters skips these. Without it,
     * anything that re-seeds — and returning from All Parameters does —
     * silently reverted a slider to whatever the model said, with no message
     * and no way to tell until the run came back the wrong size.
     *
     * Cleared when a different model is chosen, because the settings then
     * belong to that model rather than to this session.
     */
    val touched: Set<String> = emptySet(),
    val clip: DiffusionClip? = null,
    val frameIndex: Int = 0,
    val playing: Boolean = false,
    val loraOutcome: List<String> = emptyList(),
    val error: String? = null,
    val errorHint: String? = null,
) {
    /** What the runtime is doing, apart from what this screen still needs. */
    val runPhase: ai.ondevice.core.RunPhase
        get() = ai.ondevice.core.runPhaseOf(
            stopping = unloading || cancelling,
            loading = loadingModel,
            running = generating,
        )

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
                installedRoles = installedRoles,
                arrivingRoles = arrivingRoles,
                bareDenoiser = bareDenoiser,
            )
        }

    /** Seconds left at the rate the last step took, or 0 before there is one. */
    val etaSeconds: Long
        get() = if (secondsPerStep > 0f && progressSteps > step) {
            ((progressSteps - step) * secondsPerStep).toLong()
        } else {
            0L
        }

    /** The slots a download is on its way to filling. */
    val arrivingRoles: Set<ai.ondevice.core.AttachmentRole>
        get() = installing.mapNotNull { it.attachmentRole }.toSet()
}
