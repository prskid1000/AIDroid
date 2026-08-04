package ai.ondevice.engine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.ModelAttachment
import ai.ondevice.core.SparseParams
import ai.ondevice.core.VideoConditioning
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Diffusion, for real (SPEC §5). */
class DiffusionEngine(
    private val context: Context,
    private val capabilities: ai.ondevice.data.hf.DeviceCapabilities,
) {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    @Volatile
    private var handle: Long = 0L

    @Volatile
    var loadedModelId: String? = null
        private set

    /**
     * One load at a time, whatever the screens ask for.
     *
     * `nativeLoad` is a blocking JNI call that runs for minutes and cannot be
     * interrupted — cancelling the coroutine waiting on it does not reach it.
     * So a Cancel during a load left the call running with no handle recorded,
     * `isCurrent` answering false, and the next Generate starting a *second*
     * `new_sd_ctx` beside the first. Two multi-gigabyte allocations at once is
     * not an error ggml returns; it is an abort, and the process goes with it.
     * That is the crash, and it lands on the press *after* the cancel, which is
     * why it read as "cancelling mid-load crashes the app".
     */
    private val loadLock = kotlinx.coroutines.sync.Mutex()

    /**
     * A Cancel pressed while the loader was running, waiting for it to stop.
     *
     * There is no cancelling `new_sd_ctx` — upstream has no flag to set and no
     * callback to refuse from, and the time is spent inside file reads and
     * allocations rather than in a graph. What the app can honour is the
     * *intent*: let the load finish and drop the weights the moment it does,
     * rather than seating gigabytes nobody asked to keep.
     */
    @Volatile
    private var loadCancelled: Boolean = false

    /** True while `nativeLoad` is running and has not yet returned a handle. */
    @Volatile
    var loading: Boolean = false
        private set

    val available: Boolean get() = SdBridge.available
    val isLoaded: Boolean get() = handle != 0L

    fun isCurrent(modelId: String): Boolean = isLoaded && loadedModelId == modelId

    /**
     * @param attachments every component the caller knows about, **armed or
     *   not**. Passing only the armed ones made switching one off do nothing:
     *   with no enabled attachment for the role, the lookup below fell through
     *   to the stored path and loaded it anyway. Off has to mean off, or the
     *   switch is decoration — and it is the switch the error message tells
     *   people to use when a component is what broke the load.
     */
    suspend fun load(
        modelId: String,
        modelPath: String,
        attachments: List<ModelAttachment> = emptyList(),
        threads: Int = capabilities.inferenceThreads,
        params: ai.ondevice.core.SparseParams = ai.ondevice.core.SparseParams.EMPTY,
    ): Result<Unit> = withContext(Dispatchers.IO) {
        // Held across the whole call, including `unload`: a free racing another
        // thread's load is the same fault seen from the other side.
        //
        // `withLock` is not cancellable-safe to skip — a caller cancelled while
        // queued here never enters, which is the right answer for a second
        // Generate pressed during a load nobody wants any more.
        loadLock.withLock {
            loadCancelled = false
            loading = true
            try {
                loadLocked(modelId, modelPath, attachments, threads, params)
            } finally {
                loading = false
            }
        }
    }

    private fun loadLocked(
        modelId: String,
        modelPath: String,
        attachments: List<ModelAttachment>,
        threads: Int,
        params: ai.ondevice.core.SparseParams,
    ): Result<Unit> {
        return runCatching {
            check(SdBridge.available) {
                SdBridge.loadError ?: "The stable-diffusion.cpp runtime is not installed in this build."
            }
            unload()

            // These are load-time in sd.cpp — they change the context, not the
            // run — so they are resolved here and the rest at generate time.
            //
            // The stored parameter is the fallback for a role the caller said
            // nothing about, which is every run started somewhere with no
            // component sheet. Where the caller did mention the role, the
            // caller's arming decides and the stored path does not override it.
            fun pathFor(role: AttachmentRole): String {
                val known = attachments.filter { it.role == role }
                if (known.isNotEmpty()) return known.firstOrNull { it.enabled }?.path.orEmpty()
                return params.string(role.paramKey).orEmpty()
            }

            // Every load-time component, named the way the manifest names it.
            //
            // Derived from LOAD_TIME_ROLES rather than listed again here: the
            // two lists were the same list written twice, and the second one
            // was what decided whether a component reached the loader at all.
            val components = buildJsonObject {
                LOAD_TIME_ROLES.forEach { role ->
                    pathFor(role).takeIf { it.isNotBlank() }?.let { put(role.paramKey, it) }
                }
            }

            // The rest of sd_ctx_params_t, keyed by its own field names.
            //
            // Only what the caller actually set is sent, so upstream's default
            // stands for everything else. `wtype` blank keeps the file's own
            // precision, which is what most loads should still do — the
            // conversion runs on the way in, and costs load time and a working
            // buffer that have not been measured on hardware.
            val settings = buildJsonObject {
                LOAD_SETTING_KEYS.forEach { key ->
                    params[key]?.let { put(key, it) }
                }
            }
            val weightType = params.string(WEIGHT_TYPE_KEY).orEmpty().trim()

            val newHandle = SdBridge.nativeLoad(
                modelPath = modelPath,
                componentsJson = components.toString(),
                settingsJson = settings.toString(),
                threads = threads,
            )
            check(newHandle != 0L) { "The runtime returned no handle for $modelPath." }
            // A Cancel that arrived while the loader was working. The weights
            // are on the heap by now — there was no interrupting the call — so
            // honouring the press means giving them straight back rather than
            // seating gigabytes for a run nobody is waiting for.
            if (loadCancelled) {
                SdBridge.nativeFree(newHandle)
                loadCancelled = false
                android.util.Log.i(TAG, "load of ${File(modelPath).name} finished after Cancel; freed")
                throw LoadCancelled()
            }
            handle = newHandle
            // Only the roles just handed to the loader.
            //
            // This used to walk every role there is, which meant a stored
            // upscaler or LoRA appeared in the list of what the context was
            // holding. Neither is a load-time part — `sd_ctx_params_t` has no
            // field for either, and sd.cpp applies a LoRA at generate time and
            // runs an upscaler in a context of its own — so the screen was
            // naming two files as resident that the loader had never been told
            // about.
            // What the loader took, in its own words — not what was sent to it.
            //
            // These differ more often than they look like they would: FLUX.2
            // Klein carries no encoders and SDXL sometimes carries all of them,
            // so a checkpoint can silently decline a file that was passed. The
            // card says "In memory", which is a claim about memory, and only
            // the runtime can make it.
            val taken = loadedComponents()
            residentComponents = taken.mapNotNull { (role, path) ->
                role?.let {
                    // Its size too. The checkpoint's was on screen and the
                    // components' were not, so a 2.4 GB encoder and a 168 MB
                    // decoder read as the same weight — and the encoder is
                    // usually the reason a bundle does not fit.
                    ResidentPart(it, File(path).name, File(path).length())
                }
            }
            residentModel = File(modelPath).let { "${it.name} · ${formatBytes(it.length())}" }
            residentBytes = File(modelPath).length() +
                taken.sumOf { (_, path) -> File(path).length().takeIf { it > 0 } ?: 0L }
            lastUnloadReason = null
            android.util.Log.i(
                TAG,
                "loaded ${File(modelPath).name} (${File(modelPath).length() / 1024 / 1024} MB) " +
                    "threads=$threads " +
                    "wtype=${weightType.ifBlank { "as stored" }} " +
                    "attached=" + residentComponents.map { it.role.name }
                    .ifEmpty { listOf("none") }.joinToString("+"),
            )
            loadedModelId = modelId
            // What this context can make, asked of it rather than inferred from
            // the architecture's name. An SD 1.x with a motion module answers
            // yes to both, and no table of names could know that — it depends
            // on whether a file was attached.
            supportsImage = SdBridge.nativeSupportsImage(newHandle)
            supportsVideo = SdBridge.nativeSupportsVideo(newHandle)
            // Which of the two supplied frames this particular checkpoint will
            // read. Asked per checkpoint and not per family, because the Wan
            // variants disagree: a T2V one drops a first frame in silence.
            val desc = SdBridge.nativeModelDesc(newHandle).takeIf { it.isNotBlank() }
            modelDesc = desc
            supportsStartFrame = supportsVideo && VideoConditioning.supportsStartFrame(desc)
            supportsEndFrame = supportsVideo && VideoConditioning.supportsEndFrame(desc)
            // What the loader decided this checkpoint is, now that it has read
            // the tensors. Nothing else in the app can know it as reliably.
            detectedVersion = SdBridge.nativeDetectedVersion().takeIf { it.isNotBlank() }
            bareDiffusion = SdBridge.nativeIsBareDiffusion()
            detectedVersion?.let {
                android.util.Log.i(TAG, "recognised as $it" + if (bareDiffusion) " (denoiser only)" else "")
            }
            Unit
        }
    }

    /**
     * stable-diffusion.cpp's own name for the loaded checkpoint — "Flux.2
     * klein", "SDXL" — or null when nothing is loaded.
     */
    @Volatile
    var detectedVersion: String? = null
        private set

    /**
     * Whether the loaded file is the denoiser on its own.
     *
     * A full checkpoint carries its text encoders and its VAE and needs nothing
     * supplied; a quantised release carries neither. Which one a file is cannot
     * be read off its architecture — SDXL ships both ways — so this is the
     * loader's finding, not a guess.
     */
    @Volatile
    var bareDiffusion: Boolean = false
        private set

    /**
     * What the loaded context is holding, as role → the file's own name.
     *
     * Empty when nothing is loaded. The screen reads this to say what is
     * resident rather than what is merely chosen — the two differ every time a
     * component is switched off, and differed silently until now.
     */
    @Volatile
    var residentComponents: List<ResidentPart> = emptyList()
        private set

    /**
     * The checkpoint itself, as its file name and size on disk.
     *
     * Separate from [residentComponents] because it is not a component, and
     * absent from it for exactly that reason — which meant that a
     * self-contained model holding no attachments produced an empty list, and
     * the screen's "In memory" card, which renders only when the list is not
     * empty, disappeared. SDXL turbo is the case: 3.94 GB resident, nothing
     * attached, nothing on screen saying so. The one model that needs no help
     * was the one the app went quiet about.
     */
    @Volatile
    var residentModel: String? = null
        private set

    /**
     * Roughly what is resident, in bytes on disk.
     *
     * The weights dominate a diffusion context by so much that their size on
     * disk is the honest answer to "what is this costing" — nearer than any
     * figure the app could get from the runtime, which reports none.
     * Understates by the working buffers and overstates by anything mmap has
     * not faulted in, and is labelled as approximate for both reasons.
     */
    @Volatile
    var residentBytes: Long = 0L
        private set

    /**
     * What the loaded context can generate, as the runtime answered.
     *
     * Both are false when nothing is loaded. They are not opposites: an SD 1.x
     * with a motion module attached can do either, and which architectures
     * those are is upstream's `sd_version_supports_animatediff`, not a list
     * kept here.
     */
    @Volatile
    var supportsImage: Boolean = false
        private set

    @Volatile
    var supportsVideo: Boolean = false
        private set

    /** What the loaded denoiser called itself, or null when it never said. */
    @Volatile
    var modelDesc: String? = null
        private set

    /**
     * Whether a supplied first — or last — frame is read by this checkpoint.
     *
     * Two flags rather than one: the plain Wan I2V checkpoints take a start
     * frame and have nowhere to put an end one, so offering both would still
     * be offering something that does nothing.
     */
    @Volatile
    var supportsStartFrame: Boolean = true
        private set

    @Volatile
    var supportsEndFrame: Boolean = true
        private set

    /**
     * Why the last context was dropped, when it was dropped for a reason.
     *
     * Unloading is mostly invisible and mostly should be — but not when the app
     * does it on the user's behalf. Running the upscaler drops the denoiser to
     * make room, so the next generate reloads for minutes with nothing having
     * explained why. Null after a load, and after an unload nobody needs told
     * about.
     */
    @Volatile
    var lastUnloadReason: String? = null
        private set

    /**
     * How much of each LoRA the last run applied, straight from sd.cpp's tally.
     *
     * Counting is upstream's, not the app's. A gate written here would need a
     * table of which LoRA suits which architecture, kept by hand and wrong the
     * week a new one ships; the runtime has already matched every tensor by
     * name and knows the answer exactly.
     */
    private fun loraReport(): List<LoraApplication> = runCatching {
        (json.parseToJsonElement(SdBridge.nativeLoraReport()) as kotlinx.serialization.json.JsonArray)
            .map { it.jsonObject }
            .map {
                LoraApplication(
                    file = it["file"]?.jsonPrimitive?.content.orEmpty(),
                    applied = it["applied"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0,
                    total = it["total"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0,
                )
            }
    }.getOrDefault(emptyList())

    /**
     * The components the loader reported taking, as role → path.
     *
     * A role the app has no name for is kept with a null role rather than
     * dropped, so its bytes still count toward what is resident — the runtime
     * loading something this app cannot name is a fact about memory either way.
     */
    private fun loadedComponents(): List<Pair<AttachmentRole?, String>> = runCatching {
        (json.parseToJsonElement(SdBridge.nativeLoadedComponents()) as kotlinx.serialization.json.JsonArray)
            .map { it.jsonObject }
            .mapNotNull { row ->
                val path = row["path"]?.jsonPrimitive?.content.orEmpty()
                if (path.isBlank()) return@mapNotNull null
                val name = row["role"]?.jsonPrimitive?.content
                AttachmentRole.entries.firstOrNull { it.name == name } to path
            }
    }.getOrDefault(emptyList())

    /**
     * The runtime's working-memory reservations, one line per module.
     *
     * Read fresh rather than cached: the decode's buffers are not reserved
     * until the decode, so a value captured at load time would report half the
     * run and call it the run.
     */
    val buffers: List<RuntimeBuffer>
        get() = runCatching {
            (json.parseToJsonElement(SdBridge.nativeBuffers()) as kotlinx.serialization.json.JsonArray)
                .map { it.jsonObject }
                .mapNotNull { row ->
                    val what = row["what"]?.jsonPrimitive?.content.orEmpty().trim()
                    if (what.isBlank()) return@mapNotNull null
                    RuntimeBuffer(
                        what = what,
                        computeMb = row["computeMb"]?.jsonPrimitive?.content?.toDoubleOrNull() ?: 0.0,
                        cacheMb = row["cacheMb"]?.jsonPrimitive?.content?.toDoubleOrNull() ?: 0.0,
                    )
                }
                .filter { it.computeMb > 0.0 || it.cacheMb > 0.0 }
        }.getOrDefault(emptyList())

    /** The loader's own account of where it has got to, or null between loads. */
    val loadStage: String?
        get() = if (SdBridge.available) SdBridge.nativeLoadStage().takeIf { it.isNotBlank() } else null

    /** @param because what to tell the user, when the app unloaded on their behalf. */
    fun unload(because: String? = null) {
        if (handle != 0L) {
            android.util.Log.i(
                TAG,
                "unloading " + (residentModel ?: loadedModelId ?: "?") +
                    residentComponents.joinToString("") { " -${it.role.name}" } +
                    " (freeing ~${formatBytes(residentBytes)})" +
                    (because?.let { ": $it" } ?: ""),
            )
            SdBridge.nativeFree(handle)
            handle = 0L
            lastUnloadReason = because
        }
        residentComponents = emptyList()
        residentModel = null
        residentBytes = 0L
        loadedModelId = null
        supportsImage = false
        supportsVideo = false
        modelDesc = null
        supportsStartFrame = true
        supportsEndFrame = true
    }

    fun applyParams(params: SparseParams): ParamReport {
        if (handle == 0L) return ParamReport(rejected = params.keys.toList())
        val report = json.parseToJsonElement(
            SdBridge.nativeApplyParams(handle, params.toJsonString()),
        ).jsonObject
        return ParamReport(
            applied = report["applied"]?.let { el ->
                (el as? kotlinx.serialization.json.JsonArray)?.map { it.jsonPrimitive.content }
            } ?: emptyList(),
            rejected = report["rejected"]?.let { el ->
                (el as? kotlinx.serialization.json.JsonArray)?.map { it.jsonPrimitive.content }
            } ?: emptyList(),
        )
    }

    /**
     * Stop what is running, whichever of the two things that is.
     *
     * During a run this reaches the graph callback and the next ggml node
     * declines to execute. During a load there is nothing to reach — upstream
     * offers no cancel for `new_sd_ctx` — so the press is remembered and the
     * weights are freed the moment the call returns.
     */
    fun cancel() {
        if (loading) loadCancelled = true
        if (handle != 0L) SdBridge.nativeCancel(handle)
    }

    /** Generate, emitting progress and live previews while it runs. */
    fun generate(request: DiffusionRequest): Flow<DiffusionEvent> = channelFlow {
        if (handle == 0L) {
            send(DiffusionEvent.Failed("No diffusion model is loaded.", "Models → Add an SD or SDXL repo."))
            return@channelFlow
        }

        val report = applyParams(request.params)
        android.util.Log.i(
            TAG,
            "params applied=${report.applied.size} rejected=${
                report.rejected.ifEmpty { listOf("none") }.joinToString(",")
            }",
        )

        val init = request.initImageUri?.let { decodeRgb(it) }
        val reference = request.referenceImageUri?.let { decodeRgb(it) }
        val control = request.controlImageUri?.let { decodeRgb(it) }
        val style = request.styleImageUri?.let { decodeRgb(it) }
        val identity = request.identityImageUri?.let { decodeRgb(it) }
        val mask = request.maskPngPath?.let { decodeRgbFromFile(it) }

        // Attachments the *runtime* takes per-run, as a role-tagged list.
        val perRun = listOf(AttachmentRole.LORA, AttachmentRole.CONTROLNET)
        val ticked = request.attachments.filter { it.enabled && it.role in perRun }
        val attachmentsJson = buildJsonArray {
            ticked.forEach { attachment ->
                add(
                    buildJsonObject {
                        put("role", attachment.role.name)
                        put("path", attachment.path)
                        put("weight", attachment.weight)
                    },
                )
            }
            // What is chosen but was never armed on the Image screen — which is
            // every run started from anywhere else. A LoRA key holds a stack,
            // so this is one entry per file rather than one per role.
            perRun
                .filterNot { role -> ticked.any { it.role == role } }
                .forEach { role ->
                    ai.ondevice.core.WeightedPaths.parse(request.params[role.paramKey])
                        .forEach { chosen ->
                            add(
                                buildJsonObject {
                                    put("role", role.name)
                                    put("path", chosen.path)
                                    put("weight", chosen.weight)
                                },
                            )
                        }
                }
        }.toString()

        var done = false
        val started = System.currentTimeMillis()
        val worker = launch(Dispatchers.Default) {
            try {
                val bytes = SdBridge.nativeGenerate(
                    handle,
                    init?.pixels, init?.width ?: 0, init?.height ?: 0,
                    mask?.pixels, mask?.width ?: 0, mask?.height ?: 0,
                    control?.pixels, control?.width ?: 0, control?.height ?: 0,
                    reference?.pixels, reference?.width ?: 0, reference?.height ?: 0,
                    style?.pixels, style?.width ?: 0, style?.height ?: 0,
                    identity?.pixels, identity?.width ?: 0, identity?.height ?: 0,
                    attachmentsJson,
                )
                if (bytes == null) {
                    android.util.Log.e(TAG, "the run returned no pixels")
                    send(DiffusionEvent.Failed("The run produced no image.", null))
                } else {
                    val image = unpack(bytes)
                    android.util.Log.i(
                        TAG,
                        "generated ${image.summary()} in " +
                            "${(System.currentTimeMillis() - started) / 1000f}s",
                    )
                    loraReport().forEach {
                        android.util.Log.i(TAG, "lora ${it.file}: ${it.applied}/${it.total} applied")
                    }
                    send(DiffusionEvent.Completed(image, loraReport()))
                }
            } catch (t: Throwable) {
                android.util.Log.e(TAG, "generation failed", t)
                send(
                    DiffusionEvent.Failed(
                        t.message ?: "The diffusion run failed.",
                        "Lower the size, reduce steps, or turn on vae_tiling.",
                    ),
                )
            } finally {
                done = true
            }
        }

        var lastPreviewSerial = -1
        while (isActive && !done) {
            delay(POLL_MILLIS)
            val progress = runCatching {
                json.parseToJsonElement(SdBridge.nativeProgress(handle)).jsonObject
            }.getOrNull() ?: continue

            val step = progress["step"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0
            val steps = progress["steps"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0
            val secondsPerStep = progress["secondsPerStep"]?.jsonPrimitive?.content?.toFloatOrNull() ?: 0f
            val phase = when (progress["phase"]?.jsonPrimitive?.content) {
                "sampling" -> DiffusionPhase.SAMPLING
                "decoding" -> DiffusionPhase.DECODING
                else -> DiffusionPhase.PREPARING
            }
            val stage = progress["stage"]?.jsonPrimitive?.content?.takeIf { it.isNotBlank() }
            send(DiffusionEvent.Progress(step, steps, secondsPerStep, phase, stage))

            val serial = progress["previewSerial"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0
            if (serial != lastPreviewSerial) {
                lastPreviewSerial = serial
                SdBridge.nativePreview(handle)?.let { send(DiffusionEvent.Preview(unpack(it))) }
            }
        }
        worker.join()
    }

    /**
     * Generate a clip, emitting the same progress and previews a still does.
     *
     * The frames land on disk and the flow carries their paths. Everything
     * about the run — sampler, schedule, size, LoRAs — is the image path's,
     * because upstream's two parameter structs agree about all of it; what
     * differs is a handful of fields and one absence, and the absence is
     * IP-Adapter, which `sd_vid_gen_params_t` has no room for at all.
     */
    fun generateVideo(request: VideoRequest): Flow<DiffusionEvent> = channelFlow {
        if (handle == 0L) {
            send(DiffusionEvent.Failed("No model is loaded.", "Models → pick a video model."))
            return@channelFlow
        }
        if (!supportsVideo) {
            send(
                DiffusionEvent.Failed(
                    "This model does not generate video.",
                    "An SD 1.x checkpoint can, once a motion module is attached under Components.",
                ),
            )
            return@channelFlow
        }

        val report = applyParams(request.params)
        android.util.Log.i(
            TAG,
            "video params applied=${report.applied.size} rejected=${
                report.rejected.ifEmpty { listOf("none") }.joinToString(",")
            }",
        )

        val init = request.initImageUri?.let { decodeRgb(it) }
        val end = request.endImageUri?.let { decodeRgb(it) }
        val control = request.controlImageUri?.let { decodeRgb(it) }

        val perRun = listOf(AttachmentRole.LORA, AttachmentRole.CONTROLNET)
        val ticked = request.attachments.filter { it.enabled && it.role in perRun }
        val attachmentsJson = buildJsonArray {
            ticked.forEach { attachment ->
                add(
                    buildJsonObject {
                        put("role", attachment.role.name)
                        put("path", attachment.path)
                        put("weight", attachment.weight)
                    },
                )
            }
            perRun
                .filterNot { role -> ticked.any { it.role == role } }
                .forEach { role ->
                    ai.ondevice.core.WeightedPaths.parse(request.params[role.paramKey])
                        .forEach { chosen ->
                            add(
                                buildJsonObject {
                                    put("role", role.name)
                                    put("path", chosen.path)
                                    put("weight", chosen.weight)
                                },
                            )
                        }
                }
        }.toString()

        // A directory per run, so a cancelled or failed clip leaves its own
        // mess and not a mixture of two.
        val outputDir = File(context.filesDir, "video/${System.currentTimeMillis()}")
        outputDir.mkdirs()

        var done = false
        val started = System.currentTimeMillis()
        val worker = launch(Dispatchers.Default) {
            try {
                val manifest = SdBridge.nativeGenerateVideo(
                    handle,
                    init?.pixels, init?.width ?: 0, init?.height ?: 0,
                    end?.pixels, end?.width ?: 0, end?.height ?: 0,
                    control?.pixels, control?.width ?: 0, control?.height ?: 0,
                    attachmentsJson,
                    outputDir.absolutePath,
                )
                if (manifest == null) {
                    outputDir.deleteRecursively()
                    send(DiffusionEvent.Failed("The run produced no frames.", null))
                } else {
                    val clip = parseClip(manifest)
                    android.util.Log.i(
                        TAG,
                        "generated ${clip.frames.size} frames at ${clip.width}x${clip.height} in " +
                            "${(System.currentTimeMillis() - started) / 1000f}s" +
                            (clip.audioPath?.let { " with audio" } ?: ""),
                    )
                    loraReport().forEach {
                        android.util.Log.i(TAG, "lora ${it.file}: ${it.applied}/${it.total} applied")
                    }
                    send(DiffusionEvent.ClipCompleted(clip, loraReport()))
                }
            } catch (t: Throwable) {
                outputDir.deleteRecursively()
                android.util.Log.e(TAG, "video generation failed", t)
                send(
                    DiffusionEvent.Failed(
                        t.message ?: "The run failed.",
                        "Fewer frames, a smaller size, or vae_tiling.",
                    ),
                )
            } finally {
                done = true
            }
        }

        var lastPreviewSerial = -1
        while (isActive && !done) {
            delay(POLL_MILLIS)
            val progress = runCatching {
                json.parseToJsonElement(SdBridge.nativeProgress(handle)).jsonObject
            }.getOrNull() ?: continue

            val step = progress["step"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0
            val steps = progress["steps"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0
            val secondsPerStep = progress["secondsPerStep"]?.jsonPrimitive?.content?.toFloatOrNull() ?: 0f
            val phase = when (progress["phase"]?.jsonPrimitive?.content) {
                "sampling" -> DiffusionPhase.SAMPLING
                "decoding" -> DiffusionPhase.DECODING
                else -> DiffusionPhase.PREPARING
            }
            val stage = progress["stage"]?.jsonPrimitive?.content?.takeIf { it.isNotBlank() }
            send(DiffusionEvent.Progress(step, steps, secondsPerStep, phase, stage))

            val serial = progress["previewSerial"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0
            if (serial != lastPreviewSerial) {
                lastPreviewSerial = serial
                SdBridge.nativePreview(handle)?.let { send(DiffusionEvent.Preview(unpack(it))) }
            }
        }
        worker.join()
    }

    private fun parseClip(manifest: String): DiffusionClip {
        val root = json.parseToJsonElement(manifest).jsonObject
        val dir = root["dir"]?.jsonPrimitive?.content.orEmpty()
        val frames = (root["frames"] as? kotlinx.serialization.json.JsonArray)
            ?.map { "$dir/${it.jsonPrimitive.content}" }
            .orEmpty()
        return DiffusionClip(
            directory = dir,
            frames = frames,
            width = root["width"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0,
            height = root["height"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0,
            fps = root["fps"]?.jsonPrimitive?.content?.toIntOrNull() ?: 16,
            audioPath = root["audio"]?.jsonPrimitive?.content?.takeIf { it.isNotBlank() }
                ?.let { "$dir/$it" },
        )
    }

    // — image marshalling —

    private data class Rgb(val pixels: ByteArray, val width: Int, val height: Int)

    private fun decodeRgb(uri: String): Rgb? = runCatching {
        val bitmap = context.contentResolver.openInputStream(Uri.parse(uri))?.use {
            BitmapFactory.decodeStream(it)
        } ?: return null
        toRgb(bitmap)
    }.getOrNull()

    private fun decodeRgbFromFile(path: String): Rgb? = runCatching {
        toRgb(BitmapFactory.decodeFile(path) ?: return null)
    }.getOrNull()

    /**
     * sd.cpp wants tightly packed RGB, and Android hands back ARGB_8888.
     *
     * The shape is left alone. This used to floor both sides to a multiple of
     * 64, which was wrong twice over: upstream resizes every supplied picture
     * to the *requested* dimensions itself, so the work was redundant and cost
     * a second resample; and flooring changes the aspect ratio, so a 512×288
     * first frame arrived as 512×256 and the clip came out 2:1 instead of the
     * 16:9 that was asked for. Sixty-four was never the right number either —
     * it is SD's, while Wan needs 32 and Flux 16 — but no number is, because
     * this is not the place the decision belongs.
     *
     * The one thing worth doing here is bounding the size. A camera photo is
     * twelve megapixels, which is thirty-six megabytes of packed RGB handed to
     * a process already holding several gigabytes of weights, and every one of
     * those pixels is about to be thrown away by the resize upstream does.
     */
    private fun toRgb(source: Bitmap): Rgb {
        val longest = maxOf(source.width, source.height)
        val scale = if (longest > MAX_SUPPLIED_EDGE) MAX_SUPPLIED_EDGE.toFloat() / longest else 1f
        val width = (source.width * scale).toInt().coerceAtLeast(1)
        val height = (source.height * scale).toInt().coerceAtLeast(1)
        val bitmap = if (width != source.width || height != source.height) {
            Bitmap.createScaledBitmap(source, width, height, true)
        } else {
            source
        }

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val out = ByteArray(width * height * 3)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            out[i * 3] = ((pixel shr 16) and 0xFF).toByte()
            out[i * 3 + 1] = ((pixel shr 8) and 0xFF).toByte()
            out[i * 3 + 2] = (pixel and 0xFF).toByte()
        }
        return Rgb(out, width, height)
    }

    /** Upscale a finished picture with an installed ESRGAN model. */
    suspend fun upscale(
        image: DiffusionImage,
        esrganPath: String,
        factor: Int = 0,
        threads: Int = capabilities.inferenceThreads,
        tileSize: Int = UPSCALE_TILE,
    ): Result<DiffusionImage> = withContext(Dispatchers.Default) {
        runCatching {
            check(SdBridge.available) {
                SdBridge.loadError ?: "The stable-diffusion.cpp runtime is not installed in this build."
            }
            check(esrganPath.isNotBlank()) {
                "No upscaler is attached. Install an ESRGAN model and tick it under Attachments."
            }

            // The upscaler is its own context and shares nothing with the
            // denoiser, so holding both is holding one for no reason — and the
            // one being held is the larger. A 4B checkpoint plus its text
            // encoder is over 4 GB resident; asking for an ESRGAN graph on top
            // of that is what the kernel kills the app for, with no Java
            // exception and nothing in the crash buffer to explain it.
            // Generating again reloads on its own.
            unload("the upscaler needs the memory the denoiser was holding; generating again reloads it")
            val rgb = ByteArray(image.width * image.height * 3)
            var at = 0
            image.pixels.forEach { pixel ->
                rgb[at] = ((pixel shr 16) and 0xFF).toByte()
                rgb[at + 1] = ((pixel shr 8) and 0xFF).toByte()
                rgb[at + 2] = (pixel and 0xFF).toByte()
                at += 3
            }
            val out = SdBridge.nativeUpscale(
                esrganPath = esrganPath,
                rgb = rgb,
                width = image.width,
                height = image.height,
                factor = factor,
                threads = threads,
                tileSize = tileSize,
            ) ?: error("The upscaler returned nothing.")
            unpack(out)
        }
    }

    /** Unpack the 8-byte header the native side prepends. */
    private fun unpack(bytes: ByteArray): DiffusionImage {
        val buffer = ByteBuffer.wrap(bytes, 0, 8).order(ByteOrder.LITTLE_ENDIAN)
        val width = buffer.int
        val height = buffer.int
        val pixels = IntArray(width * height)
        var offset = 8
        for (i in pixels.indices) {
            val r = bytes[offset].toInt() and 0xFF
            val g = bytes[offset + 1].toInt() and 0xFF
            val b = bytes[offset + 2].toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            offset += 3
        }
        return DiffusionImage(width, height, pixels)
    }

    /** Decimal GB and MB, which is how every model repo states a size. */
    private fun formatBytes(bytes: Long): String = when {
        bytes >= 1_000_000_000L -> String.format("%.2f GB", bytes / 1_000_000_000.0)
        bytes >= 1_000_000L -> String.format("%.0f MB", bytes / 1_000_000.0)
        else -> "$bytes B"
    }

    companion object {
        /**
         * The manifest key naming the precision to load at.
         *
         * Named here because this is the only thing that reads it. It reaches
         * sd.cpp as a `nativeLoad` argument rather than through the parameter
         * table, the way the component paths and `threads` do — sd.cpp settles
         * a weight type while building the context and there is no way to
         * change one afterwards, which is what `requiresReload` says about it.
         */
        const val WEIGHT_TYPE_KEY = "type"

        private const val TAG = "DiffusionEngine"

        /**
         * The longest edge a supplied picture is carried at.
         *
         * Not a quality ceiling — upstream resizes to the requested dimensions
         * anyway, and those are far below this on a phone. It is a bound on
         * what a twelve-megapixel camera photo costs on its way through.
         */
        private const val MAX_SUPPLIED_EDGE = 2048

        /**
         * The roles `sd_ctx_params_t` has a path for.
         *
         * This decides two things at once, which is why it is one list: what
         * gets sent to the loader, and what the screen reports as resident.
         * A role missing from here is a file the app can hold, display and
         * believe it has attached, and never mention to the runtime.
         */
        private val LOAD_TIME_ROLES = listOf(
            AttachmentRole.VAE,
            AttachmentRole.CONTROLNET,
            AttachmentRole.CLIP_L,
            AttachmentRole.CLIP_G,
            AttachmentRole.T5XXL,
            AttachmentRole.IP_ADAPTER,
            AttachmentRole.EMBEDDING,
            AttachmentRole.CLIP_VISION,
            AttachmentRole.LLM_ENCODER,
            AttachmentRole.UNCOND_DIFFUSION,
            AttachmentRole.HIGH_NOISE_DIFFUSION,
            AttachmentRole.MOTION_MODULE,
            AttachmentRole.AUDIO_VAE,
            AttachmentRole.PHOTO_MAKER,
            AttachmentRole.PULID,
            AttachmentRole.LLM_VISION,
        )

        /**
         * The non-path fields of `sd_ctx_params_t`, by their own names.
         *
         * Load-time because sd.cpp settles each of them while building the
         * context; changing one on a live context is not possible, which is
         * what `requiresReload` says about them in the manifest. Anything not
         * set by the caller is left at upstream's default rather than restated
         * here — a default copied into two places is a default that drifts.
         */
        val LOAD_SETTING_KEYS = listOf(
            WEIGHT_TYPE_KEY,
            "tensor_type_rules",
            "enable_mmap",
            "flash_attn",
            "diffusion_flash_attn",
            "diffusion_conv_direct",
            "vae_conv_direct",
            "force_sdxl_vae_conv_scale",
            "max_vram",
            "stream_layers",
            "eager_load",
            "auto_fit",
            "vae_format",
            "prediction",
            "rng_type",
            "sampler_rng_type",
            "lora_apply_mode",
        )

        /** Fast enough to look live, slow enough not to spin a core polling. */
        private const val POLL_MILLIS = 250L

        /**
         * Side of the square the upscaler works on at a time.
         *
         * Zero means "no tiling", which is what this used to pass: a 512-square
         * input then runs 23 RRDB blocks over the whole frame at once, and the
         * 4x stages carry 64-channel tensors at 1024 and 2048 square —
         * gigabytes of intermediates for a picture that fits in 12 MB. Tiles
         * cost seams at worst; the alternative costs the process.
         */
        private const val UPSCALE_TILE = 128
    }
}

data class DiffusionRequest(
    val params: SparseParams,
    /** Where an img2img run starts from, travelled away from by `strength`. */
    val initImageUri: String? = null,
    /**
     * The picture an edit model is *shown*. Kontext and FLUX.2 read this and
     * change what the prompt asks for, leaving the rest alone; it is a
     * different thing from [initImageUri] and the two do not mix.
     */
    val referenceImageUri: String? = null,
    val controlImageUri: String? = null,
    /**
     * The picture an IP-Adapter takes its style from.
     *
     * A third distinct thing, and sd.cpp gives it a third field: not the map a
     * ControlNet reads, not the picture an edit model is shown. Nothing ever
     * filled it, so an IP-Adapter loaded, cost its weights and a 2.4 GB vision
     * encoder, and was handed nothing to look at.
     */
    val styleImageUri: String? = null,
    /**
     * The face PhotoMaker and PuLID are asked to keep.
     *
     * `sd_pm_params_t.id_images` and the PuLID weight were left as
     * `sd_img_gen_params_init` wrote them — no images, so no identity. Both
     * adapters could be attached and both loaded, which is the expensive half
     * of a feature that never did anything.
     */
    val identityImageUri: String? = null,
    val maskPngPath: String? = null,
    val attachments: List<ModelAttachment> = emptyList(),
)

/**
 * What a video run asks for.
 *
 * Deliberately not [DiffusionRequest] with extra fields: three of that type's
 * five picture slots have no field in `sd_vid_gen_params_t` — there is no
 * IP-Adapter, no inpainting mask and no reference image for an edit model — and
 * a request type that accepts them would be a promise nothing keeps.
 */
data class VideoRequest(
    val params: SparseParams,
    /** The first frame. Without one the clip is made from the prompt alone. */
    val initImageUri: String? = null,
    /**
     * The last frame.
     *
     * With both ends given the model travels between them, which is a different
     * request from "animate this" and has no equivalent in image generation.
     */
    val endImageUri: String? = null,
    /** One control map, held across every frame. */
    val controlImageUri: String? = null,
    val attachments: List<ModelAttachment> = emptyList(),
)

/**
 * A finished clip, as files.
 *
 * Paths rather than pixels: a five-second 480p clip is ~147 MB of raw RGB, and
 * the screen shows one frame at a time.
 */
data class DiffusionClip(
    val directory: String,
    /** Absolute paths, in order. */
    val frames: List<String>,
    val width: Int,
    val height: Int,
    val fps: Int,
    /** LTX-AV's soundtrack as a WAV, or null — no other architecture returns one. */
    val audioPath: String? = null,
) {
    val durationSeconds: Float get() = if (fps > 0) frames.size / fps.toFloat() else 0f
}

data class DiffusionImage(val width: Int, val height: Int, val pixels: IntArray) {
    fun toBitmap(): Bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)

    /** PNG bytes with the generation parameters in a `tEXt` chunk, so the file alone is reproducible without this app's database (SPEC §5.4). */
    fun toPng(parametersJson: String): ByteArray {
        val body = ByteArrayOutputStream()
        toBitmap().compress(Bitmap.CompressFormat.PNG, 100, body)
        return PngText.withTextChunk(body.toByteArray(), "parameters", parametersJson)
    }
}

/**
 * One component the loader reported taking, and what it costs.
 *
 * [bytes] is its size on disk, which is the honest figure for what it occupies
 * — the runtime reports none of its own, and the weights dominate a diffusion
 * context by enough that the file size is nearer than anything this app could
 * compute.
 */
data class ResidentPart(
    val role: AttachmentRole,
    val fileName: String,
    val bytes: Long,
)

/**
 * One module's working memory, as the runtime reported it.
 *
 * Two figures because ggml reserves two things and reports them separately.
 * [computeMb] is the graph allocator's reservation for a graph's intermediate
 * tensors. [cacheMb] is a module's own persistent cache — for Wan's decoder,
 * the feature maps a causal 3D convolution carries from one frame to the next,
 * which is why it grows with frame count and the graph buffer does not.
 *
 * Neither includes the weights, the latents or the decoded frames, so neither
 * is "what the run is using" and the pair does not become that by being added.
 */
data class RuntimeBuffer(
    /** The module, in the runtime's own spelling — `wan_vae`, `t5`. */
    val what: String,
    val computeMb: Double,
    val cacheMb: Double,
)

/** Which part of the run is happening. */
enum class DiffusionPhase(val label: String) {
    PREPARING("preparing"),
    SAMPLING("sampling"),
    DECODING("decoding"),
}

/**
 * One LoRA and how much of it landed, counted by the runtime.
 *
 * `applied == 0` is the case worth saying out loud: the file loaded, cost its
 * time, and changed nothing, because its tensor names belong to a different
 * architecture. sd.cpp reports that and continues, so without this the picture
 * is identical to one where the LoRA worked.
 */
data class LoraApplication(val file: String, val applied: Int, val total: Int) {
    val landed: Boolean get() = applied > 0
}

sealed interface DiffusionEvent {
    data class Progress(
        val step: Int,
        val steps: Int,
        val secondsPerStep: Float,
        val phase: DiffusionPhase,
        /** The runtime's own last word, where it has said one. */
        val stage: String? = null,
    ) : DiffusionEvent
    data class Preview(val image: DiffusionImage) : DiffusionEvent
    data class Completed(
        val image: DiffusionImage,
        /** What each attached LoRA managed to change, or empty when none was attached. */
        val loras: List<LoraApplication> = emptyList(),
    ) : DiffusionEvent
    /** The video counterpart, carrying paths because the frames are on disk. */
    data class ClipCompleted(
        val clip: DiffusionClip,
        val loras: List<LoraApplication> = emptyList(),
    ) : DiffusionEvent
    data class Failed(val message: String, val suggestion: String?) : DiffusionEvent

}

/**
 * A load that finished after the person had already said they did not want it.
 *
 * Its own type so the screens can tell it from a load that went wrong. Both
 * come back as a failed [Result]; only one of them is worth a red banner.
 */
class LoadCancelled : Exception("The load was cancelled.")
