package ai.ondevice.engine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.ModelAttachment
import ai.ondevice.core.SparseParams
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
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
        runCatching {
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

            val newHandle = SdBridge.nativeLoad(
                modelPath = modelPath,
                vaePath = pathFor(AttachmentRole.VAE),
                taesdPath = pathFor(AttachmentRole.TAESD),
                controlNetPath = pathFor(AttachmentRole.CONTROLNET),
                clipLPath = pathFor(AttachmentRole.CLIP_L),
                clipGPath = pathFor(AttachmentRole.CLIP_G),
                t5xxlPath = pathFor(AttachmentRole.T5XXL),
                ipAdapterPath = pathFor(AttachmentRole.IP_ADAPTER),
                embeddingsPath = pathFor(AttachmentRole.EMBEDDING),
                clipVisionPath = pathFor(AttachmentRole.CLIP_VISION),
                llmPath = pathFor(AttachmentRole.LLM_ENCODER),
                threads = threads,
            )
            check(newHandle != 0L) { "The runtime returned no handle for $modelPath." }
            handle = newHandle
            android.util.Log.i(
                TAG,
                "loaded ${File(modelPath).name} (${File(modelPath).length() / 1024 / 1024} MB) " +
                    "threads=$threads " +
                    "attached=" + AttachmentRole.entries
                    .mapNotNull { role -> pathFor(role).takeIf { it.isNotBlank() }?.let { role.name } }
                    .ifEmpty { listOf("none") }.joinToString("+"),
            )
            loadedModelId = modelId
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

    fun unload() {
        if (handle != 0L) {
            SdBridge.nativeFree(handle)
            handle = 0L
        }
        loadedModelId = null
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

    fun cancel() {
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
                    send(DiffusionEvent.Completed(image))
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
            send(DiffusionEvent.Progress(step, steps, secondsPerStep, phase))

            val serial = progress["previewSerial"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0
            if (serial != lastPreviewSerial) {
                lastPreviewSerial = serial
                SdBridge.nativePreview(handle)?.let { send(DiffusionEvent.Preview(unpack(it))) }
            }
        }
        worker.join()
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

    /** sd.cpp wants tightly packed RGB, and Android hands back ARGB_8888. */
    private fun toRgb(source: Bitmap): Rgb {
        val width = (source.width / 64).coerceAtLeast(1) * 64
        val height = (source.height / 64).coerceAtLeast(1) * 64
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
            unload()
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

    private companion object {
        const val TAG = "DiffusionEngine"

        /** Fast enough to look live, slow enough not to spin a core polling. */
        const val POLL_MILLIS = 250L

        /**
         * Side of the square the upscaler works on at a time.
         *
         * Zero means "no tiling", which is what this used to pass: a 512-square
         * input then runs 23 RRDB blocks over the whole frame at once, and the
         * 4x stages carry 64-channel tensors at 1024 and 2048 square —
         * gigabytes of intermediates for a picture that fits in 12 MB. Tiles
         * cost seams at worst; the alternative costs the process.
         */
        const val UPSCALE_TILE = 128
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
    val maskPngPath: String? = null,
    val attachments: List<ModelAttachment> = emptyList(),
)

data class DiffusionImage(val width: Int, val height: Int, val pixels: IntArray) {
    fun toBitmap(): Bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)

    /** PNG bytes with the generation parameters in a `tEXt` chunk, so the file alone is reproducible without this app's database (SPEC §5.4). */
    fun toPng(parametersJson: String): ByteArray {
        val body = ByteArrayOutputStream()
        toBitmap().compress(Bitmap.CompressFormat.PNG, 100, body)
        return PngText.withTextChunk(body.toByteArray(), "parameters", parametersJson)
    }
}

/** Which part of the run is happening. */
enum class DiffusionPhase(val label: String) {
    PREPARING("preparing"),
    SAMPLING("sampling"),
    DECODING("decoding"),
}

sealed interface DiffusionEvent {
    data class Progress(
        val step: Int,
        val steps: Int,
        val secondsPerStep: Float,
        val phase: DiffusionPhase,
    ) : DiffusionEvent
    data class Preview(val image: DiffusionImage) : DiffusionEvent
    data class Completed(val image: DiffusionImage) : DiffusionEvent
    data class Failed(val message: String, val suggestion: String?) : DiffusionEvent

}
