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

/**
 * Diffusion, for real (SPEC §5).
 *
 * The engine knows nothing about model families. A run is a base model plus a
 * list of [ModelAttachment]s, each carrying a role and a path, and the runtime
 * is the thing that decides whether a given ControlNet or LoRA is loadable
 * against the loaded base. That is not laziness — the alternative is a
 * compatibility table this app would have to keep correct for every
 * architecture ever released, which is exactly the model-locking §1.3 forbids.
 */
class DiffusionEngine(
    private val context: Context,
) {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    @Volatile
    private var handle: Long = 0L

    @Volatile
    var loadedModelId: String? = null
        private set

    val available: Boolean get() = SdBridge.available
    val isLoaded: Boolean get() = handle != 0L

    suspend fun load(
        modelId: String,
        modelPath: String,
        attachments: List<ModelAttachment> = emptyList(),
        threads: Int = 0,
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            check(SdBridge.available) {
                SdBridge.loadError ?: "The stable-diffusion.cpp runtime is not installed in this build."
            }
            unload()

            // These three are load-time in sd.cpp — they change the context, not
            // the run — so they are resolved here and the rest at generate time.
            fun pathFor(role: AttachmentRole) =
                attachments.firstOrNull { it.enabled && it.role == role }?.path.orEmpty()

            val newHandle = SdBridge.nativeLoad(
                modelPath = modelPath,
                vaePath = pathFor(AttachmentRole.VAE),
                taesdPath = pathFor(AttachmentRole.TAESD),
                controlNetPath = pathFor(AttachmentRole.CONTROLNET),
                threads = threads,
            )
            check(newHandle != 0L) { "The runtime returned no handle for $modelPath." }
            handle = newHandle
            loadedModelId = modelId
        }
    }

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

    /**
     * Generate, emitting progress and live previews while it runs.
     *
     * `generate_image` blocks, so it goes on its own coroutine and a second one
     * polls. SPEC §5.4 wants intermediate latents rather than a spinner, and
     * that is what the preview stream is — the actual denoising state, decoded
     * by TAESD, not an animation standing in for one.
     */
    fun generate(request: DiffusionRequest): Flow<DiffusionEvent> = channelFlow {
        if (handle == 0L) {
            send(DiffusionEvent.Failed("No diffusion model is loaded.", "Models → Add an SD or SDXL repo."))
            return@channelFlow
        }

        applyParams(request.params)

        val init = request.initImageUri?.let { decodeRgb(it) }
        val control = request.controlImageUri?.let { decodeRgb(it) }
        val mask = request.maskPngPath?.let { decodeRgbFromFile(it) }

        // Attachments the *runtime* takes per-run, as a role-tagged list.
        val attachmentsJson = buildJsonArray {
            request.attachments.filter { it.enabled }.forEach { attachment ->
                add(
                    buildJsonObject {
                        put("role", attachment.role.name)
                        put("path", attachment.path)
                        put("weight", attachment.weight)
                    },
                )
            }
        }.toString()

        var done = false
        val worker = launch(Dispatchers.Default) {
            try {
                val bytes = SdBridge.nativeGenerate(
                    handle,
                    init?.pixels, init?.width ?: 0, init?.height ?: 0,
                    mask?.pixels, mask?.width ?: 0, mask?.height ?: 0,
                    control?.pixels, control?.width ?: 0, control?.height ?: 0,
                    attachmentsJson,
                )
                if (bytes == null) {
                    send(DiffusionEvent.Failed("The run produced no image.", null))
                } else {
                    val image = unpack(bytes)
                    send(DiffusionEvent.Completed(image))
                }
            } catch (t: Throwable) {
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

    /**
     * sd.cpp wants tightly packed RGB, and Android hands back ARGB_8888. The
     * dimensions are forced to a multiple of 64 because the latent space is
     * 8×-downsampled and the samplers assume it divides cleanly; a 999-pixel
     * input silently produces a misaligned tensor otherwise.
     */
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
        /** Fast enough to look live, slow enough not to spin a core polling. */
        const val POLL_MILLIS = 250L
    }
}

data class DiffusionRequest(
    val params: SparseParams,
    val initImageUri: String? = null,
    val controlImageUri: String? = null,
    val maskPngPath: String? = null,
    val attachments: List<ModelAttachment> = emptyList(),
)

data class DiffusionImage(val width: Int, val height: Int, val pixels: IntArray) {
    fun toBitmap(): Bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)

    /**
     * PNG bytes with the generation parameters in a `tEXt` chunk, so the file
     * alone is reproducible without this app's database (SPEC §5.4).
     */
    fun toPng(parametersJson: String): ByteArray {
        val body = ByteArrayOutputStream()
        toBitmap().compress(Bitmap.CompressFormat.PNG, 100, body)
        return PngText.withTextChunk(body.toByteArray(), "parameters", parametersJson)
    }
}

/**
 * Which part of the run is happening. sd.cpp reports loading, tiled VAE decode
 * and sampling through one untagged callback, so the phase is inferred natively
 * and carried here — the screen can then say "preparing" instead of pretending
 * the loader's tensor count is a sampler step.
 */
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
