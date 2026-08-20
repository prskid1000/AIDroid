package ai.ondevice.proxy

import ai.ondevice.core.Modality
import ai.ondevice.core.SparseParams
import ai.ondevice.engine.DiffusionEvent
import ai.ondevice.engine.DiffusionOutcome
import ai.ondevice.engine.DiffusionRequest
import ai.ondevice.engine.VideoRequest
import io.ktor.http.ContentType
import io.ktor.http.content.PartData
import io.ktor.http.content.forEachPart
import io.ktor.server.request.receiveMultipart
import io.ktor.utils.io.toByteArray
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.io.File

/**
 * `/v1/images/generations`, `/edits` and `/upscales`.
 *
 * The OpenAI images shape, with two departures that are stated rather than
 * hidden. `b64_json` is the default response format instead of `url`, because
 * this device has no public URL to hand out and a link nobody can fetch is
 * worse than no link. And a `path` field sits beside it, because a client on
 * the same tailnet as the phone — or the model itself, through the tool version
 * of this — usually wants the file rather than four megabytes of base64.
 */
suspend fun ProxyCall.images(mode: ImageMode) {
    val body = body()
    if (mode == ImageMode.UPSCALE) return upscale(body)
    phase("Making a picture")

    val prompt = body.str("prompt").orEmpty()
    if (prompt.isBlank()) throw ProxyRefusal.badRequest("`prompt` is required.")

    val model = body.str("model")
        ?.let { resolveModel(it, setOf(Modality.DIFFUSION), "image generation") }
        ?: defaultModel(Modality.DIFFUSION, "image generation", ProxySpecs.DEFAULT_IMAGE)

    val size = parseSize(body.str("size"))
    var params = runner.paramsFor(
        model,
        SparseParams.of(
            "prompt" to prompt,
            "negative_prompt" to body.str("negative_prompt").orEmpty(),
            "width" to size.first,
            "height" to size.second,
        ),
    )
    body.i("steps")?.let { params = params.with("steps", it) }
    body.f("cfg_scale")?.let { params = params.with("cfg_scale", it) }
    body.i("seed")?.let { params = params.with("seed", it) }
    body.f("strength")?.let { params = params.with("strength", it) }

    // The picture an edit starts from, however it arrived: a path this device
    // can already see, or base64 on the wire.
    val source = body.str("image")?.let { value ->
        if (value.startsWith("data:") || value.length > PATH_CEILING) {
            media.writeBase64(value.substringAfter(',', value), "image/png")
        } else {
            value.takeIf { File(it).isFile }
        }
    }
    if (mode == ImageMode.EDIT && source == null) {
        throw ProxyRefusal.badRequest(
            "`image` is required to edit one, as a path on this device or as base64.",
        )
    }

    val mask = body.str("mask")?.let { value ->
        if (value.startsWith("data:")) {
            media.writeBase64(value.substringAfter(','), "image/png")
        } else {
            value.takeIf { File(it).isFile }
        }
    }

    val stamp = System.currentTimeMillis()
    val destination = File(runner.scratchDir("images"), "http-$stamp.png")
    val wantsBase64 = body.str("response_format") != "path"
    val streaming = body.b("stream") == true

    log.update(requestId) { it.copy(streaming = streaming) }

    if (streaming) {
        streamImage(model, params, source, mask, destination, wantsBase64)
        return
    }

    val path = runner.exclusive(runner.runtimeFor(model), waitMillis = gateWait()) {
        runner.loadDiffusion(model, params)
        val outcome = collectLast(
            runner.image(
                DiffusionRequest(params = params, initImageUri = source, maskPngPath = mask),
                destination,
                onProgress = { noteSteps(it) },
            ),
        )
        (outcome as? DiffusionOutcome.Image)?.path
    } ?: throw ProxyRefusal(500, "api_error", "The run produced no picture.", null)

    runner.touch(model.id)
    json(imageBody(stamp, path, wantsBase64))
}

/**
 * `/v1/images/upscales` — not an OpenAI route, and named as an extension.
 *
 * It exists because the capability does: this app carries an ESRGAN path that
 * nothing in either protocol has a slot for, and the alternative to a route is
 * a capability reachable only by tapping the screen. The upscaler is its own
 * context and shares nothing with the denoiser, so the engine drops the
 * denoiser first — holding both is what the kernel kills the process for, with
 * no exception and nothing in the crash buffer to explain it.
 */
private suspend fun ProxyCall.upscale(body: JsonObject) {
    phase("Enlarging a picture")
    val sourcePath = body.str("image")?.let { value ->
        if (value.startsWith("data:") || value.length > PATH_CEILING) {
            media.writeBase64(value.substringAfter(',', value), "image/png")
        } else {
            value.takeIf { File(it).isFile }
        }
    } ?: throw ProxyRefusal.badRequest(
        "`image` is required — a path on this device, or base64.",
    )

    val model = body.str("model")
        ?.let { resolveModel(it, setOf(Modality.DIFFUSION), "upscaling") }
        ?: defaultModel(Modality.DIFFUSION, "upscaling", ProxySpecs.DEFAULT_IMAGE)

    // The ESRGAN graph, from the model's own stored parameters. Named in the
    // refusal rather than left as "it failed": an upscaler is a separate
    // download, and somebody who has not made one is not going to guess that
    // from a generic error.
    val esrgan = body.str("upscale_model")
        ?: runner.paramsFor(model, SparseParams.EMPTY).string("upscale_model")
    if (esrgan.isNullOrBlank()) {
        throw ProxyRefusal.badRequest(
            "No upscaler is attached to `${model.id}`.",
            "Install an ESRGAN model — its filename gives away the role, so it appears " +
                "under Attachments once downloaded — or pass `upscale_model` as a path.",
        )
    }

    val factor = body.i("factor") ?: 0
    val wantsBase64 = body.str("response_format") != "path"

    val decoded = withContext(Dispatchers.IO) {
        android.graphics.BitmapFactory.decodeFile(sourcePath)
    } ?: throw ProxyRefusal.badRequest("The image at $sourcePath could not be read.")

    val pixels = IntArray(decoded.width * decoded.height)
    decoded.getPixels(pixels, 0, decoded.width, 0, 0, decoded.width, decoded.height)

    val stamp = System.currentTimeMillis()
    val destination = File(runner.scratchDir("images"), "upscaled-$stamp.png")

    val bigger = runner.exclusive(runner.runtimeFor(model), waitMillis = gateWait()) {
        runner.upscale(
            ai.ondevice.engine.DiffusionImage(decoded.width, decoded.height, pixels),
            esrgan,
            factor,
        )
    }
    withContext(Dispatchers.IO) {
        destination.writeBytes(bigger.toPng(SparseParams.of("upscale_model" to esrgan).toJsonString()))
    }
    runner.touch(model.id)

    json(imageBody(stamp, destination.absolutePath, wantsBase64))
}

/**
 * The same run, with progress.
 *
 * `partial_image` is OpenAI's name for a frame carrying an incomplete picture;
 * this device's diffusion engine reports steps rather than intermediate images
 * unless previews are on, so a frame carries whichever it has. The step count
 * and the seconds-per-step are the two numbers that make a progress bar honest
 * on hardware where a single step is measured in seconds.
 */
private suspend fun ProxyCall.streamImage(
    model: ai.ondevice.data.db.ModelEntity,
    params: SparseParams,
    source: String?,
    mask: String?,
    destination: File,
    wantsBase64: Boolean,
) {
    stream { emit ->
        runCatching {
            runner.exclusive(runner.runtimeFor(model), waitMillis = gateWait()) {
                runner.loadDiffusion(model, params)
                runner.image(
                    DiffusionRequest(params = params, initImageUri = source, maskPngPath = mask),
                    destination,
                ).collect { outcome ->
                    when (outcome) {
                        is DiffusionOutcome.Progress -> {
                            noteSteps(outcome.event)
                            emit(Sse.data(progressJson(outcome.event)))
                        }
                        is DiffusionOutcome.Image -> emit(
                            Sse.data(
                                imageBody(
                                    System.currentTimeMillis(),
                                    outcome.path,
                                    wantsBase64,
                                    type = "image_generation.completed",
                                ),
                            ),
                        )
                        else -> Unit
                    }
                }
            }
            runner.touch(model.id)
        }.onFailure { failure ->
            val refusal = ChatPipeline.refusalFor(failure)
            emit(Sse.data(refusal.body(Protocol.OPENAI)))
        }
        emit(Sse.DONE)
    }
}

private fun progressJson(event: DiffusionEvent.Progress): String = encode(
    buildJsonObject {
        put("type", "image_generation.partial_image")
        put("step", event.step)
        put("steps", event.steps)
        put("phase", event.phase.label)
        if (event.secondsPerStep > 0f) put("seconds_per_step", event.secondsPerStep)
        event.detail?.let { put("detail", it) }
    },
)

private fun imageBody(
    stamp: Long,
    path: String,
    wantsBase64: Boolean,
    type: String? = null,
): String = encode(
    buildJsonObject {
        type?.let { put("type", it) }
        put("created", stamp / 1000)
        put(
            "data",
            JsonArray(
                listOf(
                    buildJsonObject {
                        // Always the path. A client on the tailnet can fetch the
                        // file directly and a model calling this as a tool wants
                        // nothing else; four megabytes of base64 in a tool
                        // result would eat the context it was called from.
                        put("path", path)
                        if (wantsBase64) {
                            put(
                                "b64_json",
                                android.util.Base64.encodeToString(
                                    File(path).readBytes(),
                                    android.util.Base64.NO_WRAP,
                                ),
                            )
                        }
                    },
                ),
            ),
        )
    },
)

/**
 * `/v1/audio/speech`.
 *
 * Answers with the WAV itself, because that is what the OpenAI route does and
 * what every client of it expects — `response_format: "path"` is the escape
 * hatch for anything that would rather have the file where it lies.
 */
suspend fun ProxyCall.speech() {
    phase("Speaking")
    val body = body()
    val text = body.str("input")
        ?: throw ProxyRefusal.badRequest("`input` is required.")

    val model = body.str("model")
        ?.takeIf { it != DEFAULT_TTS_ALIAS }
        ?.let { resolveModel(it, setOf(Modality.TEXT_TO_SPEECH), "speech") }
        ?: defaultModel(Modality.TEXT_TO_SPEECH, "speech", ProxySpecs.DEFAULT_VOICE)

    var params = runner.paramsFor(model, SparseParams.EMPTY)
    // The request first, then the configured default, then the model's own
    // stored parameters, then whatever the engine offers first.
    (body.str("voice")?.takeIf { it.isNotBlank() }
        ?: config.defaultModel(ProxySpecs.TTS_VOICE).takeIf { it.isNotBlank() })
        ?.let { params = params.with("voice", it) }
    body.f("speed")?.let { params = params.with("speed", it) }
    body.str("language")?.let { params = params.with("lang_code", it) }

    val destination = File(
        runner.scratchDir("speech"),
        "http-${System.currentTimeMillis()}.wav",
    )
    val file = runner.exclusive(runner.runtimeFor(model), waitMillis = gateWait()) {
        runner.speak(model, params, text, destination)
    }
    runner.touch(model.id)

    // Named rather than silently ignored. This app synthesises WAV and has no
    // encoder for anything else; a client asking for mp3 and receiving a WAV
    // labelled mp3 is a bug it cannot diagnose from its own side.
    val wanted = body.str("response_format")
    if (wanted != null && wanted != "wav" && wanted != "path") {
        throw ProxyRefusal.badRequest(
            "This device synthesises WAV only; `$wanted` is not available.",
            "Ask for `wav`, or `path` to be told where the file is.",
        )
    }

    if (wanted == "path") {
        json(
            encode(
                buildJsonObject {
                    put("path", file.absolutePath)
                    put("format", "wav")
                    put("bytes", file.length())
                },
            ),
        )
    } else {
        bytes(file.readBytes(), ContentType("audio", "wav"))
    }
}

/**
 * `/v1/audio/transcriptions` and `/v1/audio/translations`.
 *
 * The real route is multipart, and this accepts JSON with a `file` path as well
 * — because the caller is very often the same machine, or the model itself
 * through the tool, and making it base64 a forty-minute recording to hand it to
 * a program on the same device is work for nothing.
 */
suspend fun ProxyCall.transcription(translate: Boolean) {
    phase(if (translate) "Translating" else "Transcribing")
    val contentType = call.request.headers["Content-Type"].orEmpty()
    val route = if (translate) "translation" else "transcription"

    val (audio, requestedModel, options) = if (contentType.startsWith("multipart/")) {
        readMultipart()
    } else {
        val body = body()
        val path = body.str("file")
            ?: throw ProxyRefusal.badRequest(
                "`file` is required — a path on this device, or send multipart/form-data.",
            )
        Triple(File(path), body.str("model"), body)
    }

    if (!audio.isFile) {
        throw ProxyRefusal.badRequest("There is no readable file at ${audio.absolutePath}.")
    }

    val model = requestedModel
        ?.takeIf { it != DEFAULT_STT_ALIAS }
        ?.let { resolveModel(it, setOf(Modality.SPEECH_TO_TEXT), route) }
        ?: defaultModel(Modality.SPEECH_TO_TEXT, route, ProxySpecs.DEFAULT_SPEECH)

    var params = runner.paramsFor(model, SparseParams.EMPTY)
    if (translate) params = params.with("translate", true)
    options.str("language")?.let { params = params.with("language", it) }
    options.str("prompt")?.let { params = params.with("prompt", it) }
    options.f("temperature")?.let { params = params.with("temperature", it) }

    val segments = runner.exclusive(runner.runtimeFor(model), waitMillis = gateWait()) {
        runner.transcribe(model, params, audio)
    }
    runner.touch(model.id)

    val text = segments.joinToString(" ") { it.text }.trim()

    when (options.str("response_format")) {
        "text" -> json(encode(buildJsonObject { put("text", text) }))
        "srt" -> json(
            encode(
                buildJsonObject {
                    put("text", text)
                    put("srt", ai.ondevice.core.TranscriptSegments.encode(segments))
                },
            ),
        )
        // `verbose_json` is the OpenAI name for "with the timings", which is
        // what whisper.cpp produces anyway and what a caller wanting subtitles
        // needs. Default is the plain shape, exactly as the real route does.
        "verbose_json" -> json(
            encode(
                buildJsonObject {
                    put("task", if (translate) "translate" else "transcribe")
                    put("text", text)
                    put(
                        "segments",
                        JsonArray(
                            segments.mapIndexed { index, segment ->
                                buildJsonObject {
                                    put("id", index)
                                    put("start", segment.startMillis / 1000.0)
                                    put("end", segment.endMillis / 1000.0)
                                    put("text", segment.text)
                                }
                            },
                        ),
                    )
                },
            ),
        )
        else -> json(encode(buildJsonObject { put("text", text) }))
    }
}

/**
 * Read the multipart form the real route uses.
 *
 * The bytes are copied in before anything else happens to them, which is the
 * same discipline the share-sheet path follows for a `content://` grant: a
 * stream that dies while something downstream is still holding a reference to
 * it fails minutes later, somewhere unrelated to the cause.
 */
private suspend fun ProxyCall.readMultipart(): Triple<File, String?, JsonObject> {
    val parts = call.receiveMultipart()
    var file: File? = null
    var model: String? = null
    val fields = mutableMapOf<String, kotlinx.serialization.json.JsonElement>()

    parts.forEachPart { part ->
        when (part) {
            is PartData.FileItem -> {
                val extension = part.originalFileName
                    ?.substringAfterLast('.', "")
                    ?.takeIf { it.isNotBlank() }
                    ?: "wav"
                val bytes = part.provider().toByteArray()
                media.writeBytes(bytes, extension)?.let { file = File(it) }
            }
            is PartData.FormItem -> {
                if (part.name == "model") model = part.value
                part.name?.let { fields[it] = kotlinx.serialization.json.JsonPrimitive(part.value) }
            }
            else -> Unit
        }
        part.dispose()
    }

    return Triple(
        file ?: throw ProxyRefusal.badRequest("No `file` part in the form."),
        model,
        JsonObject(fields),
    )
}

/**
 * `/v1/videos`.
 *
 * Answers immediately with a job id and carries on in the app's own scope. A
 * Wan run measured on this device is three quarters of an hour at 384 square:
 * no client's idle timeout survives that, and neither does a phone's Wi-Fi as
 * it walks between access points. Holding the connection would mean the work
 * completing and the answer having nowhere to go.
 */
suspend fun ProxyCall.createVideo() {
    val body = body()
    val prompt = body.str("prompt")
        ?: throw ProxyRefusal.badRequest("`prompt` is required.")

    val model = body.str("model")
        ?.let { resolveModel(it, setOf(Modality.DIFFUSION), "video generation") }
        ?: defaultModel(Modality.DIFFUSION, "video generation", ProxySpecs.DEFAULT_VIDEO)

    var params = runner.paramsFor(
        model,
        SparseParams.of(
            "prompt" to prompt,
            "negative_prompt" to body.str("negative_prompt").orEmpty(),
        ),
    )
    body.i("seconds")?.let { seconds ->
        val fps = params.int("fps") ?: DEFAULT_FPS
        params = params.with("video_frames", seconds * fps)
    }
    body.i("frames")?.let { params = params.with("video_frames", it) }
    body.i("seed")?.let { params = params.with("seed", it) }
    parseSizeOrNull(body.str("size"))?.let { (width, height) ->
        params = params.with("width", width).with("height", height)
    }

    val job = jobs.create(model.id, prompt)

    // The app's scope, not the request's. The whole point is that the run
    // outlives the connection that asked for it.
    //
    // Which is also why the wake lock is taken here rather than inherited: the
    // request that started this is answered and gone within the second, and the
    // forty minutes of work that follow have nothing else holding the CPU or
    // keeping the service alive. A clip is the longest thing this device does
    // and the one most likely to be running with the screen off.
    scope.launch {
        ai.ondevice.engine.InferenceService.holdingWakeLock(context) {
            renderClip(job, model, params, body)
        }
    }

    json(videoJson(jobs.get(job.id) ?: job))
}

private suspend fun ProxyCall.renderClip(
    job: VideoJobs.Job,
    model: ai.ondevice.data.db.ModelEntity,
    params: SparseParams,
    body: JsonObject,
) {
    jobs.update(job.id) { it.copy(state = VideoJobs.State.RUNNING) }
    runCatching {
        // No wait limit. The request that started this was answered and hung up
        // within the second, so there is nobody to refuse to — a clip queued
        // behind a conversation should wait for it, not fail.
        runner.exclusive(runner.runtimeFor(model)) {
            runner.loadDiffusion(model, params)
            runner.activeCancel?.let { jobs.attachCancel(job.id, it) }
            runner.clip(
                VideoRequest(
                    params = params,
                    initImageUri = body.str("first_frame"),
                    endImageUri = body.str("last_frame"),
                ),
            ) { progress ->
                jobs.update(job.id) {
                    it.copy(
                        step = progress.step,
                        steps = progress.steps,
                        phase = progress.phase.label,
                        secondsPerStep = progress.secondsPerStep,
                    )
                }
            }.collect { outcome ->
                if (outcome is DiffusionOutcome.Clip) {
                    jobs.update(job.id) {
                        it.copy(
                            state = VideoJobs.State.COMPLETED,
                            completedAt = System.currentTimeMillis(),
                            directory = outcome.clip.directory,
                            frames = outcome.clip.frames,
                            fps = outcome.clip.fps,
                            audioPath = outcome.clip.audioPath,
                        )
                    }
                }
            }
        }
        runner.touch(model.id)
    }.onFailure { failure ->
        // A cancel arrives here as a failure too, and the two must not read the
        // same on the wire: one is something that went wrong and the other is
        // something that was asked for.
        if (jobs.get(job.id)?.state != VideoJobs.State.CANCELLED) {
            val refusal = ChatPipeline.refusalFor(failure)
            jobs.update(job.id) {
                it.copy(
                    state = VideoJobs.State.FAILED,
                    completedAt = System.currentTimeMillis(),
                    error = refusal.message,
                    suggestion = refusal.suggestion,
                )
            }
        }
    }
    jobs.finished(job.id)
}

/** Diffusion counts steps; the notification is the only place they show once
 *  the app is off screen. */
private fun ProxyCall.noteSteps(event: DiffusionEvent.Progress) {
    log.update(requestId) { it.copy(step = event.step, steps = event.steps) }
}

// — small shared helpers —

private suspend fun collectLast(flow: kotlinx.coroutines.flow.Flow<DiffusionOutcome>): DiffusionOutcome? {
    var last: DiffusionOutcome? = null
    flow.collect { last = it }
    return last
}

/** `1024x1024`, or a single number, or the model's own default. */
private fun parseSize(raw: String?): Pair<Int, Int> =
    parseSizeOrNull(raw) ?: (DEFAULT_EDGE to DEFAULT_EDGE)

private fun parseSizeOrNull(raw: String?): Pair<Int, Int>? {
    val value = raw?.trim()?.lowercase()?.takeIf { it.isNotBlank() && it != "auto" } ?: return null
    val parts = value.split('x', '×')
    return when (parts.size) {
        2 -> {
            val width = parts[0].trim().toIntOrNull() ?: return null
            val height = parts[1].trim().toIntOrNull() ?: return null
            width to height
        }
        1 -> parts[0].toIntOrNull()?.let { it to it }
        else -> null
    }
}

/**
 * How long a string may be before it is assumed to be base64 rather than a path.
 *
 * Crude, and deliberately generous: no filesystem path on Android runs to four
 * kilobytes, and no useful base64 image is shorter than that.
 */
private const val PATH_CEILING = 4096

private const val DEFAULT_EDGE = 512
private const val DEFAULT_FPS = 16

/**
 * The names OpenAI clients hardcode.
 *
 * A client sending `tts-1` or `whisper-1` has not chosen a model, it has
 * repeated the only name its SDK knows. Treated as "whichever is installed"
 * rather than refused, because refusing would be technically correct and
 * useless.
 */
private const val DEFAULT_TTS_ALIAS = "tts-1"
private const val DEFAULT_STT_ALIAS = "whisper-1"
