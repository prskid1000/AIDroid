package ai.ondevice.tools

import ai.ondevice.core.Fmt
import ai.ondevice.core.Modality
import ai.ondevice.core.SparseParams
import ai.ondevice.core.Tier
import ai.ondevice.engine.DiffusionOutcome
import ai.ondevice.engine.DiffusionRequest
import ai.ondevice.engine.ModelRunner
import ai.ondevice.engine.ToolSpec
import ai.ondevice.engine.VideoRequest
import ai.ondevice.params.ParamSpec
import kotlinx.coroutines.flow.last
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import java.io.File

/**
 * The other four modalities, as tools a text model can call.
 *
 * This is what makes images, clips, speech and transcription reachable from the
 * Anthropic surface at all: the Messages API is chat-only, there is no
 * `/v1/images` in that protocol, and inventing one under that name would
 * produce something no client speaks. So the answer is the mechanism telecode
 * calls managed tools — the server injects a schema, intercepts the call, runs
 * it itself, and hands back the result without the client ever seeing that a
 * second engine was involved.
 *
 * A [ToolProvider] rather than something the proxy owns privately, for two
 * reasons. It appears in the Tools screen with its own switch and its own
 * settings rows, so what a remote client can make this phone do is visible in
 * the same place as everything else it can do. And the Chat tab gets it too:
 * asking the conversation for a picture is the same call over the same code.
 */
class MediaToolProvider(
    private val runner: ModelRunner,
    private val settings: ToolSettings = ToolSettings.EMPTY,
    /**
     * Whether the video tool is offered.
     *
     * A clip is tens of minutes on this hardware, which is longer than any
     * client's patience and longer than most of their timeouts. Offered only
     * when something is prepared to wait — the proxy's job route is, a chat
     * turn is not.
     */
    private val offerVideo: Boolean = false,
) : ToolProvider {

    override val id: String = ID

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    override suspend fun specs(): List<ToolSpec> = buildList {
        if (runner.defaultFor(Modality.DIFFUSION) != null) {
            add(GENERATE_IMAGE)
            add(EDIT_IMAGE)
            if (offerVideo) add(GENERATE_VIDEO)
        }
        if (runner.defaultFor(Modality.TEXT_TO_SPEECH) != null) add(SPEAK)
        if (runner.defaultFor(Modality.SPEECH_TO_TEXT) != null) add(TRANSCRIBE)
    }

    override fun settings(): List<ParamSpec> = listOf(
        ToolSettings.int(
            "generate_image", "steps", DEFAULT_STEPS, 1, 100,
            label = "Steps",
            help = "Denoising steps for a picture a model asked for. The default is low on " +
                "purpose: a tool call is usually a step in a longer answer, and a minute is " +
                "already a long time to make somebody wait mid-sentence.",
        ),
        ToolSettings.int(
            "generate_image", "size", DEFAULT_SIZE, 256, 2048,
            label = "Size",
            help = "The square edge, in pixels, when the model does not ask for one.",
            tier = Tier.ADVANCED,
        ),
    )

    // A "keep in the library" switch stood here and is gone rather than wired.
    // What a tool makes lands in the proxy's scratch folder and is swept after
    // a day; whether an HTTP caller should be able to fill this device's
    // gallery is a decision nobody has taken yet. A switch that stores an
    // answer nothing reads is worse than no switch.

    override suspend fun call(name: String, argumentsJson: String): ToolResult {
        val input = runCatching { json.parseToJsonElement(argumentsJson) as? JsonObject }
            .getOrNull() ?: JsonObject(emptyMap())
        return runCatching {
            when (name) {
                GENERATE_IMAGE.name -> generateImage(input, edit = false)
                EDIT_IMAGE.name -> generateImage(input, edit = true)
                GENERATE_VIDEO.name -> generateVideo(input)
                SPEAK.name -> speak(input)
                TRANSCRIBE.name -> transcribe(input)
                else -> ToolResult("No tool named \"$name\" is available.", isError = true, providerId = ID)
            }
        }.getOrElse { failure ->
            // The engines produce a suggestion beside the message and it is
            // worth as much to a model as it is to a person — "lower the
            // context size" is an instruction it can act on by itself.
            val suggestion = (failure as? ai.ondevice.engine.ModelRunFailure)?.suggestion
            ToolResult(
                listOfNotNull(failure.message, suggestion).joinToString("\n\n"),
                isError = true,
                providerId = ID,
            )
        }
    }

    private suspend fun generateImage(input: JsonObject, edit: Boolean): ToolResult {
        val prompt = input.text("prompt")
            ?: return ToolResult("`prompt` is required.", isError = true, providerId = ID)
        val model = runner.model(input.text("model").orEmpty())
            ?: runner.defaultFor(Modality.DIFFUSION)
            ?: return ToolResult(
                "No image model is installed on this device.",
                isError = true,
                providerId = ID,
            )

        val source = input.text("image")?.takeIf { edit }
        if (edit && source == null) {
            return ToolResult("`image` is required to edit one.", isError = true, providerId = ID)
        }

        val size = input.int("size") ?: settings.int("generate_image.size", DEFAULT_SIZE)
        var params = runner.paramsFor(
            model,
            SparseParams.of(
                "prompt" to prompt,
                "negative_prompt" to input.text("negative").orEmpty(),
                "steps" to (input.int("steps") ?: settings.int("generate_image.steps", DEFAULT_STEPS)),
                "width" to size,
                "height" to size,
            ),
        )
        input.int("seed")?.let { params = params.with("seed", it) }
        input.number("strength")?.let { params = params.with("strength", it.toFloat()) }

        val destination = File(
            runner.scratchDir("images"),
            "tool-${System.currentTimeMillis()}.png",
        )

        val runtime = runner.runtimeFor(model)
        val path = runner.exclusive(runtime) {
            runner.loadDiffusion(model, params)
            val outcome = runner.image(
                DiffusionRequest(params = params, initImageUri = source),
                destination,
            ).last()
            (outcome as? DiffusionOutcome.Image)?.path
        } ?: return ToolResult("The run produced no picture.", isError = true, providerId = ID)

        runner.touch(model.id)
        return ToolResult(
            "Made a ${size}x$size picture with ${model.displayName}.\nSaved to: $path\n\n" +
                "Refer to it by that path; the person can see it.",
            providerId = ID,
        )
    }

    private suspend fun generateVideo(input: JsonObject): ToolResult {
        val prompt = input.text("prompt")
            ?: return ToolResult("`prompt` is required.", isError = true, providerId = ID)
        val model = runner.model(input.text("model").orEmpty())
            ?: runner.defaultFor(Modality.DIFFUSION)
            ?: return ToolResult(
                "No diffusion model is installed on this device.",
                isError = true,
                providerId = ID,
            )

        val params = runner.paramsFor(
            model,
            SparseParams.of("prompt" to prompt, "negative_prompt" to input.text("negative").orEmpty()),
        )
        val runtime = runner.runtimeFor(model)
        val clip = runner.exclusive(runtime) {
            runner.loadDiffusion(model, params)
            val outcome = runner.clip(
                VideoRequest(
                    params = params,
                    initImageUri = input.text("first"),
                    endImageUri = input.text("last"),
                ),
            ).last()
            (outcome as? DiffusionOutcome.Clip)?.clip
        } ?: return ToolResult("The run produced no clip.", isError = true, providerId = ID)

        runner.touch(model.id)
        return ToolResult(
            "Made a ${clip.frames.size}-frame clip at ${clip.fps} fps " +
                "(${"%.1f".format(clip.durationSeconds)}s).\nFrames in: ${clip.directory}",
            providerId = ID,
        )
    }

    private suspend fun speak(input: JsonObject): ToolResult {
        val text = input.text("text")
            ?: return ToolResult("`text` is required.", isError = true, providerId = ID)
        val model = runner.model(input.text("model").orEmpty())
            ?: runner.defaultFor(Modality.TEXT_TO_SPEECH)
            ?: return ToolResult(
                "No voice is installed on this device.",
                isError = true,
                providerId = ID,
            )

        var params = runner.paramsFor(model, SparseParams.EMPTY)
        input.text("voice")?.let { params = params.with("voice", it) }
        input.number("speed")?.let { params = params.with("speed", it.toFloat()) }

        val destination = File(
            runner.scratchDir("speech"),
            "tool-${System.currentTimeMillis()}.wav",
        )
        val runtime = runner.runtimeFor(model)
        val file = runner.exclusive(runtime) {
            runner.speak(model, params, text, destination)
        }
        runner.touch(model.id)
        return ToolResult(
            "Spoke ${text.length} characters with ${model.displayName}.\n" +
                "Saved to: ${file.absolutePath} (${Fmt.bytes(file.length())})",
            providerId = ID,
        )
    }

    private suspend fun transcribe(input: JsonObject): ToolResult {
        val path = input.text("audio")
            ?: return ToolResult("`audio` is required.", isError = true, providerId = ID)
        val file = File(path)
        if (!file.isFile) {
            return ToolResult("There is no file at $path.", isError = true, providerId = ID)
        }
        val model = runner.model(input.text("model").orEmpty())
            ?: runner.defaultFor(Modality.SPEECH_TO_TEXT)
            ?: return ToolResult(
                "No speech model is installed on this device.",
                isError = true,
                providerId = ID,
            )

        var params = runner.paramsFor(model, SparseParams.EMPTY)
        if (input.bool("translate") == true) params = params.with("translate", true)
        input.text("language")?.let { params = params.with("language", it) }

        val runtime = runner.runtimeFor(model)
        val segments = runner.exclusive(runtime) {
            runner.transcribe(model, params, file)
        }
        runner.touch(model.id)
        val text = segments.joinToString(" ") { it.text }.trim()
        return ToolResult(
            text.ifBlank { "The recording produced no words." },
            providerId = ID,
        )
    }

    // — JSON readers, kept private so the schemas below are the only contract —

    private fun JsonObject.text(key: String): String? =
        (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.content?.takeIf { it.isNotBlank() }

    private fun JsonObject.int(key: String): Int? = (this[key] as? JsonPrimitive)?.intOrNull

    private fun JsonObject.number(key: String): Double? = (this[key] as? JsonPrimitive)?.doubleOrNull

    private fun JsonObject.bool(key: String): Boolean? = (this[key] as? JsonPrimitive)?.booleanOrNull

    companion object {
        const val ID = "media"

        /**
         * Every tool this provider can offer, whatever is installed.
         *
         * The screen lists what the provider *is*, not what happens to be
         * installed right now — a row that shrinks because a model was deleted
         * would read as the feature breaking rather than as the model going.
         */
        fun toolNames(): List<String> = listOf(
            GENERATE_IMAGE.name, EDIT_IMAGE.name, GENERATE_VIDEO.name,
            SPEAK.name, TRANSCRIBE.name,
        )

        private const val DEFAULT_STEPS = 8
        private const val DEFAULT_SIZE = 512

        val GENERATE_IMAGE = ToolSpec(
            name = "generate_image",
            description = "Draw a picture from a text description, on this device. " +
                "Returns the file path of the saved PNG. Takes a minute or more.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "prompt": {"type": "string", "description": "What to draw."},
                    "negative": {"type": "string", "description": "What to keep out of it."},
                    "size": {"type": "integer", "description": "Square edge in pixels. Larger is slower."},
                    "steps": {"type": "integer", "description": "Denoising steps. More is slower and usually better."},
                    "seed": {"type": "integer", "description": "Set to repeat a previous picture exactly."},
                    "model": {"type": "string", "description": "A specific installed model id. Omit for the last one used."}
                  },
                  "required": ["prompt"]
                }
            """.trimIndent(),
        )

        val EDIT_IMAGE = ToolSpec(
            name = "edit_image",
            description = "Change an existing picture according to a description. " +
                "Returns the file path of the result.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "image": {"type": "string", "description": "Absolute path of the picture to change."},
                    "prompt": {"type": "string", "description": "What it should become."},
                    "negative": {"type": "string", "description": "What to keep out of it."},
                    "strength": {"type": "number", "description": "0 keeps the original, 1 ignores it. Around 0.6 is a strong edit."},
                    "steps": {"type": "integer", "description": "Denoising steps."},
                    "seed": {"type": "integer", "description": "Set to repeat a previous result exactly."},
                    "model": {"type": "string", "description": "A specific installed model id."}
                  },
                  "required": ["image", "prompt"]
                }
            """.trimIndent(),
        )

        val GENERATE_VIDEO = ToolSpec(
            name = "generate_video",
            description = "Make a short clip from a description, on this device. " +
                "This takes tens of minutes — say so before calling it.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "prompt": {"type": "string", "description": "What should happen in the clip."},
                    "negative": {"type": "string", "description": "What to keep out of it."},
                    "first": {"type": "string", "description": "Path of a picture to start from."},
                    "last": {"type": "string", "description": "Path of a picture to end on."},
                    "model": {"type": "string", "description": "A specific installed model id."}
                  },
                  "required": ["prompt"]
                }
            """.trimIndent(),
        )

        val SPEAK = ToolSpec(
            name = "speak",
            description = "Say something out loud, on this device. Returns the path of a WAV file.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "text": {"type": "string", "description": "What to say."},
                    "voice": {"type": "string", "description": "A voice id. Omit for the model's default."},
                    "speed": {"type": "number", "description": "1.0 is normal."},
                    "model": {"type": "string", "description": "A specific installed voice id."}
                  },
                  "required": ["text"]
                }
            """.trimIndent(),
        )

        val TRANSCRIBE = ToolSpec(
            name = "transcribe",
            description = "Turn a recording into text, on this device. " +
                "Takes the path of an audio or video file already on this device.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "audio": {"type": "string", "description": "Absolute path of the recording."},
                    "language": {"type": "string", "description": "Two-letter code. Omit to detect it."},
                    "translate": {"type": "boolean", "description": "Translate into English rather than transcribing as spoken."},
                    "model": {"type": "string", "description": "A specific installed speech model id."}
                  },
                  "required": ["audio"]
                }
            """.trimIndent(),
        )
    }
}
