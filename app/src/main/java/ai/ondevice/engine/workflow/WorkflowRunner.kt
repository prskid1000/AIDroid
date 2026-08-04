package ai.ondevice.engine.workflow

import ai.ondevice.core.Modality
import ai.ondevice.core.SparseParams
import ai.ondevice.core.workflow.NodeKind
import ai.ondevice.core.workflow.NodeRecord
import ai.ondevice.core.workflow.PortType
import ai.ondevice.core.workflow.WorkflowGraph
import ai.ondevice.core.workflow.WorkflowTemplate
import ai.ondevice.data.ModelStorage
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.engine.DiffusionEngine
import ai.ondevice.engine.DiffusionEvent
import ai.ondevice.engine.DiffusionRequest
import ai.ondevice.engine.EngineManager
import ai.ondevice.engine.EngineMessage
import ai.ondevice.engine.GenerateRequest
import ai.ondevice.engine.GenerationEvent
import ai.ondevice.engine.RuntimeRegistry
import ai.ondevice.engine.Transcriber
import ai.ondevice.engine.VideoRequest
import ai.ondevice.speech.SpeechRequest
import ai.ondevice.speech.SpeechSynthesizer
import ai.ondevice.speech.SynthProvider
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.JsonPrimitive
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/** One value travelling along an edge. Paths, never pixels — see PortType. */
data class PortValue(val type: PortType, val text: String = "", val paths: List<String> = emptyList()) {
    val path: String? get() = paths.firstOrNull()

    companion object {
        fun text(value: String) = PortValue(PortType.TEXT, text = value)
        fun file(type: PortType, path: String) = PortValue(type, paths = listOf(path))
        fun list(paths: List<String>) = PortValue(PortType.LIST, paths = paths)
    }
}

/** What a step is doing, reported the way the media screens already report. */
enum class NodeRunState { WAITING, LOADING, RUNNING, DONE, FAILED, SKIPPED, CANCELLED }

data class NodeProgress(
    val state: NodeRunState = NodeRunState.WAITING,
    val phaseLabel: String = "",
    val step: Int = 0,
    val steps: Int = 0,
    val secondsPerStep: Float = 0f,
    val message: String = "",
)

/**
 * How the runner talks back while it works.
 *
 * A callback rather than a flow because the runner has to report from inside
 * several different engines' own flows, and threading one flow through all of
 * them buys nothing the caller can use.
 */
interface RunReporter {
    fun onNode(nodeId: String, progress: NodeProgress)
    fun onLoading(what: List<String>, stage: String?)
    fun onUnload(because: String)
    /** A step wants a person to choose. Returns the chosen path, or null to stop. */
    suspend fun awaitChoice(nodeId: String, options: List<String>): String?
}

/**
 * Runs a graph.
 *
 * The one thing this class exists to do that no existing code does: **unload
 * the outgoing engine before loading an incoming one of a different runtime**.
 * Each engine already guards itself — the diffusion engine unloads inside the
 * lock it loads under, the llama manager warm-swaps — but nothing has ever
 * asked the *other* engine to let go, because until now no single run used
 * two. A chat model and a diffusion model resident together is roughly
 * fourteen gigabytes on a phone with fifteen, and that is a kill rather than
 * an error.
 */
@Singleton
class WorkflowRunner @Inject constructor(
    private val db: OnDeviceDatabase,
    private val engines: EngineManager,
    private val diffusion: DiffusionEngine,
    private val transcriber: Transcriber,
    private val synthesizer: SpeechSynthesizer,
    private val storage: ModelStorage,
    private val toolProviders: ai.ondevice.tools.ToolProviderFactory,
    private val prefs: ai.ondevice.data.prefs.AppPrefs,
) {

    /** Which runtime is holding weights right now, so we know when to let go. */
    private var resident: String? = null

    /** Set while a step is inside an engine, so Cancel can reach the native call. */
    @Volatile
    var activeCancel: (() -> Unit)? = null
        private set

    suspend fun run(
        runId: String,
        graph: WorkflowGraph,
        reporter: RunReporter,
    ): Result<Map<String, PortValue>> = runCatching {
        val models = db.models().getInstalled().associateBy { it.id }
        val outputs = LinkedHashMap<String, PortValue>()
        val outDir = File(storage.root(), "workflows/$runId").apply { mkdirs() }

        var index = 0
        while (index < graph.nodes.size) {
            val node = graph.nodes[index]
            if (!node.enabled) {
                reporter.onNode(node.id, NodeProgress(NodeRunState.SKIPPED))
                index++
                continue
            }

            when (NodeKind.of(node.type)) {
                // A bracket that repeats the span below it. Bounded always —
                // an unbounded loop where one pass is minutes is a way to cook
                // the phone, so the count is required and the condition may
                // only cut it short.
                NodeKind.RepeatStart -> {
                    val end = spanEnd(graph, index, NodeKind.RepeatEnd.type)
                    val times = node.params.int("times", 2).coerceIn(1, MAX_REPEATS)
                    repeat(times) { pass ->
                        outputs["${node.id}:pass"] = PortValue.text((pass + 1).toString())
                        runSpan(graph, index + 1, end, node.id, outputs, outDir, models, reporter)
                    }
                    index = end + 1
                }

                NodeKind.ForEachStart -> {
                    val end = spanEnd(graph, index, NodeKind.ForEachEnd.type)
                    val items = resolve(node, "items", outputs, graph)?.paths.orEmpty()
                    val collected = mutableListOf<String>()
                    items.take(MAX_REPEATS).forEach { item ->
                        outputs["${node.id}:item"] = PortValue.file(PortType.FILE, item)
                        runSpan(graph, index + 1, end, node.id, outputs, outDir, models, reporter)
                        graph.nodes.getOrNull(end)?.let { endNode ->
                            resolve(endNode, "collect", outputs, graph)?.path?.let(collected::add)
                        }
                    }
                    graph.nodes.getOrNull(end)?.let { outputs["${it.id}:items"] = PortValue.list(collected) }
                    index = end + 1
                }

                NodeKind.Batch -> {
                    val end = spanEnd(graph, index, NodeKind.RepeatEnd.type)
                    val times = node.params.int("times", 2).coerceIn(1, MAX_REPEATS)
                    repeat(times) { pass ->
                        // A fresh seed per pass, and the model stays resident
                        // across all of them — the whole point of this node.
                        outputs["${node.id}:seed"] =
                            PortValue.text((System.currentTimeMillis() + pass).toString())
                        runSpan(graph, index + 1, end, node.id, outputs, outDir, models, reporter)
                    }
                    index = end + 1
                }

                NodeKind.Branch -> {
                    val condition = node.params.string("condition", "")
                    val rendered = WorkflowTemplate.render(condition) { reference ->
                        outputs[reference]?.let { it.text.ifBlank { it.path.orEmpty() } }
                    }
                    if (!WorkflowTemplate.truthy(rendered)) {
                        // Skip to the matching end, marking what was skipped.
                        val end = spanEnd(graph, index, NodeKind.RepeatEnd.type)
                        (index..end).forEach {
                            graph.nodes.getOrNull(it)?.let { n ->
                                reporter.onNode(n.id, NodeProgress(NodeRunState.SKIPPED))
                            }
                        }
                        index = end + 1
                    } else {
                        index++
                    }
                }

                NodeKind.RepeatEnd, NodeKind.ForEachEnd, NodeKind.Note -> index++

                else -> {
                    runNode(node, graph, outputs, outDir, models, reporter)
                    index++
                }
            }
        }
        outputs
    }

    private suspend fun runSpan(
        graph: WorkflowGraph,
        from: Int,
        toExclusive: Int,
        @Suppress("UNUSED_PARAMETER") owner: String,
        outputs: MutableMap<String, PortValue>,
        outDir: File,
        models: Map<String, ModelEntity>,
        reporter: RunReporter,
    ) {
        for (i in from until toExclusive) {
            val node = graph.nodes.getOrNull(i) ?: return
            if (!node.enabled) continue
            when (NodeKind.of(node.type)) {
                NodeKind.RepeatEnd, NodeKind.ForEachEnd, NodeKind.Note -> Unit
                else -> runNode(node, graph, outputs, outDir, models, reporter)
            }
        }
    }

    /** Where a bracket closes, or the end of the list when it never does. */
    private fun spanEnd(graph: WorkflowGraph, from: Int, endType: String): Int {
        var depth = 0
        for (i in (from + 1) until graph.nodes.size) {
            val type = graph.nodes[i].type
            if (type == NodeKind.RepeatStart.type || type == NodeKind.ForEachStart.type ||
                type == NodeKind.Batch.type || type == NodeKind.Branch.type
            ) {
                depth++
            }
            if (type == endType || type == NodeKind.RepeatEnd.type || type == NodeKind.ForEachEnd.type) {
                if (depth == 0) return i
                depth--
            }
        }
        return graph.nodes.size - 1
    }

    private suspend fun runNode(
        node: NodeRecord,
        graph: WorkflowGraph,
        outputs: MutableMap<String, PortValue>,
        outDir: File,
        models: Map<String, ModelEntity>,
        reporter: RunReporter,
    ) {
        reporter.onNode(node.id, NodeProgress(NodeRunState.RUNNING))
        runCatching {
            when (val kind = NodeKind.of(node.type)) {
                NodeKind.Input, NodeKind.LibraryItem -> {
                    val type = PortType.valueOf(node.params.string("portType", "TEXT"))
                    val value = if (type == PortType.TEXT) {
                        PortValue.text(node.params.string("text", ""))
                    } else {
                        PortValue.file(type, node.params.string("path", ""))
                    }
                    outputs["${node.id}:value"] = value
                    outputs[node.id] = value
                }

                NodeKind.Script -> {
                    /*
                     * Two languages, and which one is used is the author's
                     * choice rather than a guess at their intent.
                     *
                     * The template covers what most steps want — put this
                     * step's answer inside that sentence — and reads as the
                     * text it produces, which a person editing a prompt can
                     * see at a glance. JavaScript covers the rest: anything
                     * with a loop, a condition, or a shape to build.
                     *
                     * Keeping both is not indecision. A template that has to
                     * be written as a program to interpolate one value is a
                     * worse template, and a program that has to be written as
                     * a template to branch is not a program at all.
                     */
                    val script = node.params.string("script")
                    val text = if (script.isNotBlank()) {
                        ai.ondevice.engine.QuickJsBridge.eval(
                            source = script,
                            steps = stepsFor(outputs),
                            timeoutMillis = node.params.int("timeout_ms", 2_000).toLong(),
                        ).getOrThrow()
                    } else {
                        WorkflowTemplate.render(node.params.string("template", "")) { ref ->
                            outputs[ref]?.let { it.text.ifBlank { it.path.orEmpty() } }
                        }
                    }
                    put(outputs, node, "text", PortValue.text(text))
                }

                NodeKind.Extract -> {
                    val source = resolve(node, "text", outputs, graph)?.text.orEmpty()
                    val pattern = node.params.string("pattern", "")
                    val found = runCatching {
                        Regex(pattern).find(source)?.let { m ->
                            m.groupValues.getOrNull(1)?.takeIf { it.isNotEmpty() } ?: m.value
                        }
                    }.getOrNull().orEmpty()
                    put(outputs, node, "text", PortValue.text(found))
                }

                NodeKind.TextSplit -> {
                    val source = resolve(node, "text", outputs, graph)?.text.orEmpty()
                    val pieces = when (node.params.string("by", "paragraph")) {
                        "sentence" -> source.split(Regex("(?<=[.!?])\\s+"))
                        "line" -> source.lines()
                        else -> source.split(Regex("\\n\\s*\\n"))
                    }.map { it.trim() }.filter { it.isNotEmpty() }
                    // Pieces travel as a list of inline strings; a list of text
                    // needs no files behind it.
                    outputs["${node.id}:pieces"] = PortValue(PortType.LIST, paths = pieces)
                }

                NodeKind.TextJoin -> {
                    val pieces = resolve(node, "pieces", outputs, graph)?.paths.orEmpty()
                    val joined = pieces.joinToString(node.params.string("separator", "\n\n"))
                    put(outputs, node, "text", PortValue.text(joined))
                }

                NodeKind.Pick -> {
                    val options = resolve(node, "options", outputs, graph)?.paths.orEmpty()
                    val chosen = reporter.awaitChoice(node.id, options)
                        ?: throw IllegalStateException("Nothing was chosen.")
                    put(outputs, node, "chosen", PortValue.file(PortType.FILE, chosen))
                }

                NodeKind.Output -> {
                    val value = resolve(node, "value", outputs, graph)
                    outputs["${node.id}:kept"] = value ?: PortValue.text("")
                }

                NodeKind.Processor -> runProcessor(node, graph, outputs, outDir, models, reporter)

                /*
                 * The cheap ones — no model, no load, milliseconds each.
                 *
                 * They carry more of this feature's weight than they look
                 * like they should. Fitting a picture to the size the next
                 * model wants is the difference between a scene and a smear;
                 * pulling one frame out of a clip is the only way video
                 * composes with anything; and splitting long text is what
                 * makes an hour of transcript fit a context window at all.
                 */
                NodeKind.Resize -> {
                    val source = resolve(node, "image", outputs, graph)?.path
                        ?: throw IllegalStateException("No picture was given to this step.")
                    val width = node.params.int("width", 0)
                    val height = node.params.int("height", 0)
                    val mode = node.params.string("mode", "fit")
                    val file = File(outDir, "${'$'}{node.id}.png")
                    resizeTo(File(source), file, width, height, mode)
                    put(outputs, node, "image", PortValue.file(PortType.IMAGE, file.absolutePath))
                }

                NodeKind.FrameExtract -> {
                    val clip = resolve(node, "clip", outputs, graph)
                        ?: throw IllegalStateException("No clip was given to this step.")
                    val frames = clip.paths
                    if (frames.isEmpty()) throw IllegalStateException("That clip has no frames.")
                    // Negative counts back from the end, so "the last frame"
                    // does not need to know how long the clip turned out.
                    val asked = node.params.int("index", 0)
                    val at = (if (asked < 0) frames.size + asked else asked)
                        .coerceIn(0, frames.lastIndex)
                    val file = File(outDir, "${'$'}{node.id}.png")
                    File(frames[at]).copyTo(file, overwrite = true)
                    put(outputs, node, "image", PortValue.file(PortType.IMAGE, file.absolutePath))
                }

                NodeKind.Assemble -> {
                    val images = resolve(node, "images", outputs, graph)?.paths.orEmpty()
                    if (images.isEmpty()) throw IllegalStateException("No pictures to assemble.")
                    val dir = File(outDir, node.id).apply { mkdirs() }
                    val frames = images.mapIndexed { i, path ->
                        val target = File(dir, String.format("frame_%04d.png", i))
                        File(path).copyTo(target, overwrite = true)
                        target.absolutePath
                    }
                    outputs["${'$'}{node.id}:clip"] = PortValue(PortType.CLIP, paths = frames)
                }

                NodeKind.Tool -> {
                    val name = node.params.string("tool")
                    if (name.isBlank()) throw IllegalStateException("No tool chosen for this step.")
                    val arguments = resolve(node, "arguments", outputs, graph)?.text
                        ?: node.params.string("arguments", "{}")
                    val registry = toolProviders.registry(
                        enabled = prefs.enabledToolProviders.first(),
                    )
                    val result = registry.call(name, arguments.ifBlank { "{}" })
                    if (result.isError) throw IllegalStateException(result.text)
                    put(outputs, node, "text", PortValue.text(result.text))
                }

                is NodeKind.Unknown -> throw IllegalStateException(
                    "This step was saved by a newer version of the app and cannot be run here.",
                )

                else -> Unit
            }
        }.onSuccess {
            reporter.onNode(node.id, NodeProgress(NodeRunState.DONE))
        }.onFailure { failure ->
            reporter.onNode(
                node.id,
                NodeProgress(NodeRunState.FAILED, message = failure.message.orEmpty()),
            )
            throw failure
        }
    }

    // ── models ───────────────────────────────────────────────────────────

    private suspend fun runProcessor(
        node: NodeRecord,
        graph: WorkflowGraph,
        outputs: MutableMap<String, PortValue>,
        outDir: File,
        models: Map<String, ModelEntity>,
        reporter: RunReporter,
    ) {
        val model = ResidencyPlanner.modelFor(node, models)
            ?: throw IllegalStateException("No model chosen for this step.")
        val runtimeId = ResidencyPlanner.runtimeFor(model)

        makeRoomFor(runtimeId, reporter)
        reporter.onNode(node.id, NodeProgress(NodeRunState.LOADING))
        reporter.onLoading(listOf(model.label), null)

        val stored = SparseParams.parse(model.paramOverridesJson)
        // The editor keeps its own bookkeeping in the same bag — which model
        // was chosen, what shape that made the step — and none of it is a
        // runtime parameter. Passing them through would have every run report
        // a handful of rejected keys it could do nothing about.
        val settings = node.params.toMap().filterKeys { it !in BOOKKEEPING }
        val params = stored.overlaidWith(SparseParams(settings))

        when (model.modality) {
            Modality.TEXT, Modality.VISION ->
                runText(node, graph, outputs, model, params, reporter)
            Modality.DIFFUSION ->
                runDiffusion(node, graph, outputs, outDir, model, params, reporter)
            Modality.SPEECH_TO_TEXT ->
                runTranscribe(node, graph, outputs, outDir, model, params, reporter)
            Modality.TEXT_TO_SPEECH ->
                runSpeak(node, graph, outputs, outDir, model, params)
            else -> throw IllegalStateException("This model cannot be run as a step.")
        }
        resident = runtimeId
    }

    /**
     * Let the other engine go before this one loads.
     *
     * The line that does not exist anywhere else in this app, and the reason a
     * workflow can cross runtimes at all. Each engine unloads *itself* before
     * loading — none of them knows about the others, because until now none of
     * them had to.
     */
    private suspend fun makeRoomFor(runtimeId: String, reporter: RunReporter) {
        val holding = resident
        if (holding == null || holding == runtimeId) return

        val because = "a workflow step needs $runtimeId"
        reporter.onUnload(because)
        when (holding) {
            RuntimeRegistry.STABLE_DIFFUSION -> diffusion.unload(because)
            RuntimeRegistry.LLAMA -> engines.unload()
            RuntimeRegistry.WHISPER -> transcriber.unload()
            RuntimeRegistry.KOKORO, RuntimeRegistry.OMNIVOICE -> {
                synthesizer.unload(SynthProvider.KOKORO)
                synthesizer.unload(SynthProvider.OMNIVOICE)
            }
        }
        resident = null
    }

    private suspend fun runText(
        node: NodeRecord,
        graph: WorkflowGraph,
        outputs: MutableMap<String, PortValue>,
        model: ModelEntity,
        params: SparseParams,
        reporter: RunReporter,
    ) {
        engines.load(model, SparseParams.parse(model.paramOverridesJson)).getOrThrow()
        val engine = engines.llama ?: throw IllegalStateException("The text runtime is not installed.")
        engine.applyParams(params)

        val prompt = resolve(node, "prompt", outputs, graph)?.text.orEmpty()
        val system = resolve(node, "system", outputs, graph)?.text
        val image = resolve(node, "image", outputs, graph)?.path

        // The interface has no cancel — only the concrete engine does, which
        // is also how the chat screen reaches it.
        activeCancel = { (engine as? ai.ondevice.engine.LlamaEngine)?.cancel() }
        val answer = StringBuilder()
        var thought = 0
        try {
            engine.generate(
                GenerateRequest(
                    messages = listOf(EngineMessage(role = "user", content = prompt)),
                    params = params,
                    systemPrompt = system,
                    imagePaths = listOfNotNull(image),
                ),
            ).collect { event ->
                when (event) {
                    is GenerationEvent.Token -> answer.append(event.text)
                    // Counted, not kept. A reasoning model can spend its whole
                    // budget here and answer with nothing, and the step after
                    // is then handed an empty string — which fails somewhere
                    // else entirely, saying something that sounds unrelated.
                    is GenerationEvent.ThinkingDelta -> thought += event.text.length
                    is GenerationEvent.Failed -> throw IllegalStateException(event.message)
                    else -> Unit
                }
            }
        } finally {
            activeCancel = null
        }
        val text = answer.toString().trim()
        if (text.isEmpty()) {
            // Said here, where it happened, rather than two steps later.
            throw IllegalStateException(
                if (thought > 0) {
                    "This model reasoned for ${'$'}thought characters and then answered with " +
                        "nothing — the whole budget went to thinking. Raise the token limit, " +
                        "or turn thinking off in the model's parameters."
                } else {
                    "This model answered with nothing."
                },
            )
        }
        put(outputs, node, "text", PortValue.text(text))
    }

    private suspend fun runDiffusion(
        node: NodeRecord,
        graph: WorkflowGraph,
        outputs: MutableMap<String, PortValue>,
        outDir: File,
        model: ModelEntity,
        params: SparseParams,
        reporter: RunReporter,
    ) {
        diffusion.load(model.id, model.localPath, emptyList(), params = SparseParams.parse(model.paramOverridesJson))
            .getOrThrow()

        val prompt = resolve(node, "prompt", outputs, graph)?.text.orEmpty()
        val negative = resolve(node, "negative", outputs, graph)?.text.orEmpty()
        val makesVideo = diffusion.supportsVideo

        val withPrompt = params
            .with("prompt", prompt)
            .with("negative_prompt", negative)

        activeCancel = { diffusion.cancel() }
        try {
            if (makesVideo) {
                var written: List<String> = emptyList()
                diffusion.generateVideo(
                    VideoRequest(
                        params = withPrompt,
                        initImageUri = resolve(node, "first", outputs, graph)?.path,
                        endImageUri = resolve(node, "last", outputs, graph)?.path,
                    ),
                ).collect { event ->
                    when (event) {
                        is DiffusionEvent.Progress -> reporter.onNode(
                            node.id,
                            NodeProgress(
                                NodeRunState.RUNNING,
                                phaseLabel = event.phase.label,
                                step = event.step,
                                steps = event.steps,
                                secondsPerStep = event.secondsPerStep,
                            ),
                        )
                        is DiffusionEvent.ClipCompleted -> written = event.clip.frames
                        is DiffusionEvent.Failed -> throw IllegalStateException(event.message)
                        else -> Unit
                    }
                }
                outputs["${node.id}:clip"] = PortValue(PortType.CLIP, paths = written)
            } else {
                var savedPath: String? = null
                diffusion.generate(
                    DiffusionRequest(
                        params = withPrompt,
                        initImageUri = resolve(node, "init", outputs, graph)?.path,
                        controlImageUri = resolve(node, "control", outputs, graph)?.path,
                        maskPngPath = resolve(node, "mask", outputs, graph)?.path,
                    ),
                ).collect { event ->
                    when (event) {
                        is DiffusionEvent.Progress -> reporter.onNode(
                            node.id,
                            NodeProgress(
                                NodeRunState.RUNNING,
                                phaseLabel = event.phase.label,
                                step = event.step,
                                steps = event.steps,
                                secondsPerStep = event.secondsPerStep,
                            ),
                        )
                        is DiffusionEvent.Completed -> {
                            val file = File(outDir, "${node.id}.png")
                            file.outputStream().use { out ->
                                event.image.toBitmap()
                                    .compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)
                            }
                            savedPath = file.absolutePath
                        }
                        is DiffusionEvent.Failed -> throw IllegalStateException(event.message)
                        else -> Unit
                    }
                }
                put(
                    outputs, node, "image",
                    PortValue.file(PortType.IMAGE, savedPath ?: error("The run produced no picture.")),
                )
            }
        } finally {
            activeCancel = null
        }
    }

    private suspend fun runTranscribe(
        node: NodeRecord,
        graph: WorkflowGraph,
        outputs: MutableMap<String, PortValue>,
        outDir: File,
        model: ModelEntity,
        params: SparseParams,
        @Suppress("UNUSED_PARAMETER") reporter: RunReporter,
    ) {
        if (!transcriber.isCurrent(model.id)) {
            transcriber.load(model.id, model.localPath, params).getOrThrow()
        }
        val audio = resolve(node, "audio", outputs, graph)?.path
            ?: throw IllegalStateException("No recording was given to this step.")
        activeCancel = { transcriber.cancel() }
        val segments = try {
            transcriber.transcribeFile(File(audio)).getOrThrow()
        } finally {
            activeCancel = null
        }
        val text = segments.joinToString(" ") { it.text }.trim()
        put(outputs, node, "text", PortValue.text(text))

        val subtitles = File(outDir, "${node.id}.srt")
        subtitles.writeText(ai.ondevice.core.TranscriptSegments.encode(segments))
        outputs["${node.id}:subtitles"] = PortValue.file(PortType.FILE, subtitles.absolutePath)
    }

    private suspend fun runSpeak(
        node: NodeRecord,
        graph: WorkflowGraph,
        outputs: MutableMap<String, PortValue>,
        outDir: File,
        model: ModelEntity,
        params: SparseParams,
    ) {
        val text = resolve(node, "text", outputs, graph)?.text.orEmpty()
        val directory = File(model.localPath).let { if (it.isDirectory) it else it.parentFile }
        synthesizer.useKokoroModel(directory)
        synthesizer.useOmniVoiceModel(directory)
        val destination = File(outDir, "${node.id}.wav")
        // Which engine, asked of the folder rather than assumed.
        //
        // This defaulted to Kokoro whatever was chosen, so pointing a step at
        // OmniVoice ran Kokoro instead — quietly, since both make a WAV. The
        // synthesiser can tell them apart by what is in the directory, which
        // is the same structural test the Voice tab uses and needs no names.
        val provider = when {
            params.string("provider") == SynthProvider.OMNIVOICE.name.lowercase() ->
                SynthProvider.OMNIVOICE
            params.string("provider") == SynthProvider.KOKORO.name.lowercase() ->
                SynthProvider.KOKORO
            directory != null && synthesizer.omniVoiceLooksInstalled(directory) ->
                SynthProvider.OMNIVOICE
            else -> SynthProvider.KOKORO
        }
        synthesizer.synthesizeToFile(
            SpeechRequest(
                text = text,
                voiceId = params.string("voice"),
                speed = params.float("speed") ?: 1.0f,
                provider = provider,
            ),
            destination,
        ).getOrThrow()
        put(outputs, node, "audio", PortValue.file(PortType.AUDIO, destination.absolutePath))
    }

    /**
     * Fit a picture to a size, by one of three readings of "fit".
     *
     * `fit` keeps the whole picture and lets the shape change; `cover` keeps
     * the shape and loses the edges; `stretch` keeps neither. Which one is
     * wanted depends on whether the next model cares more about seeing
     * everything or about being handed the aspect ratio it was trained on,
     * and that is not something this code can guess.
     */
    private fun resizeTo(source: File, target: File, width: Int, height: Int, mode: String) {
        val bitmap = android.graphics.BitmapFactory.decodeFile(source.absolutePath)
            ?: throw IllegalStateException("That file is not a picture this device can read.")
        val w = if (width > 0) width else bitmap.width
        val h = if (height > 0) height else bitmap.height

        val out = when (mode) {
            "stretch" -> android.graphics.Bitmap.createScaledBitmap(bitmap, w, h, true)
            "cover" -> {
                val scale = maxOf(w.toFloat() / bitmap.width, h.toFloat() / bitmap.height)
                val scaled = android.graphics.Bitmap.createScaledBitmap(
                    bitmap,
                    (bitmap.width * scale).toInt().coerceAtLeast(w),
                    (bitmap.height * scale).toInt().coerceAtLeast(h),
                    true,
                )
                android.graphics.Bitmap.createBitmap(
                    scaled,
                    ((scaled.width - w) / 2).coerceAtLeast(0),
                    ((scaled.height - h) / 2).coerceAtLeast(0),
                    w.coerceAtMost(scaled.width),
                    h.coerceAtMost(scaled.height),
                )
            }
            else -> {
                val scale = minOf(w.toFloat() / bitmap.width, h.toFloat() / bitmap.height)
                android.graphics.Bitmap.createScaledBitmap(
                    bitmap,
                    (bitmap.width * scale).toInt().coerceAtLeast(1),
                    (bitmap.height * scale).toInt().coerceAtLeast(1),
                    true,
                )
            }
        }
        target.outputStream().use {
            out.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it)
        }
    }

    // ── plumbing ─────────────────────────────────────────────────────────

    private fun put(
        outputs: MutableMap<String, PortValue>,
        node: NodeRecord,
        name: String,
        value: PortValue,
    ) {
        outputs["${node.id}:$name"] = value
        outputs[node.id] = value
    }

    /**
     * Everything produced so far, in the shape a script sees it.
     *
     * `steps["2"].text` and `steps["2"].path`, keyed by the step's own id and
     * by `id:output` both, so a script can name a value the same way a slot
     * binding does. Paths and not bytes, for the reason PortType gives.
     */
    private fun stepsFor(outputs: Map<String, PortValue>): kotlinx.serialization.json.JsonObject =
        kotlinx.serialization.json.JsonObject(
            outputs.mapValues { (_, value) ->
                kotlinx.serialization.json.JsonObject(
                    mapOf(
                        "type" to kotlinx.serialization.json.JsonPrimitive(value.type.name),
                        "text" to kotlinx.serialization.json.JsonPrimitive(value.text),
                        "path" to kotlinx.serialization.json.JsonPrimitive(value.path.orEmpty()),
                        "paths" to kotlinx.serialization.json.JsonArray(
                            value.paths.map { kotlinx.serialization.json.JsonPrimitive(it) },
                        ),
                    ),
                )
            },
        )

    /** What is bound to a slot, or null when nothing is. */
    private fun resolve(
        node: NodeRecord,
        slot: String,
        outputs: Map<String, PortValue>,
        @Suppress("UNUSED_PARAMETER") graph: WorkflowGraph,
    ): PortValue? {
        val reference = node.slots[slot] ?: return null
        return outputs[reference] ?: outputs[reference.substringBefore(':')]
    }

    private fun kotlinx.serialization.json.JsonObject.int(key: String, fallback: Int): Int =
        (this[key] as? JsonPrimitive)?.content?.toIntOrNull() ?: fallback

    private fun kotlinx.serialization.json.JsonObject.string(key: String, fallback: String = ""): String =
        (this[key] as? JsonPrimitive)?.content ?: fallback

    private fun kotlinx.serialization.json.JsonObject.toMap(): Map<String, kotlinx.serialization.json.JsonElement> =
        entries.associate { it.key to it.value }

    private companion object {
        /**
         * A ceiling on any loop.
         *
         * Not a guess at what anyone wants — a bound on what a mistake can
         * cost. One diffusion pass here is minutes, so a loop with a typo in
         * its count is an afternoon of a hot phone.
         */
        const val MAX_REPEATS = 32

        /** Editor-only keys, kept out of what reaches an engine. */
        val BOOKKEEPING = setOf("model", "shape", "portType", "times", "by", "separator", "template", "pattern", "condition", "mode", "tool", "script", "timeout_ms")
    }
}
