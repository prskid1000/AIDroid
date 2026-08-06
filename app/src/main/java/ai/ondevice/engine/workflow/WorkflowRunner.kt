package ai.ondevice.engine.workflow

import ai.ondevice.core.Export
import ai.ondevice.core.Modality
import ai.ondevice.core.SparseParams
import ai.ondevice.core.workflow.NodeKind
import ai.ondevice.core.workflow.NodeRecord
import ai.ondevice.core.workflow.PortType
import ai.ondevice.core.workflow.Spans
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
data class PortValue(
    val type: PortType,
    val text: String = "",
    val paths: List<String> = emptyList(),
    /**
     * For a LIST, what one of its items is.
     *
     * A list carries its items in [paths] whatever they are, because a list of
     * four file paths and a list of four paragraphs are the same shape. They
     * are not the same *kind*, though, and nothing downstream could tell them
     * apart: a for-each over split text handed each paragraph on as a file
     * path, so the model reading it got an empty prompt and said so in a way
     * that sounded like a different problem entirely. The element type travels
     * with the value — a port that takes "many of whatever this is" is the only
     * kind this app needs, so the distinction cannot live on the port.
     */
    val elementType: PortType = PortType.FILE,
) {
    val path: String? get() = paths.firstOrNull()

    /**
     * What this is worth as text, whichever half it arrived in.
     *
     * A list reads as all of it. Taking the first item was the obvious
     * reading of "the path", and it meant a template naming a list quietly
     * showed one entry of four with nothing to say the rest had been dropped.
     */
    val asText: String get() = when {
        text.isNotBlank() -> text
        type == PortType.LIST -> paths.joinToString(", ")
        else -> path.orEmpty()
    }

    /** One item of a list, as the kind of value it actually is. */
    fun item(value: String): PortValue =
        if (elementType == PortType.TEXT) text(value) else file(elementType, value)

    companion object {
        fun text(value: String) = PortValue(PortType.TEXT, text = value)
        fun file(type: PortType, path: String) = PortValue(type, paths = listOf(path))
        fun list(paths: List<String>, elementType: PortType = PortType.FILE) =
            PortValue(PortType.LIST, paths = paths, elementType = elementType)
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

    /**
     * A step wants something to leave the app.
     *
     * Reported rather than done here for the reason written on [Handoff]: the
     * runner has no activity behind it, and starting one from the background is
     * refused. The caller decides between firing it now and parking it in a
     * notification.
     */
    fun onHandoff(handoff: Handoff)
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

    /** The seed a Batch has set for the pass being run, if one has. */
    private var batchSeed: Long? = null

    /** Everything one run needs, so the recursion carries one parameter. */
    private class RunContext(
        val graph: WorkflowGraph,
        val outputs: MutableMap<String, PortValue>,
        val outDir: File,
        val models: Map<String, ModelEntity>,
        val reporter: RunReporter,
    ) {
        val types: List<String> = graph.nodes.map { it.type }
    }

    suspend fun run(
        runId: String,
        graph: WorkflowGraph,
        reporter: RunReporter,
    ): Result<Map<String, PortValue>> = runCatching {
        val models = db.models().getInstalled().associateBy { it.id }
        val outputs = LinkedHashMap<String, PortValue>()
        val outDir = File(storage.root(), "workflows/$runId").apply { mkdirs() }
        batchSeed = null
        execute(RunContext(graph, outputs, outDir, models, reporter), 0, graph.nodes.size, "")
        outputs
    }

    /**
     * Run the steps in `[from, toExclusive)` — the whole graph at the top, and
     * one loop's body inside it.
     *
     * Recursive, and that is the point. A loop body used to be run by a flat
     * `for` that handed every step to the step runner, so a Repeat inside a
     * For-each matched no arm there and did nothing at all: the inner loop's
     * body never ran, and the run reported success. A body is a graph, so the
     * thing that runs a graph is the thing that runs a body.
     */
    private suspend fun execute(
        c: RunContext,
        from: Int,
        toExclusive: Int,
        passLabel: String,
    ) {
        val graph = c.graph
        val outputs = c.outputs
        val outDir = c.outDir
        val models = c.models
        val reporter = c.reporter

        var index = from
        while (index < toExclusive) {
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
                    val end = Spans.end(c.types, index, toExclusive)
                    val times = node.params.int("times", 2).coerceIn(1, MAX_REPEATS)
                    val gathered = mutableListOf<String>()
                    var itemType = PortType.FILE
                    repeat(times) { pass ->
                        outputs["${node.id}:pass"] = PortValue.text((pass + 1).toString())
                        execute(c, index + 1, end, "$passLabel.${pass + 1}")
                        collectPass(c, end, gathered)?.let { itemType = it }
                    }
                    publishGathered(c, end, gathered, itemType)
                    markDone(c, index, end)
                    index = end + 1
                }

                NodeKind.ForEachStart -> {
                    val end = Spans.end(c.types, index, toExclusive)
                    val list = resolve(node, "items", outputs, graph)
                    val items = list?.paths.orEmpty()
                    val gathered = mutableListOf<String>()
                    var itemType = PortType.FILE
                    if (items.size > MAX_REPEATS) {
                        // Said, not swallowed. A loop that quietly does 32 of
                        // 200 reads as "it worked" and is found much later.
                        reporter.onNode(
                            node.id,
                            NodeProgress(
                                NodeRunState.RUNNING,
                                message = "Only the first $MAX_REPEATS of ${items.size} items " +
                                    "will be run — that is this app's ceiling on a loop.",
                            ),
                        )
                    }
                    items.take(MAX_REPEATS).forEachIndexed { pass, item ->
                        // As the kind of value it is. A list of paragraphs and
                        // a list of file paths are the same shape and not the
                        // same thing; see PortValue.elementType.
                        outputs["${node.id}:item"] = (list ?: PortValue.list(emptyList())).item(item)
                        execute(c, index + 1, end, "$passLabel.${pass + 1}")
                        collectPass(c, end, gathered)?.let { itemType = it }
                    }
                    publishGathered(c, end, gathered, itemType)
                    markDone(c, index, end)
                    index = end + 1
                }

                NodeKind.Batch -> {
                    val end = Spans.end(c.types, index, toExclusive)
                    val times = node.params.int("times", 2).coerceIn(1, MAX_REPEATS)
                    val gathered = mutableListOf<String>()
                    var itemType = PortType.FILE
                    repeat(times) { pass ->
                        // A fresh seed per pass, and the model stays resident
                        // across all of them — the whole point of this node.
                        //
                        // Set on the runner rather than only published as an
                        // output: nothing binds a slot to a seed, so a value
                        // sitting in the map was a promise the run never kept
                        // and every pass sampled identically.
                        batchSeed = System.currentTimeMillis() + pass
                        outputs["${node.id}:seed"] = PortValue.text(batchSeed.toString())
                        execute(c, index + 1, end, "$passLabel.${pass + 1}")
                        collectPass(c, end, gathered)?.let { itemType = it }
                    }
                    batchSeed = null
                    publishGathered(c, end, gathered, itemType)
                    markDone(c, index, end)
                    index = end + 1
                }

                NodeKind.Branch -> {
                    val condition = node.params.string("condition", "")
                    val rendered = WorkflowTemplate.render(condition) { reference ->
                        lookup(outputs, reference)
                    }
                    if (!WorkflowTemplate.truthy(rendered)) {
                        // Skip to the matching end, marking what was skipped.
                        val end = Spans.end(c.types, index, toExclusive)
                        (index..end.coerceAtMost(graph.nodes.lastIndex)).forEach {
                            graph.nodes.getOrNull(it)?.let { n ->
                                reporter.onNode(n.id, NodeProgress(NodeRunState.SKIPPED))
                            }
                        }
                        index = end + 1
                    } else {
                        // Taken. Said so, rather than left looking unvisited.
                        markDone(c, index)
                        index++
                    }
                }

                // Not steps, but they are rows in the list and a row that
                // never changes reads as one that did not run.
                NodeKind.RepeatEnd, NodeKind.ForEachEnd, NodeKind.Note -> {
                    markDone(c, index)
                    index++
                }

                else -> {
                    runNode(node, graph, outputs, outDir, models, reporter, passLabel)
                    index++
                }
            }
        }
    }

    /**
     * What one pass of a loop kept, taken from the closing step's slot.
     *
     * As text *or* as a path, whichever the value carries. It used to read the
     * path only, so a loop collecting anything a model said gathered nothing
     * and handed on an empty list — a failure that looks exactly like a loop
     * that never ran.
     */
    private fun collectPass(c: RunContext, end: Int, into: MutableList<String>): PortType? {
        val closer = c.graph.nodes.getOrNull(end) ?: return null
        val value = resolve(closer, "collect", c.outputs, c.graph) ?: return null
        val piece = value.asText
        if (piece.isBlank()) return null
        into += piece
        return value.type
    }

    /** Hand the whole harvest to the closing step, for whatever comes after. */
    private fun publishGathered(
        c: RunContext,
        end: Int,
        gathered: List<String>,
        itemType: PortType,
    ) {
        val closer = c.graph.nodes.getOrNull(end) ?: return
        c.outputs["${closer.id}:items"] = PortValue.list(gathered, itemType)
    }

    /**
     * Say that a bracket finished.
     *
     * The brackets are not steps and were never reported on, so they sat at
     * "waiting" for the whole run and the tally counted them against it: a
     * for-each that did everything asked of it finished at "5 of 7 done",
     * which reads as two steps quietly failing rather than as success.
     */
    private fun markDone(c: RunContext, vararg indices: Int) {
        indices.forEach { at ->
            c.graph.nodes.getOrNull(at)?.let {
                c.reporter.onNode(it.id, NodeProgress(NodeRunState.DONE))
            }
        }
    }

    private suspend fun runNode(
        node: NodeRecord,
        graph: WorkflowGraph,
        outputs: MutableMap<String, PortValue>,
        outDir: File,
        models: Map<String, ModelEntity>,
        reporter: RunReporter,
        /**
         * Which pass of which loop this is, as a suffix for anything written.
         *
         * Empty outside a loop. Every file a step writes was named after the
         * step alone, so the second pass of a loop overwrote the first — and
         * the first pass's value, already gathered into a list, went on
         * pointing at a file whose contents had changed underneath it. Four
         * pictures from a Batch were one picture, four times over.
         */
        passLabel: String = "",
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
                            lookup(outputs, ref)
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
                    // needs no files behind it. Marked as text, so a for-each
                    // over them hands each one on as prose rather than as a
                    // path to a file that was never written.
                    outputs["${node.id}:pieces"] = PortValue.list(pieces, PortType.TEXT)
                }

                NodeKind.TextJoin -> {
                    val pieces = resolve(node, "pieces", outputs, graph)?.paths.orEmpty()
                    val joined = pieces.joinToString(node.params.string("separator", "\n\n"))
                    put(outputs, node, "text", PortValue.text(joined))
                }

                NodeKind.Pick -> {
                    val list = resolve(node, "options", outputs, graph)
                    val options = list?.paths.orEmpty()
                    if (options.isEmpty()) {
                        throw IllegalStateException(
                            "There is nothing to choose between — the step above produced an " +
                                "empty list.",
                        )
                    }
                    val chosen = reporter.awaitChoice(node.id, options)
                        ?: throw IllegalStateException("Nothing was chosen.")
                    // As the kind of thing it was in the list. Choosing between
                    // four sentences used to hand on a *file path* that read
                    // "the first sentence", and the step after it went looking
                    // for a file by that name.
                    put(outputs, node, "chosen", (list ?: PortValue.list(emptyList())).item(chosen))
                }

                /*
                 * Keep — which until now kept nothing.
                 *
                 * It recorded the value in the run's own map and stopped
                 * there, so the one step whose whole purpose is to put a
                 * result where the rest of the app can find it left the
                 * gallery empty. Everything a workflow made lived in a
                 * per-run scratch folder named after a UUID and was findable
                 * only by knowing that UUID.
                 *
                 * Where a thing goes is decided by what it is, using the same
                 * folders the tabs already write to, so a picture made by a
                 * graph and a picture made by the Stills tab sit together.
                 */
                NodeKind.Output -> {
                    val value = resolve(node, "value", outputs, graph)
                        ?: throw IllegalStateException("Nothing was given to this step to keep.")
                    val kept = keep(value)
                    outputs["${node.id}:kept"] = kept
                    outputs[node.id] = kept
                    reporter.onNode(
                        node.id,
                        NodeProgress(
                            NodeRunState.RUNNING,
                            message = when {
                                kept.paths.size > 1 -> "Kept ${kept.paths.size} files."
                                kept.path != null -> "Kept ${File(kept.path!!).name}."
                                else -> "Kept."
                            },
                        ),
                    )
                }

                /*
                 * Send — which stages, describes, and stops.
                 *
                 * Nothing here starts an activity. A run holds no activity and
                 * an app in the background is refused one, so a `startActivity`
                 * from this coroutine would be dropped with the step already
                 * reported done — a result that never arrives and nothing in
                 * the log to say why. What leaves this method is a description;
                 * see Handoff for who acts on it.
                 */
                NodeKind.Send -> {
                    val value = resolve(node, "value", outputs, graph)
                        ?: throw IllegalStateException("Nothing was given to this step to send.")
                    val subject = resolve(node, "subject", outputs, graph)?.asText
                        ?: WorkflowTemplate.render(node.params.string("subject")) { ref ->
                            lookup(outputs, ref)
                        }
                    val target = HandoffTarget.of(node.params.string("target"))
                    // Text goes as text and not as a file it would have to be
                    // written to first: a mail composer handed a .txt shows an
                    // attachment where the body should be.
                    val export = if (value.type == PortType.TEXT || value.path == null) {
                        null
                    } else {
                        stageForSending(value, node, passLabel)
                    }
                    reporter.onHandoff(
                        Handoff(
                            nodeId = node.id,
                            export = export,
                            text = if (export == null) value.asText else "",
                            subject = subject,
                            target = target,
                            packageName = node.params.string("package").takeIf { it.isNotBlank() },
                            label = node.params.string("appLabel"),
                        ),
                    )
                }

                NodeKind.Processor ->
                    runProcessor(node, graph, outputs, outDir, models, reporter, passLabel)

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
                    val file = File(outDir, "${node.id}$passLabel.png")
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
                    val file = File(outDir, "${node.id}$passLabel.png")
                    File(frames[at]).copyTo(file, overwrite = true)
                    put(outputs, node, "image", PortValue.file(PortType.IMAGE, file.absolutePath))
                }

                NodeKind.Assemble -> {
                    val images = resolve(node, "images", outputs, graph)?.paths.orEmpty()
                    if (images.isEmpty()) throw IllegalStateException("No pictures to assemble.")
                    val dir = File(outDir, "${node.id}$passLabel").apply { mkdirs() }
                    val frames = images.mapIndexed { i, path ->
                        val target = File(dir, String.format("frame_%04d.png", i))
                        File(path).copyTo(target, overwrite = true)
                        target.absolutePath
                    }
                    // Under the step's own id. This key was written as a
                    // literal `${node.id}:clip` — an over-escaped template that
                    // Kotlin took at its word — so nothing downstream could
                    // ever bind to it and the step was, in effect, not there.
                    val clip = PortValue(PortType.CLIP, paths = frames)
                    outputs["${node.id}:clip"] = clip
                    outputs[node.id] = clip
                }

                NodeKind.Tool -> {
                    val name = node.params.string("tool")
                    if (name.isBlank()) throw IllegalStateException("No tool chosen for this step.")
                    // A bound slot wins; otherwise what was typed, with earlier
                    // steps substituted into it the same way a template does —
                    // a tool call whose query is the previous step's answer is
                    // the ordinary case, not the clever one.
                    val arguments = resolve(node, "arguments", outputs, graph)?.text
                        ?: WorkflowTemplate.render(node.params.string("arguments", "{}")) { ref ->
                            lookup(outputs, ref)
                        }
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
        passLabel: String,
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
        val base = stored.overlaidWith(SparseParams(settings))
        // A Batch's whole promise is a different picture each pass, and it can
        // only keep it if the seed reaches the sampler. A step that pins its
        // own seed still wins — asking for the same seed every pass is a
        // strange thing to want, but it is a thing the author said.
        val params = batchSeed
            ?.takeIf { !settings.containsKey("seed") }
            ?.let { base.with("seed", it.toString()) }
            ?: base

        when (model.modality) {
            Modality.TEXT, Modality.VISION ->
                runText(node, graph, outputs, model, params, reporter)
            Modality.DIFFUSION ->
                runDiffusion(node, graph, outputs, outDir, model, params, reporter, passLabel)
            Modality.SPEECH_TO_TEXT ->
                runTranscribe(node, graph, outputs, outDir, model, params, reporter, passLabel)
            Modality.TEXT_TO_SPEECH ->
                runSpeak(node, graph, outputs, outDir, model, params, passLabel)
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
                    "This model reasoned for $thought characters and then answered with " +
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
        passLabel: String,
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
                            val file = File(outDir, "${node.id}$passLabel.png")
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
        passLabel: String,
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

        val subtitles = File(outDir, "${node.id}$passLabel.srt")
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
        passLabel: String,
    ) {
        val text = resolve(node, "text", outputs, graph)?.text.orEmpty()
        val directory = File(model.localPath).let { if (it.isDirectory) it else it.parentFile }
        synthesizer.useKokoroModel(directory)
        synthesizer.useOmniVoiceModel(directory)
        val destination = File(outDir, "${node.id}$passLabel.wav")
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
     * Put a value where the rest of the app looks for that kind of thing.
     *
     * The folders are the ones the tabs already use — no new place to look,
     * and no registry to keep in step, because the library in this app is the
     * filesystem. A copy rather than a move: the run's own folder stays intact
     * so a graph that keeps something halfway through can go on using it.
     */
    private fun keep(value: PortValue): PortValue {
        val stamp = System.currentTimeMillis()
        fun copyInto(directory: File, path: String, index: Int): String {
            val source = File(path)
            val extension = source.extension.ifBlank { "bin" }
            val target = File(
                directory.apply { mkdirs() },
                "workflow_$stamp${if (index > 0) "_$index" else ""}.$extension",
            )
            source.copyTo(target, overwrite = true)
            return target.absolutePath
        }

        return when (value.type) {
            PortType.TEXT -> {
                val target = File(storage.transcriptsDir(), "workflow_$stamp.txt")
                target.writeText(value.text)
                PortValue.file(PortType.FILE, target.absolutePath)
            }
            PortType.IMAGE ->
                PortValue.file(
                    PortType.IMAGE,
                    copyInto(storage.galleryDir(), value.path.orEmpty(), 0),
                )
            PortType.AUDIO ->
                PortValue.file(
                    PortType.AUDIO,
                    copyInto(storage.speechDir(), value.path.orEmpty(), 0),
                )
            // A clip is its frames, and they belong together — one folder in
            // the gallery rather than N loose stills that no longer read as a
            // sequence.
            PortType.CLIP -> {
                val dir = File(storage.galleryDir(), "workflow_$stamp").apply { mkdirs() }
                val frames = value.paths.mapIndexed { i, path ->
                    val target = File(dir, String.format("frame_%04d.png", i))
                    File(path).copyTo(target, overwrite = true)
                    target.absolutePath
                }
                PortValue(PortType.CLIP, paths = frames)
            }
            PortType.LIST -> {
                if (value.elementType == PortType.TEXT) {
                    val target = File(storage.transcriptsDir(), "workflow_$stamp.txt")
                    target.writeText(value.paths.joinToString("\n\n"))
                    PortValue.file(PortType.FILE, target.absolutePath)
                } else {
                    val directory = if (value.elementType == PortType.AUDIO) {
                        storage.speechDir()
                    } else {
                        storage.galleryDir()
                    }
                    PortValue.list(
                        value.paths.mapIndexed { i, path -> copyInto(directory, path, i) },
                        value.elementType,
                    )
                }
            }
            // A slot type, never a value type — see PortType.ANY. Kept as an
            // arm rather than an else so that adding a real port type later
            // still fails to compile here, which is the point of the when.
            PortType.FILE, PortType.ANY ->
                PortValue.file(
                    PortType.FILE,
                    copyInto(storage.galleryDir(), value.path.orEmpty(), 0),
                )
        }
    }

    /**
     * Copy what a Send step is sending into the one folder a URI can be made of.
     *
     * A run writes into `workflows/<runId>/`, which the file provider knows
     * nothing about — asking it for a URI there throws, and it would throw
     * inside a step that had already succeeded. The exports folder is declared
     * in `file_paths.xml` and is where every other outbound artifact in this app
     * is staged, so this is the same route the Library's share button takes.
     */
    private fun stageForSending(value: PortValue, node: NodeRecord, passLabel: String): Export {
        val source = File(value.path ?: error("There is no file to send."))
        val extension = source.extension.ifBlank { "bin" }
        val name = "workflow_${System.currentTimeMillis()}$passLabel.$extension"
        val target = File(storage.exportsDir(), name)
        source.copyTo(target, overwrite = true)
        return Export(
            staged = target,
            suggestedName = node.label.ifBlank { name }.let {
                if (it.endsWith(".$extension")) it else "$it.$extension"
            },
            mime = Export.mimeFor(name),
        )
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

    /**
     * What a template's `2.text` names among the outputs.
     *
     * The two halves of this app spell a reference differently and neither can
     * change: a slot binding is `id:output`, because that is the key an output
     * is stored under, and a template writes `id.output`, because the
     * expression grammar reads a bare word and a word cannot contain a colon.
     * Nothing bridged them, so `{{ 2.text }}` — the syntax in the editor's own
     * help text and in the template's own docstring — found nothing, was left
     * on the page as the literal characters it was written as, and every
     * condition on an Only-if was therefore false. The node had never once
     * taken its branch.
     *
     * Three readings, most specific first: the reference as written, the
     * dotted form as a slot binding, and the step alone.
     */
    private fun lookup(outputs: Map<String, PortValue>, reference: String): String? =
        outputs[reference]?.asText
            ?: outputs[reference.replace('.', ':')]?.asText
            ?: outputs[reference.substringBefore('.')]?.asText

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
        val BOOKKEEPING = setOf(
            "model", "shape", "portType", "times", "by", "separator", "template",
            "pattern", "condition", "mode", "tool", "script", "timeout_ms",
            // Trigger and hand-off bookkeeping. None of these is a runtime
            // parameter, and passing them through would have every run report a
            // handful of rejected keys nobody can do anything about.
            "from", "target", "package", "appLabel", "subject",
        )
    }
}
