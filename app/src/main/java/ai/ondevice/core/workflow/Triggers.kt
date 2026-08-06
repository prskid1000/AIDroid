package ai.ondevice.core.workflow

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * A workflow started by something outside this app.
 *
 * Everything here is a pure function over a graph and a payload, and that is
 * deliberate: the share sheet, the shortcut ranking and the text-selection menu
 * cannot be exercised without a device, so the parts that decide *which
 * workflow matches what was shared* are kept where a unit test can reach them.
 * `TriggerActivity` is then only plumbing — read the intent, copy the bytes in,
 * ask this file.
 */
object Triggers {

    // ── where an Input gets its value ────────────────────────────────────

    /** Typed into the editor, which is where every Input started. */
    const val FROM_TYPED = "typed"

    /** Filled by whatever app shared into this one. */
    const val FROM_SHARED = "shared"

    /** The param an Input stores that choice under. */
    const val PARAM_FROM = "from"

    // ── the categories that put a workflow in a share sheet ──────────────

    /**
     * These are matched against `<share-target>` entries in `res/xml/shortcuts.xml`.
     *
     * One category per share-target there, and the whole set on the shortcut
     * here. The matching rule between the two sets is not clearly documented —
     * intersection or contains-all — and with exactly one category per target
     * the answer does not matter, which is why it is written that way.
     */
    const val CATEGORY_TEXT = "ai.ondevice.category.TEXT"
    const val CATEGORY_IMAGE = "ai.ondevice.category.IMAGE"
    const val CATEGORY_AUDIO = "ai.ondevice.category.AUDIO"
    const val CATEGORY_ANY = "ai.ondevice.category.ANY"

    /**
     * What kind of value a mime type is, for the four ports that can arrive.
     *
     * Video is a `FILE` and not a `CLIP`: a clip in this app is a directory of
     * frames with a frame rate, and an mp4 someone shared is neither. Calling
     * it a clip would hand a frame-stepping step a container it cannot read,
     * and the failure would surface two steps later.
     */
    fun portFor(mime: String): PortType = when {
        mime.startsWith("image/") -> PortType.IMAGE
        mime.startsWith("audio/") -> PortType.AUDIO
        // A text file is a file. It also carries its text — see TriggerValue —
        // so it satisfies a TEXT slot without pretending not to be a file.
        else -> PortType.FILE
    }

    /** Whether a mime type is one this app can read as text. */
    fun readableAsText(mime: String, name: String): Boolean =
        mime.startsWith("text/") ||
            mime in TEXTUAL_MIMES ||
            name.substringAfterLast('.', "").lowercase() in TEXT_EXTENSIONS

    // ── what a graph takes ───────────────────────────────────────────────

    /** The Input steps waiting to be filled by another app. */
    fun sharedInputs(graph: WorkflowGraph): List<NodeRecord> =
        graph.nodes.filter {
            it.enabled &&
                it.type == NodeKind.Input.type &&
                it.params.text(PARAM_FROM) == FROM_SHARED
        }

    /**
     * The port types this graph will accept from outside.
     *
     * Derived, never declared. A workflow that has no such Input is not a share
     * target at all, and there is nothing to keep in step when one is added —
     * the same discipline that has a Processor read its shape off the model's
     * own row rather than carry a list of names.
     */
    fun accepts(graph: WorkflowGraph): List<PortType> =
        sharedInputs(graph).map { it.declaredPort() }

    /**
     * Which share sheets this workflow should appear in.
     *
     * A graph wanting a `FILE` appears in all of them, because an image, a
     * recording and a document are all files — the lattice in
     * [PortType.satisfies], applied at publish time rather than restated in
     * XML.
     */
    fun categoriesFor(graph: WorkflowGraph): Set<String> {
        val wanted = accepts(graph)
        if (wanted.isEmpty()) return emptySet()
        return buildSet {
            if (PortType.TEXT in wanted) add(CATEGORY_TEXT)
            if (PortType.IMAGE in wanted) add(CATEGORY_IMAGE)
            if (PortType.AUDIO in wanted) add(CATEGORY_AUDIO)
            if (PortType.FILE in wanted || PortType.ANY in wanted) {
                add(CATEGORY_ANY)
                add(CATEGORY_TEXT)
                add(CATEGORY_IMAGE)
                add(CATEGORY_AUDIO)
            }
        }
    }

    // ── matching, and filling ────────────────────────────────────────────

    /**
     * Whether what arrived can fill every Input this graph is waiting on.
     *
     * Greedy and in order: each value is spent once. A graph wanting a picture
     * and a line of text matches a share carrying both, and does not match a
     * share carrying two pictures — which is the honest answer rather than the
     * convenient one, because filling a text slot with a file path is how a
     * model comes to be handed the word "/storage/emulated/0/…".
     */
    fun matches(graph: WorkflowGraph, payload: TriggerPayload): Boolean =
        assign(graph, payload).size == sharedInputs(graph).size

    /**
     * The graph as it will actually run, with the payload written into it.
     *
     * Rewritten before the run rather than threaded through it: the run stores
     * a snapshot of the graph it ran, so putting the values in here means the
     * history records exactly what was worked on. The runner needs no notion of
     * a trigger at all.
     */
    fun fill(graph: WorkflowGraph, payload: TriggerPayload): WorkflowGraph {
        val assigned = assign(graph, payload)
        if (assigned.isEmpty()) return graph
        return graph.copy(
            nodes = graph.nodes.map { node ->
                val value = assigned[node.id] ?: return@map node
                val params = node.params.toMutableMap()
                if (node.declaredPort() == PortType.TEXT) {
                    params["text"] = JsonPrimitive(value.text)
                } else {
                    params["path"] = JsonPrimitive(value.path)
                }
                node.copy(params = JsonObject(params))
            },
        )
    }

    /** Which value fills which Input, or fewer entries than Inputs if it cannot. */
    private fun assign(
        graph: WorkflowGraph,
        payload: TriggerPayload,
    ): Map<String, TriggerValue> {
        val spare = payload.values.toMutableList()
        val out = LinkedHashMap<String, TriggerValue>()
        sharedInputs(graph).forEach { node ->
            val wanted = node.declaredPort()
            val found = spare.firstOrNull { it.canFill(wanted) } ?: return@forEach
            spare.remove(found)
            out[node.id] = found
        }
        return out
    }

    private fun NodeRecord.declaredPort(): PortType =
        runCatching { PortType.valueOf(params.text("portType", "TEXT")) }
            .getOrDefault(PortType.TEXT)

    private fun JsonObject.text(key: String, fallback: String = ""): String =
        (this[key] as? JsonPrimitive)?.content ?: fallback

    private val TEXTUAL_MIMES = setOf(
        "application/json", "application/xml", "application/x-yaml",
        "application/javascript", "application/x-sh", "application/rtf",
    )

    private val TEXT_EXTENSIONS = setOf(
        "txt", "md", "markdown", "json", "yaml", "yml", "xml", "csv", "tsv", "log",
        "srt", "vtt", "kt", "java", "py", "js", "ts", "c", "cpp", "h", "rs", "go",
        "sh", "toml", "ini", "cfg", "sql", "html", "css",
    )
}

/**
 * One thing another app handed over.
 *
 * [text] and [path] are both populated for a shared text file, and that is the
 * point of carrying them separately rather than as a tagged union: the same
 * `.srt` satisfies a step wanting prose *and* a step wanting a file, and
 * deciding which it "really" is at the boundary would be guessing at a graph
 * this class has not seen.
 */
data class TriggerValue(
    val type: PortType,
    val text: String = "",
    val path: String = "",
    val displayName: String = "",
) {
    fun canFill(slot: PortType): Boolean = when (slot) {
        PortType.TEXT -> text.isNotBlank()
        PortType.ANY -> text.isNotBlank() || path.isNotBlank()
        PortType.LIST -> false
        else -> path.isNotBlank() && type.satisfies(slot)
    }

    /** One line naming this, for the sheet that asks whether to run. */
    val summary: String
        get() = when {
            displayName.isNotBlank() -> displayName
            text.isNotBlank() -> text.take(60).replace('\n', ' ')
            else -> path.substringAfterLast('/')
        }
}

/**
 * Everything that arrived with one trigger.
 *
 * [fromPackage] is for the confirmation line only and must not be what a trust
 * decision keys on — `getReferrer()` is set by the caller and a caller that
 * wants to lie about it can.
 */
data class TriggerPayload(
    val values: List<TriggerValue> = emptyList(),
    val fromPackage: String? = null,
    /**
     * `ACTION_PROCESS_TEXT` only: false means the caller will accept the
     * selection being replaced with what comes back.
     */
    val readOnly: Boolean = true,
    /**
     * Files that were offered and could not be opened.
     *
     * Counted separately from [values] because the two failures need different
     * words and lead to different places. A share carrying nothing is the
     * sending app's doing; a share carrying a file this app was refused access
     * to is a permission that did not travel — usually a caller that left off
     * the read grant, or one that expired before the copy. Collapsing them
     * reported a permission error as an empty share, which sends somebody
     * looking at the wrong end of the problem entirely.
     */
    val unreadable: Int = 0,
) {
    val isEmpty: Boolean get() = values.isEmpty()

    /** Offered something and none of it could be opened. */
    val allUnreadable: Boolean get() = values.isEmpty() && unreadable > 0
}
