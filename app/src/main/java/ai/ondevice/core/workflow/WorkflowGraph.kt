package ai.ondevice.core.workflow

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject

/**
 * A workflow, as it is stored and as it is run.
 *
 * One JSON document rather than normalised node and edge tables. Every
 * structured thing in this schema is already a JSON column — parameter
 * overrides, companion paths, transcript segments, resource traces — and SPEC
 * §11 gives the reason for the first of those: a key the current build does
 * not understand is kept inert rather than dropped, so a newer build's file
 * survives an older build reading it. That argument is stronger for a graph
 * than for a parameter set, because normalising would mean a Room migration
 * every time a node type gained a field.
 */
@Serializable
data class WorkflowGraph(
    val schemaVersion: Int = SCHEMA_VERSION,
    val nodes: List<NodeRecord> = emptyList(),
) {
    /** The node this reference names, or null when it names nothing here. */
    fun producerOf(reference: String): NodeRecord? {
        val id = reference.substringBefore(':')
        return nodes.firstOrNull { it.id == id }
    }

    fun indexOf(nodeId: String): Int = nodes.indexOfFirst { it.id == nodeId }

    /**
     * Every node before this one, which is exactly the set it may read from.
     *
     * Backward-only references are what make a cycle unrepresentable: there is
     * no sort, no cycle detector and no error state for one, because the list
     * order *is* the topological order. It is the largest simplification the
     * step-list shape buys, and the reason a graph on a phone does not need a
     * canvas to be correct.
     */
    fun sourcesFor(nodeId: String): List<NodeRecord> {
        val at = indexOf(nodeId)
        return if (at <= 0) emptyList() else nodes.take(at)
    }

    fun withNode(node: NodeRecord): WorkflowGraph =
        copy(nodes = nodes.map { if (it.id == node.id) node else it })

    fun encode(): String = JSON.encodeToString(serializer(), this)

    companion object {
        const val SCHEMA_VERSION = 1

        /**
         * Lenient, and unknown keys ignored — the settings every other decode
         * in this app uses, for the same reason: a graph written by a later
         * build must open rather than throw.
         */
        val JSON = Json {
            ignoreUnknownKeys = true
            isLenient = true
            encodeDefaults = true
        }

        fun decode(text: String?): WorkflowGraph {
            if (text.isNullOrBlank()) return WorkflowGraph()
            return runCatching { JSON.decodeFromString(serializer(), text) }
                .getOrElse { WorkflowGraph() }
        }
    }
}

/**
 * One step.
 *
 * [type] is a string and not a sealed class *on the wire*. kotlinx's
 * polymorphic decoding throws on a discriminator it does not know, so a graph
 * saved by a build that understands one more node type would fail to open here
 * — losing the whole workflow rather than one step. A flat record decodes
 * whatever arrives; [NodeKind.of] then makes sense of it in Kotlin, with an
 * [NodeKind.Unknown] arm for the rest.
 */
@Serializable
data class NodeRecord(
    val id: String,
    val type: String,
    /** What the author called this step; the kind's own name when blank. */
    val label: String = "",
    /**
     * Slot name to a producer reference, `"<nodeId>"` or `"<nodeId>:<output>"`.
     *
     * Absent means unbound, which is a step that cannot run yet rather than an
     * error — the editor says so where it can be fixed.
     */
    val slots: Map<String, String> = emptyMap(),
    /** This step's own settings, in the app's one parameter currency. */
    val params: JsonObject = JsonObject(emptyMap()),
    /** Off keeps a step in the list and out of the run. */
    val enabled: Boolean = true,
)

/**
 * What a value on an edge is.
 *
 * Values are *paths*, never pixels or samples. Four nodes holding four decoded
 * bitmaps is four times width by height by four bytes, in a process already
 * holding ten gigabytes of weights — and this codebase has made the call once
 * already, in the note on DiffusionClip about a five-second clip being 147 MB
 * of raw RGB.
 */
enum class PortType(val label: String) {
    TEXT("Text"),
    IMAGE("Image"),
    AUDIO("Audio"),
    CLIP("Clip"),
    FILE("File"),

    /**
     * Several of one type, produced by a batch or a split and consumed by a
     * for-each. The element type travels with the value rather than with the
     * port, because a slot that takes "many of whatever this is" is the only
     * kind this app needs.
     */
    LIST("List"),
    ;

    /**
     * Whether a value of this type can be handed to a slot wanting [slot].
     *
     * One rule, in one direction: the three media types are all files, and
     * nothing is a media type but itself. A lattice with coercions in every
     * direction is how a graph comes to run and produce nonsense.
     */
    fun satisfies(slot: PortType): Boolean = when {
        this == slot -> true
        slot == FILE -> this in setOf(IMAGE, AUDIO, CLIP)
        else -> false
    }
}

/** One input a node takes. */
data class SlotSpec(
    val name: String,
    val type: PortType,
    val label: String,
    /** A step with an unbound required slot cannot run. */
    val required: Boolean = true,
    val help: String = "",
)

/** One value a node produces. */
data class OutputSpec(
    val name: String,
    val type: PortType,
    val label: String,
)
