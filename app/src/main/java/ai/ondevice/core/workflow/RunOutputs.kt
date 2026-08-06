package ai.ondevice.core.workflow

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * What a run had produced when it was last written down.
 *
 * Kept so a run the system kills part-way through can carry on rather than start
 * again. At these lengths that is not a nicety: a graph is several generations
 * end to end, and a phone holding ten gigabytes of weights is the first thing
 * Android reclaims — so being killed at step five of seven is an ordinary
 * event, and repeating five model loads to get back to it is most of an hour.
 *
 * Written to `workflow_runs.nodeStatesJson` as the run goes rather than at the
 * end, because a row written at the end is a row that is never written for
 * exactly the runs this exists for.
 *
 * Values are paths and text, never bytes — the artefacts themselves are already
 * on disk under `workflows/<runId>/`, and this only records where they went and
 * what kind of thing each one is.
 */
@Serializable
data class RunOutputs(
    /** Output key — `"<nodeId>"` or `"<nodeId>:<output>"` — to what it held. */
    val values: Map<String, StoredValue> = emptyMap(),
) {
    /** The steps that finished, which is what a resume is allowed to skip. */
    val completedNodes: Set<String>
        get() = values.keys.map { it.substringBefore(':') }.toSet()

    fun encode(): String = JSON.encodeToString(serializer(), this)

    companion object {
        val JSON = Json {
            ignoreUnknownKeys = true
            isLenient = true
            encodeDefaults = true
        }

        fun decode(text: String?): RunOutputs {
            if (text.isNullOrBlank()) return RunOutputs()
            return runCatching { JSON.decodeFromString(serializer(), text) }
                .getOrElse { RunOutputs() }
        }
    }
}

/** One value, in the shape it can be written down and read back. */
@Serializable
data class StoredValue(
    val type: String = PortType.TEXT.name,
    val text: String = "",
    val paths: List<String> = emptyList(),
    val elementType: String = PortType.FILE.name,
)

/**
 * Whether a graph can be picked up part-way at all.
 *
 * A loop is the exception, and refusing it is the honest answer rather than a
 * missing feature. "This step finished" is a fact about a step that runs once;
 * inside a Repeat or a For-each a step runs many times and finishing the third
 * pass says nothing about the fourth. Resuming such a graph would silently skip
 * the rest of the loop and report success, which is worse than running it again.
 */
fun WorkflowGraph.canResume(): Boolean = nodes.none {
    it.enabled && it.type in LOOPING
}

private val LOOPING = setOf(
    NodeKind.RepeatStart.type, NodeKind.RepeatEnd.type,
    NodeKind.ForEachStart.type, NodeKind.ForEachEnd.type,
    NodeKind.Batch.type,
)
