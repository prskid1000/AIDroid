package ai.ondevice.engine.workflow

import ai.ondevice.core.Modality
import ai.ondevice.core.workflow.NodeKind
import ai.ondevice.core.workflow.NodeRecord
import ai.ondevice.core.workflow.WorkflowGraph
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.engine.RuntimeRegistry

/**
 * Which model has to be resident for a step, and what that costs.
 *
 * A workflow is the first thing in this app that crosses runtimes inside one
 * run. Until now each tab spoke to one engine and each engine looked after
 * itself: the diffusion engine unloads before it loads, holding a lock across
 * both halves, and the llama manager warm-swaps. But nothing has ever asked
 * whether *two* engines are resident at once, because nothing has ever put a
 * chat model and a diffusion model in the same sentence. Both can be, and on a
 * phone holding ten gigabytes of one already, the second is what the kernel
 * kills the process for.
 *
 * So the plan is computed before the run rather than decided during it, and
 * shown before it starts. A schedule nobody can see is one nobody can debug
 * when it costs forty-five minutes.
 */
data class Residency(
    val runtimeId: String,
    val modelId: String,
    val label: String,
    val bytes: Long,
    /** The steps this residency covers, in the order they will run. */
    val nodeIds: List<String>,
)

data class ResidencyPlan(
    val residencies: List<Residency> = emptyList(),
    /** Steps that need no model at all — the cheap ones. */
    val modellessNodes: Int = 0,
) {
    val loadCount: Int get() = residencies.size

    /** What the largest single residency asks for, which is what must fit. */
    val peakBytes: Long get() = residencies.maxOfOrNull { it.bytes } ?: 0L

    /** Everything read from storage across the run, loads repeated included. */
    val totalBytes: Long get() = residencies.sumOf { it.bytes }
}

object ResidencyPlanner {

    /**
     * Work out the loads a graph implies.
     *
     * Adjacent steps on the same model are one residency and one load — the
     * engines already answer `isCurrent(modelId)` and skip the work, so this
     * only has to avoid *interleaving* them. Interleaving is the failure mode
     * worth naming: a graph that alternates chat, image, chat, image pays four
     * loads for two models, and on this hardware that is minutes of pure
     * waiting either side of every step.
     */
    fun plan(graph: WorkflowGraph, models: Map<String, ModelEntity>): ResidencyPlan {
        val residencies = mutableListOf<Residency>()
        var modelless = 0

        graph.nodes.filter { it.enabled }.forEach { node ->
            val model = modelFor(node, models)
            if (model == null) {
                modelless++
                return@forEach
            }
            val runtimeId = runtimeFor(model)
            val last = residencies.lastOrNull()
            if (last != null && last.modelId == model.id && last.runtimeId == runtimeId) {
                // Same model, still resident: one load, two steps.
                residencies[residencies.lastIndex] =
                    last.copy(nodeIds = last.nodeIds + node.id)
            } else {
                residencies += Residency(
                    runtimeId = runtimeId,
                    modelId = model.id,
                    label = model.label,
                    bytes = model.sizeBytes,
                    nodeIds = listOf(node.id),
                )
            }
        }
        return ResidencyPlan(residencies, modelless)
    }

    /** The model a step needs, or null when it needs none. */
    fun modelFor(node: NodeRecord, models: Map<String, ModelEntity>): ModelEntity? {
        if (NodeKind.of(node.type) != NodeKind.Processor) return null
        val id = node.params["model"]?.let {
            (it as? kotlinx.serialization.json.JsonPrimitive)?.content
        } ?: return null
        return models[id]
    }

    /** Which engine runs it — from the row's own modality, never a name. */
    fun runtimeFor(model: ModelEntity): String = when (model.modality) {
        Modality.DIFFUSION -> RuntimeRegistry.STABLE_DIFFUSION
        Modality.SPEECH_TO_TEXT -> RuntimeRegistry.WHISPER
        Modality.TEXT_TO_SPEECH -> RuntimeRegistry.KOKORO
        else -> RuntimeRegistry.LLAMA
    }
}
