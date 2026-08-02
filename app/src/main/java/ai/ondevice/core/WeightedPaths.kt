package ai.ondevice.core

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.put

/** One file attached with a strength. */
data class WeightedPath(val path: String, val weight: Float = 1.0f)

/**
 * How a role that takes several files at once is stored.
 *
 * Every role used to hold one path as a bare string, which is right for a VAE
 * and wrong for a LoRA: sd.cpp takes an array of them, each with its own
 * multiplier, and stacking two at different strengths is the ordinary way they
 * are used. So a multiple role stores `[{"path": …, "weight": …}, …]` instead.
 *
 * A bare string still reads as one entry at full strength. That is not
 * politeness towards old data — it is what every single-file role in this app
 * stores, and reading both shapes here means one function answers "what is
 * attached under this key" for all of them.
 */
object WeightedPaths {

    fun parse(element: JsonElement?): List<WeightedPath> = when (element) {
        null -> emptyList()
        is JsonArray -> element.mapNotNull { entry -> one(entry) }
        else -> listOfNotNull(one(element))
    }

    private fun one(element: JsonElement): WeightedPath? = when (element) {
        is JsonObject -> {
            val path = (element["path"] as? JsonPrimitive)?.contentOrNull.orEmpty()
            path.takeIf { it.isNotBlank() }?.let {
                WeightedPath(it, (element["weight"] as? JsonPrimitive)?.floatOrNull ?: 1.0f)
            }
        }
        is JsonPrimitive -> element.contentOrNull
            ?.takeIf { it.isNotBlank() }
            ?.let { WeightedPath(it) }
        else -> null
    }

    /** Null when nothing is left, because an empty array is not an answer. */
    fun toJson(items: List<WeightedPath>): JsonArray? = items
        .filter { it.path.isNotBlank() }
        .takeIf { it.isNotEmpty() }
        ?.let { kept ->
            buildJsonArray {
                kept.forEach { item ->
                    add(
                        buildJsonObject {
                            put("path", item.path)
                            put("weight", item.weight)
                        },
                    )
                }
            }
        }
}
