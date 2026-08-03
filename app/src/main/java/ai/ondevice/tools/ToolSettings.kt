package ai.ondevice.tools

import ai.ondevice.core.SparseParams
import ai.ondevice.params.ParamSpec
import ai.ondevice.params.ParamType
import ai.ondevice.core.Tier
import kotlinx.serialization.json.JsonPrimitive

/**
 * The settings a tool exposes, and what they are currently set to.
 *
 * Deliberately [ParamSpec] rather than a second description of what a number
 * is: the app already has one, the screens already render it, and the numbers
 * a tool wants tuned — a result count, a timeout, a size cap — are the same
 * shape as the numbers a sampler wants tuned. A second little settings type
 * would mean a second slider, a second clamp and a second set of defaults that
 * drift from the ones in the code.
 *
 * Keys are namespaced `<tool>.<setting>` and [ParamSpec.group] carries the tool
 * name, so one flat map covers every tool and the screen can still group by the
 * tool it belongs to.
 */
class ToolSettings(
    private val values: SparseParams = SparseParams.EMPTY,
    private val specs: List<ParamSpec> = emptyList(),
) {

    /** The stored value, else the spec's own default, else [fallback]. */
    fun int(key: String, fallback: Int): Int =
        values.int(key) ?: specDefault(key)?.content?.toIntOrNull() ?: fallback

    fun bool(key: String, fallback: Boolean): Boolean =
        values.bool(key) ?: specDefault(key)?.content?.toBooleanStrictOrNull() ?: fallback

    fun string(key: String, fallback: String): String =
        values.string(key) ?: specDefault(key)?.content ?: fallback

    private fun specDefault(key: String): JsonPrimitive? =
        specs.firstOrNull { it.key == key }?.default as? JsonPrimitive

    companion object {
        val EMPTY = ToolSettings()

        /**
         * A whole number a tool takes, as a row the params screens can render.
         *
         * The helpers exist so a provider declares a setting in one line. The
         * default lives here and nowhere else — the tool reads it back through
         * [int], so there is exactly one number to change.
         */
        fun int(
            tool: String,
            name: String,
            default: Int,
            min: Int,
            max: Int,
            label: String,
            help: String,
            tier: Tier = Tier.BASIC,
        ) = ParamSpec(
            key = "$tool.$name",
            group = tool,
            type = ParamType.INT,
            default = JsonPrimitive(default),
            min = min.toDouble(),
            max = max.toDouble(),
            step = 1.0,
            tier = tier,
            label = label,
            help = help,
        )

        fun bool(
            tool: String,
            name: String,
            default: Boolean,
            label: String,
            help: String,
            tier: Tier = Tier.BASIC,
        ) = ParamSpec(
            key = "$tool.$name",
            group = tool,
            type = ParamType.BOOL,
            default = JsonPrimitive(default),
            tier = tier,
            label = label,
            help = help,
        )

        fun string(
            tool: String,
            name: String,
            default: String,
            label: String,
            help: String,
            tier: Tier = Tier.EXPERT,
        ) = ParamSpec(
            key = "$tool.$name",
            group = tool,
            type = ParamType.STRING,
            default = JsonPrimitive(default),
            tier = tier,
            label = label,
            help = help,
        )
    }
}
