package ai.ondevice.params

import ai.ondevice.core.Tier
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull

/** SPEC §16.1 — the manifest. */
@Serializable
data class ParamManifest(
    val manifestVersion: Int,
    val runtimes: Map<String, RuntimeParams> = emptyMap(),
) {
    fun paramsFor(runtimeId: String): List<ParamSpec> = runtimes[runtimeId]?.params.orEmpty()

    fun spec(runtimeId: String, key: String): ParamSpec? =
        runtimes[runtimeId]?.params?.firstOrNull { it.key == key }
}

/** `sourceCommit`, `buildTag` and `jniContract` used to sit here as provenance. */
@Serializable
data class RuntimeParams(
    val params: List<ParamSpec> = emptyList(),
)

@Serializable
data class ParamSpec(
    /** Canonical name — this is the wire key passed through JNI verbatim. */
    val key: String,
    val group: String = "other",
    val type: ParamType,
    val default: JsonElement? = null,
    val min: Double? = null,
    val max: Double? = null,
    val step: Double? = null,
    val values: List<String> = emptyList(),
    val tier: Tier = Tier.EXPERT,
    val label: String = key,
    val help: String = "",
    /** Batched into a single reload rather than applied per-edit (SPEC §9). */
    val requiresReload: Boolean = false,
    /** Version gating — hidden unless the loaded runtime build is in range. */
    val sinceBuild: String? = null,
    val untilBuild: String? = null,
    val dependsOn: Dependency? = null,
    val appliesTo: AppliesTo? = null,
    /** Provenance; also drives the raw-args escape hatch. */
    val cliFlag: String? = null,
) {
    val defaultDisplay: String get() = default?.let { renderJson(it) } ?: "—"

    val isRange: Boolean
        get() = (type == ParamType.FLOAT || type == ParamType.INT) && min != null && max != null
}

/** §16.4 — adding a new *type* is the only case that requires app code. */
@Serializable
enum class ParamType {
    @SerialName("int") INT,
    @SerialName("float") FLOAT,
    @SerialName("bool") BOOL,
    @SerialName("enum") ENUM,
    @SerialName("string") STRING,
    @SerialName("text") TEXT,

    /**
     * A long blob the runtime parses — a Jinja chat template, a GBNF grammar.
     *
     * Its own type rather than a longer [TEXT] because three things about it
     * differ, and all three follow from what it holds rather than from which
     * key it is. It is hundreds of lines, so it belongs behind a reveal instead
     * of an always-open box that pushes the rest of the screen away. Its
     * default comes from the loaded model rather than from the manifest, so
     * "reset" means "give me the model's own back" and cannot be rendered as
     * `reset to <value>` — printing a five-thousand-character template into a
     * link is not a link. And an edit is committed deliberately, because a
     * half-typed template is a broken one and every keystroke would apply it.
     *
     * The chat settings sheet already treated the chat template this way and
     * the parameter screen did not, which is how the same setting came to have
     * two different affordances. This is that behaviour, moved to where the
     * type system can hand it to every parameter shaped like it.
     */
    @SerialName("code") CODE,
    @SerialName("string[]") STRING_ARRAY,
    @SerialName("int[]") INT_ARRAY,
    @SerialName("map") MAP,
    @SerialName("path") PATH,

    /**
     * Several installed files at once, each with its own strength.
     *
     * A LoRA is the case: sd.cpp takes an array of them and applies all of
     * them, and stacking a style at 0.8 under a character at 0.6 is the
     * ordinary way they are used. Rendering that as one dropdown would make the
     * runtime's own capability unreachable from the app.
     */
    @SerialName("weighted_paths") WEIGHTED_PATHS,
    @SerialName("ordered_list") ORDERED_LIST,
}

/** Conditional visibility, e.g. YaRN params only when `rope_scaling_type = yarn`. */
@Serializable
data class Dependency(
    val key: String,
    val equals: JsonElement? = null,
    val notEquals: JsonElement? = null,
)

@Serializable
data class AppliesTo(
    val modality: List<String>? = null,
    val arch: List<String>? = null,
    /**
     * `image`, `video`, or both — which kind of output this setting reaches.
     *
     * Diffusion's two screens share one runtime and one parameter set, so
     * `modality` cannot separate them: a Wan checkpoint and an SDXL one are
     * both `diffusion`. Without this, `ip_adapter_strength` and `batch_count`
     * were offered on a video model that has no field for either, and
     * `video_frames` needed a hand-typed list of every video architecture to
     * keep itself off a still.
     *
     * Answered from [ai.ondevice.core.DiffusionFamily.isVideo], which is
     * derived from upstream's own branch — not from a list kept here that
     * stops being true the week an architecture is added.
     */
    val output: List<String>? = null,
)

internal fun renderJson(element: JsonElement): String = when (element) {
    is JsonPrimitive -> when {
        element.isString -> element.content
        element.booleanOrNull != null -> if (element.booleanOrNull == true) "on" else "off"
        element.intOrNull != null -> element.intOrNull.toString()
        element.doubleOrNull != null -> {
            val d = element.doubleOrNull!!
            if (d == d.toLong().toDouble()) d.toLong().toString() else String.format("%.2f", d)
        }
        else -> element.content
    }
    is kotlinx.serialization.json.JsonArray ->
        element.joinToString(", ") { (it as? JsonPrimitive)?.content ?: it.toString() }
    else -> element.toString()
}
