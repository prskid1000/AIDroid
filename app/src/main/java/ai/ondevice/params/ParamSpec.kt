package ai.ondevice.params

import ai.ondevice.core.Tier
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull

/**
 * SPEC §16.1 — the manifest.
 *
 * This is what makes §1.5 real: **there are zero hardcoded parameter widgets in
 * the app.** Every parameter in §4–7 is a row in this document, and the UI is a
 * generic renderer over it. The test the spec sets is unambiguous — if adding a
 * new upstream parameter requires touching Kotlin UI source, the design has
 * been violated (Appendix A #9).
 *
 * The manifest is generated in CI from pinned upstream sources (§16.2) and
 * consumed here as signed data. The app never parses upstream C++ (Appendix A
 * #13).
 */
@Serializable
data class ParamManifest(
    val manifestVersion: Int,
    val generatedAt: String = "",
    val runtimes: Map<String, RuntimeParams> = emptyMap(),
) {
    fun paramsFor(runtimeId: String): List<ParamSpec> = runtimes[runtimeId]?.params.orEmpty()

    fun buildTagFor(runtimeId: String): String? = runtimes[runtimeId]?.buildTag

    fun spec(runtimeId: String, key: String): ParamSpec? =
        runtimes[runtimeId]?.params?.firstOrNull { it.key == key }
}

@Serializable
data class RuntimeParams(
    val sourceCommit: String = "",
    val buildTag: String = "",
    val jniContract: Int = 3,
    val params: List<ParamSpec> = emptyList(),
)

@Serializable
data class ParamSpec(
    /** Canonical name — this is the wire key passed through JNI verbatim. */
    val key: String,
    /** Section heading: model, sampling, generation, vision, vad, … */
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

/**
 * §16.4 — adding a new *type* is the only case that requires app code. Adding a
 * new *parameter* of an existing type requires none.
 */
@Serializable
enum class ParamType {
    @SerialName("int") INT,
    @SerialName("float") FLOAT,
    @SerialName("bool") BOOL,
    @SerialName("enum") ENUM,
    @SerialName("string") STRING,
    @SerialName("text") TEXT,
    @SerialName("string[]") STRING_ARRAY,
    @SerialName("int[]") INT_ARRAY,
    @SerialName("map") MAP,
    @SerialName("path") PATH,
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
