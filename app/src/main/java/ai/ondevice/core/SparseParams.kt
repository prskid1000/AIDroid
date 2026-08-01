package ai.ondevice.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive

/** A sparse key→value parameter set. */
@JvmInline
value class SparseParams(val values: Map<String, JsonElement> = emptyMap()) {

    val keys: Set<String> get() = values.keys
    val isEmpty: Boolean get() = values.isEmpty()

    operator fun get(key: String): JsonElement? = values[key]
    operator fun contains(key: String): Boolean = key in values

    fun float(key: String): Float? = values[key]?.asPrimitive()?.doubleOrNull?.toFloat()
    fun int(key: String): Int? = values[key]?.asPrimitive()?.intOrNull
    fun bool(key: String): Boolean? = values[key]?.asPrimitive()?.booleanOrNull
    fun string(key: String): String? = (values[key] as? JsonPrimitive)?.takeIf { it.isString }?.content
    fun stringList(key: String): List<String>? = (values[key] as? kotlinx.serialization.json.JsonArray)
        ?.mapNotNull { (it as? JsonPrimitive)?.content }

    fun with(key: String, value: JsonElement): SparseParams = SparseParams(values + (key to value))
    fun with(key: String, value: Float) = with(key, JsonPrimitive(value))
    fun with(key: String, value: Int) = with(key, JsonPrimitive(value))
    fun with(key: String, value: Boolean) = with(key, JsonPrimitive(value))
    fun with(key: String, value: String) = with(key, JsonPrimitive(value))
    fun with(key: String, value: List<String>) =
        with(key, kotlinx.serialization.json.JsonArray(value.map { JsonPrimitive(it) }))

    /** Clearing a key means "back to the runtime default", not "store a null". */
    fun without(key: String): SparseParams = SparseParams(values - key)

    /** Layer another set on top. */
    fun overlaidWith(other: SparseParams): SparseParams = SparseParams(values + other.values)

    fun toJsonString(): String = JSON.encodeToString(JsonObject.serializer(), JsonObject(values))

    private fun JsonElement.asPrimitive(): JsonPrimitive? = this as? JsonPrimitive

    companion object {
        val EMPTY = SparseParams()

        val JSON = Json {
            ignoreUnknownKeys = true
            isLenient = true
            encodeDefaults = true
            explicitNulls = false
        }

        /** Parse from storage. */
        fun parse(json: String?): SparseParams {
            if (json.isNullOrBlank()) return EMPTY
            return runCatching {
                SparseParams(JSON.parseToJsonElement(json).let { it as JsonObject }.toMap())
            }.getOrElse { EMPTY }
        }

        fun of(vararg pairs: Pair<String, Any?>): SparseParams = SparseParams(
            pairs.mapNotNull { (k, v) ->
                val element: JsonElement = when (v) {
                    null -> JsonNull
                    is JsonElement -> v
                    is Boolean -> JsonPrimitive(v)
                    is Int -> JsonPrimitive(v)
                    is Long -> JsonPrimitive(v)
                    is Float -> JsonPrimitive(v)
                    is Double -> JsonPrimitive(v)
                    is String -> JsonPrimitive(v)
                    is List<*> -> kotlinx.serialization.json.JsonArray(
                        v.map { JsonPrimitive(it?.toString() ?: "") },
                    )
                    else -> JsonPrimitive(v.toString())
                }
                k to element
            }.toMap(),
        )
    }
}

/** Render a JSON value the way the parameter rows display it. */
fun JsonElement.displayValue(): String = when (this) {
    is JsonPrimitive -> when {
        isString -> content
        booleanOrNull != null -> if (boolean()) "on" else "off"
        intOrNull != null -> intOrNull.toString()
        doubleOrNull != null -> formatNumber(doubleOrNull!!)
        else -> content
    }
    is kotlinx.serialization.json.JsonArray -> jsonArray.joinToString(", ") { it.jsonPrimitive.content }
    else -> toString()
}

private fun JsonPrimitive.boolean(): Boolean = booleanOrNull == true

private fun formatNumber(d: Double): String =
    if (d == d.toLong().toDouble()) d.toLong().toString() else String.format("%.2f", d)
