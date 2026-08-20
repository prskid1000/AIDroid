package ai.ondevice.proxy

import ai.ondevice.engine.EngineMessage
import ai.ondevice.engine.ToolSpec
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** The one parser. Lenient because clients are, and unknown keys are the norm. */
val ProxyJson: Json = Json {
    ignoreUnknownKeys = true
    isLenient = true
    encodeDefaults = false
    explicitNulls = false
}

// — JsonObject readers, so the codecs read like the protocol they decode —

internal fun JsonObject.str(key: String): String? =
    (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.content

internal fun JsonObject.strOrAny(key: String): String? =
    (this[key] as? JsonPrimitive)?.content

internal fun JsonObject.num(key: String): Double? =
    (this[key] as? JsonPrimitive)?.doubleOrNull

internal fun JsonObject.i(key: String): Int? = (this[key] as? JsonPrimitive)?.intOrNull

internal fun JsonObject.f(key: String): Float? = num(key)?.toFloat()

internal fun JsonObject.b(key: String): Boolean? = (this[key] as? JsonPrimitive)?.booleanOrNull

internal fun JsonObject.arr(key: String): JsonArray? = this[key] as? JsonArray

internal fun JsonObject.obj(key: String): JsonObject? = this[key] as? JsonObject

internal fun JsonObject.strings(key: String): List<String> =
    arr(key)?.mapNotNull { (it as? JsonPrimitive)?.content }.orEmpty()

/**
 * Which protocol a request arrived on.
 *
 * Carried on the request rather than inferred at the far end, because two of
 * the routes (`/v1/models`, `/v1/messages/count_tokens`) are shape-sniffed from
 * headers and the answer has to travel with the request rather than be worked
 * out twice.
 */
enum class Protocol { ANTHROPIC, OPENAI }

/**
 * A chat request, after either protocol has been decoded.
 *
 * This is the whole of the "internal shape" telecode needs 1500 lines to
 * maintain, and it is small for one reason: we are the upstream. There is no
 * second serialisation to an OpenAI body on the way out — the next thing that
 * happens to this is `LlamaEngine.generate`.
 */
data class ChatRequest(
    val protocol: Protocol,
    /** Exactly what the client asked for, so the answer can echo it back. */
    val requestedModel: String,
    val messages: List<EngineMessage>,
    val system: String?,
    val tools: List<ToolSpec>,
    val stream: Boolean,
    /** Sampling and generation keys, already translated to this app's names. */
    val params: ai.ondevice.core.SparseParams,
    /** Images that arrived on the final user turn, for a vision model. */
    val imagePaths: List<String> = emptyList(),
    /** `tool_choice`, normalised. Null = the model decides. */
    val forcedTool: String? = null,
)

/**
 * How a request is refused, in a way both protocols can render.
 *
 * A type rather than an exception message, because the [suggestion] has to
 * survive to the wire: the engines here already produce "not enough memory,
 * lower the context size or pick a smaller quant", and throwing the second
 * half away at the protocol boundary is exactly the silent failure SPEC 1.2
 * forbids.
 */
class ProxyRefusal(
    val status: Int,
    val type: String,
    override val message: String,
    val suggestion: String? = null,
    /** Seconds, for the two statuses that mean "later, not never". */
    val retryAfter: Int? = null,
) : Exception(message) {

    fun body(protocol: Protocol): String {
        val text = if (suggestion.isNullOrBlank()) message else "$message\n\n$suggestion"
        return when (protocol) {
            Protocol.ANTHROPIC -> ProxyJson.encodeToString(
                JsonObject.serializer(),
                JsonObject(
                    mapOf(
                        "type" to JsonPrimitive("error"),
                        "error" to JsonObject(
                            mapOf(
                                "type" to JsonPrimitive(type),
                                "message" to JsonPrimitive(text),
                            ),
                        ),
                    ),
                ),
            )
            Protocol.OPENAI -> ProxyJson.encodeToString(
                JsonObject.serializer(),
                JsonObject(
                    mapOf(
                        "error" to JsonObject(
                            mapOf(
                                "type" to JsonPrimitive(type),
                                "message" to JsonPrimitive(text),
                                "code" to JsonPrimitive(type),
                            ),
                        ),
                    ),
                ),
            )
        }
    }

    companion object {
        fun badRequest(message: String, suggestion: String? = null) =
            ProxyRefusal(400, "invalid_request_error", message, suggestion)

        fun unauthorized(message: String) =
            ProxyRefusal(401, "authentication_error", message)

        fun notFound(message: String, suggestion: String? = null) =
            ProxyRefusal(404, "not_found_error", message, suggestion)

        fun conflict(message: String, suggestion: String? = null) =
            ProxyRefusal(409, "conflict_error", message, suggestion)

        fun busy(message: String, retryAfter: Int) =
            ProxyRefusal(429, "rate_limit_error", message, retryAfter = retryAfter)

        fun unavailable(message: String, suggestion: String? = null, retryAfter: Int? = null) =
            ProxyRefusal(503, "overloaded_error", message, suggestion, retryAfter)

        fun notImplemented(message: String, suggestion: String? = null) =
            ProxyRefusal(501, "not_implemented_error", message, suggestion)
    }
}

/** SSE framing, in the two shapes the protocols use. */
object Sse {
    /** Anthropic names every event; clients dispatch on the name, not the payload. */
    fun named(event: String, data: String): String = "event: $event\ndata: $data\n\n"

    /** OpenAI sends bare data lines. */
    fun data(data: String): String = "data: $data\n\n"

    const val DONE = "data: [DONE]\n\n"

    /** A comment line. Keeps a connection alive without meaning anything. */
    const val KEEPALIVE = ": keepalive\n\n"
}

internal fun jsonObjectOf(vararg pairs: Pair<String, JsonElement>): JsonObject =
    JsonObject(pairs.toMap())

internal fun encode(element: JsonElement): String =
    ProxyJson.encodeToString(JsonElement.serializer(), element)

/**
 * Strip a client's own bookkeeping out of a message.
 *
 * Two things, both of which a coding client re-sends every single turn and
 * neither of which the model on this phone can do anything with: the
 * `<system-reminder>` blocks, and the `<total_tokens>` budget line. On a
 * desktop this is tidiness; at 8k of context it is several percent of the
 * budget, every turn.
 */
internal fun stripClientBookkeeping(text: String): String {
    if (text.isEmpty()) return text
    var out = REMINDER.replace(text, "")
    out = TOTAL_TOKENS.replace(out, "")
    // Collapse the run of blank lines a removal leaves behind, so the prompt
    // prefix stays byte-identical between turns and the cache keeps matching.
    return out.replace(BLANK_RUN, "\n\n").trim()
}

private val REMINDER = Regex(
    "<system-reminder>.*?</system-reminder>",
    setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE),
)
private val TOTAL_TOKENS = Regex(
    "^\\s*<total_tokens>.*?</total_tokens>\\s*$",
    setOf(RegexOption.MULTILINE, RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE),
)
private val BLANK_RUN = Regex("\n{3,}")

/** The tool-call id shape both protocols accept. */
internal fun newCallId(): String = "call_" + java.util.UUID.randomUUID().toString().replace("-", "").take(20)

internal fun newMessageId(): String = "msg_" + java.util.UUID.randomUUID().toString().replace("-", "").take(24)

internal fun newCompletionId(): String =
    "chatcmpl-" + java.util.UUID.randomUUID().toString().replace("-", "").take(16)

/** Shared by the two decoders: a JSON schema for a tool, however it was wrapped. */
internal fun toolSpecFrom(name: String, description: String, schema: JsonElement?): ToolSpec =
    ToolSpec(
        name = name,
        description = description,
        parametersJson = schema?.let { encode(it) } ?: """{"type":"object","properties":{}}""",
    )

internal fun JsonElement.asTextContent(): String = when (this) {
    is JsonPrimitive -> if (isString) content else content
    is JsonArray -> jsonArray.joinToString("\n") { block ->
        (block as? JsonObject)?.str("text").orEmpty()
    }.trim()
    is JsonObject -> str("text").orEmpty()
}
