package ai.ondevice.tools

import ai.ondevice.core.Fmt
import ai.ondevice.core.Modality
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.hf.DeviceCapabilities
import ai.ondevice.engine.ToolSpec
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * The tools the app carries itself.
 *
 * Three of them need nothing but this device. [web_search] is the exception and
 * the only one that leaves it: it is offered only when a [WebSearch] is passed,
 * so a build or a screen that has no business going out simply does not get it.
 */
class BuiltInToolProvider(
    private val db: OnDeviceDatabase,
    private val capabilities: DeviceCapabilities,
    private val web: WebSearch? = null,
) : ToolProvider {

    override val id: String = ID

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    override suspend fun specs(): List<ToolSpec> = listOfNotNull(
        web?.let {
            ToolSpec(
                name = "web_search",
                description = "Search the public web and read the results. Use it for anything " +
                    "that happened, changed or was published after your training data: current " +
                    "events, today's prices, a library's latest version, a page the user names. " +
                    "Do not use it for arithmetic, for what is already in this conversation, or " +
                    "for what the other tools here answer. Cite the URLs it returns.",
                parametersJson = """
                    {
                      "type": "object",
                      "properties": {
                        "query": {
                          "type": "string",
                          "description": "What to search for, as you would type it into a search box."
                        },
                        "read_pages": {
                          "type": "integer",
                          "description": "How many of the top results to open and read in full, 0 to 3. Snippets alone (0) answer most factual questions and are far faster; ask for 1 or 2 when the question needs what an article actually says.",
                          "minimum": 0,
                          "maximum": 3
                        }
                      },
                      "required": ["query"]
                    }
                """.trimIndent(),
            )
        },
        ToolSpec(
            name = "get_current_time",
            description = "The current date and time on this device. Use it whenever the answer " +
                "depends on today's date — the model's own sense of 'now' is its training cutoff.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "timezone": {
                      "type": "string",
                      "description": "IANA zone, e.g. Europe/London. Defaults to the device's own."
                    }
                  }
                }
            """.trimIndent(),
        ),
        ToolSpec(
            name = "calculate",
            description = "Evaluate an arithmetic expression exactly. Supports + - * / % ^, " +
                "parentheses, and the constants pi and e. Use it rather than doing multi-digit " +
                "arithmetic in your head.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "expression": { "type": "string", "description": "e.g. (2*36*32768*1024*2)/1e9" }
                  },
                  "required": ["expression"]
                }
            """.trimIndent(),
        ),
        ToolSpec(
            name = "device_status",
            description = "Free and total RAM, free storage, and which models are installed on " +
                "this device. Use it when asked whether something will fit or run here.",
            parametersJson = """{ "type": "object", "properties": {} }""",
        ),
    )

    override suspend fun call(name: String, argumentsJson: String): ToolResult {
        val args = runCatching { json.parseToJsonElement(argumentsJson).jsonObject }.getOrNull()

        return when (name) {
            "web_search" -> {
                val search = web ?: return fail("Web search is not available in this build.")
                val query = args?.get("query")?.jsonPrimitive?.content?.trim().orEmpty()
                if (query.length < 2) return fail("web_search needs a \"query\" of at least two characters.")
                val pages = (args?.get("read_pages")?.jsonPrimitive?.content?.toIntOrNull() ?: 0)
                    .coerceIn(0, MAX_PAGES_READ)
                runCatching { search.search(query, MAX_RESULTS, pages) }.fold(
                    onSuccess = { results ->
                        android.util.Log.i(
                            "BuiltInTools",
                            "web_search \"$query\" → ${results.size} result(s), $pages page(s) read",
                        )
                        ok(search.render(query, results))
                    },
                    // The model is told what failed so it can say so, rather
                    // than filling the gap with something plausible.
                    onFailure = { fail("The web search failed: ${it.message ?: "no connection"}.") },
                )
            }

            "get_current_time" -> {
                val zone = args?.get("timezone")?.jsonPrimitive?.content
                val format = SimpleDateFormat("EEEE, d MMMM yyyy 'at' HH:mm:ss z", Locale.UK).apply {
                    timeZone = zone?.let { TimeZone.getTimeZone(it) } ?: TimeZone.getDefault()
                }
                ok(format.format(Date()))
            }

            "calculate" -> {
                val expression = args?.get("expression")?.jsonPrimitive?.content
                    ?: return fail("calculate needs an \"expression\".")
                runCatching { Arithmetic.evaluate(expression) }
                    .fold(
                        // Whole results print without a trailing ".0"; anything
                        // else keeps enough digits to be checkable.
                        onSuccess = { value ->
                            val rendered = if (value == Math.floor(value) && !value.isInfinite() &&
                                Math.abs(value) < 1e15
                            ) {
                                value.toLong().toString()
                            } else {
                                String.format("%.10g", value).trimEnd('0').trimEnd('.')
                            }
                            ok("$expression = $rendered")
                        },
                        onFailure = { fail("Could not evaluate \"$expression\": ${it.message}") },
                    )
            }

            "device_status" -> {
                val models = db.models().getAll()
                ok(
                    buildString {
                        appendLine("RAM: ${Fmt.bytes(capabilities.availableRamBytes)} free of ${Fmt.bytes(capabilities.totalRamBytes)}")
                        appendLine("Storage: ${Fmt.bytes(capabilities.freeStorageBytes)} free")
                        appendLine("Performance cores: ${capabilities.performanceCores}")
                        appendLine("Installed models: ${models.size}")
                        models.take(20).forEach { model ->
                            appendLine("  - ${model.displayName} (${model.modality.name.lowercase()}, ${Fmt.bytes(model.sizeBytes)})")
                        }
                    }.trim(),
                )
            }

            else -> fail("No built-in tool named \"$name\".")
        }
    }

    private fun ok(text: String) = ToolResult(text, providerId = ID)
    private fun fail(text: String) = ToolResult(text, isError = true, providerId = ID)

    companion object {
        const val ID = "built-in"

        /** Enough to answer from, few enough to fit beside the conversation. */
        const val MAX_RESULTS = 5
        const val MAX_PAGES_READ = 3

        /** What the Tools screen lists, from here rather than from a second copy. */
        fun toolNames(webSearchAvailable: Boolean): List<String> = buildList {
            if (webSearchAvailable) add("web_search")
            add("get_current_time")
            add("calculate")
            add("device_status")
        }
    }
}

/** A small recursive-descent evaluator. */
internal object Arithmetic {

    fun evaluate(input: String): Double {
        val parser = Parser(input)
        val value = parser.expression()
        parser.skipSpace()
        require(parser.atEnd) { "unexpected \"${parser.rest}\"" }
        require(value.isFinite()) { "the result is not a finite number" }
        return value
    }

    private class Parser(private val text: String) {
        private var index = 0

        val atEnd: Boolean get() = index >= text.length
        val rest: String get() = text.substring(index).take(12)

        fun skipSpace() {
            while (index < text.length && text[index].isWhitespace()) index++
        }

        fun expression(): Double {
            var value = term()
            while (true) {
                skipSpace()
                when {
                    consume('+') -> value += term()
                    consume('-') -> value -= term()
                    else -> return value
                }
            }
        }

        private fun term(): Double {
            var value = unary()
            while (true) {
                skipSpace()
                when {
                    consume('*') -> value *= unary()
                    consume('/') -> {
                        val divisor = unary()
                        require(divisor != 0.0) { "division by zero" }
                        value /= divisor
                    }
                    consume('%') -> {
                        val divisor = unary()
                        require(divisor != 0.0) { "modulo by zero" }
                        value %= divisor
                    }
                    else -> return value
                }
            }
        }

        /** Sign binds *looser* than the exponent, so `-2^2` is −4. */
        private fun unary(): Double {
            skipSpace()
            if (consume('-')) return -unary()
            if (consume('+')) return unary()
            return power()
        }

        private fun power(): Double {
            val base = atom()
            skipSpace()
            // Right-associative, as everyone writing 2^3^2 expects. The right
            // operand is a unary so `2^-3` reads the way it looks.
            return if (consume('^')) Math.pow(base, unary()) else base
        }

        private fun atom(): Double {
            skipSpace()
            if (consume('(')) {
                val value = expression()
                skipSpace()
                require(consume(')')) { "missing closing bracket" }
                return value
            }
            if (text.startsWith("pi", index, ignoreCase = true)) { index += 2; return Math.PI }
            if (index < text.length && (text[index] == 'e' || text[index] == 'E') &&
                (index + 1 >= text.length || !text[index + 1].isDigit())
            ) {
                index++
                return Math.E
            }

            val start = index
            while (index < text.length && (text[index].isDigit() || text[index] == '.')) index++
            // Scientific notation: 1e9, 2.5e-3.
            if (index < text.length && (text[index] == 'e' || text[index] == 'E') && index > start) {
                val mark = index
                index++
                if (index < text.length && (text[index] == '+' || text[index] == '-')) index++
                if (index < text.length && text[index].isDigit()) {
                    while (index < text.length && text[index].isDigit()) index++
                } else {
                    index = mark
                }
            }
            require(index > start) { "expected a number at position $start" }
            return text.substring(start, index).toDouble()
        }

        private fun consume(char: Char): Boolean {
            if (index < text.length && text[index] == char) {
                index++
                return true
            }
            return false
        }
    }
}
