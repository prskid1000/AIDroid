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
 * Tools that need nothing but this device.
 *
 * The selection is deliberately narrow. Every one of these answers a question a
 * local model genuinely cannot answer on its own — the clock it has no access
 * to, arithmetic it will get subtly wrong, and the state of the very device it
 * is running on — and not one of them touches the network. That is the whole
 * point: the built-in set is the part of tool use that costs the user nothing
 * in privacy, so it is the part that can be on by default.
 *
 * Anything that leaves the handset lives behind an MCP server the user has
 * added by hand (see [McpToolProvider]).
 */
class BuiltInToolProvider(
    private val db: OnDeviceDatabase,
    private val capabilities: DeviceCapabilities,
) : ToolProvider {

    override val id: String = ID

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    override suspend fun specs(): List<ToolSpec> = listOf(
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
    }
}

/**
 * A small recursive-descent evaluator.
 *
 * Deliberately *not* a scripting engine. A tool the model can drive is an
 * attack surface, and `eval` on model-authored text with anything more than
 * arithmetic in scope is how a chat app grows a remote-code path. The grammar
 * here has no identifiers beyond two constants and no way to reach the host.
 */
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
            var value = power()
            while (true) {
                skipSpace()
                when {
                    consume('*') -> value *= power()
                    consume('/') -> {
                        val divisor = power()
                        require(divisor != 0.0) { "division by zero" }
                        value /= divisor
                    }
                    consume('%') -> {
                        val divisor = power()
                        require(divisor != 0.0) { "modulo by zero" }
                        value %= divisor
                    }
                    else -> return value
                }
            }
        }

        private fun power(): Double {
            val base = unary()
            skipSpace()
            // Right-associative, as everyone writing 2^3^2 expects.
            return if (consume('^')) Math.pow(base, power()) else base
        }

        private fun unary(): Double {
            skipSpace()
            if (consume('-')) return -unary()
            if (consume('+')) return unary()
            return atom()
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
