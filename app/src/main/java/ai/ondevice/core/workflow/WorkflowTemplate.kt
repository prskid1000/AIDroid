package ai.ondevice.core.workflow

/**
 * The Script step's language: a template with expressions in it.
 *
 * Deliberately not a general one. There is no scripting surface in this app to
 * reuse — no JS engine, no evaluator beyond a small arithmetic parser the
 * calculator tool uses — so a real language would mean embedding one, and
 * embedding one is a native-build decision with a sandbox, a memory bound and
 * an interrupt behind it. That decision should be made deliberately and early
 * rather than arrived at because someone wanted a `for` loop.
 *
 * What this covers is what "transform" turns out to mean in practice on this
 * device: taking what an earlier step said and shaping it into what the next
 * one needs. Turning a transcript into a prompt. Trimming a model's preamble.
 * Joining pieces. Pulling a number out to decide whether to go round again.
 *
 * It is deterministic, has no I/O, cannot loop, and runs in microseconds — so
 * it needs no timeout, no cancellation and no permission, which are three
 * things a real language would have needed and a graph node is a bad place to
 * ask for.
 *
 * ```
 * A summary of {{ 1.text }}:
 * {{ trim(2.text) }}
 * {{ upper(slice(2.text, 0, 40)) }}
 * ```
 */
object WorkflowTemplate {

    /** What an expression can see: earlier steps' outputs, by reference. */
    fun interface Lookup {
        /** `"2"` or `"2.text"`, or null when nothing answers to it. */
        fun value(reference: String): String?
    }

    /**
     * Render [template], replacing each `{{ … }}` with what it evaluates to.
     *
     * An expression that cannot be evaluated is left as it was written rather
     * than replaced with nothing: a prompt that silently loses a clause is
     * harder to notice than one that visibly still has braces in it.
     */
    fun render(template: String, lookup: Lookup): String {
        val out = StringBuilder()
        var i = 0
        while (i < template.length) {
            val open = template.indexOf("{{", i)
            if (open < 0) {
                out.append(template, i, template.length)
                break
            }
            val close = template.indexOf("}}", open + 2)
            if (close < 0) {
                out.append(template, i, template.length)
                break
            }
            out.append(template, i, open)
            val source = template.substring(open + 2, close).trim()
            out.append(evaluate(source, lookup).getOrElse { "{{$source}}" })
            i = close + 2
        }
        return out.toString()
    }

    /** Evaluate one expression, without the braces. */
    fun evaluate(source: String, lookup: Lookup): Result<String> = runCatching {
        Parser(source, lookup).parseAll()
    }

    /**
     * Whether an expression reads as true.
     *
     * Used by the conditions on Only-if and Repeat. Empty, `false`, `0` and
     * `no` are false; everything else is true. Named rather than inferred so
     * that a condition which is a bare reference — `{{3.text}}` — behaves the
     * way somebody would expect rather than the way a parser would.
     */
    fun truthy(rendered: String): Boolean {
        val value = rendered.trim().lowercase()
        return value.isNotEmpty() && value !in FALSEY
    }

    private val FALSEY = setOf("false", "0", "no", "off", "null", "none")

    /**
     * A recursive-descent parser over a small expression grammar.
     *
     * Values are strings; arithmetic and comparison coerce to numbers where
     * both sides look like numbers and compare as text otherwise, which is the
     * behaviour that surprises people least when the inputs came out of a
     * language model.
     */
    private class Parser(private val src: String, private val lookup: Lookup) {
        private var at = 0

        fun parseAll(): String {
            val value = comparison()
            skipSpace()
            require(at >= src.length) { "unexpected '${src.substring(at)}'" }
            return value
        }

        private fun comparison(): String {
            var left = additive()
            while (true) {
                skipSpace()
                val op = OPERATORS.firstOrNull { src.startsWith(it, at) } ?: return left
                at += op.length
                val right = additive()
                left = compare(left, right, op).toString()
            }
        }

        private fun compare(left: String, right: String, op: String): Boolean {
            val a = left.trim().toDoubleOrNull()
            val b = right.trim().toDoubleOrNull()
            return if (a != null && b != null) {
                when (op) {
                    "==" -> a == b; "!=" -> a != b
                    ">=" -> a >= b; "<=" -> a <= b
                    ">" -> a > b; else -> a < b
                }
            } else {
                val c = left.compareTo(right)
                when (op) {
                    "==" -> c == 0; "!=" -> c != 0
                    ">=" -> c >= 0; "<=" -> c <= 0
                    ">" -> c > 0; else -> c < 0
                }
            }
        }

        private fun additive(): String {
            var left = multiplicative()
            while (true) {
                skipSpace()
                when {
                    peek() == '+' -> {
                        at++
                        val right = multiplicative()
                        val a = left.trim().toDoubleOrNull()
                        val b = right.trim().toDoubleOrNull()
                        // Numbers add; anything else joins. A model's output is
                        // text far more often than it is a number.
                        left = if (a != null && b != null) number(a + b) else left + right
                    }
                    peek() == '-' -> {
                        at++
                        val right = multiplicative()
                        left = number(left.num() - right.num())
                    }
                    else -> return left
                }
            }
        }

        private fun multiplicative(): String {
            var left = unary()
            while (true) {
                skipSpace()
                val c = peek()
                if (c != '*' && c != '/' && c != '%') return left
                at++
                val right = unary()
                left = number(
                    when (c) {
                        '*' -> left.num() * right.num()
                        '/' -> left.num() / right.num()
                        else -> left.num() % right.num()
                    },
                )
            }
        }

        private fun unary(): String {
            skipSpace()
            if (peek() == '-') {
                at++
                return number(-unary().num())
            }
            return primary()
        }

        private fun primary(): String {
            skipSpace()
            when (peek()) {
                '(' -> {
                    at++
                    val value = comparison()
                    skipSpace()
                    require(peek() == ')') { "expected )" }
                    at++
                    return value
                }
                '"', '\'' -> return quoted()
            }
            val word = word()
            require(word.isNotEmpty()) { "expected a value at $at" }
            skipSpace()
            if (peek() == '(') return call(word)
            // A bare word is a reference to an earlier step, or itself.
            return lookup.value(word) ?: word
        }

        private fun call(name: String): String {
            at++ // (
            val args = mutableListOf<String>()
            skipSpace()
            if (peek() == ')') {
                at++
            } else {
                while (true) {
                    args += comparison()
                    skipSpace()
                    when (peek()) {
                        ',' -> at++
                        ')' -> { at++; break }
                        else -> throw IllegalArgumentException("expected , or ) in $name()")
                    }
                }
            }
            return apply(name, args)
        }

        private fun apply(name: String, a: List<String>): String = when (name) {
            "trim" -> a.first().trim()
            "lower" -> a.first().lowercase()
            "upper" -> a.first().uppercase()
            "length" -> a.first().length.toString()
            "replace" -> a[0].replace(a[1], a.getOrElse(2) { "" })
            "join" -> a.drop(1).joinToString(a[0])
            "split" -> a[0].split(a.getOrElse(1) { "\n" }).joinToString("\n")
            "slice" -> {
                val s = a[0]
                val from = a.getOrElse(1) { "0" }.num().toInt().coerceIn(0, s.length)
                val to = a.getOrElse(2) { s.length.toString() }.num().toInt().coerceIn(from, s.length)
                s.substring(from, to)
            }
            "match" -> Regex(a[1]).find(a[0])?.let { m ->
                m.groupValues.getOrNull(1)?.takeIf { it.isNotEmpty() } ?: m.value
            } ?: ""
            "round" -> number(kotlin.math.round(a.first().num()))
            "min" -> number(a.minOf { it.num() })
            "max" -> number(a.maxOf { it.num() })
            "default" -> a[0].ifBlank { a.getOrElse(1) { "" } }
            else -> throw IllegalArgumentException("no function named $name")
        }

        private fun quoted(): String {
            val quote = src[at]
            at++
            val out = StringBuilder()
            while (at < src.length && src[at] != quote) {
                if (src[at] == '\\' && at + 1 < src.length) at++
                out.append(src[at])
                at++
            }
            require(at < src.length) { "unterminated string" }
            at++
            return out.toString()
        }

        private fun word(): String {
            val start = at
            while (at < src.length && (src[at].isLetterOrDigit() || src[at] == '_' || src[at] == '.')) at++
            return src.substring(start, at)
        }

        private fun peek(): Char = if (at < src.length) src[at] else ' '

        private fun skipSpace() {
            while (at < src.length && src[at].isWhitespace()) at++
        }

        private fun String.num(): Double =
            trim().toDoubleOrNull() ?: throw IllegalArgumentException("'$this' is not a number")

        private fun number(value: Double): String =
            if (value == value.toLong().toDouble()) value.toLong().toString() else value.toString()

        companion object {
            /** Two-character forms first, or `>=` would parse as `>` then `=`. */
            val OPERATORS = listOf("==", "!=", ">=", "<=", ">", "<")
        }
    }
}
