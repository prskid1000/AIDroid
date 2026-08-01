package ai.ondevice.tools

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** The `calculate` tool's evaluator. */
class ArithmeticTest {

    private fun eval(expression: String) = Arithmetic.evaluate(expression)

    private fun assertEvaluates(expected: Double, expression: String) {
        assertEquals(expression, expected, eval(expression), 1e-9)
    }

    // — precedence and associativity, the answers a person would check —

    @Test
    fun `multiplication binds tighter than addition`() {
        assertEvaluates(14.0, "2 + 3 * 4")
        assertEvaluates(20.0, "(2 + 3) * 4")
    }

    @Test
    fun `subtraction is left associative`() {
        // 10-3-2 is 5, not 9. Getting this wrong is the classic parser bug.
        assertEvaluates(5.0, "10 - 3 - 2")
        assertEvaluates(2.0, "12 / 3 / 2")
    }

    @Test
    fun `exponentiation is right associative and beats unary minus`() {
        assertEvaluates(512.0, "2 ^ 3 ^ 2")
        assertEvaluates(-4.0, "-2 ^ 2")
    }

    @Test
    fun `unary minus stacks`() {
        assertEvaluates(3.0, "--3")
        assertEvaluates(-3.0, "-+3")
    }

    @Test
    fun `modulo and division share a level`() {
        assertEvaluates(1.0, "10 % 3")
        assertEvaluates(2.0, "10 % 4 / 1")
    }

    @Test
    fun `whitespace anywhere is fine`() {
        assertEvaluates(7.0, "  1   +   2 * 3  ")
    }

    // — constants and notation —

    @Test
    fun `pi and e are the only names it knows`() {
        assertEquals(Math.PI, eval("pi"), 1e-12)
        assertEquals(Math.E, eval("e"), 1e-12)
        assertEvaluates(Math.PI * 2, "2pi".let { "2 * pi" })
    }

    @Test
    fun `scientific notation parses, and e stays a constant when it is one`() {
        assertEvaluates(1_000_000_000.0, "1e9")
        assertEvaluates(0.0025, "2.5e-3")
        // The tricky one: `e` alone is Euler's number, `1e9` is a literal, and
        // `2*e` must not be read as a malformed exponent.
        assertEquals(2 * Math.E, eval("2 * e"), 1e-12)
    }

    // — refusals: these are the reason the class exists —

    @Test
    fun `division by zero is refused, not infinity`() {
        val failure = assertThrows(IllegalArgumentException::class.java) { eval("1 / 0") }
        assertTrue(failure.message.orEmpty(), failure.message.orEmpty().contains("zero"))
        assertThrows(IllegalArgumentException::class.java) { eval("1 % 0") }
    }

    @Test
    fun `a non-finite result is refused`() {
        // Otherwise the model is handed "Infinity" and told it is a number.
        assertThrows(IllegalArgumentException::class.java) { eval("1e308 * 10") }
    }

    @Test
    fun `identifiers are not a thing`() {
        // The whole grammar: numbers, brackets, six operators, pi and e. If any
        // of these ever start evaluating, something has grown a host reach.
        listOf(
            "System.exit(0)",
            "java.lang.Runtime",
            "foo(1)",
            "1 + bar",
            "\$(whoami)",
            "0x10",
        ).forEach { hostile ->
            assertThrows(hostile, IllegalArgumentException::class.java) { eval(hostile) }
        }
    }

    @Test
    fun `trailing rubbish is refused rather than ignored`() {
        // "2 + 3 rm -rf" must not quietly answer 5.
        assertThrows(IllegalArgumentException::class.java) { eval("2 + 3 rm -rf /") }
        assertThrows(IllegalArgumentException::class.java) { eval("5;") }
    }

    @Test
    fun `unbalanced brackets are refused`() {
        assertThrows(IllegalArgumentException::class.java) { eval("(1 + 2") }
        assertThrows(IllegalArgumentException::class.java) { eval("1 + 2)") }
    }

    @Test
    fun `an empty expression is refused`() {
        assertThrows(IllegalArgumentException::class.java) { eval("") }
        assertThrows(IllegalArgumentException::class.java) { eval("   ") }
    }

    @Test
    fun `a dangling operator is refused`() {
        assertThrows(IllegalArgumentException::class.java) { eval("2 +") }
        assertThrows(IllegalArgumentException::class.java) { eval("* 2") }
    }
}
