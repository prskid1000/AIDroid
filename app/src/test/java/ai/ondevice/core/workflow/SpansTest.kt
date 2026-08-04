package ai.ondevice.core.workflow

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Where a loop's body begins and ends.
 *
 * Every one of these was a real defect rather than a hypothetical. The span
 * logic decides which steps run and how many times, and when it is wrong the
 * run still finishes and still reports success — so there is nothing to notice
 * except a result that is quietly short.
 */
class SpansTest {

    private val repeat = NodeKind.RepeatStart.type
    private val endRepeat = NodeKind.RepeatEnd.type
    private val forEach = NodeKind.ForEachStart.type
    private val endForEach = NodeKind.ForEachEnd.type
    private val batch = NodeKind.Batch.type
    private val branch = NodeKind.Branch.type
    private val step = NodeKind.Script.type

    @Test
    fun `closes at the matching end`() {
        val types = listOf(repeat, step, step, endRepeat, step)
        assertEquals(3, Spans.end(types, 0))
    }

    /**
     * The bug that made a Batch quietly drop the last step of a graph.
     *
     * An unclosed bracket used to report `size - 1`, so the body ran up to but
     * not including the final step and the run resumed past it. A Batch with a
     * Keep at the bottom repeated everything and kept nothing.
     */
    @Test
    fun `an unclosed bracket reaches one past the end`() {
        val types = listOf(batch, step, step)
        assertEquals(3, Spans.end(types, 0))
    }

    @Test
    fun `a nested bracket does not steal the outer end`() {
        //  0 forEach
        //  1   repeat
        //  2     step
        //  3   endRepeat
        //  4 endForEach
        val types = listOf(forEach, repeat, step, endRepeat, endForEach)
        assertEquals(4, Spans.end(types, 0))
        assertEquals(3, Spans.end(types, 1))
    }

    @Test
    fun `two brackets in a row close independently`() {
        val types = listOf(repeat, step, endRepeat, forEach, step, endForEach)
        assertEquals(2, Spans.end(types, 0))
        assertEquals(5, Spans.end(types, 3))
    }

    /**
     * A body cannot reach past its parent for a closer.
     *
     * This is what makes the recursion safe: an inner bracket that the author
     * forgot to close stops at the end of the body it is in, rather than
     * swallowing the outer loop's closer and unbalancing everything after.
     */
    @Test
    fun `the limit bounds the search`() {
        val types = listOf(forEach, repeat, step, endForEach)
        assertEquals(3, Spans.end(types, 1, limit = 3))
    }

    @Test
    fun `a condition is a bracket too`() {
        val types = listOf(branch, step, endRepeat, step)
        assertEquals(2, Spans.end(types, 0))
    }

    @Test
    fun `an empty body is allowed`() {
        val types = listOf(repeat, endRepeat)
        assertEquals(1, Spans.end(types, 0))
    }
}
