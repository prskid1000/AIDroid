package ai.ondevice.core.workflow

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertFalse
import org.junit.Test

/**
 * How a template names an earlier step.
 *
 * The two halves of the app spell a reference differently — a slot binding is
 * `id:output`, a template writes `id.output` — because the expression grammar
 * reads a bare word and a word cannot contain a colon. Nothing bridged the two,
 * so the syntax in the editor's own help text resolved to nothing and was left
 * on the page as literal braces. The Only-if step was the casualty: its
 * condition was always the unrendered text, which is never a number and never
 * compares true, so the branch was never taken.
 *
 * These tests fix the bridging rule in place. The lookup they exercise is the
 * one the runner installs, written here the same way.
 */
class TemplateReferenceTest {

    /** The map a run builds: an output under both spellings, as `put` writes it. */
    private val outputs = mapOf(
        "a1b2" to "the whole answer",
        "a1b2:text" to "the whole answer",
        "c3d4:pieces" to "one",
        "e5f6" to "42",
        "e5f6:text" to "42",
    )

    private val lookup = WorkflowTemplate.Lookup { reference ->
        outputs[reference]
            ?: outputs[reference.replace('.', ':')]
            ?: outputs[reference.substringBefore('.')]
    }

    @Test
    fun `a dotted reference finds the slot-spelled output`() {
        assertEquals("the whole answer", WorkflowTemplate.render("{{ a1b2.text }}", lookup))
    }

    @Test
    fun `a bare step reference still works`() {
        assertEquals("the whole answer", WorkflowTemplate.render("{{ a1b2 }}", lookup))
    }

    @Test
    fun `an output with no bare form is still reachable`() {
        assertEquals("one", WorkflowTemplate.render("{{ c3d4.pieces }}", lookup))
    }

    @Test
    fun `an unknown output falls back to the step itself`() {
        assertEquals("the whole answer", WorkflowTemplate.render("{{ a1b2.nosuch }}", lookup))
    }

    @Test
    fun `references compose with the functions`() {
        assertEquals("THE WHOLE", WorkflowTemplate.render("{{ upper(slice(a1b2.text, 0, 9)) }}", lookup))
    }

    /** The Only-if step, which is the whole reason this matters. */
    @Test
    fun `a condition over a reference decides`() {
        assertTrue(WorkflowTemplate.truthy(WorkflowTemplate.render("{{ e5f6.text > 40 }}", lookup)))
        assertFalse(WorkflowTemplate.truthy(WorkflowTemplate.render("{{ e5f6.text > 50 }}", lookup)))
        assertTrue(WorkflowTemplate.truthy(WorkflowTemplate.render("{{ length(a1b2.text) > 8 }}", lookup)))
    }

    /**
     * What the bug looked like, and why it was so quiet.
     *
     * A reference that resolves to nothing comes back as the word itself
     * rather than as empty or as braces — a bare word is a valid expression,
     * so nothing throws and the `{{…}}` fallback never fires. The step then
     * carries on with the literal text "nothing.text", which is non-empty and
     * therefore reads as true in any condition. A branch guarded that way took
     * itself every time, and looked from the outside like a branch that worked.
     */
    @Test
    fun `an unresolved reference becomes its own name, and reads as true`() {
        val empty = WorkflowTemplate.Lookup { null }
        val rendered = WorkflowTemplate.render("{{ nothing.text }}", empty)
        assertEquals("nothing.text", rendered)
        assertTrue(WorkflowTemplate.truthy(rendered))
    }

    /** A malformed expression does keep its braces, which is the visible case. */
    @Test
    fun `a broken expression keeps its braces`() {
        val empty = WorkflowTemplate.Lookup { null }
        assertEquals("{{nosuchfn(1)}}", WorkflowTemplate.render("{{ nosuchfn(1) }}", empty))
    }
}
