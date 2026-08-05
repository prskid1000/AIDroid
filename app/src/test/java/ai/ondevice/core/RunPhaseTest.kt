package ai.ondevice.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The precedence, which is the whole reason this type exists.
 *
 * Every screen kept these flags already; what none of them agreed on was which
 * one wins when two are true at once. An unload sets stopping while generating
 * is still true — the native call has not returned yet — and both Image and
 * Video read that as "still running", so they went on showing live progress
 * over an enabled Cancel for a run that was being torn down.
 */
class RunPhaseTest {

    @Test
    fun `stopping wins over running`() {
        assertEquals(
            "a run being torn down is not a run in progress; reading it as one is " +
                "what left Cancel enabled while the memory was being freed",
            RunPhase.Stopping,
            runPhaseOf(stopping = true, running = true),
        )
    }

    @Test
    fun `stopping wins over loading too`() {
        assertEquals(RunPhase.Stopping, runPhaseOf(stopping = true, loading = true))
    }

    @Test
    fun `loading wins over running`() {
        assertEquals(
            "a generate queued behind a load has not started",
            RunPhase.Loading,
            runPhaseOf(loading = true, running = true),
        )
    }

    @Test
    fun `no flags is idle`() {
        assertEquals(RunPhase.Idle, runPhaseOf())
    }

    @Test
    fun `progress is hidden while stopping`() {
        assertFalse(
            "the numbers still arriving belong to a run that has been told to stop, " +
                "and showing them says it is still going",
            RunPhase.Stopping.showsProgress,
        )
        assertTrue(RunPhase.Running.showsProgress)
        assertTrue(RunPhase.Loading.showsProgress)
        assertFalse(RunPhase.Idle.showsProgress)
    }

    @Test
    fun `stopping offers a control that cannot be pressed`() {
        val control = RunPhase.Stopping.control()
        assertEquals("Stopping…", control?.label)
        assertEquals(
            "a second press is what produced five concurrent unloads of one handle",
            false,
            control?.enabled,
        )
    }

    @Test
    fun `loading and running both offer Cancel`() {
        listOf(RunPhase.Loading, RunPhase.Running).forEach { phase ->
            assertEquals("Cancel", phase.control()?.label)
            assertEquals(true, phase.control()?.enabled)
        }
    }

    /**
     * Idle has no label of its own, so a screen cannot render "Cancel" over a
     * tab with no model installed. What to offer when nothing is running is the
     * screen's question, not this type's.
     */
    @Test
    fun `idle has no control`() {
        assertNull(RunPhase.Idle.control())
    }

    @Test
    fun `busy counts loading`() {
        assertTrue(RunPhase.Loading.busy)
        assertTrue(RunPhase.Running.busy)
        assertTrue(RunPhase.Stopping.busy)
        assertFalse(RunPhase.Idle.busy)
    }
}
