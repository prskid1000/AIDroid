package ai.ondevice.ui

import ai.ondevice.core.RunPhase
import ai.ondevice.engine.DiffusionPhase
import ai.ondevice.ui.components.runStatusLine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * That a run being stopped says so, whichever screen is asking.
 *
 * Two phases describe a run and they answer different questions: RunPhase is
 * what the app is doing, DiffusionPhase is what the runtime is doing inside it.
 * A stopping run is still *preparing* in the runtime's sense — the encode it
 * was told to abandon has not returned — and the clip screen printed only the
 * runtime's word, so pressing Cancel left "preparing" on screen as though
 * nothing had happened. The still screen had the answer, inline in one
 * composable, which is precisely why the other screen did not.
 */
class RunStatusTest {

    private fun line(
        run: RunPhase,
        stage: DiffusionPhase = DiffusionPhase.PREPARING,
        step: Int = 0,
    ) = runStatusLine(run, stage, step, idle = "No clip yet", sampling = "sampling")

    @Test
    fun `stopping outranks the runtime's own phase`() {
        DiffusionPhase.entries.forEach { stage ->
            assertTrue(
                "stopping during $stage must say so; printing the runtime's phase " +
                    "is what made a pressed Cancel look ignored",
                line(RunPhase.Stopping, stage, step = 3).startsWith("stopping"),
            )
        }
    }

    /**
     * The one case where stopping is not immediate, and the line says why
     * rather than leaving a silent wait that reads as a hang.
     */
    @Test
    fun `stopping during the prompt encode explains the wait`() {
        assertEquals(
            "stopping · the prompt encode can't be interrupted, so it finishes first",
            line(RunPhase.Stopping, DiffusionPhase.PREPARING),
        )
    }

    @Test
    fun `idle and loading are not the runtime's business`() {
        assertEquals("No clip yet", line(RunPhase.Idle, DiffusionPhase.SAMPLING, step = 9))
        assertEquals("loading model…", line(RunPhase.Loading, DiffusionPhase.SAMPLING))
    }

    @Test
    fun `a running clip reports the runtime's phase`() {
        assertTrue(line(RunPhase.Running, DiffusionPhase.PREPARING).startsWith("preparing"))
        assertTrue(line(RunPhase.Running, DiffusionPhase.DECODING).startsWith("decoding"))
        assertEquals("warming up…", line(RunPhase.Running, DiffusionPhase.SAMPLING, step = 0))
        assertEquals("sampling", line(RunPhase.Running, DiffusionPhase.SAMPLING, step = 2))
    }

    /**
     * The sampling sentence is the only genuinely screen-specific one — a still
     * is waiting on its first preview, a clip is not previewed at all — so it is
     * a parameter rather than a branch on the caller.
     */
    @Test
    fun `each screen keeps its own word for sampling`() {
        assertEquals(
            "clip words",
            runStatusLine(
                RunPhase.Running, DiffusionPhase.SAMPLING, step = 4,
                idle = "No clip yet", sampling = "clip words",
            ),
        )
        assertEquals(
            "still words",
            runStatusLine(
                RunPhase.Running, DiffusionPhase.SAMPLING, step = 4,
                idle = "No preview yet", sampling = "still words",
            ),
        )
    }
}
