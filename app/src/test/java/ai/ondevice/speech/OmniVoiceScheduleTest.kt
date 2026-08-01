package ai.ondevice.speech

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** The commit schedule, pinned to the run that was listened to and approved. */
class OmniVoiceScheduleTest {

    private val engine = OmniVoiceEngine()

    @Test
    fun `matches the reference implementation for a short clip`() {
        // 50 frames, 32 steps, t_shift 0.1 — port_algorithm.wav.
        val schedule = engine.buildSchedule(frames = 50, steps = 32, shift = 0.1f)

        assertEquals(32, schedule.size)
        assertEquals(
            listOf(2, 2, 2, 2, 2, 2, 2, 3),
            schedule.take(8),
        )
        assertEquals(83, schedule.last())
        assertEquals(50 * OmniVoiceEngine.CODEBOOKS, schedule.sum())
    }

    @Test
    fun `matches the reference implementation for a longer clip`() {
        // 130 frames — port_longer.wav.
        val schedule = engine.buildSchedule(frames = 130, steps = 32, shift = 0.1f)

        assertEquals(
            listOf(4, 4, 4, 4, 5, 5, 5, 6),
            schedule.take(8),
        )
        assertEquals(238, schedule.last())
        assertEquals(130 * OmniVoiceEngine.CODEBOOKS, schedule.sum())
    }

    /** The property the vocoder depends on. */
    @Test
    fun `always fills the grid exactly`() {
        for (frames in listOf(1, 2, 7, 50, 130, 373, 750)) {
            for (steps in listOf(1, 4, 5, 32, 64)) {
                for (shift in listOf(0.01f, 0.1f, 0.5f, 1f)) {
                    val schedule = engine.buildSchedule(frames, steps, shift)
                    assertEquals(
                        "frames=$frames steps=$steps shift=$shift",
                        frames * OmniVoiceEngine.CODEBOOKS,
                        schedule.sum(),
                    )
                    assertTrue(
                        "frames=$frames steps=$steps shift=$shift committed a negative count",
                        schedule.all { it >= 0 },
                    )
                }
            }
        }
    }

    /** A shift of 1.0 is upstream's "no shift", and it should spread the commits evenly rather than being a special case in the arithmetic. */
    @Test
    fun `an unshifted schedule is even`() {
        val schedule = engine.buildSchedule(frames = 64, steps = 32, shift = 1f)
        // 512 slots over 32 steps, so 16 each barring rounding at the tail.
        assertTrue(schedule.toString(), schedule.dropLast(1).all { it == 16 })
        assertEquals(64 * OmniVoiceEngine.CODEBOOKS, schedule.sum())
    }

    @Test
    fun `clamps a shift the user could type`() {
        for (shift in listOf(-1f, 0f, 0.0001f, 2f, Float.NaN)) {
            val schedule = engine.buildSchedule(frames = 50, steps = 32, shift = shift)
            assertEquals(
                "shift=$shift",
                50 * OmniVoiceEngine.CODEBOOKS,
                schedule.sum(),
            )
        }
    }
}
