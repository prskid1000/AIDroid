package ai.ondevice.engine

import ai.ondevice.core.Verdict
import ai.ondevice.data.hf.CompatibilityGate
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The one rule that decides whether a declared architecture and a runtime's
 * architecture are the same thing, and what a miss is allowed to cost.
 *
 * Both halves matter. The rule catches the disagreement that is common — a
 * GGUF naming the family where the runtime names the version — and it misses
 * the disagreement that is arbitrary, where the two spellings differ in the
 * middle. What makes the misses survivable is the second half: a miss is a
 * caution, never a refusal.
 */
class ArchitectureMatchTest {

    private fun matches(declared: String, known: String) =
        RuntimeRegistry.namesMatch(declared, known)

    @Test
    fun `a family name matches the versions of that family`() {
        // What Wan 2.2 TI2V 5B's GGUF actually declares, against what
        // stable-diffusion.cpp actually lists. Exact comparison refused it.
        assertTrue(matches("wan", "wan2_2_ti2v"))
        assertTrue(matches("wan", "wan2"))
        assertTrue(matches("flux", "flux2_klein"))
    }

    @Test
    fun `case and surrounding space are not a difference`() {
        assertTrue(matches("  WAN  ", "wan2_2_ti2v"))
        assertTrue(matches("Whisper", "whisper"))
    }

    @Test
    fun `a short string cannot claim a family`() {
        // Two characters would let "sd" match every SD version there is, and
        // an empty architecture match everything.
        assertFalse(matches("sd", "sdxl"))
        assertFalse(matches("", "wan2"))
        assertFalse(matches("wan", ""))
    }

    @Test
    fun `the specific name does not match a shorter one by itself`() {
        // Direction matters: the declared name is the one allowed to be the
        // family. Reversing it would make every version match every other.
        assertFalse(matches("wan2_2_ti2v", "wan"))
    }

    @Test
    fun `names that differ in the middle are missed, and that is the point`() {
        // LTX ships GGUFs declaring `ltxv`; the runtime enumerates `ltxav`.
        // No rule short of a hand-written table catches this, and a table is
        // what this replaced. The next test is why the miss is affordable.
        assertFalse(matches("ltxv", "ltxav"))
    }

    @Test
    fun `an unrecognised architecture does not stop a download`() {
        assertTrue(Verdict.UNSUPPORTED_ARCH.runnable)
        // What does stop one is the device's own answer, not a name.
        assertFalse(Verdict.WONT_FIT.runnable)
        assertFalse(Verdict.NOT_RUNNABLE.runnable)
    }

    @Test
    fun `a name miss does not hide the arithmetic that is certain`() {
        // 14 GB of weights on a phone with 4 GB free. Both things are true —
        // the name is unrecognised and it does not fit — and only one of them
        // is worth saying, because only one of them is beyond doubt.
        val estimate = CompatibilityGate.estimate(
            weightsBytes = 14_000_000_000L,
            layers = null,
            contextTokens = 0,
            embeddingLengthKv = null,
            embeddingLength = null,
        )
        val verdict = CompatibilityGate.verdict(
            estimate = estimate,
            availableRamBytes = 4_000_000_000L,
            freeStorageBytes = 200_000_000_000L,
            storageReserveBytes = 0L,
            archSupported = false,
            hasRuntimeForFormat = true,
        )
        assertTrue(verdict == Verdict.WONT_FIT)
    }
}
