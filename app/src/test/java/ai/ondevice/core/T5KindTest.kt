package ai.ondevice.core

import ai.ondevice.core.DiffusionFamily.T5Kind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * That the two models called "T5-XXL" stay two models.
 *
 * They share a slot, a role, a runtime argument and a declared architecture.
 * The only thing that separates them is size, and the only reason it works is
 * that the gap is a whole vocabulary wide.
 */
class T5KindTest {

    @Test
    fun `each encoder is recognised from its own parameter count`() {
        // The counts Hugging Face reports for city96's two conversions.
        assertEquals(T5Kind.T5_V1_1, T5Kind.of(4_762_310_656L))
        assertEquals(T5Kind.UMT5, T5Kind.of(5_680_910_336L))
    }

    @Test
    fun `neither is mistaken for the other`() {
        // 0.92B apart, which is the 256k vocabulary against the 32k one.
        assertEquals(T5Kind.T5_V1_1, T5Kind.of(4_800_000_000L))
        assertEquals(T5Kind.UMT5, T5Kind.of(5_650_000_000L))
    }

    @Test
    fun `an encoder that is neither is left unclaimed`() {
        // Forcing a stranger into the nearer window is how a wrong answer gets
        // stated confidently. Null means the caller says nothing.
        assertNull(T5Kind.of(5_200_000_000L))
        assertNull(T5Kind.of(300_000_000L))
        assertNull(T5Kind.of(null))
        assertNull(T5Kind.of(0L))
    }

    @Test
    fun `the families disagree about which T5 they read`() {
        assertEquals(T5Kind.UMT5, DiffusionFamily.forName("wan2_2_ti2v")?.t5)
        assertEquals(T5Kind.T5_V1_1, DiffusionFamily.forName("flux")?.t5)
        assertEquals(T5Kind.T5_V1_1, DiffusionFamily.forName("sd3")?.t5)
        assertEquals(T5Kind.T5_V1_1, DiffusionFamily.forName("chroma")?.t5)
    }

    @Test
    fun `a family that reads no T5 claims none`() {
        // SDXL conditions on two CLIPs, Z-Image on a language model. Claiming a
        // T5 for either would invent a mismatch to warn about.
        assertNull(DiffusionFamily.forName("sdxl")?.t5)
        assertNull(DiffusionFamily.forName("z-image")?.t5)
        assertNull(DiffusionFamily.forName("sd1")?.t5)
    }
}
