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
    fun `each encoder is recognised by its own vocabulary`() {
        // Read out of the two files' headers, not off a model page: every
        // other field they carry is identical.
        assertEquals(T5Kind.T5_V1_1, T5Kind.of(32_128))
        assertEquals(T5Kind.UMT5, T5Kind.of(256_384))
    }

    @Test
    fun `a tokenizer that is neither is left unclaimed`() {
        // Nearest-match is how a wrong answer gets stated confidently. A
        // vocabulary is exact or it is somebody else's.
        assertNull(T5Kind.of(32_000))
        assertNull(T5Kind.of(250_000))
        assertNull(T5Kind.of(null))
        assertNull(T5Kind.of(0))
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
