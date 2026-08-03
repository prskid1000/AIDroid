package ai.ondevice.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * That the name a GGUF declares resolves to a family.
 *
 * The table is keyed by the versions stable-diffusion.cpp prints, and a
 * checkpoint's header carries the family instead — so the architecture whose
 * parts are the least interchangeable was the one nothing was known about, and
 * "nothing known" means "claim nothing", which is how a FLUX decoder came to be
 * armed against Wan.
 */
class FamilyPrefixTest {

    @Test
    fun `a declared family resolves to its version's entry`() {
        // What QuantStack's Wan 2.2 TI2V 5B GGUF actually says.
        assertEquals(
            DiffusionFamily.forName("wan2_2_ti2v"),
            DiffusionFamily.forName("wan"),
        )
    }

    @Test
    fun `resolving by family does not change what it reads`() {
        val wan = DiffusionFamily.forName("wan")
        assertEquals(setOf("t5xxl"), wan?.encoders)
        assertEquals(DiffusionFamily.T5Kind.UMT5, wan?.t5)
    }

    @Test
    fun `a direct match still wins`() {
        // "flux" is a needle in its own right, so it must not be re-read as a
        // prefix of "flux.2 klein" — the two build different conditioners.
        assertEquals(setOf("clip_l", "t5xxl"), DiffusionFamily.forName("flux")?.encoders)
        assertEquals(setOf("llm"), DiffusionFamily.forName("flux.2")?.encoders)
    }

    @Test
    fun `a short string cannot claim a family`() {
        // "sd" begins sd3, sdxl, sdxs and two more, and they disagree about
        // every encoder they read.
        assertNull(DiffusionFamily.forName("sd"))
    }

    @Test
    fun `a name nothing begins is still unknown`() {
        assertNull(DiffusionFamily.forName("zzz"))
        assertNull(DiffusionFamily.forName("some model published next year"))
    }
}
