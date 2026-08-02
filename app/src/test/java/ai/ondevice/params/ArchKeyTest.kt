package ai.ondevice.params

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

/**
 * That a parameter gated on an architecture survives being told the
 * architecture by the runtime rather than by the file's own metadata.
 */
class ArchKeyTest {

    private fun key(name: String) = ParamRepository.archKey(name)

    @Test
    fun `the runtime's label and the manifest's gate are the same architecture`() {
        assertEquals(key("sdxl"), key("SDXL"))
        assertEquals(key("flux2_klein"), key("Flux.2 klein"))
        assertEquals(key("sd3"), key("SD3.x"))
        assertEquals(key("sd1"), key("SD1.x"))
        assertEquals(key("flux"), key("Flux"))
    }

    @Test
    fun `two versions of one family are still two architectures`() {
        // FLUX.1 reads its prompt with CLIP-L and T5; FLUX.2 with a language
        // model. Collapsing them would offer each the other's encoders.
        assertNotEquals(key("flux"), key("flux2"))
        assertNotEquals(key("sd1"), key("sd2"))
        assertNotEquals(key("sdxl"), key("sdxl_refiner"))
        assertNotEquals(key("sd3"), key("sdxl"))
    }

    @Test
    fun `the trailing x is only dropped where it stands for a point release`() {
        assertEquals("sdxl", key("SDXL"))
        assertEquals("sdxs", key("sdxs"))
        assertEquals("flux2klein", key("flux2_klein"))
    }
}
