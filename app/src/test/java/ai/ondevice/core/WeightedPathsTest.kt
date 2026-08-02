package ai.ondevice.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** That a role holding several files reads back as several files. */
class WeightedPathsTest {

    private fun params(json: String) = SparseParams.parse(json)

    @Test
    fun `a stack keeps its order and its strengths`() {
        val stored = params(
            """{"loras":[{"path":"/m/style.safetensors","weight":0.8},
                         {"path":"/m/face.safetensors","weight":0.35}]}""",
        )
        val read = WeightedPaths.parse(stored["loras"])
        assertEquals(2, read.size)
        assertEquals("/m/style.safetensors", read[0].path)
        assertEquals(0.8f, read[0].weight, 0.0001f)
        assertEquals(0.35f, read[1].weight, 0.0001f)
    }

    @Test
    fun `the bare string every single-file role stores reads as one entry`() {
        val read = WeightedPaths.parse(params("""{"vae":"/m/sdxl.vae.safetensors"}""")["vae"])
        assertEquals(listOf(WeightedPath("/m/sdxl.vae.safetensors", 1.0f)), read)
    }

    @Test
    fun `a cleared slot is not an attachment`() {
        assertTrue(WeightedPaths.parse(params("""{"vae":""}""")["vae"]).isEmpty())
        assertTrue(WeightedPaths.parse(null).isEmpty())
        assertTrue(WeightedPaths.parse(params("""{"loras":[]}""")["loras"]).isEmpty())
    }

    @Test
    fun `an entry with no path is dropped rather than passed to the runtime`() {
        val read = WeightedPaths.parse(
            params("""{"loras":[{"path":"","weight":1.0},{"path":"/m/a.safetensors"}]}""")["loras"],
        )
        assertEquals(listOf(WeightedPath("/m/a.safetensors", 1.0f)), read)
    }

    @Test
    fun `what is written is what comes back`() {
        val entries = listOf(WeightedPath("/m/a.safetensors", 0.6f), WeightedPath("/m/b.gguf", 1.4f))
        val json = WeightedPaths.toJson(entries)!!
        assertEquals(entries, WeightedPaths.parse(json))
    }

    @Test
    fun `an empty stack is stored as nothing, not as an empty array`() {
        assertNull(WeightedPaths.toJson(emptyList()))
        assertNull(WeightedPaths.toJson(listOf(WeightedPath("", 1f))))
    }
}
