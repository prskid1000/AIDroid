package ai.ondevice.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * That a render never lands on a path an earlier render is recorded at.
 *
 * Images were named after their seed alone. A seed is the one input people
 * reuse deliberately — you fix it precisely so two runs differ in one setting
 * and nothing else — so the comparison overwrote its own first half. Losing the
 * file was the smaller problem: the database keeps a row per render, so the
 * older row went on pointing at a path that now held the newer picture, and a
 * gallery meant to show two results showed the second one twice.
 */
class GalleryNamingTest {

    @Test
    fun `the same seed twice is two files`() {
        val dir = createTempDir()
        try {
            val first = ModelStorage.uniqueFile(dir, "812934177", "png")
            first.writeText("first")
            val second = ModelStorage.uniqueFile(dir, "812934177", "png")

            assertEquals("the seed still names the first one", "812934177.png", first.name)
            assertTrue("the second render would overwrite the first", first != second)
            assertTrue("the first render is gone", first.exists())
            assertEquals("812934177-2.png", second.name)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun `a suffixed name counts on its own`() {
        val dir = createTempDir()
        try {
            // An upscale of the same picture is a different artifact from the
            // picture, so it must not consume the picture's numbering.
            ModelStorage.uniqueFile(dir, "7", "png").writeText("render")
            val upscaled = ModelStorage.uniqueFile(dir, "7-x2", "png")
            assertEquals("7-x2.png", upscaled.name)

            upscaled.writeText("upscaled")
            assertEquals("7-x2-2.png", ModelStorage.uniqueFile(dir, "7-x2", "png").name)
            assertEquals("7-2.png", ModelStorage.uniqueFile(dir, "7", "png").name)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun `it keeps counting past the second collision`() {
        val dir = createTempDir()
        try {
            repeat(4) { ModelStorage.uniqueFile(dir, "5", "png").writeText("x") }
            assertEquals("5-5.png", ModelStorage.uniqueFile(dir, "5", "png").name)
        } finally {
            dir.deleteRecursively()
        }
    }

    private fun createTempDir(): File =
        File.createTempFile("gallery", "").let {
            it.delete()
            it.mkdirs()
            it
        }
}
