package ai.ondevice.data.log

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * The tail, the trim, and the line that will not parse.
 *
 * All three are invisible on a device: a trim that keeps the wrong end is a log
 * screen that shows old lines and drops the ones from the failure being looked
 * at, and neither the screen nor the file says anything is wrong.
 */
class LogFileTest {

    @get:Rule
    val folder = TemporaryFolder()

    private fun log(keep: Int) = LogFile(File(folder.root, "log.jsonl"), keep)

    @Test
    fun `an empty or missing file reads as nothing`() {
        assertEquals(emptyList<String>(), log(10).read())
    }

    @Test
    fun `it keeps the newest, not the oldest`() {
        val log = log(3)
        log.append(listOf("1", "2", "3", "4", "5"))
        // Under the trim threshold, so nothing has been rewritten yet — read()
        // still has to hand back the tail rather than the whole file.
        assertEquals(listOf("3", "4", "5"), log.read())
    }

    @Test
    fun `the file is trimmed once it is twice the cap`() {
        val log = log(3)
        val file = File(folder.root, "log.jsonl")
        repeat(7) { log.append(listOf("line $it")) }

        // Seven lines against a cap of three: the append that crossed six
        // rewrites, and what is left on disk is the newest three and nothing
        // else. The point of the check is the file, not `read()` — `read()`
        // would return the same tail either way, which is exactly how an
        // unbounded file goes unnoticed.
        assertEquals(3, file.readLines().count { it.isNotBlank() })
        assertEquals(listOf("line 4", "line 5", "line 6"), log.read())
    }

    @Test
    fun `a new instance over the same file sees what the last one wrote`() {
        log(10).append(listOf("before"))
        assertEquals(listOf("before"), log(10).read())
    }

    @Test
    fun `clear empties it and leaves it readable`() {
        val log = log(10)
        log.append(listOf("a", "b"))
        log.clear()
        assertFalse(File(folder.root, "log.jsonl").exists())
        assertEquals(emptyList<String>(), log.read())
        log.append(listOf("c"))
        assertEquals(listOf("c"), log.read())
    }

    @Test
    fun `a half-written line does not take the file with it`() {
        val file = File(folder.root, "log.jsonl")
        file.writeText("first\n\nsecond\n")
        // Blank lines are what a kill mid-append leaves behind; they are dropped
        // rather than returned as records the caller then fails to decode.
        assertEquals(listOf("first", "second"), log(10).read())
        assertTrue(file.exists())
    }
}
