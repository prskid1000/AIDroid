package ai.ondevice.proxy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What gets kept of a request, and what deliberately does not.
 *
 * The failure this prevents is not a wrong answer, it is a device filling up.
 * A conversation re-sends its whole history every turn, a single screenshot on
 * one of those turns is four megabytes of base64, and two hundred records of
 * that is the ring buffer holding half a gigabyte of text nobody can read.
 */
class RequestBodyTest {

    private fun display(body: String) = RequestLog.forDisplay(body)

    @Test
    fun `an ordinary body is kept as it is`() {
        val body = """{"model":"m","messages":[{"role":"user","content":"hello"}]}"""
        assertEquals(body, display(body))
    }

    /**
     * The one that matters. Four megabytes of base64 is not legible, is not
     * diagnostic, and is the difference between a log you can keep and one you
     * cannot.
     */
    @Test
    fun `base64 is replaced by a note of its size`() {
        val payload = "A".repeat(4096)
        val body = """{"source":{"type":"base64","data":"$payload"}}"""
        val shown = display(body)

        assertFalse(shown.contains(payload))
        assertTrue(shown.contains("kB of base64"))
        // The shape around it survives, so the record still reads as JSON.
        assertTrue(shown.startsWith("""{"source":{"type":"base64","data":"""))
    }

    @Test
    fun `a data uri is recognised as base64 too`() {
        val body = """{"url":"data:image/png;base64,${"Z".repeat(1024)}"}"""
        val shown = display(body)
        assertFalse(shown.contains("Z".repeat(64)))
        assertTrue(shown.contains("kB of base64"))
    }

    /**
     * Short runs are left alone: a token, a hash and an id are all base64-ish
     * and all worth reading. The threshold is what separates a value from a
     * payload.
     */
    @Test
    fun `short base64-looking values are left alone`() {
        val body = """{"id":"msg_01abcdefGHIJKLMNOP","hash":"abc123=="}"""
        assertEquals(body, display(body))
    }

    @Test
    fun `an over-long body is capped and says how much went`() {
        val body = "x".repeat(50_000)
        val shown = display(body)
        assertTrue(shown.length < body.length)
        assertTrue(shown.contains("more characters not kept"))
    }

    @Test
    fun `an empty body stays empty rather than becoming a note`() {
        assertEquals("", display(""))
    }

    /**
     * Several payloads in one body — a conversation with three screenshots in
     * it — are each replaced, not just the first.
     */
    @Test
    fun `every payload in a body is replaced`() {
        val payload = "B".repeat(700)
        val body = """[{"data":"$payload"},{"data":"$payload"},{"data":"$payload"}]"""
        val shown = display(body)
        assertFalse(shown.contains(payload))
        assertEquals(3, Regex("kB of base64").findAll(shown).count())
    }
}
