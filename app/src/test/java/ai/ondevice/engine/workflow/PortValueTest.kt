package ai.ondevice.engine.workflow

import ai.ondevice.core.workflow.PortType
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * What a list's items turn back into when a loop hands them on one at a time.
 *
 * A list carries its items as strings whether they are paragraphs or file
 * paths, so without the element type a for-each cannot tell the two apart. It
 * used to guess "file", which meant a loop over split text handed each piece to
 * the next step as a path — and a model reading the prompt slot got an empty
 * string, then failed with a message about the model rather than about the
 * loop.
 */
class PortValueTest {

    @Test
    fun `an item of a text list is text`() {
        val list = PortValue.list(listOf("first para", "second para"), PortType.TEXT)
        val item = list.item("first para")
        assertEquals(PortType.TEXT, item.type)
        assertEquals("first para", item.text)
        assertEquals("first para", item.asText)
    }

    @Test
    fun `an item of a file list is a path`() {
        val list = PortValue.list(listOf("/a/one.png", "/a/two.png"), PortType.IMAGE)
        val item = list.item("/a/one.png")
        assertEquals(PortType.IMAGE, item.type)
        assertEquals("/a/one.png", item.path)
        assertEquals("", item.text)
        assertEquals("/a/one.png", item.asText)
    }

    /** A list left unmarked is files, which is what every producer but split makes. */
    @Test
    fun `the default element is a file`() {
        assertEquals(PortType.FILE, PortValue.list(listOf("/a/one")).elementType)
    }

    /**
     * What a loop's collect step reads.
     *
     * It took the path only, so a pass that produced text contributed nothing
     * and the harvest came out empty — indistinguishable from a loop whose body
     * never ran.
     */
    @Test
    fun `asText reads whichever half carries the value`() {
        assertEquals("said", PortValue.text("said").asText)
        assertEquals("/a/one.wav", PortValue.file(PortType.AUDIO, "/a/one.wav").asText)
        assertEquals("", PortValue(PortType.TEXT).asText)
    }
}
