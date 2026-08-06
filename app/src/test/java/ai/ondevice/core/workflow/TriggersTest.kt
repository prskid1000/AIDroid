package ai.ondevice.core.workflow

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The share sheet, the shortcut ranking and the text-selection menu need a
 * device. Deciding *which workflow can take what was shared* does not, and that
 * is the part that is wrong in ways nobody notices — a workflow quietly missing
 * from a sheet, or one offered a value it cannot use.
 */
class TriggersTest {

    private fun input(id: String, port: PortType, from: String) = NodeRecord(
        id = id,
        type = NodeKind.Input.type,
        params = JsonObject(
            mapOf(
                "portType" to JsonPrimitive(port.name),
                Triggers.PARAM_FROM to JsonPrimitive(from),
            ),
        ),
    )

    private fun graphOf(vararg nodes: NodeRecord) = WorkflowGraph(nodes = nodes.toList())

    private val someText = TriggerValue(PortType.TEXT, text = "hello")
    private val somePicture = TriggerValue(PortType.IMAGE, path = "/tmp/a.png")
    private val someRecording = TriggerValue(PortType.AUDIO, path = "/tmp/a.wav")

    /** A text file is both: prose for a TEXT slot, a file for a FILE slot. */
    private val someTextFile =
        TriggerValue(PortType.FILE, text = "the contents", path = "/tmp/notes.md")

    // ── what a graph advertises ──────────────────────────────────────────

    @Test
    fun `an input typed here is not a share target`() {
        val graph = graphOf(input("1", PortType.TEXT, Triggers.FROM_TYPED))
        assertTrue(Triggers.sharedInputs(graph).isEmpty())
        assertTrue(Triggers.categoriesFor(graph).isEmpty())
    }

    @Test
    fun `a disabled input does not advertise anything`() {
        val graph = graphOf(
            input("1", PortType.IMAGE, Triggers.FROM_SHARED).copy(enabled = false),
        )
        assertTrue(Triggers.categoriesFor(graph).isEmpty())
    }

    @Test
    fun `a text input appears only in the text sheet`() {
        val graph = graphOf(input("1", PortType.TEXT, Triggers.FROM_SHARED))
        assertEquals(setOf(Triggers.CATEGORY_TEXT), Triggers.categoriesFor(graph))
    }

    /**
     * The lattice, applied at publish time. A graph wanting a file belongs in
     * every sheet, because a picture and a recording are both files — and that
     * rule lives in PortType.satisfies, not restated in the XML.
     */
    @Test
    fun `a file input appears everywhere`() {
        val graph = graphOf(input("1", PortType.FILE, Triggers.FROM_SHARED))
        assertEquals(
            setOf(
                Triggers.CATEGORY_TEXT,
                Triggers.CATEGORY_IMAGE,
                Triggers.CATEGORY_AUDIO,
                Triggers.CATEGORY_ANY,
            ),
            Triggers.categoriesFor(graph),
        )
    }

    @Test
    fun `two inputs advertise both sheets`() {
        val graph = graphOf(
            input("1", PortType.IMAGE, Triggers.FROM_SHARED),
            input("2", PortType.TEXT, Triggers.FROM_SHARED),
        )
        assertEquals(
            setOf(Triggers.CATEGORY_IMAGE, Triggers.CATEGORY_TEXT),
            Triggers.categoriesFor(graph),
        )
    }

    // ── what fills what ──────────────────────────────────────────────────

    @Test
    fun `a picture does not fill a text slot`() {
        assertFalse(somePicture.canFill(PortType.TEXT))
        assertTrue(somePicture.canFill(PortType.IMAGE))
        assertTrue(somePicture.canFill(PortType.FILE))
    }

    /**
     * The case that would otherwise hand a model the word "/storage/…".
     *
     * Shared text has no file behind it, so it must not satisfy a slot that
     * wants one — a step reaching for a path would find the prose instead.
     */
    @Test
    fun `shared text does not fill a file slot`() {
        assertTrue(someText.canFill(PortType.TEXT))
        assertFalse(someText.canFill(PortType.FILE))
    }

    @Test
    fun `a text file fills both a text slot and a file slot`() {
        assertTrue(someTextFile.canFill(PortType.TEXT))
        assertTrue(someTextFile.canFill(PortType.FILE))
    }

    // ── matching ─────────────────────────────────────────────────────────

    @Test
    fun `a picture matches a graph wanting a picture`() {
        val graph = graphOf(input("1", PortType.IMAGE, Triggers.FROM_SHARED))
        assertTrue(Triggers.matches(graph, TriggerPayload(listOf(somePicture))))
    }

    @Test
    fun `a recording does not match a graph wanting a picture`() {
        val graph = graphOf(input("1", PortType.IMAGE, Triggers.FROM_SHARED))
        assertFalse(Triggers.matches(graph, TriggerPayload(listOf(someRecording))))
    }

    @Test
    fun `a picture with a caption matches a graph wanting both`() {
        val graph = graphOf(
            input("1", PortType.IMAGE, Triggers.FROM_SHARED),
            input("2", PortType.TEXT, Triggers.FROM_SHARED),
        )
        assertTrue(Triggers.matches(graph, TriggerPayload(listOf(somePicture, someText))))
    }

    /**
     * Each value is spent once. Two pictures cannot fill a picture slot and a
     * text slot, and answering otherwise means a model is handed a path where
     * a prompt should be.
     */
    @Test
    fun `two pictures do not match a graph wanting a picture and text`() {
        val graph = graphOf(
            input("1", PortType.IMAGE, Triggers.FROM_SHARED),
            input("2", PortType.TEXT, Triggers.FROM_SHARED),
        )
        assertFalse(
            Triggers.matches(
                graph,
                TriggerPayload(listOf(somePicture, somePicture.copy(path = "/tmp/b.png"))),
            ),
        )
    }

    // ── filling ──────────────────────────────────────────────────────────

    @Test
    fun `filling writes text into the text input and a path into the picture one`() {
        val graph = graphOf(
            input("1", PortType.IMAGE, Triggers.FROM_SHARED),
            input("2", PortType.TEXT, Triggers.FROM_SHARED),
        )
        val filled = Triggers.fill(graph, TriggerPayload(listOf(somePicture, someText)))

        assertEquals("/tmp/a.png", filled.nodes[0].params.string("path"))
        assertEquals("hello", filled.nodes[1].params.string("text"))
    }

    @Test
    fun `filling leaves an input that is typed here alone`() {
        val graph = graphOf(
            input("1", PortType.TEXT, Triggers.FROM_TYPED).copy(
                params = JsonObject(
                    mapOf(
                        "portType" to JsonPrimitive("TEXT"),
                        Triggers.PARAM_FROM to JsonPrimitive(Triggers.FROM_TYPED),
                        "text" to JsonPrimitive("what was written in the editor"),
                    ),
                ),
            ),
        )
        val filled = Triggers.fill(graph, TriggerPayload(listOf(someText)))
        assertEquals("what was written in the editor", filled.nodes[0].params.string("text"))
    }

    // ── mime ─────────────────────────────────────────────────────────────

    @Test
    fun `a video is a file and not a clip`() {
        // A clip in this app is a directory of frames with a frame rate. Calling
        // an mp4 one would hand a frame-stepping step a container it cannot read.
        assertEquals(PortType.FILE, Triggers.portFor("video/mp4"))
    }

    @Test
    fun `mime maps to the port it is`() {
        assertEquals(PortType.IMAGE, Triggers.portFor("image/png"))
        assertEquals(PortType.AUDIO, Triggers.portFor("audio/wav"))
        assertEquals(PortType.FILE, Triggers.portFor("application/pdf"))
        assertEquals(PortType.FILE, Triggers.portFor("text/plain"))
    }

    @Test
    fun `subtitles and markdown are readable as text`() {
        assertTrue(Triggers.readableAsText("text/plain", "notes.txt"))
        assertTrue(Triggers.readableAsText("application/octet-stream", "clip.srt"))
        assertTrue(Triggers.readableAsText("application/json", "data.json"))
        assertFalse(Triggers.readableAsText("image/png", "shot.png"))
    }

    private fun JsonObject.string(key: String): String =
        (this[key] as? JsonPrimitive)?.content.orEmpty()
}
