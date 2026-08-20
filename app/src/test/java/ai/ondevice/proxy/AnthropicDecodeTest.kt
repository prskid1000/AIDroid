package ai.ondevice.proxy

import kotlinx.serialization.json.JsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Anthropic content blocks, decomposed.
 *
 * This is the half of the proxy that cannot be checked on a device: a wrong
 * decomposition produces a conversation the model reads in the wrong order, and
 * what comes back is a plausible answer to a question nobody asked. There is
 * nothing to see in a screenshot.
 */
class AnthropicDecodeTest {

    private val sink = RecordingSink()

    private fun decode(json: String) =
        AnthropicCodec.decode(
            ProxyJson.parseToJsonElement(json) as JsonObject,
            midSystemPolicy = ProxySpecs.MID_DEMOTE,
            stripBookkeeping = false,
            media = sink,
        )

    @Test
    fun `string content stays one message`() {
        val request = decode(
            """{"model":"m","messages":[{"role":"user","content":"hello"}]}""",
        )
        assertEquals(1, request.messages.size)
        assertEquals("user", request.messages[0].role)
        assertEquals("hello", request.messages[0].content)
    }

    /**
     * The split that every chat template needs and this protocol does not make.
     *
     * Anthropic packs the prose and the tool call into one assistant message.
     * Left packed, the template renders a tool call as text and the model reads
     * its own previous call as something it said out loud.
     */
    @Test
    fun `assistant prose and tool call arrive together and stay together`() {
        val request = decode(
            """
            {"model":"m","messages":[
              {"role":"assistant","content":[
                {"type":"text","text":"Let me look."},
                {"type":"tool_use","id":"t1","name":"Read","input":{"path":"a.txt"}}
              ]}
            ]}
            """.trimIndent(),
        )
        assertEquals(1, request.messages.size)
        val message = request.messages[0]
        assertEquals("Let me look.", message.content)
        assertEquals(1, message.toolCalls.size)
        assertEquals("Read", message.toolCalls[0].name)
        assertEquals("t1", message.toolCalls[0].id)
        assertTrue(message.toolCalls[0].argumentsJson.contains("a.txt"))
    }

    /**
     * A tool result and the next question share a user turn, and must not share
     * a message.
     */
    @Test
    fun `tool result becomes its own message before the prose`() {
        val request = decode(
            """
            {"model":"m","messages":[
              {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":"file contents"},
                {"type":"text","text":"what does it say?"}
              ]}
            ]}
            """.trimIndent(),
        )
        assertEquals(2, request.messages.size)
        assertEquals("tool", request.messages[0].role)
        assertEquals("t1", request.messages[0].toolCallId)
        assertEquals("file contents", request.messages[0].content)
        assertEquals("user", request.messages[1].role)
        assertEquals("what does it say?", request.messages[1].content)
    }

    /**
     * The case worth the whole file.
     *
     * A tool that returns a screenshot puts an image inside a `tool_result`. A
     * `tool` message cannot carry one, so the obvious implementation drops it —
     * and the model is then asked about a picture it was never shown, which
     * reads as the model being unable to see rather than as the picture never
     * arriving.
     */
    @Test
    fun `an image inside a tool result is lifted into a following user turn`() {
        val request = decode(
            """
            {"model":"m","messages":[
              {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":[
                  {"type":"text","text":"here it is"},
                  {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}
                ]}
              ]}
            ]}
            """.trimIndent(),
        )
        assertEquals(2, request.messages.size)
        assertEquals("tool", request.messages[0].role)
        assertEquals("here it is", request.messages[0].content)
        assertEquals("user", request.messages[1].role)
        assertEquals(listOf("/tmp/AAAA.png"), request.messages[1].imagePaths)
    }

    /**
     * A URL source is refused rather than fetched.
     *
     * Fetching it would be an outbound request this app did not choose to make,
     * aimed by whoever holds the socket.
     */
    @Test
    fun `a url image is not fetched`() {
        val request = decode(
            """
            {"model":"m","messages":[
              {"role":"user","content":[
                {"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}
              ]}
            ]}
            """.trimIndent(),
        )
        assertTrue(request.messages.all { it.imagePaths.isEmpty() })
    }

    /** Prior-turn thinking is not replayed: it is not the answer and costs full price. */
    @Test
    fun `thinking blocks are dropped from history`() {
        val request = decode(
            """
            {"model":"m","messages":[
              {"role":"assistant","content":[
                {"type":"thinking","thinking":"long deliberation"},
                {"type":"text","text":"the answer"}
              ]}
            ]}
            """.trimIndent(),
        )
        assertEquals("the answer", request.messages[0].content)
    }

    @Test
    fun `system is read from a string or from blocks`() {
        assertEquals(
            "be brief",
            decode("""{"model":"m","system":"be brief","messages":[]}""").system,
        )
        assertEquals(
            "be brief\nand kind",
            decode(
                """{"model":"m","system":[{"type":"text","text":"be brief"},
                   {"type":"text","text":"and kind"}],"messages":[]}""",
            ).system,
        )
    }

    @Test
    fun `sampling keys are translated and nothing else is invented`() {
        val request = decode(
            """{"model":"m","max_tokens":256,"temperature":0.4,"top_k":20,
               "stop_sequences":["END"],"messages":[]}""",
        )
        assertEquals(256, request.params.int("n_predict"))
        assertEquals(0.4f, request.params.float("temp")!!, 0.0001f)
        assertEquals(20, request.params.int("top_k"))
        assertEquals(listOf("END"), request.params.stringList("stop"))
        // Nothing the client did not send: the model's own stored overrides
        // have to survive everything the request did not mention.
        assertNull(request.params.float("top_p"))
    }

    @Test
    fun `tool_choice is normalised`() {
        assertEquals(
            "Read",
            decode("""{"model":"m","messages":[],"tool_choice":{"type":"tool","name":"Read"}}""")
                .forcedTool,
        )
        assertEquals(
            AnthropicCodec.ANY_TOOL,
            decode("""{"model":"m","messages":[],"tool_choice":{"type":"any"}}""").forcedTool,
        )
        assertNull(decode("""{"model":"m","messages":[]}""").forcedTool)
    }

    /** A body with no model cannot be served and says so rather than guessing one. */
    @Test
    fun `a missing model is refused by name`() {
        val refusal = runCatching { decode("""{"messages":[]}""") }.exceptionOrNull()
        assertTrue(refusal is ProxyRefusal)
        assertEquals(400, (refusal as ProxyRefusal).status)
    }

    /** Deterministic, so the assertions above can name a path. */
    private class RecordingSink : MediaSink {
        override fun writeBase64(data: String, mediaType: String): String =
            "/tmp/$data." + if (mediaType.endsWith("jpeg")) "jpg" else "png"

        override fun writeBytes(bytes: ByteArray, extension: String): String =
            "/tmp/bytes.$extension"
    }
}
