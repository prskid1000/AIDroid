package ai.ondevice.proxy

import ai.ondevice.engine.GenerationEvent
import ai.ondevice.engine.StopReason
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * `GenerationEvent` on the wire, in both protocols.
 *
 * The block index is the whole of the difficulty on the Anthropic side.
 * Anthropic numbers content blocks within a message, a block must be closed
 * before one of another kind opens, and a client that sees an index it never
 * saw opened drops the message silently. None of that is visible on a device —
 * the symptom is a client showing nothing at all, which looks like the server
 * being down.
 */
class StreamWriterTest {

    private fun events(frames: String): List<Pair<String, String>> =
        frames.split("\n\n")
            .filter { it.isNotBlank() }
            .mapNotNull { block ->
                val name = block.lineSequence().firstOrNull { it.startsWith("event: ") }
                    ?.removePrefix("event: ")
                val data = block.lineSequence().firstOrNull { it.startsWith("data: ") }
                    ?.removePrefix("data: ")
                if (name != null && data != null) name to data else null
            }

    @Test
    fun `anthropic opens with message_start exactly once`() {
        val writer = AnthropicCodec.Writer("claude-x")
        val first = writer.start()
        val second = writer.start()
        assertEquals(1, events(first).count { it.first == "message_start" })
        assertEquals("", second)
    }

    /**
     * Text after thinking closes the thinking block first.
     *
     * Left open, the two deltas land on the same index with different types and
     * the client's parser has no way to render either.
     */
    @Test
    fun `switching from thinking to text closes the block`() {
        val writer = AnthropicCodec.Writer("m")
        val frames = buildString {
            append(writer.start())
            append(writer.event(GenerationEvent.ThinkingDelta("hmm")))
            append(writer.event(GenerationEvent.Token("answer", 0)))
        }
        val names = events(frames).map { it.first }
        assertEquals(
            listOf(
                "message_start",
                "content_block_start", "content_block_delta",
                "content_block_stop",
                "content_block_start", "content_block_delta",
            ),
            names,
        )
    }

    /**
     * Status lines take the low indices and content starts after them.
     *
     * A status written at an index the content later reuses makes the client
     * append the answer into the status block, so the tool line and the answer
     * merge into one paragraph.
     */
    @Test
    fun `status blocks do not collide with content blocks`() {
        val writer = AnthropicCodec.Writer("m")
        val frames = buildString {
            append(writer.start())
            append(writer.status("● ToolSearch"))
            append(writer.event(GenerationEvent.Token("hello", 0)))
        }
        val indices = events(frames)
            .filter { it.first.startsWith("content_block") }
            .map { (_, data) ->
                Regex("\"index\":(\\d+)").find(data)!!.groupValues[1].toInt()
            }
        // Status opens/deltas/stops at 0; the text block is a fresh index.
        assertEquals(listOf(0, 0, 0, 1, 1), indices)
    }

    @Test
    fun `a tool call is opened filled and closed`() {
        val writer = AnthropicCodec.Writer("m")
        val frames = buildString {
            append(writer.start())
            append(writer.event(GenerationEvent.ToolCall("Read", """{"path":"a"}""", "t1")))
        }
        val decoded = events(frames)
        assertTrue(decoded.any { it.first == "content_block_start" && it.second.contains("tool_use") })
        assertTrue(decoded.any { it.second.contains("input_json_delta") })
        assertTrue(decoded.any { it.first == "content_block_stop" })
    }

    @Test
    fun `finish closes an open block before stopping`() {
        val writer = AnthropicCodec.Writer("m")
        val frames = buildString {
            append(writer.start())
            append(writer.event(GenerationEvent.Token("hi", 0)))
            append(writer.finish(StopReason.EOS))
        }
        val names = events(frames).map { it.first }
        assertEquals("content_block_stop", names[names.size - 3])
        assertEquals("message_delta", names[names.size - 2])
        assertEquals("message_stop", names.last())
    }

    @Test
    fun `stop reasons map to the protocol's names`() {
        fun reasonFor(stop: StopReason): String {
            val writer = AnthropicCodec.Writer("m")
            writer.start()
            val data = events(writer.finish(stop)).first { it.first == "message_delta" }.second
            return Regex("\"stop_reason\":\"(\\w+)\"").find(data)!!.groupValues[1]
        }
        assertEquals("end_turn", reasonFor(StopReason.EOS))
        assertEquals("max_tokens", reasonFor(StopReason.MAX_TOKENS))
        assertEquals("max_tokens", reasonFor(StopReason.CONTEXT_FULL))
        assertEquals("stop_sequence", reasonFor(StopReason.STOP_SEQUENCE))
    }

    /**
     * Cache reads are counted separately from fresh input.
     *
     * llama.cpp reports cached prompt tokens and Anthropic has a field for
     * exactly that, so a client showing a cache-hit rate shows a true one —
     * where folding them together would report every cached turn as a full
     * re-read.
     */
    @Test
    fun `usage separates cache reads from fresh input`() {
        val writer = AnthropicCodec.Writer("m")
        writer.start()
        writer.noteUsage(input = 1000, cached = 800, output = 50, thinking = 10)
        val data = events(writer.finish(StopReason.EOS)).first { it.first == "message_delta" }.second
        assertTrue(data.contains("\"input_tokens\":200"))
        assertTrue(data.contains("\"cache_read_input_tokens\":800"))
        assertTrue(data.contains("\"output_tokens\":50"))
    }

    /** The engine's suggestion has to survive to the wire — SPEC 1.2. */
    @Test
    fun `an error carries the suggestion as well as the message`() {
        val writer = AnthropicCodec.Writer("m")
        val frames = writer.error("Not enough memory.", "Lower the context size.")
        assertTrue(frames.contains("Not enough memory."))
        assertTrue(frames.contains("Lower the context size."))
    }

    // ── OpenAI ──────────────────────────────────────────────────────────

    private fun chunks(frames: String): List<String> =
        frames.split("\n\n")
            .filter { it.startsWith("data: ") && !it.contains("[DONE]") }
            .map { it.removePrefix("data: ") }

    @Test
    fun `openai opens with the role and does not repeat it`() {
        val writer = OpenAiCodec.Writer("gpt-x")
        val first = chunks(writer.start())
        assertTrue(first.single().contains("\"role\":\"assistant\""))
        assertEquals("", writer.start())
    }

    @Test
    fun `thinking goes to reasoning_content`() {
        val writer = OpenAiCodec.Writer("m")
        val frame = chunks(writer.event(GenerationEvent.ThinkingDelta("hmm"))).single()
        assertTrue(frame.contains("\"reasoning_content\":\"hmm\""))
    }

    @Test
    fun `a tool call finishes with tool_calls not stop`() {
        val writer = OpenAiCodec.Writer("m")
        val frames = chunks(writer.finish(StopReason.EOS, hadToolCalls = true))
        assertTrue(frames.first().contains("\"finish_reason\":\"tool_calls\""))
    }

    /**
     * The usage chunk is sent unconditionally.
     *
     * A client that asked for it and never receives one reports zero tokens for
     * every request; a client that did not ask ignores an extra chunk with an
     * empty `choices` array, which is what the spec says this is.
     */
    @Test
    fun `a final usage chunk always follows the finish`() {
        val writer = OpenAiCodec.Writer("m")
        writer.noteUsage(input = 100, cached = 20, output = 7, thinking = 0)
        val frames = chunks(writer.finish(StopReason.EOS, hadToolCalls = false))
        assertEquals(2, frames.size)
        assertTrue(frames[1].contains("\"prompt_tokens\":100"))
        assertTrue(frames[1].contains("\"completion_tokens\":7"))
        assertTrue(frames[1].contains("\"cached_tokens\":20"))
        assertTrue(frames[1].contains("\"choices\":[]"))
    }
}
