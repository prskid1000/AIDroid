package ai.ondevice.proxy

import ai.ondevice.engine.EngineMessage
import ai.ondevice.engine.ToolSpec
import kotlinx.serialization.json.JsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The matching, ranking and rewriting rules — the parts with no UI at all.
 *
 * Every one of these decides something a person can only observe indirectly: a
 * tool that could not be found, a profile that did not apply, a system message
 * the template refused, a context spent on text the model cannot use.
 */
class ProxyRulesTest {

    // ── BM25 ────────────────────────────────────────────────────────────

    private val tools = listOf(
        ToolSpec("Read", "Read a file from disk", """{"type":"object","properties":{"path":{"type":"string","description":"absolute file path"}}}"""),
        ToolSpec("Write", "Write a file to disk", """{"type":"object","properties":{"path":{"type":"string"}}}"""),
        ToolSpec("WebSearch", "Search the web for pages", """{"type":"object","properties":{"query":{"type":"string"}}}"""),
        ToolSpec("Bash", "Run a shell command", """{"type":"object","properties":{"command":{"type":"string"}}}"""),
    )

    @Test
    fun `search ranks by what a tool is for`() {
        val index = ToolSearchIndex(tools)
        assertEquals("WebSearch", index.search("search the internet").first().name)
        assertEquals("Bash", index.search("run a command in a shell").first().name)
    }

    /**
     * `select:` is exact, and it matters that it is.
     *
     * Once the model has been told the names, asking for one by name is the
     * common case — and a ranked search over an exact name also returns three
     * things that merely sound similar, each of which costs context.
     */
    @Test
    fun `select loads by exact name only`() {
        val index = ToolSearchIndex(tools)
        assertEquals(listOf("Read"), index.search("select:Read").map { it.name })
        assertEquals(
            listOf("Read", "Write"),
            index.search("select:Read,Write").map { it.name }.sorted(),
        )
        assertTrue(index.search("select:NoSuchTool").isEmpty())
    }

    @Test
    fun `a query matching nothing returns nothing rather than everything`() {
        assertTrue(ToolSearchIndex(tools).search("photosynthesis").isEmpty())
        assertTrue(ToolSearchIndex(tools).search("   ").isEmpty())
    }

    @Test
    fun `parameter descriptions are searchable`() {
        // "absolute" appears only in Read's parameter description, which is
        // exactly the kind of word a model uses when it knows what it wants and
        // not what it is called.
        assertEquals("Read", ToolSearchIndex(tools).search("absolute path").first().name)
    }

    // ── mid-conversation system messages ────────────────────────────────

    private val conversation = listOf(
        EngineMessage("system", "first"),
        EngineMessage("user", "q1"),
        EngineMessage("assistant", "a1"),
        EngineMessage("system", "late"),
        EngineMessage("user", "q2"),
    )

    /**
     * Demote keeps the position and changes the role.
     *
     * The only mode that is both template-safe and cache-safe: the prompt
     * prefix stays append-only, so nothing before this turn has to be
     * re-prefilled.
     */
    @Test
    fun `demote re-roles in place`() {
        val out = applyMidSystemPolicy(conversation, ProxySpecs.MID_DEMOTE)
        assertEquals(5, out.size)
        assertEquals("user", out[3].role)
        assertEquals("late", out[3].content)
        assertEquals("system", out[0].role)
    }

    @Test
    fun `strip removes it entirely`() {
        val out = applyMidSystemPolicy(conversation, ProxySpecs.MID_STRIP)
        assertEquals(4, out.size)
        assertTrue(out.drop(1).none { it.role == "system" })
    }

    /**
     * merge_top is template-safe and pins the cache, which is why it is not the
     * default: a client emitting one system message per turn grows the front
     * block every turn, and the whole history is re-prefilled each time.
     */
    @Test
    fun `merge_top hoists into the leading block`() {
        val out = applyMidSystemPolicy(conversation, ProxySpecs.MID_MERGE_TOP)
        assertEquals(4, out.size)
        assertEquals("first\n\nlate", out[0].content)
    }

    @Test
    fun `keep leaves it alone and an unknown policy demotes`() {
        assertEquals("system", applyMidSystemPolicy(conversation, ProxySpecs.MID_KEEP)[3].role)
        assertEquals("user", applyMidSystemPolicy(conversation, "nonsense")[3].role)
    }

    @Test
    fun `a leading system block is never touched`() {
        val onlyLeading = listOf(EngineMessage("system", "s"), EngineMessage("user", "q"))
        assertEquals(onlyLeading, applyMidSystemPolicy(onlyLeading, ProxySpecs.MID_STRIP))
    }

    // ── client bookkeeping ──────────────────────────────────────────────

    /**
     * On a desktop this is tidiness. At 8k of context it is several percent of
     * the budget, re-sent on every turn of the conversation.
     */
    @Test
    fun `reminders and token budget lines are removed`() {
        val text = """
            <system-reminder>
            Do not mention this.
            </system-reminder>

            The real question.

            <total_tokens>15000 tokens left</total_tokens>
        """.trimIndent()
        assertEquals("The real question.", stripClientBookkeeping(text))
    }

    @Test
    fun `several reminders go and the prose between them stays`() {
        val text = "<system-reminder>a</system-reminder>one<system-reminder>b</system-reminder>two"
        assertEquals("onetwo", stripClientBookkeeping(text))
    }

    /**
     * A run of blank lines is collapsed rather than left behind.
     *
     * The prompt prefix has to stay byte-identical between turns or llama.cpp's
     * cache stops matching, and a removal that leaves four newlines where the
     * previous turn had two is a different prefix.
     */
    @Test
    fun `removal does not leave a growing run of blank lines`() {
        val text = "a\n\n<system-reminder>x</system-reminder>\n\n\nb"
        assertFalse(stripClientBookkeeping(text).contains("\n\n\n"))
    }

    @Test
    fun `text with nothing to strip is returned unchanged`() {
        assertEquals("plain", stripClientBookkeeping("plain"))
        assertEquals("", stripClientBookkeeping(""))
    }

    // ── profiles and aliases ────────────────────────────────────────────

    private val document = ProxyDocument(
        aliases = mapOf("claude-sonnet-4-6" to "qwen3-8b:Q4_K_M"),
        profiles = listOf(
            ProxyProfile(
                name = "coding",
                matchHeader = "User-Agent",
                matchContains = "claude-cli",
                overridesJson = """{"proxy.tool_search":false}""",
            ),
            ProxyProfile(
                name = "browser",
                matchHeader = "Referer",
                matchContains = "example.com",
            ),
            ProxyProfile(name = "known", token = "secret-token"),
        ),
    )

    @Test
    fun `the first matching profile wins, so the list is the priority`() {
        val matched = ProxyConfig.match(
            document,
            { if (it == "User-Agent") "claude-cli/2.0" else null },
            bearer = null,
        )
        assertEquals("coding", matched?.name)
    }

    @Test
    fun `matching is case-insensitive and substring`() {
        assertEquals(
            "browser",
            ProxyConfig.match(
                document,
                { if (it == "Referer") "https://EXAMPLE.com/app" else null },
                bearer = null,
            )?.name,
        )
    }

    /** A token is proof; a header is a hint. Proof is checked first. */
    @Test
    fun `a token matches before any header`() {
        assertEquals(
            "known",
            ProxyConfig.match(
                document,
                { if (it == "User-Agent") "claude-cli" else null },
                bearer = "secret-token",
            )?.name,
        )
    }

    @Test
    fun `nothing matches when no rule does`() {
        assertNull(ProxyConfig.match(document, { null }, bearer = null))
    }

    @Test
    fun `a profile override wins and everything else falls through`() {
        val profile = document.profiles.first()
        val config = ProxyConfig(document, profile)
        assertFalse(config.toolSearch)
        // Untouched by the profile, so the spec's own default applies.
        assertTrue(config.autoLoadTools)
        assertEquals(15, config.maxRoundTrips)
    }

    @Test
    fun `aliases rewrite and unknown names pass through`() {
        val config = ProxyConfig(document)
        assertEquals("qwen3-8b:Q4_K_M", config.resolveAlias("claude-sonnet-4-6"))
        assertEquals("something-else", config.resolveAlias("something-else"))
    }

    /**
     * A default that moves in a later release has to move for everyone who
     * never touched that row, which is only true while the stored map is sparse.
     */
    @Test
    fun `settings storage stays sparse`() {
        var doc = ProxyDocument.EMPTY.withSetting(ProxySpecs.PORT, 9000)
        assertEquals(1, doc.settings.keys.size)
        doc = doc.withSetting(ProxySpecs.PORT, null)
        assertTrue(doc.settings.isEmpty)
    }

    @Test
    fun `a document round-trips through storage`() {
        val encoded = document.withSetting(ProxySpecs.ENABLED, true).encode()
        val decoded = ProxyDocument.parse(encoded)
        assertEquals(document.aliases, decoded.aliases)
        assertEquals(3, decoded.profiles.size)
        assertTrue(ProxyConfig(decoded).enabled)
    }

    @Test
    fun `unreadable storage falls back to defaults rather than crashing`() {
        val config = ProxyConfig(ProxyDocument.parse("{ not json"))
        assertFalse(config.enabled)
        assertEquals(8080, config.port)
    }

    // ── reachability ────────────────────────────────────────────────────

    /**
     * `100.64.0.0/10` is the range Tailscale allocates from, and the second
     * octet running 64..127 is the whole of the test — 100.128.x is a different
     * network entirely and binding to it would be binding to the internet.
     */
    @Test
    fun `only the tailscale range counts as the tailnet`() {
        assertTrue(Reachability.isTailscale("100.64.0.1"))
        assertTrue(Reachability.isTailscale("100.127.255.254"))
        assertTrue(Reachability.isTailscale("100.94.12.7"))
        assertFalse(Reachability.isTailscale("100.63.0.1"))
        assertFalse(Reachability.isTailscale("100.128.0.1"))
        assertFalse(Reachability.isTailscale("192.168.1.5"))
        assertFalse(Reachability.isTailscale("10.0.0.1"))
        assertFalse(Reachability.isTailscale("not an address"))
    }

    @Test
    fun `loopback and all resolve without a tailnet`() {
        assertEquals(
            "127.0.0.1",
            (Reachability.resolveBindAddress(ProxySpecs.BIND_LOOPBACK)
                as Reachability.BindResult.Ok).address,
        )
        assertEquals(
            "0.0.0.0",
            (Reachability.resolveBindAddress(ProxySpecs.BIND_ALL)
                as Reachability.BindResult.Ok).address,
        )
    }

    // ── refusals ────────────────────────────────────────────────────────

    /**
     * A refusal has to be readable by whichever client asked, and the
     * suggestion has to survive: it is the half that says what to do about it.
     */
    @Test
    fun `a refusal renders in both protocols and keeps its suggestion`() {
        val refusal = ProxyRefusal.unavailable("Battery is at 8%.", "Plug it in.")
        val anthropic = refusal.body(Protocol.ANTHROPIC)
        val openai = refusal.body(Protocol.OPENAI)
        assertTrue(anthropic.contains("\"type\":\"error\""))
        assertTrue(anthropic.contains("Plug it in."))
        assertTrue(openai.contains("\"error\""))
        assertTrue(openai.contains("Battery is at 8%."))
    }

    // ── OpenAI decoding ─────────────────────────────────────────────────

    private fun openai(json: String) = OpenAiCodec.decode(
        ProxyJson.parseToJsonElement(json) as JsonObject,
        midSystemPolicy = ProxySpecs.MID_DEMOTE,
        stripBookkeeping = false,
        media = object : MediaSink {
            override fun writeBase64(data: String, mediaType: String) = "/tmp/$data.png"
            override fun writeBytes(bytes: ByteArray, extension: String) = "/tmp/b.$extension"
        },
    )

    @Test
    fun `a leading system message becomes the system prompt`() {
        val request = openai(
            """{"model":"m","messages":[
               {"role":"system","content":"be brief"},
               {"role":"user","content":"hi"}]}""",
        )
        assertEquals("be brief", request.system)
        assertEquals(1, request.messages.size)
    }

    @Test
    fun `tool calls and results round-trip`() {
        val request = openai(
            """{"model":"m","messages":[
               {"role":"assistant","content":null,"tool_calls":[
                 {"id":"c1","type":"function","function":{"name":"Read","arguments":"{\"p\":1}"}}]},
               {"role":"tool","tool_call_id":"c1","content":"done"}]}""",
        )
        assertEquals("Read", request.messages[0].toolCalls.single().name)
        assertEquals("c1", request.messages[1].toolCallId)
        assertEquals("done", request.messages[1].content)
    }

    @Test
    fun `a data uri image becomes a path and an http one does not`() {
        val request = openai(
            """{"model":"m","messages":[{"role":"user","content":[
               {"type":"image_url","image_url":{"url":"data:image/png;base64,ZZZ"}},
               {"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}""",
        )
        assertEquals(listOf("/tmp/ZZZ.png"), request.messages.single().imagePaths)
    }

    @Test
    fun `max_completion_tokens wins over the deprecated name`() {
        val request = openai(
            """{"model":"m","messages":[],"max_tokens":100,"max_completion_tokens":200}""",
        )
        assertEquals(200, request.params.int("n_predict"))
    }

    @Test
    fun `stop accepts a string or an array`() {
        assertEquals(
            listOf("END"),
            openai("""{"model":"m","messages":[],"stop":"END"}""").params.stringList("stop"),
        )
        assertEquals(
            listOf("A", "B"),
            openai("""{"model":"m","messages":[],"stop":["A","B"]}""").params.stringList("stop"),
        )
    }
}
