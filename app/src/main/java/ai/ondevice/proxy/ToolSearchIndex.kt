package ai.ondevice.proxy

import ai.ondevice.engine.ToolSpec
import kotlinx.serialization.json.JsonObject
import kotlin.math.ln

/**
 * BM25 over tool definitions, so a phone-sized context does not have to hold
 * every schema it might need.
 *
 * A close port of telecode's `tool_search.py`, including its tuning: k1 = 0.9
 * and b = 0.4 are low for BM25, which suits documents that are a name, a
 * sentence and a handful of parameter descriptions rather than prose.
 *
 * This matters more here than it does on a desktop. A coding client sends
 * upwards of forty tools; their schemas alone are several thousand tokens, and
 * a model running at 8k of context on this hardware spends a quarter of its
 * budget on tools it will not call. Holding them back behind one search tool
 * turns that into a few hundred tokens plus a round-trip when it is actually
 * needed.
 */
class ToolSearchIndex(private val tools: List<ToolSpec>) {

    private val corpus: List<List<String>> = tools.map { tokenize(searchableText(it)) }
    private val documentFrequency: Map<String, Int> = buildMap {
        corpus.forEach { document ->
            document.toSet().forEach { term -> put(term, (get(term) ?: 0) + 1) }
        }
    }
    private val averageLength: Double =
        if (corpus.isEmpty()) 1.0 else corpus.sumOf { it.size }.toDouble() / corpus.size

    /**
     * The tools a query is worth.
     *
     * `select:Name` and `select:A,B` short-circuit the ranking entirely — it is
     * how the model asks for a schema it already knows the name of, which is
     * the overwhelmingly common case once it has been told the names once. A
     * ranked search over an exact name works, but it also returns four things
     * that merely sound similar, and those cost context.
     */
    fun search(query: String, maxResults: Int = MAX_RESULTS): List<ToolSpec> {
        if (query.isBlank() || tools.isEmpty()) return emptyList()

        if (query.startsWith(SELECT_PREFIX, ignoreCase = true)) {
            val wanted = query.removePrefix(SELECT_PREFIX)
                .split(',')
                .map { it.trim() }
                .filter { it.isNotBlank() }
                .toSet()
            return tools.filter { it.name in wanted }
        }

        val terms = tokenize(query)
        if (terms.isEmpty()) return emptyList()

        val n = corpus.size
        val scored = corpus.mapIndexed { index, document ->
            val counts = document.groupingBy { it }.eachCount()
            var score = 0.0
            terms.forEach { term ->
                val tf = counts[term] ?: return@forEach
                val df = documentFrequency[term] ?: 0
                val idf = ln((n - df + 0.5) / (df + 0.5) + 1.0)
                val numerator = tf * (K1 + 1)
                val denominator = tf + K1 * (1 - B + B * document.size / averageLength)
                score += idf * numerator / denominator
            }
            index to score
        }

        return scored
            .filter { it.second > 0.0 }
            .sortedByDescending { it.second }
            .take(maxResults)
            .map { tools[it.first] }
    }

    /** Every tool, flattened: name, sentence, parameter names and their sentences. */
    private fun searchableText(tool: ToolSpec): String = buildString {
        append(tool.name).append(' ')
        append(tool.description).append(' ')
        runCatching {
            val schema = ProxyJson.parseToJsonElement(tool.parametersJson) as? JsonObject
            schema?.obj("properties")?.forEach { (name, definition) ->
                append(name).append(' ')
                (definition as? JsonObject)?.str("description")?.let { append(it).append(' ') }
            }
        }
    }

    private fun tokenize(text: String): List<String> =
        SPLIT.split(text.lowercase()).filter { it.isNotBlank() }

    companion object {
        /** telecode's BM25 tuning, and its reasoning: these are short documents. */
        private const val K1 = 0.9
        private const val B = 0.4
        const val MAX_RESULTS = 5
        private const val SELECT_PREFIX = "select:"
        private val SPLIT = Regex("[^a-z0-9]+")

        /** The meta-tool, injected whenever anything is being held back. */
        val TOOL_SEARCH_SPEC = ToolSpec(
            name = "ToolSearch",
            description = "Find and load the schemas of tools that are available but not " +
                "currently loaded. Call this before using any tool listed as unloaded. " +
                "Use \"select:ToolName\" to load one by exact name, or plain keywords to " +
                "search by what the tool does.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "query": {
                      "type": "string",
                      "description": "Keywords describing the capability needed, or \"select:Name\" to load a known tool by name."
                    },
                    "max_results": {
                      "type": "integer",
                      "description": "How many schemas to load. Defaults to 5."
                    }
                  },
                  "required": ["query"]
                }
            """.trimIndent(),
        )

        /**
         * Tool schemas, rendered for a tool result.
         *
         * A fenced JSON block per tool rather than prose, because what the
         * model needs from this is the exact parameter names — the reason it
         * was told to search in the first place was that it did not have them.
         */
        fun renderSchemas(tools: List<ToolSpec>): String = tools.joinToString("\n\n") { tool ->
            buildString {
                append("### ").append(tool.name).append('\n')
                if (tool.description.isNotBlank()) append(tool.description).append('\n')
                append("```json\n").append(tool.parametersJson).append("\n```")
            }
        }
    }
}
