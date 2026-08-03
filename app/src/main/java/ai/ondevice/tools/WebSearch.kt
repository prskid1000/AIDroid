package ai.ondevice.tools

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit

/**
 * The public web, read the way a browser reads it.
 *
 * There is no API key and no account anywhere in this app, and a search key
 * would be the first. Brave renders its results server-side, so the answers are
 * in the HTML that comes back from a plain GET — no JavaScript to run, and the
 * result links are the destinations rather than redirect wrappers.
 *
 * That makes this the one part of the app whose correctness depends on someone
 * else's markup. The selectors below avoid Svelte's per-build class hashes and
 * use only the semantic ones (`snippet`, `title`, `content`), but a redesign
 * will still break it, so a parse that finds nothing says exactly that instead
 * of pretending the web had no answer.
 */
class WebSearch(client: OkHttpClient) {

    // Longer than the app's default: a search plus its pages is several round
    // trips, and the tool call has its own 30 s ceiling above this.
    private val http = client.newBuilder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    data class Result(val title: String, val url: String, val snippet: String, val page: String = "")

    /**
     * @param fetchPages how many of the top results to open and read. Zero is
     *   snippets only, which is fast and usually enough.
     */
    suspend fun search(query: String, maxResults: Int = 5, fetchPages: Int = 0): List<Result> {
        val html = withContext(Dispatchers.IO) { get(SEARCH_URL + encode(query) + "&source=web") }
            ?: throw IllegalStateException(rateLimited.getAndSet(false).let { limited ->
                if (limited) {
                    "the search is rate limited for this network. Opening " +
                        "https://search.brave.com in a browser once usually clears it"
                } else {
                    "the search could not be reached"
                }
            })

        val results = parse(html, maxResults)
        if (results.isEmpty() || fetchPages <= 0) return results

        // Concurrent: the wall-clock is the slowest page, not their sum.
        val opened = minOf(fetchPages, results.size)
        return coroutineScope {
            results.mapIndexed { index, result ->
                if (index >= opened) {
                    async { result }
                } else {
                    async(Dispatchers.IO) {
                        result.copy(page = get(result.url)?.let(::readableText).orEmpty())
                    }
                }
            }.map { it.await() }
        }
    }

    /** The tool's answer, as the model will read it. */
    fun render(query: String, results: List<Result>): String {
        if (results.isEmpty()) {
            return "No results for \"$query\". Either there are none, or the search page changed " +
                "shape and this build can no longer read it — say so rather than inventing an answer."
        }
        return buildString {
            appendLine("Results for \"$query\":")
            results.forEachIndexed { index, result ->
                appendLine()
                appendLine("${index + 1}. ${result.title}")
                appendLine("   ${result.url}")
                if (result.snippet.isNotBlank()) appendLine("   ${result.snippet}")
                if (result.page.isNotBlank()) {
                    appendLine()
                    appendLine(result.page)
                }
            }
            appendLine()
            append("Cite the URLs above in your reply. Anything not in them, you do not know.")
        }
    }

    /**
     * Set when the last search came back 429. Brave rate-limits by IP and it is
     * worth saying so, because the remedy is unusual and it works: opening the
     * same search in a browser clears it for that address.
     */
    private val rateLimited = java.util.concurrent.atomic.AtomicBoolean(false)

    /**
     * One page the caller already has the address of, as readable text.
     *
     * Separate from [search] because knowing the URL is a different job from
     * finding it, and going through a search engine to reach a page you can
     * already name costs a round trip and returns someone else's summary of it.
     */
    suspend fun fetch(url: String): String? = withContext(Dispatchers.IO) {
        get(url)?.let { readableText(it) }?.takeIf { it.isNotBlank() }
    }

    private fun get(url: String): String? {
        val request = Request.Builder()
            .url(url)
            // Brave serves a different page to something that does not look
            // like a browser, and that page has no results in it.
            .header("User-Agent", USER_AGENT)
            .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            .header("Accept-Language", "en-US,en;q=0.9")
            .build()

        return runCatching {
            http.newCall(request).execute().use { response ->
                if (response.code == HTTP_TOO_MANY_REQUESTS) rateLimited.set(true)
                if (!response.isSuccessful) return@use null
                val type = response.header("content-type").orEmpty().lowercase()
                if (!type.contains("html") && !type.contains("xml") && type.isNotEmpty()) return@use null
                val body = response.body ?: return@use null
                // A page far larger than an article is a download, not a read.
                val bytes = body.source().let { source ->
                    source.request(MAX_PAGE_BYTES + 1)
                    source.buffer.snapshot()
                }
                if (bytes.size > MAX_PAGE_BYTES) null else bytes.utf8()
            }
        }.getOrNull()
    }

    private fun parse(html: String, limit: Int): List<Result> {
        val cleaned = BOILERPLATE.replace(COMMENTS.replace(html, ""), "")
        val seen = mutableSetOf<String>()
        val out = mutableListOf<Result>()

        // The split leaves everything before the first result in element 0.
        SNIPPET_SPLIT.split(cleaned).drop(1).forEach { block ->
            if (out.size >= limit) return@forEach
            val title = TITLE.find(block)?.groupValues?.get(1) ?: return@forEach
            val url = URL_RE.find(block)?.groupValues?.get(1) ?: return@forEach
            if (!seen.add(normalise(url))) return@forEach
            out += Result(
                title = unescape(title),
                url = url,
                snippet = CONTENT.find(block)?.groupValues?.get(1)
                    ?.let { unescape(TAGS.replace(it, "")) }
                    ?.trim()
                    ?.take(MAX_SNIPPET_CHARS)
                    .orEmpty(),
            )
        }
        return out
    }

    /** Strip the chrome, prefer the article, and keep what a reader would read. */
    private fun readableText(html: String): String {
        var text = COMMENTS.replace(html, "")
        text = CHROME.replace(text, "")
        MAIN.findAll(text).map { it.groupValues[1] }.maxByOrNull { it.length }?.let { text = it }
        text = BLOCK_TAGS.replace(text, "\n")
        text = TAGS.replace(text, "")
        text = unescape(text)
        text = SPACES.replace(text, " ")
        // What survives the tag strip is still half furniture: "LOGOUT",
        // "Loading…", a headline repeated in the nav and again in the page. So
        // drop the scraps and anything already said, keeping first occurrences
        // in order.
        val seen = mutableSetOf<String>()
        text = text.lines()
            .map { it.trim() }
            .filter { line -> line.isEmpty() || (line.length >= MIN_LINE_CHARS && seen.add(line)) }
            .joinToString("\n")
        text = BLANK_LINES.replace(text, "\n\n").trim()
        return if (text.length > MAX_PAGE_CHARS) {
            text.take(MAX_PAGE_CHARS).trimEnd() + "\n…(truncated)"
        } else {
            text
        }
    }

    /** For dedup only — the URL shown is always the one the page gave. */
    private fun normalise(url: String): String = url
        .substringBefore('#')
        .substringBefore('?')
        .removeSuffix("/")
        .removePrefix("https://")
        .removePrefix("http://")
        .removePrefix("www.")
        .lowercase()

    private fun unescape(text: String): String = text
        .replace("&quot;", "\"").replace("&#39;", "'").replace("&apos;", "'")
        .replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
        .replace("&amp;", "&")

    private fun encode(query: String): String =
        java.net.URLEncoder.encode(query, "UTF-8").replace("+", "%20")

    private companion object {
        const val SEARCH_URL = "https://search.brave.com/search?q="
        const val USER_AGENT =
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

        const val MAX_SNIPPET_CHARS = 400

        /** Below this a line is a button, a badge or a menu item, not prose. */
        const val MIN_LINE_CHARS = 25
        const val MAX_PAGE_CHARS = 3000
        const val MAX_PAGE_BYTES = 1_500_000L
        const val HTTP_TOO_MANY_REQUESTS = 429

        val COMMENTS = Regex("<!--.*?-->", RegexOption.DOT_MATCHES_ALL)
        val BOILERPLATE = Regex(
            "<(?:script|style|noscript|svg|template)\\b[^>]*>.*?</(?:script|style|noscript|svg|template)>",
            setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
        )
        val CHROME = Regex(
            "<(?:nav|header|footer|aside|form|iframe|noscript|script|style|svg|template|figure|dialog|" +
                "button|select|label|menu)" +
                "\\b[^>]*>.*?</(?:nav|header|footer|aside|form|iframe|noscript|script|style|svg|template|figure|dialog|" +
                "button|select|label|menu)>",
            setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
        )
        val MAIN = Regex(
            "<(?:main|article)\\b[^>]*>(.*?)</(?:main|article)>",
            setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
        )
        val BLOCK_TAGS = Regex(
            "</?(?:p|br|div|li|tr|h[1-6]|section|ul|ol|blockquote|pre)\\b[^>]*>",
            RegexOption.IGNORE_CASE,
        )
        val TAGS = Regex("<[^>]+>")
        val SPACES = Regex("[ \\t]+")
        val BLANK_LINES = Regex("\\n{3,}")

        // Only the class names that mean something. `svelte-jmfu5f` beside them
        // is a build hash and changes without the page changing.
        val SNIPPET_SPLIT = Regex("<div\\s+[^>]*class=\"snippet\\b[^\"]*\"[^>]*data-type=\"web\"")
        val TITLE = Regex("class=\"title[^\"]*\"\\s+title=\"([^\"]*)\"")
        val URL_RE = Regex("<a\\s+href=\"(https?://[^\"]+)\"")
        val CONTENT = Regex("<div\\s+class=\"content\\s+[^\"]*\"[^>]*>(.*?)</div>", RegexOption.DOT_MATCHES_ALL)
    }
}
