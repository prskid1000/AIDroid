package ai.ondevice.tools

import ai.ondevice.engine.ToolSpec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File

/**
 * Reading and writing files, within [Workspace]'s bounds.
 *
 * The set is the one a coding assistant actually uses — read, write, edit,
 * glob, grep — rather than one general "do something to a file" tool. They are
 * separate because their failure modes are: an edit that matches nothing must
 * say so and change nothing, and a write that silently replaces a file the
 * model meant to append to is the expensive mistake.
 */
class FileToolProvider(
    private val workspace: Workspace,
    private val settings: ToolSettings = ToolSettings.EMPTY,
) : ToolProvider {

    override val id: String = ID

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    override fun settings(): List<ai.ondevice.params.ParamSpec> = listOf(
        ToolSettings.int(
            "read_file", "default_lines", DEFAULT_LINES, 20, MAX_LINES,
            label = "Lines per read",
            help = "How much of a file comes back when the model does not say. " +
                "It can always ask for more with offset and limit.",
        ),
        ToolSettings.int(
            "read_file", "max_kb", (MAX_READ_BYTES / 1024).toInt(), 16, 4096,
            label = "Largest file",
            help = "Files over this are refused rather than read, so one big file " +
                "cannot swallow the whole conversation.",
        ),
        ToolSettings.int(
            "search_files", "max_matches", MAX_RESULTS, 10, 1000,
            label = "Matches returned",
            help = "The cap on lines search_files and list_files return before they stop.",
        ),
        ToolSettings.int(
            "search_files", "max_depth", MAX_DEPTH, 1, 30,
            label = "Folder depth",
            help = "How far down the tree a search walks.",
        ),
    )

    private val defaultLines get() = settings.int("read_file.default_lines", DEFAULT_LINES)
    private val maxReadBytes get() = settings.int("read_file.max_kb", (MAX_READ_BYTES / 1024).toInt()) * 1024L
    private val maxResults get() = settings.int("search_files.max_matches", MAX_RESULTS)
    private val maxDepth get() = settings.int("search_files.max_depth", MAX_DEPTH)

    override suspend fun specs(): List<ToolSpec> = listOf(
        ToolSpec(
            name = "read_file",
            description = "Read a text file. Returns numbered lines, so you can quote a line " +
                "number back. Large files are truncated — pass offset and limit to page through " +
                "one. Paths are relative to the workspace unless they start with /.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "path": { "type": "string", "description": "e.g. notes/todo.md" },
                    "offset": { "type": "integer", "description": "First line to return, 1-based. Default 1." },
                    "limit": { "type": "integer", "description": "How many lines. Default $DEFAULT_LINES." }
                  },
                  "required": ["path"]
                }
            """.trimIndent(),
        ),
        ToolSpec(
            name = "write_file",
            description = "Create a file or replace one entirely. Parent folders are created. " +
                "To change part of a file use edit_file instead — this overwrites everything.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                  },
                  "required": ["path", "content"]
                }
            """.trimIndent(),
        ),
        ToolSpec(
            name = "edit_file",
            description = "Replace an exact run of text in a file. old_string must appear exactly " +
                "once unless replace_all is true, so include enough surrounding text to make it " +
                "unique. Nothing is written if it does not match.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "path": { "type": "string" },
                    "old_string": { "type": "string", "description": "Exact text to find, whitespace included." },
                    "new_string": { "type": "string", "description": "What to put in its place." },
                    "replace_all": { "type": "boolean", "description": "Replace every occurrence. Default false." }
                  },
                  "required": ["path", "old_string", "new_string"]
                }
            """.trimIndent(),
        ),
        ToolSpec(
            name = "list_files",
            description = "Find files by name pattern. Supports * within a folder and ** across " +
                "folders, e.g. **/*.txt. Returns paths, newest first.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "pattern": { "type": "string", "description": "Glob, e.g. **/*.json. Default **/*." },
                    "path": { "type": "string", "description": "Folder to search from. Default the workspace." }
                  }
                }
            """.trimIndent(),
        ),
        ToolSpec(
            name = "search_files",
            description = "Search file contents with a regular expression and return matching " +
                "lines as path:line:text. Use it to find where something is written rather than " +
                "reading whole files.",
            parametersJson = """
                {
                  "type": "object",
                  "properties": {
                    "pattern": { "type": "string", "description": "Regular expression." },
                    "path": { "type": "string", "description": "Folder to search. Default the workspace." },
                    "glob": { "type": "string", "description": "Only search files matching this, e.g. *.md." },
                    "ignore_case": { "type": "boolean" }
                  },
                  "required": ["pattern"]
                }
            """.trimIndent(),
        ),
    )

    override suspend fun call(name: String, argumentsJson: String): ToolResult =
        withContext(Dispatchers.IO) {
            val args = runCatching { json.parseToJsonElement(argumentsJson).jsonObject }.getOrNull()
            fun str(key: String): String? = args?.get(key)?.jsonPrimitive?.content?.takeIf { it.isNotBlank() }
            fun int(key: String): Int? = args?.get(key)?.jsonPrimitive?.content?.toIntOrNull()
            fun bool(key: String): Boolean = args?.get(key)?.jsonPrimitive?.content == "true"

            runCatching {
                when (name) {
                    "read_file" -> readFile(str("path"), int("offset"), int("limit"))
                    "write_file" -> writeFile(str("path"), args?.get("content")?.jsonPrimitive?.content)
                    "edit_file" -> editFile(
                        str("path"),
                        args?.get("old_string")?.jsonPrimitive?.content,
                        args?.get("new_string")?.jsonPrimitive?.content,
                        bool("replace_all"),
                    )
                    "list_files" -> listFiles(str("pattern") ?: "**/*", str("path"))
                    "search_files" -> searchFiles(str("pattern"), str("path"), str("glob"), bool("ignore_case"))
                    else -> fail("No file tool named \"$name\".")
                }
            }.getOrElse { fail(it.message ?: it.toString()) }
        }

    // — the tools —

    private fun readFile(path: String?, offset: Int?, limit: Int?): ToolResult {
        val file = open(path) ?: return fail("read_file needs a \"path\".")
        if (!file.exists()) return fail("${workspace.display(file)} does not exist.")
        if (file.isDirectory) return fail("${workspace.display(file)} is a folder. Use list_files.")
        if (file.length() > maxReadBytes) {
            return fail(
                "${workspace.display(file)} is ${file.length() / 1024} KB, over the " +
                    "${maxReadBytes / 1024} KB this tool reads. Use search_files, or read a " +
                    "slice with offset and limit.",
            )
        }

        val bytes = file.readBytes()
        // A NUL in the first block is the same test `grep` uses, and the reason
        // for it is that handing a model a megabyte of mojibake costs its whole
        // context and tells it nothing.
        if (bytes.take(BINARY_SNIFF).any { it == 0.toByte() }) {
            return fail("${workspace.display(file)} looks like a binary file, not text.")
        }

        val lines = bytes.toString(Charsets.UTF_8).lines()
        val from = (offset ?: 1).coerceAtLeast(1)
        val count = (limit ?: defaultLines).coerceIn(1, MAX_LINES)
        val slice = lines.drop(from - 1).take(count)
        if (slice.isEmpty()) {
            return fail("${workspace.display(file)} has ${lines.size} lines; line $from is past the end.")
        }

        val body = slice.mapIndexed { i, line ->
            "${(from + i).toString().padStart(6)}\t${line.take(MAX_LINE_CHARS)}"
        }.joinToString("\n")
        val shown = from + slice.size - 1
        val note = if (shown < lines.size) {
            "\n\n… ${lines.size - shown} more lines. Read on with offset=${shown + 1}."
        } else {
            ""
        }
        return ok("${workspace.display(file)} (${lines.size} lines)\n$body$note")
    }

    private fun writeFile(path: String?, content: String?): ToolResult {
        val file = open(path) ?: return fail("write_file needs a \"path\".")
        if (content == null) return fail("write_file needs \"content\".")
        if (file.isDirectory) return fail("${workspace.display(file)} is a folder.")

        val existed = file.exists()
        file.parentFile?.mkdirs()
        file.writeText(content)
        val lines = content.lines().size
        return ok(
            (if (existed) "Replaced" else "Wrote") +
                " ${workspace.display(file)} — $lines line${if (lines == 1) "" else "s"}, " +
                "${content.toByteArray().size} bytes.",
        )
    }

    private fun editFile(
        path: String?,
        old: String?,
        new: String?,
        replaceAll: Boolean,
    ): ToolResult {
        val file = open(path) ?: return fail("edit_file needs a \"path\".")
        if (old == null || new == null) return fail("edit_file needs \"old_string\" and \"new_string\".")
        if (!file.exists()) return fail("${workspace.display(file)} does not exist.")
        if (old == new) return fail("old_string and new_string are identical, so there is nothing to do.")

        val text = file.readText()
        val hits = countOccurrences(text, old)
        if (hits == 0) {
            return fail(
                "That text is not in ${workspace.display(file)}. It must match exactly, " +
                    "including indentation and line breaks — read the file first.",
            )
        }
        if (hits > 1 && !replaceAll) {
            return fail(
                "That text appears $hits times in ${workspace.display(file)}. Include more " +
                    "surrounding text to pick one, or pass replace_all.",
            )
        }

        val updated = if (replaceAll) text.replace(old, new) else text.replaceFirst(old, new)
        file.writeText(updated)
        return ok(
            "Edited ${workspace.display(file)} — replaced $hits occurrence" +
                if (hits == 1) "." else "s.",
        )
    }

    private fun listFiles(pattern: String, path: String?): ToolResult {
        val base = open(path ?: ".") ?: return fail("That path cannot be used.")
        if (!base.exists()) return fail("${workspace.display(base)} does not exist.")

        val regex = globToRegex(pattern)
        val matches = base.walkTopDown()
            .maxDepth(maxDepth)
            .filter { it.isFile }
            .filter { regex.matches(it.relativeTo(base).invariantPath()) }
            .sortedByDescending { it.lastModified() }
            .take(maxResults + 1)
            .toList()

        if (matches.isEmpty()) return ok("No file under ${workspace.display(base)} matches \"$pattern\".")
        val shown = matches.take(maxResults)
        val more = if (matches.size > maxResults) "\n… and more; narrow the pattern." else ""
        return ok(shown.joinToString("\n") { workspace.display(it) } + more)
    }

    private fun searchFiles(
        pattern: String?,
        path: String?,
        glob: String?,
        ignoreCase: Boolean,
    ): ToolResult {
        if (pattern.isNullOrBlank()) return fail("search_files needs a \"pattern\".")
        val base = open(path ?: ".") ?: return fail("That path cannot be used.")
        if (!base.exists()) return fail("${workspace.display(base)} does not exist.")

        val regex = runCatching {
            Regex(pattern, if (ignoreCase) setOf(RegexOption.IGNORE_CASE) else emptySet())
        }.getOrElse { return fail("\"$pattern\" is not a valid regular expression: ${it.message}") }
        val nameFilter = glob?.let { globToRegex(it) }

        val out = StringBuilder()
        var found = 0
        // A plain loop with an index rather than forEachIndexed: the cap has to
        // stop the whole walk, and `break` cannot cross a lambda boundary.
        for (file in base.walkTopDown().maxDepth(maxDepth).filter { it.isFile }) {
            if (found >= maxResults) break
            if (nameFilter != null && !nameFilter.matches(file.relativeTo(base).invariantPath())) continue
            if (file.length() > maxReadBytes) continue
            val bytes = runCatching { file.readBytes() }.getOrNull() ?: continue
            if (bytes.take(BINARY_SNIFF).any { it == 0.toByte() }) continue

            var lineNumber = 0
            for (line in bytes.toString(Charsets.UTF_8).lineSequence()) {
                lineNumber++
                if (!regex.containsMatchIn(line)) continue
                out.append(workspace.display(file)).append(':').append(lineNumber).append(": ")
                    .append(line.trim().take(MAX_LINE_CHARS)).append('\n')
                found++
                if (found >= maxResults) break
            }
        }

        if (found == 0) return ok("Nothing under ${workspace.display(base)} matches /$pattern/.")
        val note = if (found >= MAX_RESULTS) "\n… stopped at $MAX_RESULTS matches." else ""
        return ok(out.toString().trimEnd() + note)
    }

    // — plumbing —

    private fun open(path: String?): File? =
        path?.let { workspace.resolve(it).getOrElse { error -> throw error } }

    private fun ok(text: String) = ToolResult(text, providerId = ID)
    private fun fail(text: String) = ToolResult(text, isError = true, providerId = ID)

    companion object {
        const val ID = "files"

        /** Enough to read a source file whole; small enough not to eat the context. */
        private const val DEFAULT_LINES = 400
        private const val MAX_LINES = 2000
        private const val MAX_LINE_CHARS = 400
        private const val MAX_READ_BYTES = 512 * 1024L
        private const val MAX_RESULTS = 200
        private const val MAX_DEPTH = 12
        private const val BINARY_SNIFF = 1024

        fun toolNames(): List<String> =
            listOf("read_file", "write_file", "edit_file", "list_files", "search_files")

        /**
         * A glob as a regex, with `**` crossing folders and `*` not.
         *
         * Order matters: `**` has to be consumed before `*` gets a chance at
         * it, or `**` becomes two single-segment wildcards and never matches a
         * nested path — which is the one thing people write `**` for.
         */
        fun globToRegex(glob: String): Regex {
            val out = StringBuilder()
            var i = 0
            while (i < glob.length) {
                when (val c = glob[i]) {
                    '*' ->
                        if (i + 1 < glob.length && glob[i + 1] == '*') {
                            out.append(".*")
                            i++
                            // `**/` should also match zero folders, so that
                            // `**/*.kt` finds a file sitting at the top.
                            if (i + 1 < glob.length && glob[i + 1] == '/') i++
                        } else {
                            out.append("[^/]*")
                        }
                    '?' -> out.append("[^/]")
                    '.', '(', ')', '+', '|', '^', '$', '@', '%', '{', '}', '[', ']', '\\' ->
                        out.append('\\').append(c)
                    else -> out.append(c)
                }
                i++
            }
            return Regex(out.toString())
        }

        fun countOccurrences(text: String, needle: String): Int {
            if (needle.isEmpty()) return 0
            var count = 0
            var at = text.indexOf(needle)
            while (at >= 0) {
                count++
                at = text.indexOf(needle, at + needle.length)
            }
            return count
        }
    }
}

/** Windows builds the path with backslashes; every pattern is written with slashes. */
private fun File.invariantPath(): String = path.replace(File.separatorChar, '/')
