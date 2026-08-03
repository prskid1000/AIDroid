package ai.ondevice.tools

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * That the sandbox is a sandbox.
 *
 * These are the checks the file tools and the shell both stand on, so a hole
 * here is a hole in every one of them at once. Written against real temporary
 * directories rather than fabricated paths, because the rule is about what
 * `canonicalFile` does with `..` and with symlinks, and a string comparison
 * against a made-up path proves nothing about either.
 */
class WorkspaceContainmentTest {

    @Test
    fun `a file in the root is inside`() {
        val root = tempDir()
        try {
            val file = File(root, "notes.txt").canonicalFile
            assertTrue(Workspace.isInside(file, listOf(root)))
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun `a nested file is inside`() {
        val root = tempDir()
        try {
            val file = File(root, "a/b/c.txt").canonicalFile
            assertTrue(Workspace.isInside(file, listOf(root)))
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun `walking out with dot-dot is not inside`() {
        val root = tempDir()
        try {
            // What the sandbox exists to refuse: the app's database and its
            // preferences both sit beside the workspace directory.
            val escaped = File(root, "../../databases/ondevice.db").canonicalFile
            assertFalse(
                "canonicalisation must resolve .. before the prefix is compared",
                Workspace.isInside(escaped, listOf(root)),
            )
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun `a sibling sharing the root's name prefix is not inside`() {
        val root = tempDir()
        val sibling = File(root.parentFile, root.name + "-elsewhere").apply { mkdirs() }
        try {
            val file = File(sibling, "secret.txt").canonicalFile
            assertFalse(
                "the separator on the end of the prefix is what stops this",
                Workspace.isInside(file, listOf(root)),
            )
        } finally {
            root.deleteRecursively()
            sibling.deleteRecursively()
        }
    }

    @Test
    fun `the root itself is inside`() {
        val root = tempDir()
        try {
            assertTrue(Workspace.isInside(root, listOf(root)))
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun `any one of several roots is enough`() {
        val first = tempDir()
        val second = tempDir()
        try {
            val file = File(second, "models/flux.gguf").canonicalFile
            assertTrue(Workspace.isInside(file, listOf(first, second)))
            assertFalse(Workspace.isInside(file, listOf(first)))
        } finally {
            first.deleteRecursively()
            second.deleteRecursively()
        }
    }

    // — the glob, which decides what list_files and search_files even look at —

    @Test
    fun `a double star crosses folders and a single one does not`() {
        assertTrue(FileToolProvider.globToRegex("**/*.kt").matches("a/b/Main.kt"))
        assertTrue(
            "**/ has to match zero folders too, or a top-level file is missed",
            FileToolProvider.globToRegex("**/*.kt").matches("Main.kt"),
        )
        assertFalse(FileToolProvider.globToRegex("*.kt").matches("a/Main.kt"))
        assertTrue(FileToolProvider.globToRegex("*.kt").matches("Main.kt"))
    }

    @Test
    fun `a dot in the pattern is a dot and not any character`() {
        assertFalse(FileToolProvider.globToRegex("*.md").matches("READMExmd"))
        assertTrue(FileToolProvider.globToRegex("*.md").matches("README.md"))
    }

    @Test
    fun `occurrences are counted without overlapping`() {
        assertEquals(2, FileToolProvider.countOccurrences("abcabc", "abc"))
        assertEquals(0, FileToolProvider.countOccurrences("abc", "z"))
        // The edit tool decides "is this unique" from this number, so an
        // overlap-counting version would refuse edits that are in fact fine.
        assertEquals(2, FileToolProvider.countOccurrences("aaaa", "aa"))
    }

    private fun tempDir(): File = File.createTempFile("workspace", "").let {
        it.delete()
        it.mkdirs()
        it.canonicalFile
    }
}
