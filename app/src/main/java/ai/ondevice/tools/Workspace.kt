package ai.ondevice.tools

import android.content.Context
import android.os.Build
import android.os.Environment
import java.io.File

/**
 * Which files the model may touch, and the single place that decides.
 *
 * Every file tool and the shell resolve through [resolve], so there is one
 * containment rule rather than one per tool. The check is done on the
 * *canonical* path, which is the only version of it that holds: `..` segments
 * and symlinks both resolve away, and a check written against the literal
 * string would pass `workspace/../../databases/ondevice.db` without blinking.
 */
class Workspace(
    private val context: Context,
    val scope: Scope,
) {

    enum class Scope {
        /** filesDir and the app's own external directory. Needs no permission. */
        SANDBOX,

        /** The whole of shared storage as well, once the all-files grant is given. */
        DEVICE,
    }

    /**
     * Where a relative path starts.
     *
     * Its own directory rather than filesDir itself, so the model working in
     * "." cannot walk into the Room database, the DataStore preferences or the
     * downloaded model files, all of which live beside it.
     */
    val root: File = File(context.filesDir, "workspace").apply { mkdirs() }

    /**
     * Everything reachable, widest last.
     *
     * The app's external directory is included at both scopes because it is
     * where the models, the gallery and the clips already are — a model asked
     * "how big is my checkpoint" should be able to answer without a permission
     * prompt, and that directory is the app's own either way.
     */
    private val roots: List<File> = buildList {
        add(root)
        context.getExternalFilesDir(null)?.let { add(it) }
        if (scope == Scope.DEVICE && hasAllFilesAccess(context)) {
            add(Environment.getExternalStorageDirectory())
        }
    }.mapNotNull { runCatching { it.canonicalFile }.getOrNull() }

    /** What to tell the model when it asks for something out of bounds. */
    val description: String
        get() = roots.joinToString(", ") { it.absolutePath }

    /**
     * A path the caller may use, or an error explaining why not.
     *
     * Relative paths are taken against [root]. Absolute ones are allowed only
     * where they canonicalise inside a root — which is the same test, written
     * once.
     */
    fun resolve(path: String): Result<File> {
        val raw = path.trim()
        if (raw.isEmpty()) return Result.failure(IllegalArgumentException("The path is empty."))

        val candidate = File(raw).takeIf { it.isAbsolute } ?: File(root, raw)
        // A file that does not exist yet has no canonical path of its own, so
        // the check runs against the nearest parent that does and the
        // non-existent tail is appended. Writing a new file must be allowed;
        // writing it outside must not.
        val canonical = runCatching { candidate.canonicalFile }
            .getOrElse { return Result.failure(IllegalArgumentException("\"$raw\" is not a usable path.")) }

        if (!isInside(canonical, roots)) {
            return Result.failure(
                SecurityException(
                    "\"$raw\" is outside the folders this app can use. " +
                        "Allowed: $description." +
                        if (scope == Scope.SANDBOX) {
                            " Settings → Tools → Filesystem access can widen this to the whole device."
                        } else {
                            ""
                        },
                ),
            )
        }
        return Result.success(canonical)
    }

    /** The path as the model should see it: relative to [root] where it can be. */
    fun display(file: File): String {
        val path = runCatching { file.canonicalPath }.getOrDefault(file.path)
        val prefix = root.path + File.separator
        return if (path.startsWith(prefix)) path.removePrefix(prefix) else path
    }

    companion object {

        /**
         * Whether a canonical path lies in one of [roots].
         *
         * Its own function so it can be tested without an Android context —
         * this is the whole of the containment rule, and a hole in it is the
         * difference between a sandbox and a filesystem.
         *
         * The separator on the end of the prefix is not decoration: without it
         * `/data/user/0/ai.ondevice/files/workspace-elsewhere` starts with
         * `/data/user/0/ai.ondevice/files/workspace` and would pass.
         */
        fun isInside(canonical: File, roots: List<File>): Boolean = roots.any { root ->
            canonical == root || canonical.path.startsWith(root.path + File.separator)
        }

        /**
         * Whether the all-files grant is actually held.
         *
         * Asked of the system rather than remembered, because it is revocable
         * from Settings at any time and a stored "yes" would outlive the grant
         * — the tools would offer paths that then fail on open, which reads as
         * a bug rather than as a permission that was taken away.
         */
        fun hasAllFilesAccess(context: Context): Boolean =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                Environment.isExternalStorageManager()
            } else {
                false
            }
    }
}
