package ai.ondevice.engine

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.jsonPrimitive

/**
 * The workflow Script step's language.
 *
 * QuickJS, with its standard library deliberately not linked — see
 * `quickjs_jni.cpp` and `native/VERSIONS`. What a script can reach is the
 * language and the outputs of the steps above it, and nothing else: the C code
 * that would open a file or a socket is not in the binary.
 *
 * Absent rather than fatal when the library is missing. A build without it
 * should lose the Script step and keep the other nineteen, the same way a
 * build without the diffusion runtime keeps chat.
 */
object QuickJsBridge {

    val available: Boolean = runCatching {
        System.loadLibrary("ondevice_quickjs")
        true
    }.getOrElse {
        loadError = it.message
        false
    }

    @Volatile
    var loadError: String? = null
        private set

    /** What the embedded engine calls itself. */
    val version: String get() = runCatching { nativeVersion() }.getOrDefault("unknown")

    /**
     * Run [source] with [steps] bound to a global of the same name.
     *
     * The script's value is its last expression, the way a console behaves —
     * so `steps["2"].text.trim()` is a whole script and needs no `return`.
     * An object or an array comes back as JSON so a step can hand a list on
     * without inventing a convention.
     */
    fun eval(source: String, steps: JsonObject, timeoutMillis: Long = 2_000): Result<String> {
        if (!available) {
            return Result.failure(
                IllegalStateException(
                    "This build has no script engine: ${loadError ?: "the library did not load"}.",
                ),
            )
        }
        val raw = runCatching { nativeEval(source, steps.toString(), timeoutMillis) }
            .getOrElse { return Result.failure(it) }

        val parsed = runCatching { JSON.parseToJsonElement(raw).let { it as JsonObject } }
            .getOrElse { return Result.failure(IllegalStateException("The script engine answered badly.")) }

        val ok = parsed["ok"]?.jsonPrimitive?.booleanOrNull ?: false
        return if (ok) {
            Result.success(parsed["value"]?.jsonPrimitive?.content.orEmpty())
        } else {
            Result.failure(
                IllegalStateException(
                    parsed["error"]?.jsonPrimitive?.content ?: "the script failed",
                ),
            )
        }
    }

    private val JSON = Json { ignoreUnknownKeys = true; isLenient = true }

    private external fun nativeEval(source: String, stepsJson: String, timeoutMillis: Long): String

    private external fun nativeVersion(): String
}
