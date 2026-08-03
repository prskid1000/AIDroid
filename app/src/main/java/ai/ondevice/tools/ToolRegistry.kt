package ai.ondevice.tools

import ai.ondevice.engine.ToolSpec
import kotlinx.coroutines.withTimeoutOrNull

/** What the model is allowed to do, and who decides. */
interface ToolProvider {
    val id: String
    suspend fun specs(): List<ToolSpec>
    suspend fun call(name: String, argumentsJson: String): ToolResult

    /**
     * The knobs this provider's tools expose, as rows the params screens render.
     *
     * Empty for a provider with nothing worth tuning — an MCP server's tools
     * are described by the server and take whatever arguments it declares, so
     * there is nothing here for the app to offer.
     *
     * [ai.ondevice.params.ParamSpec.group] is the tool the setting belongs to,
     * which is what lets one flat list render as a section per tool.
     */
    fun settings(): List<ai.ondevice.params.ParamSpec> = emptyList()
}

data class ToolResult(
    val text: String,
    val isError: Boolean = false,
    /** For the UI: which provider actually ran it. */
    val providerId: String = "",
)

/** The providers in play, and nothing else. */
class ToolRegistry(
    private val providers: List<ToolProvider>,
) {
    /** Every tool currently available. */
    suspend fun specs(): List<ToolSpec> {
        val seen = mutableSetOf<String>()
        val out = mutableListOf<ToolSpec>()
        providers.forEach { provider ->
            runCatching { provider.specs() }.getOrDefault(emptyList()).forEach { spec ->
                if (seen.add(spec.name)) out += spec
            }
        }
        return out
    }

    suspend fun call(name: String, argumentsJson: String): ToolResult {
        providers.forEach { provider ->
            val owns = runCatching { provider.specs() }.getOrDefault(emptyList()).any { it.name == name }
            if (!owns) return@forEach
            // A hung server must not hang the conversation. The model gets a
            // timeout it can talk about instead of a spinner that never ends.
            val result = withTimeoutOrNull(CALL_TIMEOUT_MILLIS) {
                runCatching { provider.call(name, argumentsJson) }
                    .getOrElse { ToolResult("The tool failed: ${it.message}", isError = true, providerId = provider.id) }
            }
            return result ?: ToolResult(
                "The tool did not answer within ${CALL_TIMEOUT_MILLIS / 1000} seconds.",
                isError = true,
                providerId = provider.id,
            )
        }
        return ToolResult("No tool named \"$name\" is available.", isError = true)
    }

    private companion object {
        const val CALL_TIMEOUT_MILLIS = 30_000L
    }
}
