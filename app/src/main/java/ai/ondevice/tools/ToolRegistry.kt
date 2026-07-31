package ai.ondevice.tools

import ai.ondevice.engine.ToolSpec
import kotlinx.coroutines.withTimeoutOrNull

/**
 * What the model is allowed to do, and who decides.
 *
 * A tool call is the one place a local-first app can quietly stop being local:
 * the model asks for something, and if the app just does it, the user's data
 * leaves the handset without anyone deciding. So two rules run through this
 * package:
 *
 *  - **Nothing is offered that the user has not enabled.** [ToolRegistry.specs]
 *    returns the empty list until tools are switched on, and the empty list is
 *    what the chat template sees, so the model is never even told the option
 *    exists.
 *  - **A tool result is data, not instruction.** Whatever comes back from a
 *    server is inserted as a `tool` message and nothing in this app treats its
 *    contents as a command. An MCP server is a third party; its output is no
 *    more trusted than a web page.
 *
 * The result of a call is always a string, and a failure is a *result* saying
 * what failed — never an exception that kills the turn. A model that is told
 * "that tool timed out" can say so; a crash cannot.
 */
interface ToolProvider {
    val id: String
    suspend fun specs(): List<ToolSpec>
    suspend fun call(name: String, argumentsJson: String): ToolResult
}

data class ToolResult(
    val text: String,
    val isError: Boolean = false,
    /** For the UI: which provider actually ran it. */
    val providerId: String = "",
)

/**
 * The providers in play, and nothing else.
 *
 * There is no enabled-id filter here on purpose. It used to take one, so a
 * provider could be in the list and switched off at the same time — and since
 * the factory filtered separately on a different flag, "switched off" meant two
 * incompatible things depending on which code you read. A provider that should
 * not run is now simply not constructed.
 */
class ToolRegistry(
    private val providers: List<ToolProvider>,
) {
    /**
     * Every tool currently available. Two providers offering the same name is
     * resolved first-wins in provider order, and the loser is dropped rather
     * than silently shadowing — the model cannot disambiguate names it cannot
     * see. Built-ins come first, so a server cannot displace `calculate`.
     */
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
