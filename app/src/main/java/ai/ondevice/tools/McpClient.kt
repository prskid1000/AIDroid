package ai.ondevice.tools

import ai.ondevice.data.db.McpServerEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.engine.ToolSpec
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/** A Model Context Protocol client, over Streamable HTTP. */
class McpToolProvider(
    private val server: McpServerEntity,
    private val client: OkHttpClient,
    /**
     * Supplies a bearer for this server, refreshing it if it has expired.
     *
     * A function rather than a token because a registry is built once per turn
     * and a turn can outlive an access token — capturing the string here would
     * mean the second tool call of a long conversation failing with a 401 that
     * a refresh would have prevented.
     */
    private val bearer: suspend () -> String? = { null },
) : ToolProvider {

    override val id: String = "${ID_PREFIX}${server.id}"

    private val json = Json { ignoreUnknownKeys = true; isLenient = true; encodeDefaults = true }
    private val nextId = AtomicLong(1)

    @Volatile
    private var sessionId: String? = null

    @Volatile
    private var cachedSpecs: List<ToolSpec>? = null

    private val disabledTools: Set<String> = McpTools.disabled(server)

    /** The tools this server offers, before the user's own exclusions. */
    private suspend fun offered(): List<ToolSpec> = cachedSpecs ?: withContext(Dispatchers.IO) {
        initialize()
        val result = rpc("tools/list", buildJsonObject { })
        val tools = result["tools"]?.jsonArray.orEmpty()
        val specs = tools.map { element ->
            val tool = element.jsonObject
            ToolSpec(
                name = tool["name"]?.jsonPrimitive?.content.orEmpty(),
                description = tool["description"]?.jsonPrimitive?.content.orEmpty(),
                // MCP calls it inputSchema; the chat template wants parameters.
                // Same JSON Schema either way.
                parametersJson = (tool["inputSchema"] as? JsonObject)?.toString() ?: "{\"type\":\"object\"}",
            )
        }.filter { it.name.isNotBlank() }
        cachedSpecs = specs
        specs
    }

    /** What the model is told about — the server's list, minus the tools the user switched off. */
    override suspend fun specs(): List<ToolSpec> =
        offered().filterNot { it.name in disabledTools }

    override suspend fun call(name: String, argumentsJson: String): ToolResult = withContext(Dispatchers.IO) {
        // Belt and braces. The model should not know this name, but it can
        // guess one, and a switch that only hides a tool is not a switch.
        if (name in disabledTools) {
            return@withContext ToolResult(
                "\"$name\" is switched off for ${server.name}.",
                isError = true,
                providerId = id,
            )
        }
        runCatching {
            initialize()
            val result = rpc(
                "tools/call",
                buildJsonObject {
                    put("name", name)
                    put(
                        "arguments",
                        runCatching { json.parseToJsonElement(argumentsJson) }
                            .getOrElse { buildJsonObject { } },
                    )
                },
            )

            // MCP returns content parts; text is all this app renders into the conversation, and a part it cannot render is named rather than dropped silently.
            val text = result["content"]?.jsonArray.orEmpty().joinToString("\n") { part ->
                val obj = part.jsonObject
                when (obj["type"]?.jsonPrimitive?.content) {
                    "text" -> obj["text"]?.jsonPrimitive?.content.orEmpty()
                    else -> "[${obj["type"]?.jsonPrimitive?.content ?: "unknown"} content, not shown]"
                }
            }
            ToolResult(
                text = text.ifBlank { "The tool returned nothing." },
                isError = result["isError"]?.jsonPrimitive?.content == "true",
                providerId = id,
            )
        }.getOrElse {
            ToolResult("${server.name} could not run \"$name\": ${it.message}", isError = true, providerId = id)
        }
    }

    /** Probe a server without registering it — what the "Test" button calls, and what Refresh calls to pick up tools that have appeared or gone away. */
    suspend fun probe(): McpProbe = withContext(Dispatchers.IO) {
        runCatching {
            val info = initialize(force = true)
            val offered = offered()
            McpProbe(
                ok = true,
                serverName = info["serverInfo"]?.jsonObject?.get("name")?.jsonPrimitive?.content
                    ?: server.name,
                protocolVersion = info["protocolVersion"]?.jsonPrimitive?.content.orEmpty(),
                tools = offered.map { McpTool(it.name, it.description) },
            )
        }.getOrElse { failure ->
            val unauthorized = failure as? McpUnauthorized
            McpProbe(
                ok = false,
                error = failure.message ?: failure.toString(),
                needsAuthorization = unauthorized != null,
                challenge = unauthorized?.challenge,
            )
        }
    }

    private var initialized = false

    private suspend fun initialize(force: Boolean = false): JsonObject {
        if (initialized && !force) return buildJsonObject { }
        val result = rpc(
            "initialize",
            buildJsonObject {
                put("protocolVersion", PROTOCOL_VERSION)
                putJsonObject("capabilities") { }
                putJsonObject("clientInfo") {
                    put("name", "on-device-ai")
                    put("version", "1")
                }
            },
        )
        initialized = true
        // `notifications/initialized` has no reply; a server that rejects it is
        // still usable, so a failure here is not fatal.
        runCatching { notify("notifications/initialized") }
        return result
    }

    private suspend fun rpc(method: String, params: JsonObject): JsonObject {
        val body = buildJsonObject {
            put("jsonrpc", "2.0")
            put("id", nextId.getAndIncrement())
            put("method", method)
            put("params", params)
        }

        val request = Request.Builder()
            .url(server.url)
            .post(body.toString().toRequestBody(JSON_MEDIA))
            // Streamable HTTP lets the server answer with either JSON or an SSE
            // stream, and it picks. Accepting both is what the spec requires.
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .apply {
                sessionId?.let { header("Mcp-Session-Id", it) }
                // A live OAuth token wins over a header typed by hand: the
                // header is what someone pasted before authorising, and after
                // authorising it is the stale half of the pair.
                val token = bearer()
                when {
                    token != null -> header("Authorization", "Bearer $token")
                    else -> server.authHeader?.takeIf { it.isNotBlank() }
                        ?.let { header("Authorization", it) }
                }
                header("MCP-Protocol-Version", PROTOCOL_VERSION)
            }
            .build()

        client.newCall(request).execute().use { response ->
            response.header("Mcp-Session-Id")?.let { sessionId = it }
            val raw = response.body?.string().orEmpty()
            if (response.code == HTTP_UNAUTHORIZED) {
                // Not an error string: the challenge header names where to go
                // and sign in, and the Tools screen turns it into a button.
                throw McpUnauthorized(
                    server.url,
                    response.header("WWW-Authenticate"),
                )
            }
            if (!response.isSuccessful) {
                error("HTTP ${response.code} from ${server.url}: ${raw.take(200)}")
            }

            val payload = if (raw.startsWith("event:") || raw.contains("\ndata:") || raw.startsWith("data:")) {
                // SSE: the JSON-RPC response is the first `data:` line.
                raw.lineSequence()
                    .firstOrNull { it.startsWith("data:") }
                    ?.removePrefix("data:")?.trim()
                    ?: error("The server sent an event stream with no data frame.")
            } else {
                raw
            }

            val message = json.parseToJsonElement(payload).jsonObject
            message["error"]?.jsonObject?.let { rpcError ->
                error(rpcError["message"]?.jsonPrimitive?.content ?: rpcError.toString())
            }
            return message["result"]?.jsonObject ?: buildJsonObject { }
        }
    }

    private suspend fun notify(method: String) {
        val body = buildJsonObject {
            put("jsonrpc", "2.0")
            put("method", method)
            put("params", buildJsonObject { })
        }
        val request = Request.Builder()
            .url(server.url)
            .post(body.toString().toRequestBody(JSON_MEDIA))
            .header("Accept", "application/json, text/event-stream")
            .apply {
                sessionId?.let { header("Mcp-Session-Id", it) }
                val token = bearer()
                when {
                    token != null -> header("Authorization", "Bearer $token")
                    else -> server.authHeader?.takeIf { it.isNotBlank() }
                        ?.let { header("Authorization", it) }
                }
            }
            .build()
        client.newCall(request).execute().close()
    }

    companion object {
        const val ID_PREFIX = "mcp:"
        /** The revision this client implements; sent on every request. */
        const val PROTOCOL_VERSION = "2025-06-18"
        private const val HTTP_UNAUTHORIZED = 401
        private val JSON_MEDIA = "application/json".toMediaType()

        fun httpClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .build()
    }
}

/**
 * A server that wants a sign-in.
 *
 * Its own type because it is not a failure the user can do anything about by
 * retrying — it is the one error with a next step attached, and the screen
 * turns it into an Authorize button rather than a red line.
 */
class McpUnauthorized(
    val serverUrl: String,
    /** The `WWW-Authenticate` challenge, which names where the metadata lives. */
    val challenge: String?,
) : Exception("This server needs you to sign in.")

/** One tool as a server described it. */
@kotlinx.serialization.Serializable
data class McpTool(val name: String, val description: String = "")

data class McpProbe(
    val ok: Boolean,
    val serverName: String = "",
    val protocolVersion: String = "",
    val tools: List<McpTool> = emptyList(),
    val error: String? = null,
    /**
     * The server answered 401 rather than failing.
     *
     * Kept apart from [error] because it is the one failure with a next step:
     * the server is reachable and working, it just has not been signed in to.
     */
    val needsAuthorization: Boolean = false,
    /** The `WWW-Authenticate` challenge, when the 401 carried one. */
    val challenge: String? = null,
)

/** Reading and writing the two tool lists on a server row. */
object McpTools {

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }
    private val toolList = ListSerializer(McpTool.serializer())
    private val nameList = ListSerializer(String.serializer())

    /** Accepts the old format too. */
    fun parse(raw: String?): List<McpTool> {
        val text = raw?.trim().orEmpty()
        if (text.isEmpty()) return emptyList()
        if (text.startsWith("[")) {
            return runCatching { json.decodeFromString(toolList, text) }.getOrDefault(emptyList())
        }
        return text.split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { McpTool(it) }
    }

    fun encode(tools: List<McpTool>): String = json.encodeToString(toolList, tools)

    fun disabled(server: McpServerEntity): Set<String> =
        runCatching { json.decodeFromString(nameList, server.disabledToolsJson) }
            .getOrDefault(emptyList())
            .toSet()

    fun encodeDisabled(names: Set<String>): String = json.encodeToString(nameList, names.sorted())
}

/** Builds the live provider list. */
class ToolProviderFactory(
    private val db: OnDeviceDatabase,
    private val capabilities: ai.ondevice.data.hf.DeviceCapabilities,
    private val context: android.content.Context,
    tokens: ai.ondevice.data.secure.TokenStore,
) {
    private val http = McpToolProvider.httpClient()
    private val web = WebSearch(http)

    /** Sign-in, refresh and the bearer every MCP request asks for. */
    val authorizer = McpAuthorizer(context, db, tokens, http)

    /**
     * @param enabled the provider ids the user has switched on. The shell is in
     *   this set only after its own toggle, which is why it is not a parameter
     *   of its own: one list of what is allowed, checked in one place.
     * @param fileScope how far the file tools and the shell can reach.
     */
    suspend fun registry(
        enabled: Set<String>,
        fileScope: Workspace.Scope = Workspace.Scope.SANDBOX,
        tuning: ai.ondevice.core.SparseParams = ai.ondevice.core.SparseParams.EMPTY,
    ): ToolRegistry {
        val servers = db.mcpServers().getAll().filter { it.enabled }
        val workspace by lazy { Workspace(context, fileScope) }
        return ToolRegistry(
            buildList {
                if (BuiltInToolProvider.ID in enabled) add(built(tuning))
                if (FileToolProvider.ID in enabled) add(FileToolProvider(workspace, settingsFor(files(workspace), tuning)))
                if (ShellToolProvider.ID in enabled) add(ShellToolProvider(workspace, settingsFor(shell(workspace), tuning)))
                addAll(servers.map { server -> provider(server) })
            },
        )
    }

    /**
     * Every settings row the app's own tools expose, for the screen that
     * renders them.
     *
     * Built from throwaway providers rather than from a second list: the specs
     * are declared next to the code that reads them, and a list kept here as
     * well would be the copy that goes stale.
     */
    fun allSettings(context: android.content.Context): List<ai.ondevice.params.ParamSpec> {
        val workspace = Workspace(context, Workspace.Scope.SANDBOX)
        return built(ai.ondevice.core.SparseParams.EMPTY).settings() +
            files(workspace).settings() +
            shell(workspace).settings()
    }

    private fun built(tuning: ai.ondevice.core.SparseParams) =
        BuiltInToolProvider(db, capabilities, web).let { bare ->
            BuiltInToolProvider(db, capabilities, web, settingsFor(bare, tuning))
        }

    private fun files(workspace: Workspace) = FileToolProvider(workspace)
    private fun shell(workspace: Workspace) = ShellToolProvider(workspace)

    private fun settingsFor(provider: ToolProvider, tuning: ai.ondevice.core.SparseParams) =
        ToolSettings(tuning, provider.settings())

    fun provider(server: McpServerEntity): McpToolProvider =
        McpToolProvider(server, http, bearer = { authorizer.bearer(server.id) })
}
