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

/**
 * A Model Context Protocol client, over Streamable HTTP.
 *
 * Only the HTTP transport, and that is not a shortcut. MCP's other transport is
 * stdio, which means launching a process — Android's W^X rules make that either
 * impossible or a way to sneak executable code onto the device, and SPEC §17.1
 * is explicit that this app does not do that. An HTTP endpoint the user typed
 * in is a thing they can see, revoke and reason about.
 *
 * The protocol surface used here is the minimum that makes tools work:
 * `initialize`, `tools/list`, `tools/call`. Prompts, resources and sampling are
 * not implemented — in particular *sampling* is deliberately absent, because it
 * would let a remote server drive inference on this device, which inverts the
 * entire premise of a local-first app.
 */
class McpToolProvider(
    private val server: McpServerEntity,
    private val client: OkHttpClient,
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

    /**
     * What the model is told about — the server's list, minus the tools the user
     * switched off. Not offering it is the real control: a tool the model was
     * never told exists cannot be called, whereas one that is merely refused at
     * call time still costs a round trip and an apology.
     */
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

            // MCP returns content parts; text is all this app renders into the
            // conversation, and a part it cannot render is named rather than
            // dropped silently.
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

    /**
     * Probe a server without registering it — what the "Test" button calls, and
     * what Refresh calls to pick up tools that have appeared or gone away.
     *
     * Reports everything the server offers, including tools the user has
     * switched off: this is the inventory the picker is drawn from, so hiding
     * the disabled ones here would make them unrecoverable.
     */
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
        }.getOrElse { McpProbe(ok = false, error = it.message ?: it.toString()) }
    }

    private var initialized = false

    private fun initialize(force: Boolean = false): JsonObject {
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

    private fun rpc(method: String, params: JsonObject): JsonObject {
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
                server.authHeader?.takeIf { it.isNotBlank() }?.let { header("Authorization", it) }
                header("MCP-Protocol-Version", PROTOCOL_VERSION)
            }
            .build()

        client.newCall(request).execute().use { response ->
            response.header("Mcp-Session-Id")?.let { sessionId = it }
            val raw = response.body?.string().orEmpty()
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

    private fun notify(method: String) {
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
                server.authHeader?.takeIf { it.isNotBlank() }?.let { header("Authorization", it) }
            }
            .build()
        client.newCall(request).execute().close()
    }

    companion object {
        const val ID_PREFIX = "mcp:"
        /** The revision this client implements; sent on every request. */
        const val PROTOCOL_VERSION = "2025-06-18"
        private val JSON_MEDIA = "application/json".toMediaType()

        fun httpClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .build()
    }
}

/** One tool as a server described it. */
@kotlinx.serialization.Serializable
data class McpTool(val name: String, val description: String = "")

data class McpProbe(
    val ok: Boolean,
    val serverName: String = "",
    val protocolVersion: String = "",
    val tools: List<McpTool> = emptyList(),
    val error: String? = null,
)

/**
 * Reading and writing the two tool lists on a server row.
 *
 * Both are stored as JSON in a text column rather than as their own tables. A
 * tool list is only ever read whole, alongside the server it belongs to, and is
 * replaced wholesale by the next refresh — there is nothing to query across, so
 * a join table would be structure without a use for it.
 */
object McpTools {

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }
    private val toolList = ListSerializer(McpTool.serializer())
    private val nameList = ListSerializer(String.serializer())

    /**
     * Accepts the old format too. Before tools carried descriptions this column
     * held a comma-joined list of bare names, and a server the user added last
     * week should not come back empty because the shape changed — it comes back
     * with names and no descriptions, until the next refresh fills them in.
     */
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

/**
 * Builds the live provider list. Rebuilt per turn rather than cached, so pausing
 * a server or switching off one of its tools takes effect on the next message
 * rather than the next process start.
 */
class ToolProviderFactory(
    private val db: OnDeviceDatabase,
    private val capabilities: ai.ondevice.data.hf.DeviceCapabilities,
) {
    private val http = McpToolProvider.httpClient()

    /**
     * Only providers that are actually switched on are constructed, so a paused
     * server has no representative in the registry at all — rather than one that
     * is present and filtered out somewhere further down, which is how it came
     * to be filtered in two places by two different flags that disagreed.
     */
    suspend fun registry(builtInEnabled: Boolean): ToolRegistry {
        val servers = db.mcpServers().getAll().filter { it.enabled }
        return ToolRegistry(
            buildList {
                if (builtInEnabled) add(BuiltInToolProvider(db, capabilities))
                addAll(servers.map { McpToolProvider(it, http) })
            },
        )
    }

    fun provider(server: McpServerEntity): McpToolProvider = McpToolProvider(server, http)
}
