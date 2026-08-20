package ai.ondevice.proxy

import ai.ondevice.core.SparseParams
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.intOrNull

/**
 * One client, matched by a header, with its own answers to some of the
 * settings.
 *
 * Header matching is telecode's mechanism and it ports because it is the only
 * thing that works without the client's cooperation: Claude Code cannot be
 * asked to send a profile name, but it does send `claude-cli` in its
 * User-Agent, and a browser sends its origin in `Referer`. A token would
 * identify a client more honestly, and [token] is here for the clients that
 * can be told one — the header is the fallback, not the design.
 */
@Serializable
data class ProxyProfile(
    val name: String = "",
    val matchHeader: String = "User-Agent",
    val matchContains: String = "",
    /** Matched before the header when set, because a token is proof and a header is a hint. */
    val token: String = "",
    /**
     * Extra system-prompt text for this client, appended after its own.
     *
     * The text itself rather than telecode's filename-under-`instructions/`.
     * There is no directory of instruction files on a phone, and adding one
     * would be a second place to look for something a text field already holds.
     */
    val instruction: String = "",
    /**
     * Sparse overrides, keyed exactly as [ProxySpecs] keys them.
     *
     * Sparse for the reason every other override map in this app is sparse: a
     * profile that never mentioned a setting follows the global one, including
     * when the global default moves in a later release.
     */
    val overridesJson: String = "{}",
    /**
     * Tool names this app runs itself, offered to this client.
     *
     * Null means "every enabled provider", which is what an unconfigured
     * profile should mean. An empty list means "none", which is a different
     * answer and has to stay expressible.
     */
    val injectManaged: List<String>? = null,
    /** Names removed from whatever tool list the client sent. */
    val stripTools: List<String> = emptyList(),
    /** Tools that stay loaded for this client. Empty inherits the global list. */
    val coreTools: List<String> = emptyList(),
) {
    val overrides: SparseParams get() = SparseParams.parse(overridesJson)
}

/**
 * Everything the proxy is configured to be, resolved once per request.
 *
 * Built from the sparse settings map plus the three list-shaped things that are
 * not settings. Stored whole as one JSON document in DataStore rather than
 * normalised into Room, following the argument in `docs/workflow-plan.md`:
 * normalising means a migration every time a field is added, and the schema is
 * already at fourteen migrations for things that genuinely needed rows.
 */
@Serializable
data class ProxyDocument(
    /** The `proxy.*` settings, sparse. Serialised as a JSON object string. */
    val settingsJson: String = "{}",
    /** Client-facing model name → installed model id. */
    val aliases: Map<String, String> = emptyMap(),
    /** Allowed browser origins. Empty disables CORS entirely, which is the safe default. */
    val corsOrigins: List<String> = emptyList(),
    val profiles: List<ProxyProfile> = emptyList(),
    /** Tool names that stay loaded for every client when tool search is on. */
    val coreTools: List<String> = emptyList(),
) {
    companion object {
        private val JSON = Json {
            ignoreUnknownKeys = true
            isLenient = true
            encodeDefaults = true
        }

        val EMPTY = ProxyDocument()

        fun parse(raw: String?): ProxyDocument {
            if (raw.isNullOrBlank()) return EMPTY
            return runCatching { JSON.decodeFromString(serializer(), raw) }.getOrElse { EMPTY }
        }

        fun encode(document: ProxyDocument): String =
            JSON.encodeToString(serializer(), document)
    }

    fun encode(): String = encode(this)

    val settings: SparseParams get() = SparseParams.parse(settingsJson)

    fun withSetting(key: String, value: Any?): ProxyDocument {
        val current = settings
        val next = when (value) {
            null -> current.without(key)
            is Boolean -> current.with(key, value)
            is Int -> current.with(key, value)
            is Float -> current.with(key, value)
            is String -> current.with(key, value)
            is JsonElement -> current.with(key, value)
            else -> current.with(key, value.toString())
        }
        return copy(settingsJson = next.toJsonString())
    }
}

/**
 * A settings view with a profile layered over it.
 *
 * Reads fall through in one direction — profile, then global, then the spec's
 * own default — so there is exactly one place a default is written down and it
 * is next to the sentence describing it.
 */
class ProxyConfig(
    val document: ProxyDocument,
    private val profile: ProxyProfile? = null,
) {
    private val global = document.settings
    private val overrides = profile?.overrides ?: SparseParams.EMPTY

    val profileName: String? get() = profile?.name?.takeIf { it.isNotBlank() }

    fun bool(key: String): Boolean =
        overrides.takeIf { key in it }?.bool(key)
            ?: global.bool(key)
            ?: (ProxySpecs.spec(key)?.default as? JsonPrimitive)?.booleanOrNull
            ?: false

    fun int(key: String): Int =
        overrides.takeIf { key in it }?.int(key)
            ?: global.int(key)
            ?: (ProxySpecs.spec(key)?.default as? JsonPrimitive)?.intOrNull
            ?: 0

    fun string(key: String): String =
        overrides.takeIf { key in it }?.string(key)
            ?: global.string(key)
            ?: (ProxySpecs.spec(key)?.default as? JsonPrimitive)?.content
            ?: ""

    /** Names that stay loaded when tool search is on — the profile's list wins whole. */
    val coreTools: List<String>
        get() = profile?.coreTools?.takeIf { it.isNotEmpty() } ?: document.coreTools

    val stripTools: List<String> get() = profile?.stripTools.orEmpty()

    val injectManaged: List<String>? get() = profile?.injectManaged

    val instruction: String get() = profile?.instruction.orEmpty()

    // — the handful of reads that have a name worth using —

    val enabled: Boolean get() = bool(ProxySpecs.ENABLED)
    val port: Int get() = int(ProxySpecs.PORT)
    val bind: String get() = string(ProxySpecs.BIND)
    val anthropicEnabled: Boolean get() = bool(ProxySpecs.PROTOCOL_ANTHROPIC)
    val openAiEnabled: Boolean get() = bool(ProxySpecs.PROTOCOL_OPENAI)
    val servesImages: Boolean get() = bool(ProxySpecs.SERVE_IMAGES)
    val servesAudio: Boolean get() = bool(ProxySpecs.SERVE_AUDIO)
    val servesVideo: Boolean get() = bool(ProxySpecs.SERVE_VIDEO)

    /** The model a surface uses when the request named none. Empty = most recent. */
    fun defaultModel(key: String): String = string(key).trim()
    val toolSearch: Boolean get() = bool(ProxySpecs.TOOL_SEARCH)
    val autoLoadTools: Boolean get() = bool(ProxySpecs.AUTO_LOAD_TOOLS)
    val stripReminders: Boolean get() = bool(ProxySpecs.STRIP_REMINDERS)
    val sortTools: Boolean get() = bool(ProxySpecs.SORT_TOOLS)
    val midSystem: String get() = string(ProxySpecs.MID_SYSTEM)
    val injectDate: Boolean get() = bool(ProxySpecs.INJECT_DATE)
    val location: String get() = string(ProxySpecs.LOCATION)
    val modelPolicy: String get() = string(ProxySpecs.MODEL_POLICY)
    val maxRoundTrips: Int get() = int(ProxySpecs.MAX_ROUNDTRIPS)
    val pingInterval: Int get() = int(ProxySpecs.PING_INTERVAL)
    val queueDepth: Int get() = int(ProxySpecs.QUEUE_DEPTH)
    val queueTimeoutSeconds: Int get() = int(ProxySpecs.QUEUE_TIMEOUT)
    val batteryFloor: Int get() = int(ProxySpecs.BATTERY_FLOOR)
    val chargingOnly: Boolean get() = bool(ProxySpecs.CHARGING_ONLY)
    val requireAuth: Boolean get() = bool(ProxySpecs.REQUIRE_AUTH)
    val debug: Boolean get() = bool(ProxySpecs.DEBUG)

    val corsOrigins: List<String> get() = document.corsOrigins

    /** Resolve a client-facing model name through the alias table. */
    fun resolveAlias(requested: String): String =
        document.aliases[requested] ?: requested

    /**
     * The profile whose match wins, or null.
     *
     * First match, in declared order, so the list is also the priority — the
     * same rule telecode uses, and the reason the screen lets profiles be
     * reordered rather than sorting them by name.
     */
    companion object {
        fun match(
            document: ProxyDocument,
            header: (String) -> String?,
            bearer: String?,
        ): ProxyProfile? {
            document.profiles.forEach { profile ->
                if (profile.token.isNotBlank() && profile.token == bearer) return profile
            }
            document.profiles.forEach { profile ->
                val needle = profile.matchContains
                if (needle.isBlank() || profile.matchHeader.isBlank()) return@forEach
                val value = header(profile.matchHeader).orEmpty()
                if (value.contains(needle, ignoreCase = true)) return profile
            }
            return null
        }

        fun of(document: ProxyDocument, profile: ProxyProfile? = null) =
            ProxyConfig(document, profile)
    }
}

/** Kept next to the document so the two encoders never disagree about shape. */
internal fun JsonObject.stringOrNull(key: String): String? =
    (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.content
