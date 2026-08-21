package ai.ondevice.proxy

import ai.ondevice.core.Tier
import ai.ondevice.params.ParamSpec
import ai.ondevice.params.ParamType
import kotlinx.serialization.json.JsonPrimitive

/**
 * Every proxy setting, declared as data.
 *
 * Deliberately [ParamSpec] and not a screen full of switches, for the reason
 * `ToolSettings` already gives about tool knobs: the app has one description of
 * "a number with a range, a default and a sentence about it", the screens
 * already render it, and a second little settings type would mean a second
 * slider, a second clamp and a second set of defaults that drift from these.
 *
 * The payoff is visible on the Proxy screen: it contains no `when` on a key
 * name anywhere. Adding a setting is adding a line here.
 */
object ProxySpecs {

    /** Everything the screen renders, in the order it renders it. */
    val ALL: List<ParamSpec> = listOf(

        // — network —

        bool(
            ENABLED, false,
            label = "Serve the API",
            help = "Off until asked for. Nothing listens on any port while this is off.",
            tier = Tier.BASIC,
        ),
        enum(
            BIND, BIND_TAILNET, listOf(BIND_TAILNET, BIND_LOOPBACK, BIND_ALL),
            label = "Listen on",
            help = "Tailnet binds to this device's 100.x address alone — reachable from your " +
                "other Tailscale machines and from nowhere else. Loopback is this device only, " +
                "for adb reverse. All includes whatever Wi-Fi you are on, which is a stranger's " +
                "network more often than not.",
            requiresReload = true,
        ),
        int(
            PORT, 8080, 1024, 65535,
            label = "Port",
            help = "Ports below 1024 are not available to an app.",
            requiresReload = true,
        ),
        bool(
            TLS, false,
            label = "HTTPS",
            help = "Serves TLS with a certificate this device signs for itself — there is no " +
                "authority that will issue one for an address only your tailnet can reach. A " +
                "client has to be given that certificate, and the card below sends it. Off " +
                "means plain HTTP, which over a tailnet is already encrypted between " +
                "machines by WireGuard.",
            requiresReload = true,
        ),

        // — protocols —

        bool(
            PROTOCOL_ANTHROPIC, true,
            label = "Anthropic",
            help = "Serves /v1/messages and /v1/messages/count_tokens.",
        ),
        bool(
            PROTOCOL_OPENAI, true,
            label = "OpenAI",
            help = "Serves /v1/chat/completions, and the image, audio and video routes.",
        ),

        // — surfaces —

        bool(
            SERVE_IMAGES, true,
            label = "Images",
            help = "/v1/images/generations, /edits and /upscales, on the diffusion runtime.",
        ),
        bool(
            SERVE_AUDIO, true,
            label = "Audio",
            help = "/v1/audio/speech, /transcriptions and /translations.",
        ),
        bool(
            SERVE_VIDEO, true,
            label = "Video",
            help = "/v1/videos, as a job you poll. A clip is tens of minutes on this hardware " +
                "and no HTTP connection survives that.",
        ),

        // — default models —

        string(
            DEFAULT_TEXT, "",
            label = "Chat model",
            help = "Which model answers when a request names none, or names one this device " +
                "does not have. Empty uses whichever text model was used most recently.",
        ),
        string(
            DEFAULT_IMAGE, "",
            label = "Image model",
            help = "The checkpoint /v1/images uses. Worth setting: pictures and clips are both " +
                "diffusion models, so \"most recently used\" cannot be right for both at once.",
        ),
        string(
            DEFAULT_VIDEO, "",
            label = "Video model",
            help = "The checkpoint /v1/videos uses, which is almost never the one that makes " +
                "stills.",
        ),
        string(
            DEFAULT_VOICE, "",
            label = "Voice",
            help = "The model /v1/audio/speech speaks with.",
        ),
        string(
            DEFAULT_SPEECH, "",
            label = "Speech model",
            help = "The model /v1/audio/transcriptions listens with.",
        ),

        // — behaviour —

        bool(
            TOOL_SEARCH, true,
            label = "Tool search",
            help = "Hold tool schemas back behind a ToolSearch tool and load them on demand. " +
                "A phone-sized context cannot hold forty of them, so this is closer to a " +
                "requirement here than to an optimisation.",
        ),
        bool(
            AUTO_LOAD_TOOLS, true,
            label = "Auto-load schemas",
            help = "When the model calls a held-back tool without loading it first, hand it the " +
                "schema and let it try again, instead of refusing.",
        ),
        bool(
            STRIP_REMINDERS, true,
            label = "Strip client bookkeeping",
            help = "Remove <system-reminder> blocks and per-turn <total_tokens> lines from the " +
                "history before it reaches the model. Ours are kept.",
        ),
        bool(
            SORT_TOOLS, false,
            label = "Sort tools",
            help = "Sort the tool list by name before rendering the prompt. Stabilises the " +
                "prefix when a client reorders its own tools, at the cost of overriding any " +
                "deliberate ordering.",
            tier = Tier.EXPERT,
        ),
        enum(
            MID_SYSTEM, MID_DEMOTE, listOf(MID_DEMOTE, MID_STRIP, MID_MERGE_TOP, MID_KEEP),
            label = "Mid-conversation system messages",
            help = "What to do with a system message that arrives after the conversation has " +
                "started. Chat templates like Qwen's refuse one that is not first. Demote keeps " +
                "its position and re-roles it to user — the only option that is both " +
                "template-safe and cache-safe, because merging them at the top makes the front " +
                "block grow every turn and re-prefills the whole history.",
            tier = Tier.EXPERT,
        ),
        bool(
            INJECT_DATE, true,
            label = "Inject the date",
            help = "Append today's date, and the location below when one is set, to the system " +
                "prompt. The date comes from this device; nothing is looked up over the network.",
        ),
        string(
            LOCATION, "",
            label = "Location",
            help = "Appended beside the date when set. Empty means the date alone — there is no " +
                "geo-IP lookup here and there will not be one.",
        ),
        enum(
            MODEL_POLICY, POLICY_QUEUE, listOf(POLICY_QUEUE, POLICY_REFUSE, POLICY_SWAP),
            label = "When a request wants another model",
            help = "Queue waits for whatever is running and then swaps. Refuse answers 409 and " +
                "names the model that is loaded — right for a phone in your hand. Swap evicts " +
                "immediately, which will take a conversation's context with it.",
        ),

        // — limits —

        int(
            MAX_ROUNDTRIPS, 15, 1, 50,
            label = "Max round-trips",
            help = "How many tool rounds one request may take before it gives up.",
        ),
        int(
            PING_INTERVAL, 10, 1, 60,
            label = "Ping interval",
            help = "Seconds between keep-alive frames while the model is still thinking. A long " +
                "prefill on this hardware outlasts most clients' idle timeouts.",
        ),
        int(
            QUEUE_DEPTH, 4, 1, 32,
            label = "Queue depth",
            help = "Requests allowed to wait for the engine. Beyond this the answer is 429 with " +
                "a Retry-After rather than a queue nobody can see the end of.",
        ),
        int(
            QUEUE_TIMEOUT, 120, 10, 3600,
            label = "Queue timeout",
            help = "Seconds a request may wait before it is answered 503. One engine runs at a " +
                "time, so waiting is normal and waiting forever is not.",
        ),
        int(
            BATTERY_FLOOR, 20, 0, 100,
            label = "Battery floor",
            help = "Below this percentage, generation is refused with the battery level in the " +
                "message. The socket stays open — refusing loudly beats getting quietly slower.",
        ),
        bool(
            CHARGING_ONLY, false,
            label = "Only while charging",
            help = "Accept generation only when this device is on power.",
        ),

        // — access —

        bool(
            REQUIRE_AUTH, true,
            label = "Require a token",
            help = "On by default and worth leaving on. Off, anything that can reach this " +
                "device on the network can generate on it, read files the tools can read, and " +
                "spend its battery.",
        ),

        // — diagnostics —

        bool(
            DEBUG, false,
            label = "Write request dumps",
            help = "Writes each request and its answer as JSON under the app's own files, in " +
                "proxy/logs/. The screen above already keeps both while the app is running; " +
                "this is for the run that ends with the app being killed, which is the one " +
                "worth looking at afterwards. It includes every prompt, and the previous " +
                "session's files are cleared when the server starts.",
            tier = Tier.EXPERT,
        ),
    )

    fun spec(key: String): ParamSpec? = ALL.firstOrNull { it.key == key }

    // — keys. Namespaced `proxy.` so one flat sparse map holds them all. —

    const val ENABLED = "proxy.enabled"
    const val BIND = "proxy.bind"
    const val PORT = "proxy.port"
    const val TLS = "proxy.tls"
    const val PROTOCOL_ANTHROPIC = "proxy.protocol_anthropic"
    const val PROTOCOL_OPENAI = "proxy.protocol_openai"
    const val SERVE_IMAGES = "proxy.serve_images"
    const val SERVE_AUDIO = "proxy.serve_audio"
    const val SERVE_VIDEO = "proxy.serve_video"

    const val DEFAULT_TEXT = "proxy.default_text_model"
    const val DEFAULT_IMAGE = "proxy.default_image_model"
    const val DEFAULT_VIDEO = "proxy.default_video_model"
    const val DEFAULT_VOICE = "proxy.default_voice_model"
    const val DEFAULT_SPEECH = "proxy.default_speech_model"
    const val TTS_VOICE = "proxy.tts_voice"
    const val TOOL_SEARCH = "proxy.tool_search"
    const val AUTO_LOAD_TOOLS = "proxy.auto_load_tools"
    const val STRIP_REMINDERS = "proxy.strip_reminders"
    const val SORT_TOOLS = "proxy.sort_tools"
    const val MID_SYSTEM = "proxy.mid_system_messages"
    const val INJECT_DATE = "proxy.inject_date_location"
    const val LOCATION = "proxy.location"
    const val MODEL_POLICY = "proxy.model_policy"
    const val MAX_ROUNDTRIPS = "proxy.max_roundtrips"
    const val PING_INTERVAL = "proxy.ping_interval"
    const val QUEUE_DEPTH = "proxy.queue_depth"
    const val QUEUE_TIMEOUT = "proxy.queue_timeout_sec"
    const val BATTERY_FLOOR = "proxy.battery_floor"
    const val CHARGING_ONLY = "proxy.charging_only"
    const val REQUIRE_AUTH = "proxy.require_auth"
    const val DEBUG = "proxy.debug"

    // — enum values, named so nothing compares string literals in two places —

    const val BIND_TAILNET = "tailnet"
    const val BIND_LOOPBACK = "loopback"
    const val BIND_ALL = "all"

    const val MID_DEMOTE = "demote"
    const val MID_STRIP = "strip"
    const val MID_MERGE_TOP = "merge_top"
    const val MID_KEEP = "keep"

    const val POLICY_QUEUE = "queue"
    const val POLICY_REFUSE = "refuse"
    const val POLICY_SWAP = "swap"

    /**
     * The keys a client profile is allowed to override.
     *
     * A list rather than "anything in ALL", because a profile overriding the
     * port or the bind address would be a per-request answer to a
     * process-lifetime question — and the row would be editable and inert,
     * which is the failure this codebase is least willing to ship.
     */
    val PROFILE_OVERRIDABLE: Set<String> = setOf(
        TOOL_SEARCH, AUTO_LOAD_TOOLS, STRIP_REMINDERS, SORT_TOOLS,
        MID_SYSTEM, INJECT_DATE, MAX_ROUNDTRIPS,
    )

    // — declaration helpers, so a setting is one readable line above —

    private fun bool(
        key: String,
        default: Boolean,
        label: String,
        help: String,
        tier: Tier = Tier.BASIC,
        requiresReload: Boolean = false,
    ) = ParamSpec(
        key = key, group = GROUP, type = ParamType.BOOL,
        default = JsonPrimitive(default), tier = tier,
        label = label, help = help, requiresReload = requiresReload,
    )

    private fun int(
        key: String,
        default: Int,
        min: Int,
        max: Int,
        label: String,
        help: String,
        tier: Tier = Tier.BASIC,
        requiresReload: Boolean = false,
    ) = ParamSpec(
        key = key, group = GROUP, type = ParamType.INT,
        default = JsonPrimitive(default),
        min = min.toDouble(), max = max.toDouble(), step = 1.0,
        tier = tier, label = label, help = help, requiresReload = requiresReload,
    )

    private fun enum(
        key: String,
        default: String,
        values: List<String>,
        label: String,
        help: String,
        tier: Tier = Tier.BASIC,
        requiresReload: Boolean = false,
    ) = ParamSpec(
        key = key, group = GROUP, type = ParamType.ENUM,
        default = JsonPrimitive(default), values = values,
        tier = tier, label = label, help = help, requiresReload = requiresReload,
    )

    private fun string(
        key: String,
        default: String,
        label: String,
        help: String,
        tier: Tier = Tier.BASIC,
    ) = ParamSpec(
        key = key, group = GROUP, type = ParamType.STRING,
        default = JsonPrimitive(default), tier = tier, label = label, help = help,
    )

    private const val GROUP = "proxy"
}
