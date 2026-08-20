package ai.ondevice.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.params.ParamRow
import ai.ondevice.proxy.ProxyProfile
import ai.ondevice.proxy.ProxySpecs
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCardBody
import ai.ondevice.ui.components.NDot
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NIconButton
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NSwitch
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.vm.ProxyViewModel

/**
 * Settings → Proxy.
 *
 * Every row below the status card is a [ParamRow] over a [ProxySpecs] entry —
 * label, the value in mono accent, the key on its own line, the type-chosen
 * control, the help paragraph and a "reset to …" link, all of it drawn by the
 * same code the model parameters use. There is deliberately no `when` on a key
 * name anywhere in this file: adding a setting means adding a line to
 * `ProxySpecs`, and nothing here changes.
 */
@Composable
fun ProxyScreen(
    onBack: () -> Unit,
    onOpenLog: () -> Unit,
    viewModel: ProxyViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val clipboard = LocalClipboardManager.current

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = "Proxy",
                subtitle = when {
                    state.status.listening -> "${state.status.url} · ${state.requests.count { it.inFlight }} live"
                    state.status.refusal != null -> "not listening"
                    else -> "off"
                },
                subtitleMono = state.status.listening,
                onBack = onBack,
                trailing = {
                    NIconButton(
                        NIcons.Rotate,
                        "Re-check reachability",
                        onClick = viewModel::refreshReachability,
                        size = 32.dp,
                        iconSize = 15.dp,
                    )
                },
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            StatusCard(state, viewModel, clipboard)

            SectionKicker("Access", Modifier.padding(top = 20.dp, bottom = 8.dp))
            TokenCard(state, viewModel, clipboard)
            Rows(viewModel, listOf(ProxySpecs.REQUIRE_AUTH))

            SectionKicker("Network", Modifier.padding(top = 20.dp, bottom = 8.dp))
            Rows(viewModel, listOf(ProxySpecs.ENABLED, ProxySpecs.BIND, ProxySpecs.PORT))

            SectionKicker("Protocols", Modifier.padding(top = 20.dp, bottom = 8.dp))
            Rows(viewModel, listOf(ProxySpecs.PROTOCOL_ANTHROPIC, ProxySpecs.PROTOCOL_OPENAI))

            SectionKicker("Surfaces", Modifier.padding(top = 20.dp, bottom = 8.dp))
            Rows(
                viewModel,
                listOf(ProxySpecs.SERVE_IMAGES, ProxySpecs.SERVE_AUDIO, ProxySpecs.SERVE_VIDEO),
            )
            NHelp(
                "The Anthropic protocol has no image or audio endpoints — it is a chat API and " +
                    "nothing else. Those surfaces reach it as tools the model can call instead, " +
                    "which is the only shape that endpoint has.",
                Modifier.padding(top = 6.dp),
            )

            SectionKicker("Behaviour", Modifier.padding(top = 20.dp, bottom = 8.dp))
            Rows(
                viewModel,
                listOf(
                    ProxySpecs.TOOL_SEARCH,
                    ProxySpecs.AUTO_LOAD_TOOLS,
                    ProxySpecs.STRIP_REMINDERS,
                    ProxySpecs.SORT_TOOLS,
                    ProxySpecs.MID_SYSTEM,
                    ProxySpecs.INJECT_DATE,
                    ProxySpecs.LOCATION,
                    ProxySpecs.MODEL_POLICY,
                ),
            )

            SectionKicker("Limits", Modifier.padding(top = 20.dp, bottom = 8.dp))
            Rows(
                viewModel,
                listOf(
                    ProxySpecs.MAX_ROUNDTRIPS,
                    ProxySpecs.PING_INTERVAL,
                    ProxySpecs.QUEUE_DEPTH,
                    ProxySpecs.QUEUE_TIMEOUT,
                    ProxySpecs.BATTERY_FLOOR,
                    ProxySpecs.CHARGING_ONLY,
                ),
            )

            SectionKicker("Core tools", Modifier.padding(top = 20.dp, bottom = 8.dp))
            StringList(
                values = state.document.coreTools,
                placeholder = "Read",
                onChange = viewModel::setCoreTools,
            )
            NHelp(
                "Names that stay loaded when tool search is on. Everything else is held back " +
                    "and retrieved by the model when it needs it. Empty means everything is held " +
                    "back, which is right for a small context and wrong for a chatty one.",
                Modifier.padding(top = 6.dp),
            )

            SectionKicker("Model mapping", Modifier.padding(top = 20.dp, bottom = 8.dp))
            AliasList(state, viewModel)
            NHelp(
                "Rewrites the model a request names before anything else looks at it. This is " +
                    "what lets an unmodified Claude Code or OpenAI client point at this phone: " +
                    "it sends the name it was configured with, and this turns it into whatever " +
                    "is actually installed.",
                Modifier.padding(top = 6.dp),
            )

            SectionKicker("Browser origins", Modifier.padding(top = 20.dp, bottom = 8.dp))
            OriginList(state, viewModel)
            NHelp(
                "Empty means no browser page may call this server, which is the safe default. " +
                    "Anything listed here can drive this device's models from a web page.",
                Modifier.padding(top = 6.dp),
            )

            SectionKicker("Clients", Modifier.padding(top = 20.dp, bottom = 8.dp))
            state.document.profiles.forEachIndexed { index, profile ->
                ProfileCard(
                    profile = profile,
                    expanded = state.expandedProfile == profile.name,
                    onExpand = {
                        viewModel.expandProfile(
                            if (state.expandedProfile == profile.name) null else profile.name,
                        )
                    },
                    onChange = { transform -> viewModel.updateProfile(index, transform) },
                    onOverride = { key, value -> viewModel.setProfileOverride(index, key, value) },
                    onRemove = { viewModel.removeProfile(index) },
                    modifier = Modifier.padding(bottom = 8.dp),
                )
            }
            NButton("Add a client", onClick = viewModel::addProfile, leadingIcon = NIcons.PlusThin)
            NHelp(
                "Per-client answers, matched on a request header — a User-Agent containing " +
                    "\"claude-cli\", a Referer containing your own page. A token matches first " +
                    "where one is set, because a token is proof and a header is a hint.",
                Modifier.padding(top = 6.dp),
            )

            SectionKicker("Diagnostics", Modifier.padding(top = 20.dp, bottom = 8.dp))
            Rows(viewModel, listOf(ProxySpecs.DEBUG))
            NButton(
                "Recent requests",
                onClick = onOpenLog,
                modifier = Modifier.padding(top = 8.dp).fillMaxWidth(),
                block = true,
            )

            NHelp(
                "Traffic to this server never leaves the tailnet, and this app makes no outbound " +
                    "call because of it. What a client asks for is run here, on this device.",
                Modifier.padding(top = 20.dp),
            )
        }
    }
}

/**
 * What the server is doing, and where to point a client.
 *
 * The Funnel sentence is not a caveat added for completeness: telecode's
 * equivalent card shows a public `https://…ts.net` URL, that is the shape people
 * expect, and it is not available on Android at all. Saying so here is cheaper
 * than the half hour somebody would otherwise spend looking for the setting.
 */
@Composable
private fun StatusCard(
    state: ai.ondevice.ui.vm.ProxyState,
    viewModel: ProxyViewModel,
    clipboard: androidx.compose.ui.platform.ClipboardManager,
) {
    NCard(gap = 9.dp) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            NDot(
                color = when {
                    state.status.listening -> NocturneColors.Accent
                    state.status.refusal != null -> NocturneColors.Neutral500
                    else -> NocturneColors.Neutral700
                },
                size = 6.dp,
            )
            Text(
                when {
                    state.status.listening -> "Listening"
                    state.status.refusal != null -> "Not listening"
                    else -> "Off"
                },
                style = NocturneType.Row,
                modifier = Modifier.weight(1f),
            )
            if (state.config.anthropicEnabled) NTag("Anthropic", style = NTagStyle.Accent)
            if (state.config.openAiEnabled) NTag("OpenAI", style = NTagStyle.Accent2)
        }

        state.status.url?.let { url ->
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    url,
                    style = NocturneType.MonoValue,
                    color = NocturneColors.Accent300,
                    modifier = Modifier.weight(1f),
                )
                NIconButton(
                    NIcons.Copy,
                    "Copy the address",
                    onClick = { clipboard.setText(AnnotatedString(url)) },
                    size = 30.dp,
                    iconSize = 14.dp,
                )
            }
        }

        state.status.refusal?.let { NCardBody(it) }

        if (state.status.listening) {
            NCardBody(
                if (state.status.onTailnet) {
                    "Reachable from your other Tailscale machines, and from nowhere else. " +
                        "Not from the public internet: Tailscale Funnel needs the command-line " +
                        "client, which the Android app does not ship, so there is no public " +
                        "HTTPS address to give you and no setting that would produce one."
                } else {
                    "Bound to a local address. Anything on this network can reach it, which " +
                        "includes whatever else is on this Wi-Fi."
                },
            )
            NHelp(
                "Point Claude Code here with ANTHROPIC_BASE_URL, or an OpenAI client with " +
                    "OPENAI_BASE_URL, and give it the token below.",
            )
        }

        if (!state.status.listening && state.status.refusal == null && state.config.enabled) {
            NCardBody("Turn on Serve the API below, then this card says where to point a client.")
        }

        state.tailnetAddress?.takeIf { !state.status.listening }?.let { address ->
            NHelp("Tailscale is connected at $address.")
        }
    }
}

@Composable
private fun TokenCard(
    state: ai.ondevice.ui.vm.ProxyState,
    viewModel: ProxyViewModel,
    clipboard: androidx.compose.ui.platform.ClipboardManager,
) {
    NCard(gap = 8.dp) {
        Text(
            state.revealedToken ?: state.maskedToken ?: "No token yet",
            style = NocturneType.MonoValue,
            color = if (state.maskedToken != null) NocturneColors.Accent300 else NocturneColors.TextMuted,
        )
        NCardBody(
            if (state.revealedToken != null) {
                "This is the only time it is shown in full. Copy it now — after this it is masked, " +
                    "and the only way back is a new one."
            } else {
                "Sent as `Authorization: Bearer …` or as `x-api-key`; both are accepted, because " +
                    "the two protocols disagree about which to send and this server speaks both."
            },
        )
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            state.revealedToken?.let { token ->
                NButton(
                    "Copy",
                    onClick = {
                        clipboard.setText(AnnotatedString(token))
                        viewModel.dismissRevealedToken()
                    },
                    style = NButtonStyle.Primary,
                    modifier = Modifier.weight(1f),
                )
            }
            NButton(
                if (state.maskedToken == null) "Generate a token" else "Replace it",
                onClick = viewModel::regenerateToken,
                style = if (state.maskedToken == null) NButtonStyle.Primary else NButtonStyle.Secondary,
                modifier = Modifier.weight(1f),
            )
        }
    }
}

/** A block of [ParamRow]s. The only thing this file knows about any key is its name. */
@Composable
private fun Rows(viewModel: ProxyViewModel, keys: List<String>) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    NCard(gap = 0.dp, padding = PaddingValues(horizontal = 12.dp, vertical = 2.dp)) {
        keys.forEach { key ->
            state.spec(key)?.let { spec ->
                ParamRow(
                    spec = spec,
                    values = state.settings,
                    onChange = viewModel::set,
                )
            }
        }
    }
}

/** `alias = target`, the shape telecode's panel uses and the one people read it in. */
@Composable
private fun AliasList(state: ai.ondevice.ui.vm.ProxyState, viewModel: ProxyViewModel) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        state.document.aliases.forEach { (alias, target) ->
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                NInput(
                    value = alias,
                    onValueChange = { viewModel.setAlias(it, target, replacing = alias) },
                    modifier = Modifier.weight(1f),
                    textStyle = NocturneType.MonoSm,
                    code = true,
                )
                Text("=", style = NocturneType.MonoValue, color = NocturneColors.TextMuted)
                NInput(
                    value = target,
                    onValueChange = { viewModel.setAlias(alias, it) },
                    modifier = Modifier.weight(1f),
                    textStyle = NocturneType.MonoSm,
                    code = true,
                )
                NIconButton(
                    NIcons.Cross,
                    "Remove",
                    onClick = { viewModel.removeAlias(alias) },
                    size = 30.dp,
                    iconSize = 12.dp,
                    style = NButtonStyle.Ghost,
                )
            }
        }
        var pending by rememberSaveable { mutableStateOf("") }
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            NInput(
                value = pending,
                onValueChange = { pending = it },
                placeholder = "claude-sonnet-4-6",
                modifier = Modifier.weight(1f),
                textStyle = NocturneType.MonoSm,
                code = true,
            )
            NButton(
                "Add",
                onClick = {
                    if (pending.isNotBlank()) {
                        viewModel.setAlias(pending.trim(), "")
                        pending = ""
                    }
                },
                enabled = pending.isNotBlank(),
            )
        }
    }
}

@Composable
private fun OriginList(state: ai.ondevice.ui.vm.ProxyState, viewModel: ProxyViewModel) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        state.document.corsOrigins.forEachIndexed { index, origin ->
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                NInput(
                    value = origin,
                    onValueChange = { viewModel.setCorsOrigin(index, it) },
                    placeholder = "https://example.com",
                    modifier = Modifier.weight(1f),
                    textStyle = NocturneType.MonoSm,
                    code = true,
                )
                NIconButton(
                    NIcons.Cross,
                    "Remove",
                    onClick = { viewModel.removeCorsOrigin(index) },
                    size = 30.dp,
                    iconSize = 12.dp,
                    style = NButtonStyle.Ghost,
                )
            }
        }
        NButton("Add an origin", onClick = viewModel::addCorsOrigin, leadingIcon = NIcons.PlusThin)
    }
}

/** A plain list of names, edited in place. */
@Composable
private fun StringList(
    values: List<String>,
    placeholder: String,
    onChange: (List<String>) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        values.forEachIndexed { index, value ->
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                NInput(
                    value = value,
                    onValueChange = { edited ->
                        onChange(values.toMutableList().also { it[index] = edited })
                    },
                    placeholder = placeholder,
                    modifier = Modifier.weight(1f),
                    textStyle = NocturneType.MonoSm,
                    code = true,
                )
                NIconButton(
                    NIcons.Cross,
                    "Remove",
                    onClick = { onChange(values.filterIndexed { i, _ -> i != index }) },
                    size = 30.dp,
                    iconSize = 12.dp,
                    style = NButtonStyle.Ghost,
                )
            }
        }
        NButton(
            "Add",
            onClick = { onChange(values + "") },
            leadingIcon = NIcons.PlusThin,
        )
    }
}

/**
 * One client, collapsed to its match rule until opened.
 *
 * The same shape the Tools screen uses for a provider, and for the same reason:
 * a phone screen cannot hold five expanded profiles, and the match rule is what
 * identifies one at a glance.
 */
@Composable
private fun ProfileCard(
    profile: ProxyProfile,
    expanded: Boolean,
    onExpand: () -> Unit,
    onChange: ((ProxyProfile) -> ProxyProfile) -> Unit,
    onOverride: (String, Any?) -> Unit,
    onRemove: () -> Unit,
    modifier: Modifier = Modifier,
) {
    NCard(modifier = modifier, gap = 9.dp) {
        Row(
            Modifier.fillMaxWidth().nClickableFlat(onClick = onExpand),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(profile.name.ifBlank { "unnamed" }, style = NocturneType.CardTitle)
                Text(
                    if (profile.matchContains.isBlank()) {
                        "matches nothing yet"
                    } else {
                        "${profile.matchHeader} contains \"${profile.matchContains}\""
                    },
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                )
            }
            Icon(
                if (expanded) NIcons.ChevronDown else NIcons.ChevronLeft,
                contentDescription = null,
                tint = NocturneColors.TextMuted,
                modifier = Modifier.size(15.dp),
            )
        }

        if (!expanded) return@NCard

        LabelledInput("Name", profile.name) { edited -> onChange { it.copy(name = edited) } }
        LabelledInput("Header", profile.matchHeader) { edited ->
            onChange { it.copy(matchHeader = edited) }
        }
        LabelledInput("Contains", profile.matchContains) { edited ->
            onChange { it.copy(matchContains = edited) }
        }
        LabelledInput("Token (optional)", profile.token) { edited ->
            onChange { it.copy(token = edited) }
        }
        LabelledInput("System instruction", profile.instruction) { edited ->
            onChange { it.copy(instruction = edited) }
        }

        Text(
            "Overrides",
            style = NocturneType.SectionKicker,
            color = NocturneColors.Neutral500,
            modifier = Modifier.padding(top = 4.dp),
        )
        // Only the keys a profile may sensibly answer differently. Port and
        // bind are not among them: a per-request answer to a process-lifetime
        // question would be a row that is editable and inert.
        ProxySpecs.ALL
            .filter { it.key in ProxySpecs.PROFILE_OVERRIDABLE }
            .forEach { spec ->
                ParamRow(
                    spec = spec,
                    values = profile.overrides,
                    onChange = onOverride,
                )
            }

        NButton(
            "Remove this client",
            onClick = onRemove,
            style = NButtonStyle.Ghost,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun LabelledInput(label: String, value: String, onChange: (String) -> Unit) {
    Column {
        Text(
            label,
            style = NocturneType.FieldLabel,
            color = NocturneColors.TextLabel,
            modifier = Modifier.padding(bottom = 4.dp),
        )
        NInput(
            value = value,
            onValueChange = onChange,
            textStyle = NocturneType.MonoSm,
            code = true,
        )
    }
}
