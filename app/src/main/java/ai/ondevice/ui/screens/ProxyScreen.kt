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
            Rows(
                viewModel,
                listOf(ProxySpecs.ENABLED, ProxySpecs.BIND, ProxySpecs.PORT, ProxySpecs.TLS),
            )
            if (state.config.tls) {
                CertificateCard(state, viewModel, clipboard, Modifier.padding(top = 8.dp))
            }

            SectionKicker("Staying up", Modifier.padding(top = 20.dp, bottom = 8.dp))
            ResilienceCard(state, viewModel)

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

            SectionKicker("Models", Modifier.padding(top = 20.dp, bottom = 8.dp))
            Pickers(
                viewModel,
                listOf(
                    ProxySpecs.DEFAULT_TEXT,
                    ProxySpecs.DEFAULT_IMAGE,
                    ProxySpecs.DEFAULT_VIDEO,
                    ProxySpecs.DEFAULT_VOICE,
                    ProxySpecs.TTS_VOICE,
                    ProxySpecs.DEFAULT_SPEECH,
                ),
            )
            NHelp(
                "Which model each surface uses when a request names none. Pictures and clips " +
                    "are both diffusion models, so there is one row for each — \"whichever was " +
                    "used last\" cannot be right for both at once. A request that names a model " +
                    "still wins.",
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
                addLabel = "Add a core tool",
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
                "Rewrites the model a request names before anything else looks at it — what " +
                    "lets an unmodified Claude Code or OpenAI client point here: it sends the " +
                    "name it was configured with, and this turns it into whatever is installed. " +
                    "You only need a row when the client's name is fixed. A request may also " +
                    "ask for a model by its full id, or by the name you gave it on its own " +
                    "screen.",
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
            AddButton("Add a client", onClick = viewModel::addProfile)
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
                modifier = Modifier.padding(top = 8.dp),
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

        // Both forms, each copyable, and the second is not a convenience.
        // Tailscale hands its DNS to applications one at a time on Android, so a
        // client outside that set reaches 100.x perfectly well and cannot resolve
        // a `.ts.net` name at all -- measured on this device, where the name
        // resolved for the shell and did not for the app beside it. Offering only
        // the name leaves that client with an address it cannot use and no way to
        // discover the one it can.
        state.status.url?.let { url ->
            ProxyAddressRow(url, clipboard)
        }
        state.status.addressUrl?.let { url ->
            ProxyAddressRow(url, clipboard)
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
                    "OPENAI_BASE_URL, and give it the token below." +
                    if (state.status.secure) {
                        " The certificate is this device's own, so the client also needs to be " +
                            "given it — see the card under Network."
                    } else {
                        ""
                    },
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

/**
 * The certificate, its fingerprint, and the two things anybody does with them.
 *
 * The fingerprint is the point of this card. A certificate a device signed for
 * itself is worth exactly as much as the reader's ability to check they were
 * handed the right one, and this is the only place that check can start from.
 */
@Composable
private fun CertificateCard(
    state: ai.ondevice.ui.vm.ProxyState,
    viewModel: ProxyViewModel,
    clipboard: androidx.compose.ui.platform.ClipboardManager,
    modifier: Modifier = Modifier,
) {
    val certificate = state.certificate
    NCard(gap = 8.dp, modifier = modifier) {
        Text(
            certificate?.fingerprint ?: "No certificate yet",
            style = NocturneType.MonoXs,
            color = if (certificate != null) NocturneColors.Accent300 else NocturneColors.TextMuted,
        )
        NCardBody(
            if (certificate == null) {
                "One is made the first time the server starts with HTTPS on, and kept after " +
                    "that so the fingerprint a client pinned does not move."
            } else {
                "SHA-256, and valid for " + certificate.names.joinToString(", ") + "."
            },
        )
        if (certificate != null) {
            // Copy, because the client that needs this is usually on this phone.
            // Vessel takes the certificate as pasted text and imports it into a
            // container root store, which is the store Chromium reads -- so an
            // Electron application inside the container trusts this server.
            //
            // A share sheet was here and is gone. It solved handing the file to
            // another machine, and that is not what anyone was doing with it.
            NCardBody(
                "Copy it, then paste it into the client that has to trust this server. " +
                    "In Vessel that is the container Certificates sheet; elsewhere, " +
                    "`--cacert ondevice.pem`, `NODE_EXTRA_CA_CERTS` or `REQUESTS_CA_BUNDLE`.",
            )
            NHelp(
                "Or fetch it from the server itself: curl -k " +
                    (state.status.url ?: "https://…") + "/certificate -o ondevice.pem. " +
                    "Turning the check off for that one request is not the leap it looks like " +
                    "on a tailnet — WireGuard has already proved which machine answered.",
            )
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            certificate?.let {
                NButton(
                    "Copy",
                    onClick = { clipboard.setText(AnnotatedString(it.pem)) },
                    style = NButtonStyle.Primary,
                    modifier = Modifier.weight(1f),
                )
            }
            NButton(
                if (certificate == null) "Make one now" else "Replace it",
                onClick = viewModel::regenerateCertificate,
                style = if (certificate == null) NButtonStyle.Primary else NButtonStyle.Secondary,
                modifier = Modifier.weight(1f),
            )
        }
    }
}

/**
 * What this device will let the proxy do while nobody is looking.
 *
 * **This card exists because its absence was the bug.** The API went quiet
 * overnight and there was nothing anywhere on the phone that said why: the
 * process had been killed, the restart had been refused foreground standing,
 * and the service had stopped itself with only a line in a log nobody reads.
 * Every part of that is now either fixed or said out loud, and the two things
 * that cannot be fixed from inside the app are these.
 *
 * It says nothing at all when the platform is already allowing both, which is
 * the point: a permanent row nagging about a permission that is granted is how
 * a screen teaches people to stop reading it.
 */
@Composable
private fun ResilienceCard(
    state: ai.ondevice.ui.vm.ProxyState,
    viewModel: ProxyViewModel,
) {
    val resilience = state.resilience

    // Re-read on every return to this screen. Both answers are changed out in
    // Settings, so the value this was built with is stale the moment somebody
    // acts on it -- and a card still saying "not allowed" after you allowed it
    // is worse than one that never mentioned it.
    androidx.compose.runtime.LaunchedEffect(Unit) { viewModel.refreshResilience() }

    NCard(gap = 9.dp) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            NDot(
                color = if (resilience.settled) NocturneColors.Accent else NocturneColors.Neutral500,
                size = 6.dp,
            )
            Text(
                if (resilience.settled) {
                    "Allowed to come back on its own"
                } else {
                    "This device can stop it for good"
                },
                style = NocturneType.Row,
                modifier = Modifier.weight(1f),
            )
        }

        // One line. The paragraph here explained that the system kills this app
        // for memory and that a check restarts it -- true, and an account of the
        // machinery rather than of anything to do about it. What is left is the
        // part a reader can act on; the rest is in SPEC 18.7.
        NCardBody("A check restarts the server every fifteen minutes if this device stops it.")

        if (!resilience.canRestartUnattended) {
            RestrictionRow(
                text = "Alarms & reminders is off, so the restart needs a tap.",
                action = "Allow",
                onClick = {
                    viewModel.openRestrictionSettings(ai.ondevice.ui.vm.ProxyState.Restriction.ALARMS)
                },
            )
        }

        if (!resilience.exemptFromBattery) {
            RestrictionRow(
                text = "Battery optimisation is on. Set this app to Unrestricted.",
                action = "Open",
                onClick = {
                    viewModel.openRestrictionSettings(ai.ondevice.ui.vm.ProxyState.Restriction.BATTERY)
                },
            )
        }

        if (!resilience.notificationsAllowed) {
            RestrictionRow(
                text = "Notifications are off, so it cannot tell you when it fails.",
                action = "Turn on",
                onClick = {
                    viewModel.openRestrictionSettings(
                        ai.ondevice.ui.vm.ProxyState.Restriction.NOTIFICATIONS,
                    )
                },
            )
        }
    }
}

/**
 * One thing the platform is withholding, on one line, with the tap that grants it.
 *
 * **A row appears only while it is a problem.** Nothing is drawn greyed out or
 * ticked off: a permission that has already been given has nothing to say, and a
 * row that stays behind to congratulate the reader is how this card would teach
 * them to scroll past it. So when all three are granted this whole section is
 * the heading and one sentence.
 *
 * The sentence says what to do and stops. Each of these used to open with a
 * paragraph explaining the failure it prevents, which is an argument aimed at
 * somebody who has not had that failure and is reading a settings screen. The
 * arguments are in SPEC 18.7.
 */
@Composable
private fun RestrictionRow(
    text: String,
    action: String,
    onClick: () -> Unit,
) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        NCardBody(text, Modifier.weight(1f))
        NButton(action, onClick = onClick, style = NButtonStyle.Secondary)
    }
}

/** One address the server answers on, with the button that copies it. */
@Composable
private fun ProxyAddressRow(
    url: String,
    clipboard: androidx.compose.ui.platform.ClipboardManager,
) {
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
            "Copy $url",
            onClick = { clipboard.setText(AnnotatedString(url)) },
            size = 30.dp,
            iconSize = 14.dp,
        )
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

/**
 * The default-model rows.
 *
 * Separate from [Rows] only because their options come from the device rather
 * than from the spec — the control, the label and the key line are the same
 * `ParamRow` everything else uses, and the screen still knows nothing about any
 * particular key.
 */
@Composable
private fun Pickers(viewModel: ProxyViewModel, keys: List<String>) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    NCard(gap = 0.dp, padding = PaddingValues(horizontal = 12.dp, vertical = 2.dp)) {
        keys.forEach { key ->
            state.picker(key)?.let { spec ->
                ParamRow(
                    spec = spec,
                    values = state.settings,
                    onChange = viewModel::set,
                    pathChoices = state.choices(key),
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
                    placeholder = "owner/repo:quant",
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
                style = NButtonStyle.Primary,
                enabled = pending.isNotBlank(),
                minHeight = 36.dp,
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
        AddButton("Add an origin", onClick = viewModel::addCorsOrigin)
    }
}

/** A plain list of names, edited in place. */
@Composable
private fun StringList(
    values: List<String>,
    placeholder: String,
    addLabel: String,
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
        AddButton(addLabel) { onChange(values + "") }
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
            block = true,
        )
    }
}

/**
 * The one shape every "add another of these" button on this screen takes.
 *
 * Full width and accent-outlined, which is what the token button beside it
 * already looked like. They were content-width with a small plus glyph, and at
 * that size the three of them read as incidental links wedged under their lists
 * rather than as the action each section exists for — the odd one out being the
 * only button on the screen that mattered less than the rest.
 *
 * One composable rather than the same three arguments repeated, so the next
 * list to gain an Add cannot be spelled differently.
 */
@Composable
private fun AddButton(text: String, onClick: () -> Unit) {
    NButton(
        text,
        onClick = onClick,
        style = NButtonStyle.Primary,
        block = true,
    )
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
