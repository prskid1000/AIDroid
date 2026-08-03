package ai.ondevice.ui.screens

import androidx.compose.foundation.background
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
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.data.db.McpServerEntity
import ai.ondevice.tools.McpTools
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NInput
import ai.ondevice.params.ParamRow
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
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.ToolsViewModel

@Composable
fun ToolsScreen(
    onBack: () -> Unit,
    viewModel: ToolsViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = "Tools",
                // The live count, not the installed one: a paused server's
                // tools are not offered however many it has.
                subtitle = if (state.enabled) {
                    "${state.liveToolCount} offered to the model"
                } else {
                    "off — the model is not told any exist"
                },
                subtitleMono = false,
                onBack = onBack,
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            NCard(gap = 9.dp) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(9.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("Let the model call tools", style = NocturneType.Row, modifier = Modifier.weight(1f))
                    NSwitch(state.enabled, viewModel::setEnabled)
                }
                Text(
                    "Off by default. A model that is never told tools exist cannot call one — which is " +
                        "the safe default when some of them reach a server you do not control.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            }

            // — built-ins —
            SectionKicker("Built in", Modifier.padding(top = 20.dp, bottom = 8.dp))
            ProviderRow(
                title = "Built-in",
                tools = state.builtInTools,
                on = state.builtInEnabled,
                onToggle = { viewModel.toggleProvider(ai.ondevice.tools.BuiltInToolProvider.ID) },
                settings = state.settingsFor(ai.ondevice.tools.BuiltInToolProvider.ID, state.builtInTools),
                values = state.tuning,
                expanded = state.expandedProviderId == ai.ondevice.tools.BuiltInToolProvider.ID,
                onExpand = {
                    viewModel.expandProvider(
                        if (state.expandedProviderId == ai.ondevice.tools.BuiltInToolProvider.ID) null else ai.ondevice.tools.BuiltInToolProvider.ID,
                    )
                },
                onSetting = viewModel::setToolParam,
            )
            NHelp(
                "The clock it cannot read, arithmetic it gets subtly wrong, and the state of the " +
                    "device it is running on — none of which touches the network. web_search and " +
                    "fetch_url do: they fetch search.brave.com, and the pages it is asked to open, " +
                    "with no key and no account. Your query goes out; the conversation does not.",
                Modifier.padding(top = 6.dp),
            )

            // — files —
            SectionKicker("Files", Modifier.padding(top = 20.dp, bottom = 8.dp))
            ProviderRow(
                title = "Filesystem",
                tools = state.fileTools,
                on = state.filesEnabled,
                onToggle = { viewModel.toggleProvider(ai.ondevice.tools.FileToolProvider.ID) },
                settings = state.settingsFor(ai.ondevice.tools.FileToolProvider.ID, state.fileTools),
                values = state.tuning,
                expanded = state.expandedProviderId == ai.ondevice.tools.FileToolProvider.ID,
                onExpand = {
                    viewModel.expandProvider(
                        if (state.expandedProviderId == ai.ondevice.tools.FileToolProvider.ID) null else ai.ondevice.tools.FileToolProvider.ID,
                    )
                },
                onSetting = viewModel::setToolParam,
            )
            NHelp(
                "Reading, writing and searching files, the way an editor does — an exact-match edit " +
                    "rather than a whole-file rewrite, so a change that does not match is refused " +
                    "instead of guessed at.",
                Modifier.padding(top = 6.dp),
            )

            if (state.filesEnabled || state.shellEnabled) {
                val context = androidx.compose.ui.platform.LocalContext.current
                val granted = ai.ondevice.tools.Workspace.hasAllFilesAccess(context)
                NCard(gap = 9.dp, modifier = Modifier.padding(top = 10.dp)) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(9.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            "Reach the whole device",
                            style = NocturneType.Row,
                            modifier = Modifier.weight(1f),
                        )
                        NSwitch(state.fileScopeDevice, { want ->
                            viewModel.setFileScopeDevice(want)
                            // The switch records the intent; the grant itself is
                            // a system screen, and there is no way to ask for it
                            // in-app. Sending them straight there is the only
                            // thing that makes the switch mean anything.
                            if (want && !granted) {
                                runCatching {
                                    context.startActivity(
                                        android.content.Intent(
                                            android.provider.Settings
                                                .ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                                            android.net.Uri.parse("package:${context.packageName}"),
                                        ),
                                    )
                                }
                            }
                        })
                    }
                    Text(
                        when {
                            !state.fileScopeDevice ->
                                "Off — the tools see this app's own folders only: its workspace, and " +
                                    "the models, gallery and clips beside them. Nothing else on the " +
                                    "phone is reachable, and no permission is needed."
                            granted ->
                                "On — Downloads, Documents, Pictures and the rest of internal " +
                                    "storage are readable and writable."
                            else ->
                                "Asked for, but Android has not granted it. Open All files access " +
                                    "for this app in system Settings, or turn this off — until then " +
                                    "the tools stay in the app's own folders."
                        },
                        style = NocturneType.CardBody,
                        color = if (state.fileScopeDevice && !granted) {
                            NocturneColors.Accent300
                        } else {
                            NocturneColors.Text.copy(alpha = 0.8f)
                        },
                    )
                }
            }

            // — shell —
            SectionKicker("Shell", Modifier.padding(top = 20.dp, bottom = 8.dp))
            ProviderRow(
                title = "Shell",
                tools = state.shellTools,
                on = state.shellEnabled,
                onToggle = { viewModel.toggleProvider(ai.ondevice.tools.ShellToolProvider.ID) },
                settings = state.settingsFor(ai.ondevice.tools.ShellToolProvider.ID, state.shellTools),
                values = state.tuning,
                expanded = state.expandedProviderId == ai.ondevice.tools.ShellToolProvider.ID,
                onExpand = {
                    viewModel.expandProvider(
                        if (state.expandedProviderId == ai.ondevice.tools.ShellToolProvider.ID) null else ai.ondevice.tools.ShellToolProvider.ID,
                    )
                },
                onSetting = viewModel::setToolParam,
            )
            NHelp(
                "Runs commands with the system shell, in the same folders the file tools use. " +
                    "Android ships mksh and toybox — sed, grep, find, sort, tar and about 200 more — " +
                    "and no awk, python or node. Every command is listed below with its exit code.",
                Modifier.padding(top = 6.dp),
            )

            if (state.shellRuns.isNotEmpty()) {
                Row(
                    Modifier.fillMaxWidth().padding(top = 14.dp, bottom = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    SectionKicker("Last ${state.shellRuns.size} commands", Modifier.weight(1f))
                    Text(
                        "clear",
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMuted,
                        modifier = Modifier.nClickableFlat { viewModel.clearShellLog() },
                    )
                }
                state.shellRuns.take(SHELL_LOG_SHOWN).forEach { run ->
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .background(NocturneColors.Surface, Radius.Md)
                            .ring(NocturneColors.Divider, Radius.Md)
                            .padding(horizontal = 12.dp, vertical = 9.dp)
                            .padding(bottom = 1.dp),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("$ ${run.command}", style = NocturneType.MonoXs)
                            if (run.summary.isNotBlank()) {
                                Text(
                                    run.summary,
                                    style = NocturneType.MonoXs,
                                    color = NocturneColors.TextMuted,
                                )
                            }
                        }
                        Text("${run.millis} ms", style = NocturneType.MonoXs, color = NocturneColors.TextMuted)
                        NTag(
                            when (run.exitCode) {
                                0 -> "ok"
                                ai.ondevice.tools.ShellToolProvider.TIMED_OUT -> "timeout"
                                else -> "exit ${run.exitCode}"
                            },
                            style = if (run.exitCode == 0) NTagStyle.Outline else NTagStyle.Accent,
                        )
                    }
                    androidx.compose.foundation.layout.Spacer(Modifier.padding(top = 5.dp))
                }
            }

            // — MCP —
            SectionKicker("MCP servers", Modifier.padding(top = 20.dp, bottom = 8.dp))

            if (state.servers.isEmpty()) {
                NHelp("None added. Everything below this point stays on the device until you add one.")
            }

            state.servers.forEach { server ->
                ServerCard(
                    server = server,
                    expanded = state.expandedServerId == server.id,
                    busy = state.testing,
                    onPause = { viewModel.pauseServer(server, it) },
                    onExpand = {
                        viewModel.expandServer(if (state.expandedServerId == server.id) null else server.id)
                    },
                    onToggleTool = { viewModel.toggleTool(server, it) },
                    onRefresh = { viewModel.refresh(server) },
                    onRemove = { viewModel.removeServer(server.id) },
                    authorized = server.id in state.authorizedServers,
                    authorizing = state.authorizingServerId == server.id,
                    onAuthorize = { viewModel.authorize(server) },
                    onSignOut = { viewModel.signOut(server) },
                    onSetClientId = { viewModel.setServerClientId(server, it) },
                    onSetClientSecret = { viewModel.setServerClientSecret(server, it) },
                )
            }

            NCard(gap = 8.dp) {
                Text("Add a server", style = NocturneType.CardTitleSm)
                NInput(
                    value = state.draftName,
                    onValueChange = { viewModel.setDraft(name = it) },
                    placeholder = "Name (optional)",
                    minHeight = 42.dp,
                )
                NInput(
                    value = state.draftUrl,
                    onValueChange = { viewModel.setDraft(url = it) },
                    placeholder = "https://example.com/mcp",
                    textStyle = NocturneType.MonoCode,
                    minHeight = 42.dp,
                )
                NInput(
                    value = state.draftAuth,
                    onValueChange = { viewModel.setDraft(auth = it) },
                    placeholder = "Authorization header (optional)",
                    textStyle = NocturneType.MonoCode,
                    minHeight = 42.dp,
                )

                // Closed by default. Most servers need nothing in here — the
                // app registers itself — and two empty boxes above the button
                // would read as two more things to fill in.
                Row(
                    Modifier
                        .fillMaxWidth()
                        .nClickableFlat { viewModel.toggleDraftAdvanced() }
                        .padding(vertical = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        NIcons.ChevronDown,
                        contentDescription = null,
                        tint = NocturneColors.Accent2,
                        modifier = Modifier
                            .size(14.dp)
                            .rotate(if (state.draftAdvancedOpen) 180f else 0f),
                    )
                    Text("Advanced settings", style = NocturneType.Row, color = NocturneColors.Accent2)
                }

                if (state.draftAdvancedOpen) {
                    NInput(
                        value = state.draftClientId,
                        onValueChange = { viewModel.setDraft(clientId = it) },
                        placeholder = "OAuth client ID (optional)",
                        textStyle = NocturneType.MonoCode,
                        minHeight = 42.dp,
                    )
                    NInput(
                        value = state.draftClientSecret,
                        onValueChange = { viewModel.setDraft(clientSecret = it) },
                        placeholder = "OAuth client secret (optional)",
                        textStyle = NocturneType.MonoCode,
                        minHeight = 42.dp,
                    )
                    NHelp(
                        "Only for a server that will not let apps register themselves. " +
                            "Left empty, the app asks the server for a client ID of its own.",
                    )
                }
                state.draftError?.let { error ->
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            NIcons.TriangleAlert,
                            contentDescription = null,
                            tint = NocturneColors.Neutral300,
                            modifier = Modifier.size(14.dp),
                        )
                        Text(error, style = NocturneType.Help, modifier = Modifier.weight(1f))
                    }
                }
                NButton(
                    if (state.testing) "Testing…" else "Test and add",
                    onClick = viewModel::addServer,
                    style = NButtonStyle.Primary,
                    block = true,
                )
            }

            NHelp(
                "HTTP transport only. MCP's other transport launches a process, which Android's W^X " +
                    "rules make either impossible or a way to get executable code onto the device — so " +
                    "this app does not offer it. A server is added only once it has answered and " +
                    "listed its tools.",
                Modifier.padding(top = 12.dp),
            )
            NHelp(
                "https:// everywhere except this device: plain http:// is allowed to localhost and " +
                    "127.0.0.1, because a server you run yourself is the common case and that traffic " +
                    "never leaves the handset. A LAN address is not — an MCP request carries whatever " +
                    "the model decided to send.",
                Modifier.padding(top = 8.dp),
            )
            NHelp(
                "A tool result is data, not instruction. Nothing in this app treats what a server " +
                    "returns as a command.",
                Modifier.padding(top = 8.dp),
            )
        }
    }
}

/** One server: paused or not, and — when opened — every tool it offers with its own switch. */
/**
 * What each of the app's own tool sets looks like: name, its tools, on or off,
 * and — when it has any — the settings its tools expose.
 *
 * The settings are [ai.ondevice.params.ParamSpec] and render through the same
 * [ParamRow] the model parameters use, so a slider here behaves exactly like a
 * slider on All Parameters: same clamping, same nudge buttons, same "unset
 * means the default" rule.
 */
@Composable
private fun ProviderRow(
    title: String,
    tools: List<String>,
    on: Boolean,
    onToggle: () -> Unit,
    settings: List<ai.ondevice.params.ParamSpec> = emptyList(),
    values: ai.ondevice.core.SparseParams = ai.ondevice.core.SparseParams.EMPTY,
    expanded: Boolean = false,
    onExpand: () -> Unit = {},
    onSetting: (String, Any?) -> Unit = { _, _ -> },
) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(if (on) NocturneColors.Accent900 else NocturneColors.Surface, Radius.Md)
            .ring(if (on) NocturneColors.Accent else NocturneColors.Divider, Radius.Md),
    ) {
        Row(
            Modifier
                .fillMaxWidth()
                .nClickableFlat(onClick = onToggle)
                .padding(horizontal = 12.dp, vertical = 11.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(title, style = NocturneType.CardTitleSm)
                Text(
                    tools.joinToString(" · "),
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                )
            }
            if (settings.isNotEmpty()) {
                // Its own hit target, because tapping the row means "on or
                // off" and tapping this means "show me the numbers" — one
                // control doing both would make every settings peek a toggle.
                Text(
                    if (expanded) "hide" else "settings",
                    style = NocturneType.MonoXs,
                    color = NocturneColors.Accent2,
                    modifier = Modifier
                        .nClickableFlat(onClick = onExpand)
                        .padding(horizontal = 6.dp, vertical = 4.dp),
                )
            }
            NTag(if (on) "on" else "off", style = if (on) NTagStyle.Accent else NTagStyle.Outline)
        }

        if (expanded && settings.isNotEmpty()) {
            Column(Modifier.padding(start = 12.dp, end = 12.dp, bottom = 11.dp)) {
                settings.groupBy { it.group }.forEach { (tool, rows) ->
                    Text(
                        tool,
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMuted,
                        modifier = Modifier.padding(top = 10.dp, bottom = 4.dp),
                    )
                    rows.forEach { spec ->
                        ParamRow(
                            spec = spec,
                            values = values,
                            onChange = onSetting,
                            showKeyLine = false,
                        )
                    }
                }
            }
        }
    }
}

/** Enough to see what just happened without turning the screen into a terminal. */
private const val SHELL_LOG_SHOWN = 25


@Composable
private fun ServerCard(
    server: McpServerEntity,
    expanded: Boolean,
    busy: Boolean,
    onPause: (Boolean) -> Unit,
    onExpand: () -> Unit,
    onToggleTool: (String) -> Unit,
    onRefresh: () -> Unit,
    onRemove: () -> Unit,
    authorized: Boolean,
    authorizing: Boolean,
    onAuthorize: () -> Unit,
    onSignOut: () -> Unit,
    onSetClientId: (String) -> Unit,
    onSetClientSecret: (String) -> Unit,
) {
    val tools = remember(server.lastToolsJson) { McpTools.parse(server.lastToolsJson) }
    val disabled = remember(server.disabledToolsJson) { McpTools.disabled(server) }
    val live = tools.count { it.name !in disabled }

    NCard(
        Modifier.padding(bottom = 8.dp),
        ring = if (server.enabled) NocturneColors.Accent700 else NocturneColors.Divider,
    ) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(9.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(server.name, style = NocturneType.CardTitleSm)
                Text(
                    server.url,
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                    maxLines = 1,
                    overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                )
            }
            NSwitch(checked = server.enabled, onCheckedChange = onPause)
        }

        // Sign-in, when the server wants one.
        //
        // Shown whenever it is signed in *or* it has OAuth endpoints on file —
        // the endpoints only get there by a 401 having sent us discovering, so
        // their presence is the record that this server asked. A server behind
        // a pasted header never shows any of this.
        // MCP's fourth option: ask the person.
        //
        // Only while signed out, and only once the server has shown it wants
        // OAuth — before that these two fields would be a question about
        // nothing. Held in local state so typing does not wait on a round trip
        // through Room, and written on every change so there is no Save to
        // forget.
        if (!authorized && server.oauthAuthorizeEndpoint != null) {
            var clientId by rememberSaveable(server.id) {
                mutableStateOf(server.oauthClientId.orEmpty())
            }
            Column(
                Modifier.fillMaxWidth().padding(top = 10.dp),
                verticalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                NInput(
                    value = clientId,
                    onValueChange = { clientId = it; onSetClientId(it) },
                    placeholder = "OAuth client ID (optional)",
                    textStyle = NocturneType.MonoCode,
                    minHeight = 42.dp,
                )
                NHelp(
                    "Leave the ID empty and the app registers itself. Fill in one " +
                        "you were issued to skip registration — which is what a " +
                        "server that will not let apps register themselves needs. " +
                        "A secret, if the server issues one, is set when the server is " +
                        "added and is not shown again.",
                )
            }
        }

        if (authorized || server.oauthAuthorizeEndpoint != null) {
            Row(
                Modifier.fillMaxWidth().padding(top = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(9.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text(
                        if (authorized) "Signed in" else "Needs you to sign in",
                        style = NocturneType.Row,
                    )
                    server.oauthIssuer?.takeIf { it.isNotBlank() }?.let { issuer ->
                        Text(
                            issuer,
                            style = NocturneType.MonoXs,
                            color = NocturneColors.TextMuted,
                            maxLines = 1,
                            overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                        )
                    }
                }
                NButton(
                    text = when {
                        authorizing -> "Opening…"
                        authorized -> "Sign out"
                        else -> "Authorize"
                    },
                    onClick = if (authorized) onSignOut else onAuthorize,
                    style = if (authorized) NButtonStyle.Ghost else NButtonStyle.Primary,
                    enabled = !authorizing,
                )
            }
        }

        // The count is of what is *offered*, and says so when that is fewer than the server has — a switched-off tool is the kind of thing you forget you did.
        Row(
            Modifier
                .fillMaxWidth()
                .nClickableFlat(onClick = onExpand)
                .padding(vertical = 2.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                when {
                    tools.isEmpty() -> "No tools listed — refresh to ask again"
                    !server.enabled -> "$live of ${tools.size} tools, paused"
                    live == tools.size -> "$live tools"
                    else -> "$live of ${tools.size} tools"
                },
                style = NocturneType.Help,
                color = if (server.enabled) NocturneColors.Accent300 else NocturneColors.TextMuted,
                modifier = Modifier.weight(1f),
            )
            if (tools.isNotEmpty()) {
                Text(
                    if (expanded) "⌃" else "⌄",
                    style = NocturneType.Row,
                    color = NocturneColors.Accent300,
                )
            }
        }

        if (expanded) {
            Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                tools.forEach { tool ->
                    val on = tool.name !in disabled
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .background(
                                if (on) NocturneColors.Accent900 else NocturneColors.Neutral900,
                                Radius.Sm,
                            )
                            .nClickableFlat { onToggleTool(tool.name) }
                            .padding(horizontal = 10.dp, vertical = 8.dp),
                        horizontalArrangement = Arrangement.spacedBy(9.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                tool.name,
                                style = NocturneType.MonoCode,
                                color = if (on) NocturneColors.Accent200 else NocturneColors.TextMuted,
                            )
                            if (tool.description.isNotBlank()) {
                                Text(
                                    tool.description,
                                    style = NocturneType.Help,
                                    color = NocturneColors.Text.copy(alpha = if (on) 0.75f else 0.4f),
                                )
                            }
                        }
                        NSwitch(checked = on, onCheckedChange = { onToggleTool(tool.name) })
                    }
                }
                NHelp("A tool switched off here is never mentioned to the model, so it cannot be called.")
            }
        }

        server.lastError?.let {
            Text(it, style = NocturneType.Help, color = NocturneColors.Neutral300)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
            NButton(
                if (busy) "Asking…" else "Refresh",
                onClick = onRefresh,
                modifier = Modifier.weight(1f),
                minHeight = 40.dp,
            )
            NButton(
                "Remove",
                onClick = onRemove,
                modifier = Modifier.weight(1f),
                minHeight = 40.dp,
            )
        }
    }
}
