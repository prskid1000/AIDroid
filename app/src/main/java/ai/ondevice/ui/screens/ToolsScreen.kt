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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.tools.BuiltInToolProvider
import ai.ondevice.tools.McpToolProvider
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NHelp
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
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.ToolsViewModel

/**
 * **Tools and MCP servers.**
 *
 * The one screen where this app hands something outward, so it is the one
 * screen that argues with itself in front of the user. Tool use is off until
 * asked for; the built-in set is separated from anything remote because the
 * built-ins touch nothing but this device; and an MCP server has to answer
 * before it can be added, so a URL that was never reachable cannot sit here
 * looking installed and fail mid-conversation instead.
 */
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
                subtitle = "What the model is allowed to do",
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
            SectionKicker("On this device", Modifier.padding(top = 20.dp, bottom = 8.dp))
            val builtInOn = BuiltInToolProvider.ID in state.enabledProviders
            Row(
                Modifier
                    .fillMaxWidth()
                    .background(
                        if (builtInOn) NocturneColors.Accent900 else NocturneColors.Surface,
                        Radius.Md,
                    )
                    .ring(if (builtInOn) NocturneColors.Accent else NocturneColors.Divider, Radius.Md)
                    .nClickableFlat { viewModel.toggleProvider(BuiltInToolProvider.ID) }
                    .padding(horizontal = 12.dp, vertical = 11.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text("Built-in", style = NocturneType.CardTitleSm)
                    Text(
                        state.builtInTools.joinToString(" · "),
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMuted,
                    )
                }
                NTag(if (builtInOn) "on" else "off", style = if (builtInOn) NTagStyle.Accent else NTagStyle.Outline)
            }
            NHelp(
                "The clock it cannot read, arithmetic it gets subtly wrong, and the state of the device " +
                    "it is running on. None of it touches the network.",
                Modifier.padding(top = 6.dp),
            )

            // — MCP —
            SectionKicker("MCP servers", Modifier.padding(top = 20.dp, bottom = 8.dp))

            state.servers.forEach { server ->
                val id = "${McpToolProvider.ID_PREFIX}${server.id}"
                val on = id in state.enabledProviders
                NCard(Modifier.padding(bottom = 8.dp), ring = if (on) NocturneColors.Accent700 else NocturneColors.Divider) {
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
                        NSwitch(checked = on, onCheckedChange = { viewModel.toggleProvider(id) })
                    }
                    server.lastToolsJson?.takeIf { it.isNotBlank() }?.let { tools ->
                        Text(
                            tools.split(",").joinToString(" · "),
                            style = NocturneType.MonoXs,
                            color = NocturneColors.Accent300,
                        )
                    }
                    server.lastError?.let {
                        Text(it, style = NocturneType.Help, color = NocturneColors.Neutral300)
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                        NButton(
                            "Refresh",
                            onClick = { viewModel.refresh(server) },
                            modifier = Modifier.weight(1f),
                            minHeight = 40.dp,
                        )
                        NButton(
                            "Remove",
                            onClick = { viewModel.removeServer(server.id) },
                            modifier = Modifier.weight(1f),
                            minHeight = 40.dp,
                        )
                    }
                }
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
                "HTTP only. MCP's other transport launches a process, which Android's W^X rules make " +
                    "either impossible or a way to get executable code onto the device — so this app " +
                    "does not offer it. A server is added only once it has answered and listed its tools.",
                Modifier.padding(top = 12.dp),
            )
            NHelp(
                "A tool result is data, not instruction. Nothing in this app treats what a server " +
                    "returns as a command.",
                Modifier.padding(top = 8.dp),
            )
        }
    }
}
