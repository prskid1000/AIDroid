package ai.ondevice.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.ui.BottomDestinations
import ai.ondevice.ui.components.NBottomBar
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NIconButton
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NSwitch
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.RootToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.vm.SettingsViewModel

/** Settings root. */
@Composable
fun SettingsScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onOpenModels: () -> Unit,
    onOpenTools: () -> Unit,
    onOpenProxy: () -> Unit,
    onOpenLogs: () -> Unit,
    viewModel: SettingsViewModel = hiltViewModel(),
) {
    val state by viewModel.settings.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = {
            RootToolbar("Settings") {
                // Leftmost of the three, so the group reads outward-facing to
                // inward-facing: what other machines reach, what the model
                // reaches, what this device holds.
                NIconButton(
                    NIcons.Endpoint,
                    "Proxy — serve the API to other machines",
                    onClick = onOpenProxy,
                    size = 34.dp,
                    iconSize = 15.dp,
                )
                NIconButton(
                    NIcons.Tools,
                    "Tools and MCP servers",
                    onClick = onOpenTools,
                    size = 34.dp,
                    iconSize = 15.dp,
                )
                NIconButton(
                    NIcons.Models,
                    "Models",
                    onClick = onOpenModels,
                    size = 34.dp,
                    iconSize = 15.dp,
                )
                // Last, because it is the only one that is about what already
                // happened rather than about what the device can do.
                NIconButton(
                    NIcons.Logs,
                    "Engine log",
                    onClick = onOpenLogs,
                    size = 34.dp,
                    iconSize = 15.dp,
                )
            }
        },
        bottomBar = { NBottomBar(BottomDestinations, currentRoute) { onNavigate(it.route) } },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            SectionKicker("Device", Modifier.padding(bottom = 8.dp))
            NCard(gap = 6.dp) {
                InfoRow("SoC", state.soc)
                InfoRow("RAM", Fmt.bytes(state.totalRamBytes))
                InfoRow("Cores", "${state.performanceCores} performance of ${state.totalCores}")
                InfoRow("Compute", "CPU, ${state.totalCores - 1} threads")
                InfoRow("Free storage", Fmt.bytes(state.freeStorageBytes))
                InfoRow("Models on disk", Fmt.bytes(state.storageUsedBytes))
            }
            SectionKicker("Network", Modifier.padding(top = 20.dp, bottom = 8.dp))
            NCard(gap = 10.dp) {
                // One toggle, because there is one thing that goes over the network.
                ToggleRow("Wi-Fi only downloads", state.wifiOnly, viewModel::setWifiOnly)
            }

            SectionKicker("Hugging Face token", Modifier.padding(top = 20.dp, bottom = 8.dp))
            NCard(gap = 8.dp) {
                Text(
                    state.maskedToken ?: "No token stored",
                    style = NocturneType.MonoValue,
                    color = if (state.hasToken) NocturneColors.Accent300 else NocturneColors.TextMuted,
                )
                Text(
                    "Optional, and only for gated repos. Stored in the Android Keystore and used for " +
                        "nothing else. There is no account and no login anywhere in this app.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
                // The field held "" and wrote on every keystroke, so it
                // cleared itself after each character and stored one letter at
                // a time. A token is pasted in one go and is worth nothing
                // partially typed, so it is edited here and committed on Save.
                var typed by rememberSaveable { mutableStateOf("") }
                NInput(
                    value = typed,
                    onValueChange = { typed = it },
                    placeholder = "hf_…",
                    textStyle = NocturneType.MonoValue,
                )
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    NButton(
                        "Save token",
                        onClick = {
                            viewModel.setToken(typed.trim())
                            typed = ""
                        },
                        style = NButtonStyle.Primary,
                        enabled = typed.isNotBlank(),
                        modifier = Modifier.weight(1f),
                    )
                    if (state.hasToken) {
                        NButton(
                            "Remove",
                            onClick = {
                                viewModel.setToken(null)
                                typed = ""
                            },
                            modifier = Modifier.weight(1f),
                        )
                    }
                }
            }

            // Tools and Models are toolbar icons, not rows.

            NHelp(
                "No account, no telemetry, no network after download. The only outbound calls are the " +
                    "Hugging Face API and the ones you start.",
                Modifier.padding(top = 20.dp),
            )
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = NocturneType.Row, color = NocturneColors.TextMuted, modifier = Modifier.weight(1f))
        Text(value, style = NocturneType.MonoValue)
    }
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(9.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = NocturneType.Row, modifier = Modifier.weight(1f))
        NSwitch(checked, onChange)
    }
}
