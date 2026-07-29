package ai.ondevice.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.BackendId
import ai.ondevice.core.Fmt
import ai.ondevice.core.ThermalPolicy
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.ui.BottomDestinations
import ai.ondevice.ui.components.NBottomBar
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NRadio
import ai.ondevice.ui.components.NSwitch
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.RootToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.ruleBelow
import ai.ondevice.ui.vm.SettingsViewModel

/**
 * Settings root. Backend, thermal policy, storage, token, network, and the way
 * through to Runtimes (S15).
 *
 * The closing line is a load-bearing claim rather than marketing: SPEC §13 —
 * no account, no telemetry, and no network at all after download.
 */
@Composable
fun SettingsScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onOpenRuntimes: () -> Unit,
    viewModel: SettingsViewModel = hiltViewModel(),
) {
    val state by viewModel.settings.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = { RootToolbar("Settings") },
        bottomBar = { NBottomBar(BottomDestinations, currentRoute) { onNavigate(it.route) } },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            SectionKicker("Device", Modifier.padding(bottom = 8.dp))
            NCard(gap = 6.dp) {
                InfoRow("SoC", state.soc)
                InfoRow("RAM", Fmt.bytes(state.totalRamBytes))
                InfoRow("Cores", "${state.performanceCores} performance of ${state.totalCores}")
                InfoRow("Free storage", Fmt.bytes(state.freeStorageBytes))
                InfoRow("Models on disk", Fmt.bytes(state.storageUsedBytes))
            }

            SectionKicker("Backend", Modifier.padding(top = 20.dp, bottom = 8.dp))
            NCard(gap = 7.dp) {
                listOf(
                    AppPrefs.BACKEND_AUTO,
                    BackendId.OPENCL.name,
                    BackendId.HEXAGON.name,
                    BackendId.CPU.name,
                ).forEach { mode ->
                    NRadio(
                        label = AppPrefs.backendModeLabel(mode),
                        selected = state.backendMode == mode,
                        onSelect = { viewModel.setBackendMode(mode) },
                    )
                }
            }
            NHelp(
                "Auto picks whichever backend measured fastest for each model on this device. " +
                    "Hexagon needs its runtime installed and is capped at ~3.5 GB a session.",
                Modifier.padding(top = 8.dp),
            )

            SectionKicker("Thermal policy", Modifier.padding(top = 20.dp, bottom = 8.dp))
            NCard(gap = 9.dp) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(7.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    NTag("status: ${state.thermalLabel}", style = NTagStyle.Neutral)
                    Box(Modifier.weight(1f))
                    Text(
                        "battery ${state.batteryPercent}%",
                        style = NocturneType.Meta,
                        color = NocturneColors.TextMuted,
                    )
                }
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    ThermalPolicy.entries.forEach { policy ->
                        NRadio(
                            label = policy.label,
                            selected = state.thermalPolicy == policy,
                            onSelect = { viewModel.setThermalPolicy(policy) },
                        )
                    }
                }
            }

            SectionKicker("Network", Modifier.padding(top = 20.dp, bottom = 8.dp))
            NCard(gap = 10.dp) {
                ToggleRow("Wi-Fi only downloads", state.wifiOnly, viewModel::setWifiOnly)
                ToggleRow("Check for manifest updates on Wi-Fi only", state.manifestWifiOnly, viewModel::setManifestWifiOnly)
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
                NInput(
                    value = "",
                    onValueChange = { viewModel.setToken(it.takeIf { v -> v.isNotBlank() }) },
                    placeholder = "hf_…",
                    textStyle = NocturneType.MonoValue,
                )
            }

            SectionKicker("Parameters", Modifier.padding(top = 20.dp, bottom = 8.dp))
            NCard(gap = 10.dp) {
                ToggleRow(
                    "Show all parameters",
                    state.showAllParameters,
                    viewModel::setShowAllParameters,
                )
                Text(
                    "Collapses the Basic / Advanced / Expert tiers entirely. Tiering only ever " +
                        "controlled default visibility — nothing was hidden permanently.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            }

            NButton(
                "Runtimes →",
                onClick = onOpenRuntimes,
                style = ai.ondevice.ui.components.NButtonStyle.Primary,
                block = true,
                modifier = Modifier.padding(top = 20.dp),
            )

            NHelp(
                "No account, no telemetry, no network after download. The only outbound calls are the " +
                    "Hugging Face API and the ones you start.",
                Modifier.padding(top = 16.dp),
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
