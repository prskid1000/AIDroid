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
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.ui.BottomDestinations
import ai.ondevice.ui.components.NBottomBar
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NSeg
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
 * Settings root. Backend, network, token, parameter tiering, and the way through
 * to Tools and Runtimes (S15).
 *
 * The closing line is a load-bearing claim rather than marketing: SPEC §13 —
 * no account, no telemetry, and no network at all after download.
 */
@Composable
fun SettingsScreen(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    onOpenRuntimes: () -> Unit,
    onOpenTools: () -> Unit,
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

            // Three devices, named as devices.
            //
            // The list used to open with "Auto" and then name APIs — OpenCL,
            // Hexagon HTP — which asked the wrong question twice over. Nobody
            // wants OpenCL; they want the GPU, and OpenCL is one route to it.
            // And auto was never a choice: it meant "the first backend
            // registered", an ordering dressed up as a decision.
            //
            // All three are always shown, including ones this phone cannot
            // reach, because "the NPU is not an option here" is information and
            // a missing segment is not. An unreachable one is dimmed, unselectable,
            // and named underneath (§1.2 — a refusal names what went wrong).
            SectionKicker("Compute device", Modifier.padding(top = 20.dp, bottom = 8.dp))
            val devices = listOf(BackendId.HEXAGON, BackendId.OPENCL, BackendId.CPU)
            val available = state.availableBackends
            NCard(gap = 7.dp) {
                // One row rather than three stacked radios: three one-word
                // labels naming three pieces of the same chip are a single
                // choice, and reading them as a column made them look like
                // three separate settings.
                NSeg(
                    options = devices.map { it.label },
                    selectedIndex = devices.indexOfFirst { it.name == state.backendMode }
                        .takeIf { it >= 0 } ?: devices.indexOf(BackendId.CPU),
                    onSelect = { viewModel.setBackendMode(devices[it].name) },
                    enabled = { devices[it] in available },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            // A dimmed segment says "not here" but not why, and the row has no
            // space for the reason — so the reason goes underneath, naming the
            // devices rather than leaving the user to work out which one is grey.
            val missing = devices.filterNot { it in available }
            NHelp(
                "What ggml registered on this phone, not what the build contains: a backend " +
                    "compiled in still needs the silicon and the driver behind it. The GPU runs " +
                    "through OpenCL, the NPU through Hexagon — and the NPU only has kernels for " +
                    "Q4_0, Q4_1, Q8_0, IQ4_NL and MXFP4 weights, so a K-quant model selects it " +
                    "and then does its arithmetic on the CPU regardless." +
                    if (missing.isEmpty()) {
                        ""
                    } else {
                        " " + missing.joinToString(" and ") { it.label } +
                            " did not register here, so ${if (missing.size > 1) "they are" else "it is"} " +
                            "shown but cannot be chosen."
                    },
                Modifier.padding(top = 8.dp),
            )

            // No thermal policy card. The kernel governor throttles a hot SoC on
            // its own, and the four settings that used to sit here mostly did
            // nothing — `n_threads` cannot be changed on a live llama.cpp
            // context, so two of them were inert and a third was mislabelled.

            SectionKicker("Network", Modifier.padding(top = 20.dp, bottom = 8.dp))
            NCard(gap = 10.dp) {
                // One toggle, because there is one thing that goes over the
                // network. The manifest-update toggle that used to sit beside it
                // gated a fetch that does not exist: nothing writes the manifest
                // table, so the "newer than bundled" path can never be taken.
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
                NInput(
                    value = "",
                    onValueChange = { viewModel.setToken(it.takeIf { v -> v.isNotBlank() }) },
                    placeholder = "hf_…",
                    textStyle = NocturneType.MonoValue,
                )
            }

            // No "Show all parameters" here. It was a second copy of the All tab
            // that already sits on the parameters screen, one tap from the list
            // it filters — a global preference for a per-screen choice, set in a
            // different screen from the one it changes.

            NButton(
                "Tools and MCP servers →",
                onClick = onOpenTools,
                style = ai.ondevice.ui.components.NButtonStyle.Secondary,
                block = true,
                modifier = Modifier.padding(top = 20.dp),
            )

            NButton(
                "Runtimes →",
                onClick = onOpenRuntimes,
                style = ai.ondevice.ui.components.NButtonStyle.Primary,
                block = true,
                modifier = Modifier.padding(top = 8.dp),
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
