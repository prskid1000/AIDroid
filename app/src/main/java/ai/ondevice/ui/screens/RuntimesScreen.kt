package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.core.RuntimeState
import ai.ondevice.data.db.RuntimeBundleEntity
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCardMeta
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NMetaText
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.vm.SettingsViewModel

/**
 * **S15 — Settings → Runtimes.**
 *
 * SPEC §17: engines are versioned, replaceable, separately installable
 * artifacts. What this screen has to communicate honestly is the constraint
 * behind them — Android's W^X enforcement makes true in-process `.so` hot-swap
 * impossible (§1.6, §17.1), so an update is a signed package delivered by the
 * package manager, not a downloaded library.
 *
 * §17.7 also insists update notes state what actually changed — engine version,
 * new parameters, new architectures — never "improvements and bug fixes".
 */
@Composable
fun RuntimesScreen(
    onBack: () -> Unit,
    viewModel: SettingsViewModel = hiltViewModel(),
) {
    val runtimes by viewModel.runtimes.collectAsStateWithLifecycle()
    val manifest by viewModel.manifest.collectAsStateWithLifecycle()
    val settings by viewModel.settings.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = { PushToolbar("Runtimes", onBack) },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            runtimes.forEach { runtime -> RuntimeCard(runtime, viewModel) }

            runtimes.firstOrNull { it.rolledBackFrom != null }?.let { rolled ->
                NCard(Modifier.padding(top = 8.dp, bottom = 16.dp), ring = NocturneColors.Neutral700) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            NIcons.Rotate,
                            contentDescription = null,
                            tint = NocturneColors.Neutral300,
                            modifier = Modifier.size(15.dp),
                        )
                        Text("Rolled back once", style = NocturneType.CardTitleSm)
                    }
                    Text(
                        "${rolled.rolledBackFrom} failed to initialise twice in a row, so the app " +
                            "reverted to ${rolled.buildTag} on its own and told you why. A bad upstream " +
                            "commit can't brick it.",
                        style = NocturneType.CardBody,
                        color = NocturneColors.Text.copy(alpha = 0.8f),
                    )
                }
            }

            SectionKicker("Parameter manifest", Modifier.padding(top = 8.dp, bottom = 8.dp))
            NCard(gap = 8.dp) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.Bottom,
                ) {
                    Text("v${manifest.version}", style = NocturneType.CardTitleSm, modifier = Modifier.weight(1f))
                    NTag("bundled", style = NTagStyle.Neutral)
                }
                // No "Ed25519 ✓" and no "Auto-check · Wi-Fi only". Both described
                // an over-the-air update path that does not exist: nothing writes
                // the manifest table, so the app only ever reads the copy shipped
                // inside it, and there is no signature check to pass.
                Text(
                    "Ships inside the app. Data, not code — it can retier, relabel and reveal " +
                        "parameters the installed engine already supports. It cannot add capability " +
                        "the engine lacks; that is what an engine update is for.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            }

            // §17.1/§17.2 — be explicit about why an engine update is a package.
            NCard(Modifier.padding(top = 16.dp), ring = NocturneColors.Neutral800) {
                Text("How engine updates arrive", style = NocturneType.CardTitleSm)
                Text(
                    if (settings.canSelfUpdateRuntimes) {
                        "This is the sideload build. Updates are signed packages verified against the " +
                            "pinned certificate and installed through the package manager — Android's " +
                            "W^X enforcement rejects loading a native library from writable storage, " +
                            "so a downloaded .so is never dlopen'd."
                    } else {
                        "This is the Play build. Native code arrives as Play Feature Delivery modules, " +
                            "installed read-only into the app's lib directory. Play policy forbids " +
                            "downloading executable code from anywhere else, so the in-app updater " +
                            "degrades to a store link."
                    },
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            }

            NHelp(
                "No account, no telemetry, no network after download. The only outbound calls are the " +
                    "Hugging Face API and the ones you start.",
                Modifier.padding(top = 14.dp),
            )
        }
    }
}

@Composable
private fun RuntimeCard(runtime: RuntimeBundleEntity, viewModel: SettingsViewModel) {
    val installed = runtime.state != RuntimeState.NOT_INSTALLED
    val hasUpdate = runtime.availableBuildTag != null

    NCard(
        Modifier.padding(bottom = 8.dp),
        gap = if (hasUpdate) 8.dp else 7.dp,
        ring = if (hasUpdate) NocturneColors.Accent700 else if (!installed) NocturneColors.Neutral800 else null,
    ) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.Top,
        ) {
            Column(Modifier.weight(1f)) {
                Text(
                    runtime.engine,
                    style = NocturneType.CardTitle,
                    modifier = Modifier.alpha(if (installed) 1f else 0.7f),
                )
                Text(
                    buildString {
                        if (installed) {
                            append(runtime.buildTag ?: "installed")
                            if (runtime.architectureCount > 1) {
                                append(" · ${runtime.architectureCount} architectures")
                            }
                            append(" · ${runtime.backendsJson.split(',').joinToString(", ") { backendLabel(it) }}")
                        } else {
                            append("not installed · ${Fmt.bytes(runtime.sizeBytes)}")
                        }
                    },
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                    modifier = Modifier.padding(top = 2.dp),
                )
            }
            when {
                hasUpdate -> NButton(
                    "Update",
                    onClick = { viewModel.updateRuntime(runtime.engine) },
                    style = NButtonStyle.Primary,
                    minHeight = 30.dp,
                )
                !installed -> NButton(
                    "Install",
                    onClick = { viewModel.installRuntime(runtime.engine) },
                    style = NButtonStyle.Secondary,
                    minHeight = 30.dp,
                )
                else -> Text(
                    "Up to date",
                    style = NocturneType.Input,
                    color = NocturneColors.TextMuted,
                )
            }
        }

        // §17.7 — say what actually changed.
        if (hasUpdate) {
            Column(
                Modifier
                    .fillMaxWidth()
                    .background(NocturneColors.Accent900, Radius.Sm)
                    .padding(horizontal = 10.dp, vertical = 9.dp),
            ) {
                Text(
                    "${runtime.buildTag} → ${runtime.availableBuildTag}",
                    style = NocturneType.CardTitleSm,
                    color = NocturneColors.Accent200,
                )
                Text(
                    runtime.availableNotes ?: "See the manifest diff for what changed.",
                    style = NocturneType.Help,
                    color = NocturneColors.Accent300,
                    modifier = Modifier.padding(top = 3.dp),
                )
            }
            NCardMeta {
                NMetaText(
                    "Signed bundle · jniContract ${runtime.jniContract} · previous kept for rollback",
                )
            }
        } else if (!installed) {
            Text(
                "Optional. Skipping it keeps ${Fmt.bytes(runtime.sizeBytes)} of code and every model " +
                    "for it off the device.",
                style = NocturneType.CardBody,
                color = NocturneColors.Text.copy(alpha = 0.8f),
            )
        }
    }
}

private fun backendLabel(raw: String): String = runCatching {
    // The API rather than the device, because this screen is about what the
    // build contains — "OpenCL" says which of several routes to the GPU was
    // compiled, and that is the distinction a runtime card exists to draw.
    ai.ondevice.core.BackendId.valueOf(raw.trim()).api
}.getOrDefault(raw.trim())
