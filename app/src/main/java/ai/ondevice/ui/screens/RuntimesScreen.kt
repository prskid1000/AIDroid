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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.data.db.RuntimeBundleEntity
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.vm.SettingsViewModel

/**
 * **Settings → Runtimes.** Which engines this build carries, and at which
 * upstream commit.
 *
 * There is nothing to press. The engines are compiled into the APK, so an
 * engine update is an app update — see `native/VERSIONS` for the four steps
 * that produce one. Install, Update and rollback buttons lived here for a
 * while; no code ever set the columns they read.
 */
@Composable
fun RuntimesScreen(
    onBack: () -> Unit,
    viewModel: SettingsViewModel = hiltViewModel(),
) {
    val runtimes by viewModel.runtimes.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = { PushToolbar("Runtimes", onBack) },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {
            runtimes.forEach { runtime -> RuntimeCard(runtime) }

            NHelp(
                "No account, no telemetry, no network after download. The only outbound calls are " +
                    "the Hugging Face API, the web_search tool when the model uses it, and the ones " +
                    "you start.",
                Modifier.padding(top = 14.dp),
            )
        }
    }
}

@Composable
private fun RuntimeCard(runtime: RuntimeBundleEntity) {
    NCard(Modifier.padding(bottom = 8.dp), gap = 7.dp) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.Top,
        ) {
            Column(Modifier.weight(1f)) {
                Text(runtime.engine, style = NocturneType.CardTitle)
                Text(
                    buildString {
                        append(runtime.buildTag ?: "unknown build")
                        runtime.upstreamCommit?.takeIf { it.isNotBlank() }?.let { append(" · $it") }
                        if (runtime.architectureCount > 1) {
                            append(" · ${runtime.architectureCount} architectures")
                        }
                        append(" · ${runtime.backendsJson.split(',').joinToString(", ") { it.trim() }}")
                    },
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                    modifier = Modifier.padding(top = 2.dp),
                )
            }
        }
    }
}
