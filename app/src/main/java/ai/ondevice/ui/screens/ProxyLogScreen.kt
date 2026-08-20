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
import ai.ondevice.proxy.InterceptRecord
import ai.ondevice.proxy.RequestRecord
import ai.ondevice.proxy.VideoJobs
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NDot
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NRowRule
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.vm.ProxyViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * What the server has been asked for, and what it did about it.
 *
 * The intercept trace under each row is the point of this screen. It is the
 * only place the answer to "why did that take four minutes" exists: which round
 * searched for what, which tool ran, what was blocked and what was refused. In
 * memory and cleared on restart, deliberately — the trace is worth keeping for
 * a session and the prompts inside it are not worth keeping at all.
 */
@Composable
fun ProxyLogScreen(
    onBack: () -> Unit,
    viewModel: ProxyViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    var opened by rememberSaveable { mutableStateOf<String?>(null) }

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = "Recent requests",
                subtitle = "${state.requests.size} kept · cleared when the app restarts",
                subtitleMono = false,
                onBack = onBack,
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            if (state.videoJobs.isNotEmpty()) {
                SectionKicker("Video jobs", Modifier.padding(bottom = 8.dp))
                state.videoJobs.forEach { job ->
                    VideoJobRow(job, onCancel = { viewModel.cancelVideoJob(job.id) })
                }
                SectionKicker("Requests", Modifier.padding(top = 20.dp, bottom = 8.dp))
            }

            if (state.requests.isEmpty()) {
                NCard {
                    Text(
                        "Nothing yet.",
                        style = NocturneType.Row,
                        color = NocturneColors.TextMuted,
                    )
                }
                NHelp(
                    "Every request the proxy answers appears here with the rounds it took and " +
                        "the tools it ran on the way.",
                    Modifier.padding(top = 6.dp),
                )
            }

            state.requests.forEach { record ->
                RequestRow(
                    record = record,
                    expanded = opened == record.id,
                    onToggle = { opened = if (opened == record.id) null else record.id },
                    modifier = Modifier.padding(bottom = 6.dp),
                )
            }

            if (state.requests.isNotEmpty()) {
                NButton(
                    "Clear",
                    onClick = viewModel::clearLog,
                    style = NButtonStyle.Ghost,
                    modifier = Modifier.padding(top = 12.dp).fillMaxWidth(),
                    block = true,
                )
            }
        }
    }
}

@Composable
private fun RequestRow(
    record: RequestRecord,
    expanded: Boolean,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier,
) {
    NCard(modifier = modifier, gap = 6.dp) {
        Row(
            Modifier.fillMaxWidth().nClickableFlat(onClick = onToggle),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            NDot(color = colourFor(record), size = 6.dp)
            Text(TIME.format(Date(record.startedAt)), style = NocturneType.MonoTimestamp)
            Text(
                record.path.removePrefix("/v1/"),
                style = NocturneType.MonoXs,
                color = NocturneColors.Text,
                modifier = Modifier.weight(1f),
            )
            Text(
                if (record.inFlight) "running" else "${record.durationMillis} ms",
                style = NocturneType.MonoXs,
                color = NocturneColors.TextMuted,
            )
        }

        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            NTag(
                record.protocol.name.lowercase(),
                style = if (record.protocol.name == "ANTHROPIC") NTagStyle.Accent else NTagStyle.Accent2,
            )
            if (record.streaming) NTag("stream", style = NTagStyle.Neutral)
            if (record.rounds > 1) NTag("${record.rounds} rounds", style = NTagStyle.Neutral)
            Text(
                record.client.take(40),
                style = NocturneType.Meta,
                color = NocturneColors.TextMeta,
                modifier = Modifier.weight(1f),
            )
        }

        if (!expanded) return@NCard

        NRowRule()

        // The model a client asked for and the model that ran are different
        // whenever an alias is involved, which is most of the time — and a
        // client that thinks it is talking to Sonnet deserves one place that
        // says otherwise.
        Detail("asked for", record.requestedModel.ifBlank { "—" })
        Detail("ran", record.resolvedModel.ifBlank { "—" })
        if (record.promptTokens > 0) Detail("prompt", "${record.promptTokens} tokens")
        if (record.generatedTokens > 0) {
            Detail(
                "generated",
                "${record.generatedTokens} tokens" +
                    if (record.tokensPerSecond > 0f) {
                        " · %.1f t/s".format(record.tokensPerSecond)
                    } else {
                        ""
                    },
            )
        }
        if (record.status != 0) Detail("status", record.status.toString())
        record.error?.let { Detail("error", it) }

        record.intercepts.forEach { intercept ->
            Row(
                Modifier.fillMaxWidth().padding(top = 4.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text(
                    labelFor(intercept.kind),
                    style = NocturneType.Mono2Xs,
                    color = NocturneColors.Accent300,
                )
                Column(Modifier.weight(1f)) {
                    Text(intercept.name, style = NocturneType.MonoXs)
                    if (intercept.detail.isNotBlank()) {
                        Text(
                            intercept.detail,
                            style = NocturneType.Mono2Xs,
                            color = NocturneColors.TextMuted,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun VideoJobRow(job: VideoJobs.Job, onCancel: () -> Unit) {
    NCard(gap = 6.dp, modifier = Modifier.padding(bottom = 6.dp)) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(job.id.removePrefix("video_").take(8), style = NocturneType.MonoTimestamp)
            Text(
                job.state.name.lowercase(),
                style = NocturneType.MonoXs,
                color = if (job.terminal) NocturneColors.TextMuted else NocturneColors.Accent300,
                modifier = Modifier.weight(1f),
            )
            if (!job.terminal) {
                NButton("Cancel", onClick = onCancel, style = NButtonStyle.Ghost)
            }
        }
        Text(
            job.prompt.take(90),
            style = NocturneType.Meta,
            color = NocturneColors.TextMeta,
        )
        if (!job.terminal && job.steps > 0) {
            Text(
                "step ${job.step}/${job.steps} · ${job.phase}" +
                    if (job.secondsPerStep > 0f) " · %.0f s/it".format(job.secondsPerStep) else "",
                style = NocturneType.Mono2Xs,
                color = NocturneColors.TextMuted,
            )
        }
        job.error?.let {
            Text(it, style = NocturneType.Mono2Xs, color = NocturneColors.Accent300)
        }
        if (job.frames.isNotEmpty()) {
            Text(
                "${job.frames.size} frames at ${job.fps} fps",
                style = NocturneType.Mono2Xs,
                color = NocturneColors.TextMuted,
            )
        }
    }
}

@Composable
private fun Detail(label: String, value: String) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Text(
            label,
            style = NocturneType.Mono2Xs,
            color = NocturneColors.TextMuted,
            modifier = Modifier.weight(0.35f),
        )
        Text(value, style = NocturneType.MonoXs, modifier = Modifier.weight(0.65f))
    }
}

private fun labelFor(kind: InterceptRecord.Kind): String = when (kind) {
    InterceptRecord.Kind.TOOL_SEARCH -> "search"
    InterceptRecord.Kind.RAN_TOOL -> "ran"
    InterceptRecord.Kind.AUTO_LOADED -> "loaded"
    InterceptRecord.Kind.BLOCKED -> "blocked"
    InterceptRecord.Kind.UNKNOWN_TOOL -> "unknown"
    InterceptRecord.Kind.REFUSED -> "refused"
}

private fun colourFor(record: RequestRecord) = when {
    record.inFlight -> NocturneColors.Accent
    record.error != null || record.status >= 400 -> NocturneColors.Neutral400
    else -> NocturneColors.Neutral700
}

private val TIME = SimpleDateFormat("HH:mm:ss", Locale.UK)
