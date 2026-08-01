package ai.ondevice.ui.screens

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
import ai.ondevice.core.DownloadState
import ai.ondevice.core.Fmt
import ai.ondevice.data.download.DownloadErrorKind
import ai.ondevice.data.download.DownloadJob
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCardMeta
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NMetaText
import ai.ondevice.ui.components.NProgressBar
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.ruleAbove
import ai.ondevice.ui.vm.DownloadsViewModel

/** **S4 — Download queue.** Sharded, resumable, companion-aware. */
@Composable
fun DownloadQueueScreen(
    onBack: () -> Unit,
    viewModel: DownloadsViewModel = hiltViewModel(),
) {
    val jobs by viewModel.jobs.collectAsStateWithLifecycle()
    val active = jobs.filter { it.state != DownloadState.FAILED && it.state != DownloadState.COMPLETE }
    val failed = jobs.filter { it.state == DownloadState.FAILED }
    val complete = jobs.filter { it.state == DownloadState.COMPLETE }

    PhoneScaffold(
        toolbar = {
            PushToolbar("Downloads", onBack) {
                Text("Wi-Fi only", style = NocturneType.Input, color = NocturneColors.TextMuted)
            }
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            if (jobs.isEmpty()) {
                NCard {
                    Text("Nothing downloading", style = NocturneType.CardTitleSm)
                    Text(
                        "Downloads run in a foreground service and survive app kill. Byte offsets are " +
                            "persisted, so a transfer resumes across a reboot rather than starting over.",
                        style = NocturneType.CardBody,
                        color = NocturneColors.Text.copy(alpha = 0.8f),
                    )
                }
            }

            active.forEach { job -> ActiveJobCard(job, viewModel) }

            if (failed.isNotEmpty()) {
                SectionKicker("Failed · ${failed.size}", Modifier.padding(top = 18.dp, bottom = 9.dp))
                failed.forEach { job -> FailedJobCard(job, viewModel) }
            }

            if (complete.isNotEmpty()) {
                SectionKicker("Complete · ${complete.size}", Modifier.padding(top = 18.dp, bottom = 9.dp))
                complete.forEach { job ->
                    NCard(Modifier.padding(bottom = 7.dp), gap = 6.dp) {
                        Text(job.displayName, style = NocturneType.CardTitle)
                        jobSubtitle(job)?.let {
                            Text(it, style = NocturneType.MonoXs, color = NocturneColors.TextMuted)
                        }
                        NCardMeta { NMetaText("${Fmt.bytes(job.bytesTotal)} · sha256 verified") }
                    }
                }
            }

            if (complete.isNotEmpty() || failed.isNotEmpty()) {
                NButton(
                    "Clear finished · ${complete.size + failed.size}",
                    viewModel::clearFinished,
                    style = NButtonStyle.Secondary,
                    block = true,
                    modifier = Modifier.padding(top = 10.dp),
                )
                // Said plainly because "clear" next to a list of model names reads like it might remove the models.
                NHelp(
                    "Removes the history only. Installed models and their files are untouched.",
                    Modifier.padding(top = 6.dp),
                )
            }

            NHelp(
                "Downloads run in a foreground service and survive app kill. Partial files are swept on " +
                    "next boot.",
                Modifier.padding(top = 14.dp),
            )
        }
    }
}

@Composable
private fun ActiveJobCard(job: DownloadJob, viewModel: DownloadsViewModel) {
    val running = job.state == DownloadState.RUNNING
    NCard(
        Modifier.padding(bottom = 10.dp),
        gap = 9.dp,
        ring = if (running) NocturneColors.Accent700 else null,
    ) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.Top,
        ) {
            Column(Modifier.weight(1f)) {
                Text(job.displayName, style = NocturneType.CardTitle)
                jobSubtitle(job)?.let {
                    Text(it, style = NocturneType.MonoXs, color = NocturneColors.TextMuted)
                }
                Text(
                    if (job.state == DownloadState.PAUSED) {
                        "paused — ${Fmt.percent(job.fraction)} done"
                    } else {
                        job.subtitle
                    },
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                    modifier = Modifier.padding(top = 2.dp),
                )
            }
            if (running) {
                Icon(
                    NIcons.Pause,
                    contentDescription = "Pause",
                    tint = NocturneColors.Text,
                    modifier = Modifier.size(19.dp).nClickableFlat { viewModel.pause(job.id) },
                )
            } else {
                NButton(
                    "Resume",
                    onClick = { viewModel.resume(job.id) },
                    style = NButtonStyle.Ghost,
                    minHeight = 24.dp,
                )
            }
            // Cancel was reachable only from a *failed* job, as "Discard".
            NButton(
                "Cancel",
                onClick = { viewModel.cancel(job.id) },
                style = NButtonStyle.Ghost,
                minHeight = 24.dp,
            )
        }

        NProgressBar(
            fraction = job.fraction,
            fill = if (running) NocturneColors.Accent500 else NocturneColors.Neutral600,
        )

        NCardMeta(gap = 8.dp) {
            Text(
                Fmt.percent(job.fraction),
                style = NocturneType.MonoSm,
                color = NocturneColors.Accent300,
            )
            NMetaText("·")
            NMetaText("${Fmt.bytes(job.bytesDone)} of ${Fmt.bytes(job.bytesTotal)}")
            Box(Modifier.weight(1f))
            // Rate and time remaining.
            if (job.bytesPerSecond > 0) {
                NMetaText(Fmt.transferRate(job.bytesPerSecond))
                if (running && job.etaSeconds > 0) {
                    NMetaText("·")
                    NMetaText(Fmt.eta(job.etaSeconds))
                }
            }
        }

        // The per-file breakdown: shards and companions, each with its own state.
        if (job.files.size > 1) {
            Column(
                Modifier
                    .fillMaxWidth()
                    .padding(top = 4.dp)
                    .ruleAbove()
                    .padding(top = 7.dp),
                verticalArrangement = Arrangement.spacedBy(5.dp),
            ) {
                job.files.forEach { file ->
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .alpha(if (file.complete || file.bytesDone > 0) 1f else 0.45f),
                        horizontalArrangement = Arrangement.spacedBy(7.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            when {
                                file.complete -> "✓"
                                file.bytesDone > 0 -> "↓"
                                else -> "·"
                            },
                            style = NocturneType.MonoSm,
                            color = when {
                                file.complete -> NocturneColors.Accent
                                file.bytesDone > 0 -> NocturneColors.Accent400
                                else -> NocturneColors.TextMuted
                            },
                        )
                        Text(
                            file.filename.substringAfterLast('/'),
                            style = NocturneType.MonoSm,
                            color = NocturneColors.Text.copy(alpha = 0.75f),
                            modifier = Modifier.weight(1f),
                            maxLines = 1,
                        )
                        Text(
                            file.progressLabel,
                            style = NocturneType.MonoSm,
                            color = NocturneColors.TextMuted,
                        )
                    }
                }
            }
        }

        if (job.state == DownloadState.PAUSED) {
            NCardMeta { NMetaText("Offsets persisted. Resumes across reboot.") }
        }
    }
}

/** The refusal shape again: neutral disc, no red. */
@Composable
private fun FailedJobCard(job: DownloadJob, viewModel: DownloadsViewModel) {
    val error = job.error
    NCard(Modifier.padding(bottom = 10.dp), ring = NocturneColors.Neutral700) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                if (error?.kind == DownloadErrorKind.CHECKSUM_MISMATCH) NIcons.SlashCircle else NIcons.InfoCircle,
                contentDescription = null,
                tint = NocturneColors.Neutral300,
                modifier = Modifier.size(16.dp),
            )
            Text(
                error?.kind?.title ?: "Download failed",
                style = NocturneType.CardTitleSm,
                modifier = Modifier.weight(1f),
            )
        }
        Text(
            error?.message ?: "The transfer stopped and nothing was installed.",
            style = NocturneType.CardBody,
            color = NocturneColors.Text.copy(alpha = 0.8f),
        )
        if (error?.expected != null && error.actual != null) {
            Text(
                "expected ${Fmt.shortHash(error.expected)}\ngot      ${Fmt.shortHash(error.actual)}",
                style = NocturneType.MonoXs,
                color = NocturneColors.Text.copy(alpha = 0.55f),
            )
        }
        Row(
            Modifier.fillMaxWidth().padding(top = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(7.dp),
        ) {
            NButton(
                "Re-download",
                onClick = { viewModel.retry(job.id) },
                style = NButtonStyle.Primary,
                modifier = Modifier.weight(1f),
            )
            NButton(
                "Discard",
                onClick = { viewModel.cancel(job.id) },
                style = NButtonStyle.Secondary,
                modifier = Modifier.weight(1f),
            )
        }
    }
}

/**
 * What a queued job is, when its name is not enough.
 *
 * `displayName` is the repo, and one repo can be queued several times over: the
 * SD 3.5 encoder repo yields CLIP-L, CLIP-G and a T5, each its own job, each
 * showing the same title. The file being fetched is the thing that differs.
 */
private fun jobSubtitle(job: ai.ondevice.data.download.DownloadJob): String? =
    job.files
        .map { it.filename.substringAfterLast('/') }
        .distinct()
        .takeIf { it.isNotEmpty() }
        ?.let { names ->
            if (names.size == 1) names.first() else "${names.first()} +${names.size - 1} more"
        }
