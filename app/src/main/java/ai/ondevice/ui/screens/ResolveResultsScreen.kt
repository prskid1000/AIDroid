package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import ai.ondevice.core.RefusalKind
import ai.ondevice.data.hf.RemedyAction
import ai.ondevice.data.hf.Resolution
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType

/**
 * **S5 — Honest refusal.**
 *
 * SPEC §1.2 made visible: "Won't run" is an acceptable answer; a crash is not.
 * Each of the ways a repo can be turned down gets its own card, its own
 * message, and at least one remedy that is an *action* rather than advice.
 *
 * The colour discipline matters here. There is no red on this screen: a
 * refusal is a neutral-800 disc with a cross, a caveat is an accent-800 disc
 * with a bang. Weight comes from the mark, not the hue.
 */
@Composable
fun ResolveResultsScreen(onBack: () -> Unit) {
    PhoneScaffold(
        toolbar = { PushToolbar("Resolve results", onBack) },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        Column(
            Modifier.verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SampleRefusals.forEach { refusal ->
                RefusalCard(refusal, onRemedy = {})
            }
        }
    }
}

/**
 * One refusal, rendered the same way wherever it appears — on S1 under the
 * paste field, or gathered here.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun RefusalCard(
    refusal: Resolution.Refused,
    onRemedy: (RemedyAction) -> Unit,
) {
    val caveat = refusal.kind == RefusalKind.GATED || refusal.kind == RefusalKind.UNSCANNED
    NCard(ring = if (caveat) NocturneColors.Accent700 else NocturneColors.Neutral700) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(9.dp),
            verticalAlignment = Alignment.Top,
        ) {
            RefusalMark(refusal.kind)
            Column(Modifier.weight(1f)) {
                Text(refusal.title, style = NocturneType.CardTitle)
                Text(
                    refusal.subject,
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                    modifier = Modifier.padding(top = 2.dp),
                )
            }
        }

        Text(
            refusal.detail,
            style = NocturneType.CardBody,
            color = NocturneColors.Text.copy(alpha = 0.8f),
            modifier = Modifier.padding(top = 2.dp),
        )

        refusal.working?.let { working ->
            Text(
                working,
                style = NocturneType.MonoXs,
                color = NocturneColors.Text.copy(alpha = 0.6f),
            )
        }

        val primary = refusal.remedies.filter { it.primary }
        val secondary = refusal.remedies.filterNot { it.primary }

        primary.forEach { remedy ->
            NButton(
                remedy.label,
                onClick = { onRemedy(remedy.action) },
                style = NButtonStyle.Primary,
                block = true,
                minHeight = 40.dp,
            )
        }
        if (secondary.isNotEmpty()) {
            FlowRow(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(5.dp),
                verticalArrangement = Arrangement.spacedBy(5.dp),
            ) {
                secondary.forEach { remedy ->
                    NButton(
                        remedy.label,
                        onClick = { onRemedy(remedy.action) },
                        style = NButtonStyle.Secondary,
                        modifier = Modifier.weight(1f, fill = false),
                    )
                }
            }
        }
    }
}

@Composable
private fun RefusalMark(kind: RefusalKind) {
    val (icon, ground, ink) = when (kind) {
        RefusalKind.GATED -> Triple(NIcons.Bang, NocturneColors.Accent800, NocturneColors.Accent200)
        RefusalKind.UNSCANNED, RefusalKind.PICKLE_BLOCKED ->
            Triple(NIcons.Shield, NocturneColors.Neutral800, NocturneColors.Neutral200)
        else -> Triple(NIcons.Cross, NocturneColors.Neutral800, NocturneColors.Neutral200)
    }
    Box(
        Modifier.size(20.dp).background(ground, CircleShape),
        contentAlignment = Alignment.Center,
    ) {
        Icon(icon, contentDescription = null, tint = ink, modifier = Modifier.size(11.dp))
    }
}

/**
 * The five cases the canvas shows on S5, kept as data so the screen can be
 * reached without a live network. Each is produced verbatim by
 * [ai.ondevice.data.hf.ModelResolver] at runtime.
 */
private val SampleRefusals = listOf(
    Resolution.Refused(
        kind = RefusalKind.WONT_FIT,
        title = "Won't fit",
        subject = "Llama-3.3-70B-Instruct-GGUF · Q4_K_M",
        detail = "Needs ≈ 44.2 GB resident. This device has 12 GB, 10.4 GB free.",
        working = "weights 42.5 + KV 1.5 at 4K + compute 0.2",
        remedies = listOf(
            ai.ondevice.data.hf.Remedy("Smaller quants (2)", RemedyAction.ShowSmallerQuants("")),
            ai.ondevice.data.hf.Remedy("8B sibling", RemedyAction.SearchRepo("Llama-3.1-8B-Instruct-GGUF")),
        ),
    ),
    Resolution.Refused(
        kind = RefusalKind.PYTORCH_ONLY,
        title = "PyTorch weights only",
        subject = "mistralai/Mistral-Small-3.2-24B",
        detail = "This repo ships safetensors. Converting to GGUF needs a desktop — the app won't " +
            "pretend otherwise.",
        remedies = listOf(
            ai.ondevice.data.hf.Remedy(
                "Search for Mistral-Small-3.2-24B-GGUF",
                RemedyAction.SearchRepo("Mistral-Small-3.2-24B-GGUF"),
                primary = true,
            ),
            ai.ondevice.data.hf.Remedy("bartowski", RemedyAction.OpenMirror("bartowski", "Mistral-Small-3.2-24B-GGUF")),
            ai.ondevice.data.hf.Remedy("unsloth", RemedyAction.OpenMirror("unsloth", "Mistral-Small-3.2-24B-GGUF")),
            ai.ondevice.data.hf.Remedy("mradermacher", RemedyAction.OpenMirror("mradermacher", "Mistral-Small-3.2-24B-GGUF")),
        ),
    ),
    Resolution.Refused(
        kind = RefusalKind.UNKNOWN_ARCHITECTURE,
        title = "Unsupported architecture",
        subject = "arch plamo3",
        detail = "llama.cpp b6482 — the build installed on this device — has 41 architectures and " +
            "this isn't one of them. A newer runtime may add it.",
        remedies = listOf(
            ai.ondevice.data.hf.Remedy("Check for runtime update", RemedyAction.CheckRuntimeUpdate, primary = true),
            ai.ondevice.data.hf.Remedy(
                "Upstream issues",
                RemedyAction.OpenUrl("https://github.com/ggml-org/llama.cpp/issues?q=plamo3"),
            ),
        ),
    ),
    Resolution.Refused(
        kind = RefusalKind.GATED,
        title = "Gated repo",
        subject = "google/gemma-3-27b-it-qat-q4_0-gguf",
        detail = "Accept the licence on Hugging Face, then paste a token. The token is stored in the " +
            "Android Keystore and used for nothing else.",
        remedies = listOf(
            ai.ondevice.data.hf.Remedy(
                "Open repo page",
                RemedyAction.OpenUrl("https://huggingface.co/google/gemma-3-27b-it-qat-q4_0-gguf"),
                primary = true,
            ),
            ai.ondevice.data.hf.Remedy("Enter token", RemedyAction.EnterToken),
        ),
    ),
    Resolution.Refused(
        kind = RefusalKind.UNSCANNED,
        title = "Unscanned files",
        subject = "A warning, not a block",
        detail = "Hugging Face hasn't scanned these files. GGUF has had template-injection " +
            "vulnerabilities. Pickle files, if present, are blocked outright.",
        remedies = listOf(
            ai.ondevice.data.hf.Remedy("Continue anyway", RemedyAction.ContinueAnyway("")),
        ),
    ),
)
