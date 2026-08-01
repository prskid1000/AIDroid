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

/** **S5 — Honest refusal.** SPEC §1.2 made visible: "Won't run" is an acceptable answer; a crash is not. */
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
            Text(
                "The ways a repo can be turned down. A real refusal appears under the paste " +
                    "field on Add model, naming the repo, this device and this runtime, with " +
                    "remedies you can act on.",
                style = NocturneType.CardBody,
                color = NocturneColors.Neutral400,
            )
            RefusalKind.entries.forEach { kind ->
                NCard {
                    Text(kind.heading, style = NocturneType.CardTitleSm, color = NocturneColors.Text)
                    Text(
                        kind.explanation,
                        style = NocturneType.CardBody,
                        color = NocturneColors.Text.copy(alpha = 0.8f),
                    )
                }
            }
        }
    }
}

/** One refusal, rendered the same way wherever it appears — on S1 under the paste field, or gathered here. */
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

