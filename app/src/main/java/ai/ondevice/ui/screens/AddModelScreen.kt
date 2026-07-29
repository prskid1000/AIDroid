package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
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
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.core.SpeedClass
import ai.ondevice.core.VerdictTone
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NFieldLabel
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NInput
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
import ai.ondevice.ui.theme.Touch
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.AddModelViewModel

/**
 * **S1 — Add model.**
 *
 * The paste field is the primary affordance and is deliberately first on the
 * screen (Appendix A #8): curated lists are convenience shortcuts and must
 * never be the only path. Below it, the verdict card shows the fit arithmetic
 * before anything is downloaded, and the quant list annotates every variant
 * with its size and speed class.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun AddModelScreen(
    onBack: () -> Unit,
    onShowRefusals: () -> Unit,
    onDownloadStarted: () -> Unit,
    viewModel: AddModelViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = { PushToolbar("Add model", onBack) },
        contentPadding = androidx.compose.foundation.layout.PaddingValues(
            start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp,
        ),
    ) {
        Column(Modifier.verticalScroll(rememberScrollState())) {

            NFieldLabel("Hugging Face ID, URL, or direct .gguf link")
            Row(
                Modifier.fillMaxWidth().padding(bottom = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                Box(Modifier.weight(1f)) {
                    NInput(
                        value = state.query,
                        onValueChange = viewModel::onQueryChange,
                        minHeight = 44.dp,
                        textStyle = NocturneType.MonoValue.copy(fontSize = NocturneType.MonoCode.fontSize),
                        placeholder = "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                    )
                }
                NButton(
                    text = if (state.resolving) "…" else "Resolve",
                    onClick = viewModel::resolve,
                    style = NButtonStyle.Primary,
                    minHeight = 44.dp,
                    enabled = state.query.isNotBlank() && !state.resolving,
                )
            }

            Row(Modifier.padding(bottom = 16.dp)) {
                NHelp("Also accepts a local file — ")
                Text(
                    "import .gguf from storage",
                    style = NocturneType.Help,
                    color = NocturneColors.Accent,
                    modifier = Modifier.nClickableFlat { viewModel.importLocal() },
                )
            }

            // — the resolved model, or the reason there isn't one —

            state.refusal?.let { refusal ->
                RefusalCard(refusal, onRemedy = viewModel::applyRemedy)
                NButton(
                    "See all resolve results",
                    onClick = onShowRefusals,
                    style = NButtonStyle.Secondary,
                    block = true,
                    modifier = Modifier.padding(top = 12.dp),
                )
            }

            state.resolved?.let { resolved ->
                NCard(padding = androidx.compose.foundation.layout.PaddingValues(0.dp), gap = 0.dp) {
                    Column(Modifier.padding(start = 13.dp, end = 13.dp, top = 12.dp, bottom = 11.dp)) {
                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(9.dp),
                            verticalAlignment = Alignment.Top,
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text(resolved.displayName, style = NocturneType.CardTitleLg)
                                Text(
                                    listOfNotNull(
                                        resolved.owner.takeIf { it.isNotBlank() },
                                        resolved.architecture,
                                        resolved.parameterCount?.let { paramLabel(it) },
                                    ).joinToString(" · "),
                                    style = NocturneType.MonoXs,
                                    color = NocturneColors.TextMuted,
                                    modifier = Modifier.padding(top = 2.dp),
                                )
                            }
                            state.verdict?.let { verdict ->
                                VerdictTag(verdict.verdict)
                            }
                        }

                        FlowRow(
                            Modifier.padding(top = 10.dp),
                            horizontalArrangement = Arrangement.spacedBy(5.dp),
                            verticalArrangement = Arrangement.spacedBy(5.dp),
                        ) {
                            resolved.contextLength?.let {
                                NTag("${Fmt.contextLabel(it)} ctx", style = NTagStyle.Neutral)
                            }
                            if (resolved.chatTemplate != null) {
                                NTag("chat template ✓", style = NTagStyle.Neutral)
                            }
                            NTag(
                                when {
                                    resolved.securityStatus == null -> "unscanned"
                                    resolved.securityStatus.equals("safe", true) -> "scanned clean"
                                    else -> resolved.securityStatus
                                },
                                style = NTagStyle.Neutral,
                            )
                            NTag(if (resolved.gated) "gated" else "not gated", style = NTagStyle.Outline)
                            if (resolved.metadataFromHeader) {
                                NTag("header-parsed", style = NTagStyle.Outline)
                            }
                        }
                    }

                    // The fit arithmetic, shown before download rather than after
                    // a crash. SPEC §3.3: show the sum, not a bare yes/no.
                    state.verdict?.let { verdict ->
                        Column(
                            Modifier
                                .fillMaxWidth()
                                .background(verdictGround(verdict.verdict.tone))
                                .padding(start = 13.dp, end = 13.dp, top = 11.dp, bottom = 12.dp),
                        ) {
                            Text(
                                verdict.estimate.summary(),
                                style = NocturneType.CardTitleSm,
                                color = verdictInk(verdict.verdict.tone),
                            )
                            Text(
                                verdict.estimate.shortWorking(),
                                style = NocturneType.MonoSm,
                                color = verdictInkMuted(verdict.verdict.tone),
                                modifier = Modifier.padding(top = 5.dp),
                            )
                            Text(
                                verdict.headroomNote(state.totalRamBytes),
                                style = NocturneType.MonoSm,
                                color = verdictInkMuted(verdict.verdict.tone).copy(alpha = 0.75f),
                                modifier = Modifier.padding(top = 5.dp),
                            )
                        }
                    }
                }

                SectionKicker(
                    "Quant variants · ${resolved.quants.size} in repo",
                    Modifier.padding(top = 20.dp, bottom = 8.dp),
                )

                resolved.quants.forEach { quant ->
                    val selected = quant.name == state.selectedQuant
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .padding(bottom = 6.dp)
                            .background(
                                if (selected) NocturneColors.Accent900 else NocturneColors.Surface,
                                Radius.Md,
                            )
                            .ring(if (selected) NocturneColors.Accent else NocturneColors.Divider, Radius.Md)
                            .nClickableFlat { viewModel.selectQuant(quant.name) }
                            .padding(horizontal = 12.dp, vertical = 11.dp),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                quant.name,
                                style = NocturneType.MonoValue.copy(
                                    fontSize = androidx.compose.ui.unit.TextUnit(13f, androidx.compose.ui.unit.TextUnitType.Sp),
                                ),
                                color = if (selected) NocturneColors.Accent200 else NocturneColors.Text,
                            )
                            NHelp(quant.note, Modifier.padding(top = 1.dp))
                        }
                        Column(horizontalAlignment = Alignment.End) {
                            Text(
                                Fmt.bytes(quant.totalBytes),
                                style = NocturneType.MonoValue,
                                color = NocturneColors.Text.copy(alpha = 0.85f),
                            )
                            Text(
                                quant.speedClass.label,
                                style = NocturneType.Help,
                                color = if (quant.speedClass == SpeedClass.OPENCL_FAST) {
                                    NocturneColors.Accent300
                                } else {
                                    NocturneColors.TextMuted
                                },
                            )
                        }
                    }
                }

                // The footnote has to describe the runtime that is actually
                // installed. Promising a GPU fast path on a build with no GPU
                // backend compiled in is the assertion SPEC §8.2 forbids.
                NHelp(
                    if (resolved.quants.any { it.speedClass == SpeedClass.OPENCL_FAST }) {
                        "Q4_0 hits the Adreno OpenCL fast path on this device. Other quants fall back to CPU."
                    } else {
                        "This runtime build has no GPU backend, so every quant runs on CPU. Smaller " +
                            "quants are faster here for that reason alone."
                    },
                    Modifier.padding(top = 8.dp),
                )

                if (resolved.companions.isNotEmpty()) {
                    SectionKicker("Companions · auto-paired", Modifier.padding(top = 20.dp, bottom = 8.dp))
                    resolved.companions.forEach { companion ->
                        Row(
                            Modifier.fillMaxWidth().padding(vertical = 8.dp),
                            horizontalArrangement = Arrangement.spacedBy(9.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                companion.file.filename.substringAfterLast('/'),
                                style = NocturneType.MonoXs,
                                modifier = Modifier.weight(1f),
                            )
                            Text(
                                Fmt.bytes(companion.file.sizeBytes),
                                style = NocturneType.Meta,
                                color = NocturneColors.TextMuted,
                            )
                        }
                    }
                    NHelp("Queued with the weights — a multi-file model is never hand-assembled.")
                }

                val selectedQuant = resolved.quants.firstOrNull { it.name == state.selectedQuant }
                val runnable = state.verdict?.verdict?.runnable == true
                // The required companions are downloaded with the weights, so
                // they belong in the figure on the button. Kokoro's 55 voice
                // packs are 28 MB against a 92 MB graph — quoting the weights
                // alone understates the download by a quarter, and the number a
                // user agrees to should be the number that gets transferred.
                val downloadBytes = (selectedQuant?.totalBytes ?: 0) +
                    resolved.companions.filter { it.role.required || it.autoSelected }
                        .sumOf { it.file.sizeBytes }
                NButton(
                    text = if (runnable) {
                        "Download ${Fmt.bytes(downloadBytes)}"
                    } else {
                        state.verdict?.verdict?.label ?: "Not runnable"
                    },
                    onClick = { viewModel.download(); onDownloadStarted() },
                    style = NButtonStyle.Primary,
                    block = true,
                    enabled = runnable,
                    minHeight = Touch.Primary,
                    modifier = Modifier.padding(top = 16.dp),
                )
            }

            if (state.resolved == null && state.refusal == null && !state.resolving) {
                EmptyResolveHint()
                StarterTable(onPick = { repoId ->
                    viewModel.onQueryChange(repoId)
                    viewModel.resolve()
                })
            }
        }
    }
}

@Composable
private fun EmptyResolveHint() {
    NCard(Modifier.padding(top = 8.dp)) {
        Text("Runtime-locked, not model-locked", style = NocturneType.CardTitleSm)
        Text(
            "Any model whose artifacts match a bundled runtime and fits this device is usable — " +
                "paste its ID. A model released tomorrow works without an app update, provided its " +
                "architecture is already known to the runtime.",
            style = NocturneType.CardBody,
            color = NocturneColors.Text.copy(alpha = 0.8f),
        )
    }
}

/**
 * Somewhere to start.
 *
 * "Paste a Hugging Face ID" assumes you already know one, and for two of the
 * four runtimes that is a genuinely unfair assumption: whisper's weights live
 * in a repo named after the runtime rather than the model, and Kokoro's ONNX
 * export is published by a different owner than the original. Neither is
 * guessable.
 *
 * Tapping a row fills the paste field and resolves it — the same path a typed
 * ID takes, with the same verdict and the same refusals. Nothing here is a
 * shortcut around the fit arithmetic; it only saves the typing.
 */
@Composable
private fun StarterTable(onPick: (String) -> Unit) {
    ai.ondevice.core.StarterModels.BY_MODALITY.forEach { (modality, entries) ->
        if (entries.isEmpty()) return@forEach
        SectionKicker(modality.label, Modifier.padding(top = 18.dp, bottom = 7.dp))
        StarterRows(entries, onPick)
    }

    // Add-ons last: they attach to a diffusion model, so installing one before
    // you have a base model gives you nothing to attach it to.
    SectionKicker("Image add-ons", Modifier.padding(top = 18.dp, bottom = 7.dp))
    StarterRows(ai.ondevice.core.StarterModels.ADDONS, onPick)
    NHelp(
        "These appear in the Image screen's Attachments section once installed — the app files " +
            "them by role from their filenames, so it never needs to know the model by name.",
        Modifier.padding(top = 7.dp),
    )

    NHelp(
        "Tapping one fills the field above and resolves it — the same path a pasted ID takes, " +
            "with the same fit check. These are known to work in this build; they are not the " +
            "only things that do.",
        Modifier.padding(top = 10.dp),
    )
}

@Composable
private fun StarterRows(
    entries: List<ai.ondevice.core.StarterModel>,
    onPick: (String) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        entries.forEach { entry ->
            Row(
                Modifier
                    .fillMaxWidth()
                    .background(NocturneColors.Surface, Radius.Md)
                    .ring(NocturneColors.Divider, Radius.Md)
                    .nClickableFlat { onPick(entry.repoId) }
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(9.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        entry.role?.let { NTag(it.label, style = NTagStyle.Outline) }
                        // The id is the thing being copied into the field, so it
                        // leads and it is monospaced — it is a value, not prose.
                        Text(entry.repoId, style = NocturneType.MonoValue)
                    }
                    Text(
                        entry.summary,
                        style = NocturneType.CardBody,
                        color = NocturneColors.Text.copy(alpha = 0.75f),
                        modifier = Modifier.padding(top = 2.dp),
                    )
                }
                // Stated before the tap, because the download is the part that
                // costs something.
                Text(
                    entry.sizeHint,
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                )
            }
        }
    }
}

/**
 * The verdict mark. No red, no green — the palette is mono, so "runnable" is an
 * accent tag with a check, "caveat" is an accent outline, and "no" is a neutral
 * disc with a cross. Weight comes from the mark.
 */
@Composable
fun VerdictTag(verdict: ai.ondevice.core.Verdict) {
    when (verdict.tone) {
        VerdictTone.AFFIRMATIVE -> NTag(verdict.label, style = NTagStyle.Accent, leadingIcon = NIcons.Check)
        VerdictTone.CAVEAT -> NTag(verdict.label, style = NTagStyle.Outline)
        VerdictTone.REFUSAL -> NTag(verdict.label, style = NTagStyle.Neutral, leadingIcon = NIcons.Cross)
    }
}

fun verdictGround(tone: VerdictTone) = when (tone) {
    VerdictTone.AFFIRMATIVE -> NocturneColors.Accent900
    VerdictTone.CAVEAT -> NocturneColors.Accent900
    VerdictTone.REFUSAL -> NocturneColors.Neutral900
}

fun verdictInk(tone: VerdictTone) = when (tone) {
    VerdictTone.REFUSAL -> NocturneColors.Neutral200
    else -> NocturneColors.Accent200
}

fun verdictInkMuted(tone: VerdictTone) = when (tone) {
    VerdictTone.REFUSAL -> NocturneColors.Neutral300
    else -> NocturneColors.Accent300
}

private fun paramLabel(count: Long): String = when {
    count >= 1_000_000_000 -> String.format("%.2fB", count / 1_000_000_000.0)
    count >= 1_000_000 -> String.format("%.0fM", count / 1_000_000.0)
    else -> count.toString()
}
