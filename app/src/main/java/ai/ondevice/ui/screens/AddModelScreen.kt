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
import androidx.compose.foundation.layout.widthIn
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
import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.Fmt
import ai.ondevice.core.Modality
import ai.ondevice.core.VerdictTone
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCardMeta
import ai.ondevice.ui.components.NDropdown
import ai.ondevice.ui.components.NMetaText
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
                        // Shows the shape of a Hugging Face id rather than a second copy of one.
                        placeholder = ai.ondevice.core.StarterModels.ALL.first().repoId,
                    )
                }
                NButton(
                    // A Hugging Face id has a slash in it, so text without one can only ever fail to resolve.
                    text = when {
                        state.resolving || state.searching -> "…"
                        !state.query.contains('/') -> "Search"
                        else -> "Resolve"
                    },
                    onClick = viewModel::resolve,
                    style = NButtonStyle.Primary,
                    minHeight = 44.dp,
                    enabled = state.query.isNotBlank() && !state.resolving && !state.searching,
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

            // — search results, when the query was a name rather than an id —

            if (state.searchResults.isNotEmpty()) {
                SectionKicker(
                    "Search · ${state.searchResults.size} on Hugging Face",
                    Modifier.padding(bottom = 8.dp),
                )
                state.searchResults.forEach { result ->
                    NCard(
                        Modifier
                            .padding(bottom = 7.dp)
                            .nClickableFlat { viewModel.openSearchResult(result.id) },
                        gap = 5.dp,
                    ) {
                        Text(result.id, style = NocturneType.CardTitleSm)
                        NCardMeta {
                            result.pipelineTag?.let { NMetaText(it) }
                            NMetaText("${Fmt.grouped(result.downloads.toInt())} downloads")
                            if (result.likes > 0) NMetaText("${result.likes} likes")
                        }
                    }
                }
                // Search returns whatever matches the name.
                NHelp(
                    "Ranked by downloads. Pick one to resolve it — until then nothing here has been " +
                        "checked against this device or against the bundled runtimes.",
                    Modifier.padding(bottom = 16.dp),
                )
            }

            if (state.searching) {
                NHelp("Searching Hugging Face…", Modifier.padding(bottom = 16.dp))
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

                // "Quant variants" is only true when they are quantisations of
                // one model. h94/IP-Adapter holds twelve files that are twelve
                // different adapters — SD 1.5 against SDXL, plain against plus
                // against face — and calling those precisions of each other
                // invites picking the one that cannot work with your model.
                val areQuants = resolved.quants.any {
                    it.name.matches(Regex("""(?i)(IQ|Q)\d.*|BF16|F16|F32"""))
                }
                SectionKicker(
                    if (areQuants) {
                        "Quant variants · ${resolved.quants.size} in repo"
                    } else {
                        "Files · ${resolved.quants.size} in repo · pick one"
                    },
                    Modifier.padding(top = 20.dp, bottom = 8.dp),
                )
                if (!areQuants) {
                    NHelp(
                        "These are not precisions of one model — they are separate files, and " +
                            "which of them works depends on the model you mean to use it with.",
                        Modifier.padding(bottom = 8.dp),
                    )
                }
                state.intendedRole?.takeIf { state.roleWasSuggested }?.let { role ->
                    NHelp(
                        "Narrowed to what can fill the ${role.label} slot, because that is the " +
                            "card you came from.",
                        Modifier.padding(bottom = 8.dp),
                    )
                }

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
                            // A blocked variant is shown, not hidden — a row that vanishes reads as a repo that does not have it.
                            NHelp(
                                quant.blockedReason?.let { "Cannot run here — $it" } ?: quant.note,
                                Modifier.padding(top = 1.dp),
                            )
                            // A caution is not a block.
                            quant.cautionReason?.takeIf { quant.blockedReason == null }?.let {
                                Text(
                                    it,
                                    style = NocturneType.Help,
                                    color = NocturneColors.Accent300,
                                    modifier = Modifier.padding(top = 3.dp),
                                )
                            }
                        }
                        Text(
                            Fmt.bytes(quant.totalBytes),
                            style = NocturneType.MonoValue,
                            color = NocturneColors.Text.copy(alpha = 0.85f),
                        )
                    }
                }

                NHelp(
                    "Size is the whole story: a smaller quant is faster and leaves more room, " +
                        "and pays for it in quality.",
                    Modifier.padding(top = 8.dp),
                )

                if (resolved.companions.isNotEmpty()) {
                    SectionKicker("Companions", Modifier.padding(top = 20.dp, bottom = 8.dp))

                    // One tap for "I have these already". A role is filled from
                    // the library now, so the copy in this repo is a
                    // convenience, not the only source.
                    val anyChosen = resolved.companions.any { group ->
                        (state.companionChoice[group.role] ?: group.selected).isNotEmpty()
                    }
                    Row(
                        Modifier.fillMaxWidth().padding(bottom = 8.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        NButton(
                            if (anyChosen) "Weights only" else "Restore defaults",
                            onClick = {
                                if (anyChosen) viewModel.skipAllCompanions()
                                else viewModel.restoreCompanionDefaults()
                            },
                            modifier = Modifier.weight(1f),
                        )
                    }

                    resolved.companions.forEach { group ->
                        val picked = state.companionChoice[group.role] ?: group.selected

                        val parts = group.kind == ai.ondevice.data.hf.CompanionGroup.Kind.PARTS

                        Row(
                            Modifier.fillMaxWidth().padding(top = 6.dp, bottom = 2.dp),
                            horizontalArrangement = Arrangement.spacedBy(9.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(group.role.label, style = NocturneType.CardTitleSm, modifier = Modifier.weight(1f))
                            // Fifty-five voice packs is a lot of tapping to say
                            // "actually, none of these" — and one file is one
                            // tap too many when the answer is "I have it".
                            val allPicked = picked.size == group.candidates.size
                            Text(
                                if (picked.isEmpty()) {
                                    if (parts) "all" else "take it"
                                } else if (parts && !allPicked) {
                                    "all"
                                } else {
                                    "none"
                                },
                                style = NocturneType.Meta,
                                color = NocturneColors.Accent,
                                modifier = Modifier
                                    .nClickableFlat {
                                        viewModel.chooseAllCompanions(
                                            group.role,
                                            picked.isEmpty() || (parts && !allPicked),
                                        )
                                    }
                                    .padding(horizontal = 8.dp, vertical = 2.dp),
                            )
                            Text(
                                Fmt.bytes(group.candidates.filter { it.file.filename in picked }
                                    .sumOf { it.file.sizeBytes }),
                                style = NocturneType.Meta,
                                color = NocturneColors.TextMuted,
                            )
                        }
                        group.note?.let { NHelp(it, Modifier.padding(bottom = 4.dp)) }

                        // Skipping is allowed and is not free. Say which of the
                        // two it is rather than disabling the control.
                        if (picked.isEmpty() && group.role.requiredBy(resolved.architecture)) {
                            NHelp(
                                "Nothing taken. The model needs a ${group.role.label.lowercase()} " +
                                    "from somewhere before it runs — install one separately and " +
                                    "choose it under All Parameters.",
                                Modifier.padding(bottom = 4.dp),
                            )
                        }

                        group.candidates.forEach { candidate ->
                            val chosen = candidate.file.filename in picked
                            Row(
                                Modifier
                                    .fillMaxWidth()
                                    .padding(bottom = 4.dp)
                                    .background(
                                        if (chosen) NocturneColors.Accent900 else NocturneColors.Surface,
                                        Radius.Md,
                                    )
                                    .ring(
                                        if (chosen) NocturneColors.Accent else NocturneColors.Divider,
                                        Radius.Md,
                                    )
                                    .nClickableFlat {
                                        viewModel.chooseCompanion(group.role, candidate.file.filename)
                                    }
                                    .padding(horizontal = 12.dp, vertical = 9.dp),
                                horizontalArrangement = Arrangement.spacedBy(9.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Text(
                                    candidate.file.filename.substringAfterLast('/'),
                                    style = NocturneType.MonoXs,
                                    modifier = Modifier.weight(1f),
                                )
                                Text(
                                    Fmt.bytes(candidate.file.sizeBytes),
                                    style = NocturneType.Meta,
                                    color = NocturneColors.TextMuted,
                                )
                            }
                        }
                    }
                    NHelp(
                        "Where a repo ships the same file at several precisions, or several " +
                            "things that fill the same slot, one is chosen for you. Everything " +
                            "here is a tap away from being changed.",
                        Modifier.padding(top = 4.dp),
                    )
                }

                // Type and role, stated rather than guessed.
                SectionKicker("What this is", Modifier.padding(top = 20.dp, bottom = 8.dp))

                NHelp("Type", Modifier.padding(bottom = 4.dp))
                // UNKNOWN is what the resolver said when it could not tell. It is
                // not something a person would ever mean to choose.
                val modalities = Modality.entries.filterNot { it == Modality.UNKNOWN }
                NDropdown(
                    options = modalities.map { it.label },
                    selected = state.selectedModality?.label,
                    onSelect = { label ->
                        modalities.firstOrNull { it.label == label }?.let(viewModel::setModality)
                    },
                    placeholder = "Choose what this model is…",
                )

                NHelp("Role", Modifier.padding(top = 10.dp, bottom = 4.dp))
                val baseLabel = "Base model — nothing hangs off it"
                val roleLabels = listOf(baseLabel) + AttachmentRole.entries.map { it.label }
                NDropdown(
                    options = roleLabels,
                    selected = when {
                        !state.roleAnswered -> null
                        state.selectedRole == null -> baseLabel
                        else -> state.selectedRole?.label
                    },
                    onSelect = { label ->
                        viewModel.setRole(AttachmentRole.entries.firstOrNull { it.label == label })
                    },
                    placeholder = "Base model, or which add-on slot…",
                )

                val selectedQuant = resolved.quants.firstOrNull { it.name == state.selectedQuant }
                val runnable = state.verdict?.verdict?.runnable == true
                // The companions are downloaded with the weights, so they belong in the figure on the button.
                val downloadBytes = (selectedQuant?.totalBytes ?: 0) + state.companionBytes
                // A variant can be blocked while the *model* is fine, so this is a separate question from the verdict.
                val blocked = selectedQuant?.blockedReason
                NButton(
                    text = when {
                        !runnable -> state.verdict?.verdict?.label ?: "Not runnable"
                        blocked != null -> "This variant cannot run here"
                        !state.classified -> "Choose a type and a role"
                        else -> "Download ${Fmt.bytes(downloadBytes)}"
                    },
                    onClick = { viewModel.download(); onDownloadStarted() },
                    style = NButtonStyle.Primary,
                    block = true,
                    enabled = runnable && state.classified && blocked == null,
                    minHeight = Touch.Primary,
                    modifier = Modifier.padding(top = 16.dp),
                )
            }

            if (state.resolved == null && state.refusal == null && !state.resolving) {
                StarterTable(onPick = viewModel::pickStarter)
            }
        }
    }
}

/** Somewhere to start. */
@Composable
private fun StarterTable(onPick: (ai.ondevice.core.StarterModel) -> Unit) {
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
    onPick: (ai.ondevice.core.StarterModel) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        entries.forEach { entry ->
            Row(
                Modifier
                    .fillMaxWidth()
                    .background(NocturneColors.Surface, Radius.Md)
                    .ring(NocturneColors.Divider, Radius.Md)
                    .nClickableFlat { onPick(entry) }
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
                // costs something. Capped, because it is a figure beside a
                // name and not a second description: an over-long one used to
                // take the row and leave the repo id wrapping four characters
                // at a time.
                Text(
                    entry.sizeHint,
                    style = NocturneType.MonoXs,
                    color = NocturneColors.TextMuted,
                    textAlign = androidx.compose.ui.text.style.TextAlign.End,
                    modifier = Modifier.widthIn(max = 118.dp),
                )
            }
        }
    }
}

/** The verdict mark. */
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
