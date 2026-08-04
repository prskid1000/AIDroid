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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.engine.RuntimeRegistry
import ai.ondevice.params.ParamRow
import ai.ondevice.params.ParamType
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NCardKicker
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NTextArea
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.ParamsViewModel

/** **S8 — All parameters.** The proof of SPEC §1.5. */
@Composable
fun AllParametersScreen(
    onBack: () -> Unit,
    onOpenSamplerChain: () -> Unit,
    initialRuntime: String = RuntimeRegistry.LLAMA,
    /** Whose overrides to edit; null lets the screen pick the one in use. */
    initialModelId: String? = null,
    // Activity-scoped: the sampler-chain screen edits the same parameter set, so
    // the two must see one instance rather than each loading its own copy.
    viewModel: ParamsViewModel = activityParamsViewModel(),
) {
    LaunchedEffect(initialRuntime, initialModelId) {
        viewModel.setRuntime(initialRuntime, initialModelId)
    }

    val state by viewModel.state.collectAsStateWithLifecycle()

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = "Parameters",
                subtitle = listOfNotNull(
                    // The model leads: it is what these values are being saved
                    // against, and the screen never used to say.
                    state.modelLabel,
                    "${state.runtimeId} ${state.buildTag}",
                    "manifest v${state.manifestVersion}",
                ).joinToString(" · "),
                onBack = onBack,
                trailing = {
                    Icon(
                        NIcons.Rotate,
                        contentDescription = "Reset all",
                        tint = NocturneColors.Text,
                        modifier = Modifier.size(19.dp).nClickableFlat { viewModel.resetAll() },
                    )
                },
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 20.dp),
    ) {
        // One list, in full. There used to be Basic / Advanced / Expert / All
        // pills over it, which is four ways of looking at one thing and a
        // fourth that made the other three redundant. Search narrows it when
        // it needs narrowing.
        NInput(
            value = state.query,
            onValueChange = viewModel::setQuery,
            placeholder = "Search ${state.totalCount} parameters",
            minHeight = 40.dp,
            modifier = Modifier.padding(bottom = 10.dp),
        )

        Column(Modifier.verticalScroll(rememberScrollState())) {

            state.unsavedReason?.let { reason ->
                NCard(Modifier.padding(bottom = 10.dp), ring = NocturneColors.Neutral700) {
                    Text("These are not being saved", style = NocturneType.CardTitleSm)
                    Text(
                        reason,
                        style = NocturneType.CardBody,
                        color = NocturneColors.Text.copy(alpha = 0.8f),
                    )
                }
            }

            NHelp(
                "${state.visible.shownCount} available · ${state.visible.disabledCount} shown but " +
                    "not usable here, each with the reason under it",
                Modifier.padding(bottom = 4.dp),
            )

            // Reload-required edits are batched and applied once, not per-edit.
            if (state.needsReload) {
                NCard(
                    Modifier.padding(vertical = 8.dp),
                    ring = NocturneColors.Accent700,
                ) {
                    Text(
                        "${state.pendingReloadKeys.size} change(s) need a model reload",
                        style = NocturneType.CardTitleSm,
                    )
                    Text(
                        state.pendingReloadKeys.joinToString(", "),
                        style = NocturneType.MonoXs,
                        color = NocturneColors.Accent300,
                    )
                    if (state.reloadsOnNextRun) {
                        NHelp("This runtime loads at the start of a run, so the next one picks them up.")
                    }
                    NButton(
                        "Apply and reload",
                        onClick = viewModel::applyPendingReload,
                        style = NButtonStyle.Primary,
                        block = true,
                    )
                }
            }

            // Mirostat replaces the chain entirely — say so rather than
            // silently ignoring the user's ordering.
            if (viewModel.mirostatActive()) {
                NCard(Modifier.padding(bottom = 10.dp), ring = NocturneColors.Neutral700) {
                    Text("Mirostat is active", style = NocturneType.CardTitleSm)
                    Text(
                        "It replaces the sampler chain entirely, so top-k, top-p, min-p and the chain " +
                            "ordering have no effect until you set it back to 0.",
                        style = NocturneType.CardBody,
                        color = NocturneColors.Text.copy(alpha = 0.8f),
                    )
                }
            }

            var lastGroup: String? = null
            state.visible.rows.forEach { row ->
                val spec = row.spec
                if (spec.group != lastGroup) {
                    SectionKicker(groupLabel(spec.group), Modifier.padding(top = 16.dp, bottom = 4.dp))
                    lastGroup = spec.group
                }

                if (spec.type == ParamType.ORDERED_LIST) {
                    // The sampler chain needs a drag handle per row, so it gets
                    // its own screen; the manifest row is the way in.
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .padding(vertical = 11.dp)
                            .nClickableFlat(onClick = onOpenSamplerChain),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(spec.label, style = NocturneType.Row)
                            Text(
                                viewModel.samplerOrder().joinToString(" → "),
                                style = NocturneType.MonoXs,
                                color = NocturneColors.Accent300,
                            )
                        }
                        Text("→", style = NocturneType.Row, color = NocturneColors.Accent)
                    }
                } else {
                    ParamRow(
                        spec = spec,
                        values = state.values,
                        onChange = viewModel::setValue,
                        pathChoices = state.pathChoices,
                        disabledBecause = row.disabledBecause,
                    )
                }
            }

            // §16.6 — the escape hatch.
            NCard(
                Modifier.padding(top = 16.dp),
                ring = NocturneColors.Accent800,
            ) {
                NCardKicker("Escape hatch · §16.6")
                Text("Raw parameters", style = NocturneType.CardTitleSm)
                // A text area, not a line: this is where JSON gets typed by
                // hand, and JSON has newlines in it. It scrolls, so a long
                // object stays editable instead of running off the side.
                NTextArea(
                    value = state.rawJson,
                    onValueChange = viewModel::setRawJson,
                    minHeight = 120.dp,
                    textStyle = NocturneType.MonoCode,
                )
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        when (state.rawJsonParses) {
                            null -> "Empty"
                            true -> "Parses"
                            false -> "Not valid JSON yet"
                        },
                        style = NocturneType.Help,
                        color = if (state.rawJsonParses == false) {
                            NocturneColors.Neutral300
                        } else {
                            NocturneColors.Accent300
                        },
                        modifier = Modifier.weight(1f),
                    )
                    NButton(
                        "Format",
                        onClick = viewModel::formatRawJson,
                        enabled = state.rawJsonParses == true,
                        style = NButtonStyle.Ghost,
                    )
                }
                Text(
                    "Passed straight through to the runtime. Unknown keys are reported, never fatal — " +
                        "so anything the loaded .so supports is always reachable.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
                state.rawError?.let {
                    Text(it, style = NocturneType.Help, color = NocturneColors.Neutral300)
                }
                state.lastReport?.takeIf { it.hasRejections }?.let { report ->
                    Text(
                        "Runtime rejected: ${report.rejected.joinToString(", ")} — kept in your preset " +
                            "anyway, inert, in case a later build supports them.",
                        style = NocturneType.Help,
                        color = NocturneColors.Accent300,
                    )
                }
                NButton(
                    "Apply raw JSON",
                    onClick = viewModel::applyRawJson,
                    style = NButtonStyle.Secondary,
                    block = true,
                )
            }

            NHelp(
                "Manifest v${state.manifestVersion} (bundled v${state.bundledVersion}). It can retier, " +
                    "relabel and reveal parameters the installed engine already supports — it cannot " +
                    "add capability the engine lacks.",
                Modifier.padding(top = 14.dp),
            )
        }
    }
}

/**
 * The heading a parameter group gets.
 *
 * A component and the settings that belong to it are one thing to think about,
 * and the manifest now files them that way — the ControlNet's path and its
 * strength under one heading, the IP-Adapter's path, its strength and the
 * encoder it looks through under another. They used to be two headings apart,
 * which made the strength read as a global setting and the path as a component
 * with nothing to tune.
 *
 * `encoder` groups CLIP-L, CLIP-G, T5-XXL and the LLM because they are four
 * spellings of one job — turning the prompt into conditioning — rather than
 * four unrelated files.
 */
private fun groupLabel(group: String): String = when (group) {
    // Shared
    "generation" -> "Generation"
    "sampling" -> "Sampling"
    "model" -> "Model"
    "lora" -> "LoRA"
    "voice" -> "Voice"
    "output" -> "Output"

    // llama.cpp
    "dynatemp" -> "Dynamic temperature"
    "repetition" -> "Repetition"
    "dry" -> "DRY repetition"
    "xtc" -> "XTC"
    "mirostat" -> "Mirostat"
    "constraints" -> "Constrained output"
    "context" -> "Context & KV cache"
    "rope" -> "RoPE & long context"
    "loading" -> "Loading & threads"
    "vision" -> "Vision"
    "chat" -> "Chat template"

    // whisper.cpp
    "transcribe" -> "Transcription"
    "decoding" -> "Decoding"
    "segments" -> "Segments"
    "thresholds" -> "Quality thresholds"
    "audio" -> "Audio window"

    // stable-diffusion.cpp
    "prompt" -> "Prompt"
    "size" -> "Size"
    "guidance" -> "Guidance"
    "img2img" -> "Image to image"
    "encoder" -> "Prompt encoder"
    "decoder" -> "Decoder"
    "controlnet" -> "ControlNet"
    "ipadapter" -> "IP-Adapter"
    "embeddings" -> "Embeddings"
    "postprocess" -> "Post-processing"

    // kokoro
    "synthesis" -> "Synthesis"

    else -> group
}
