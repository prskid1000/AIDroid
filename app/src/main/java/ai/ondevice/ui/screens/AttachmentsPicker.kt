package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.MissingComponent
import ai.ondevice.core.ModelAttachment
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring

/**
 * The components a run is built from, for whichever screen is asking.
 *
 * One implementation because it is one idea. This lived inside the Image
 * screen, and when Video needed the same thing it got a plain list of switches
 * instead — which lost the family grouping, the warnings about a part the model
 * cannot run without, the "installed but not chosen" case, and the n/a state
 * for a component the architecture has no use for. Those are not decoration:
 * they are the difference between a screen that is empty because there is
 * nothing to add and one that is empty because the run is about to fail.
 *
 * @param strengthFor the per-run dial a role takes, or null where it takes
 *   none. It differs by screen — a ControlNet is weighted by `control_strength`
 *   on a still and by `vace_strength` on a clip — which is why it is asked for
 *   rather than assumed.
 */
@Composable
fun AttachmentsPicker(
    available: List<ModelAttachment>,
    armedCount: Int,
    missing: List<MissingComponent>,
    unchosenRoles: List<AttachmentRole>,
    /** What the loader called this checkpoint, for the "not used by…" line. */
    architectureLabel: String?,
    onToggle: (String) -> Unit,
    onWeight: (String, Float) -> Unit,
    strengthFor: (AttachmentRole) -> Float? = { null },
    onStrength: (AttachmentRole, Float) -> Unit = { _, _ -> },
    /** Said when the library holds nothing this model can take. */
    emptyHelp: String = "A prompt encoder, a VAE, a LoRA, a ControlNet — whichever of them " +
        "this architecture can take. Downloading one is enough for the parts a run cannot do " +
        "without; the rest are chosen under All Parameters.",
) {
    if (available.isEmpty()) {
        SectionKicker("Components", Modifier.padding(top = 20.dp, bottom = 8.dp))
        MissingCards(missing)
        NCard {
            // Two different situations wore the same sentence: a library with
            // no add-ons in it, and a library holding several of one role with
            // nobody having said which. Only the second is a decision waiting.
            if (unchosenRoles.isEmpty()) {
                Text("Nothing chosen for this model", style = NocturneType.CardTitleSm)
                Text(
                    emptyHelp,
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            } else {
                Text("Installed, not chosen", style = NocturneType.CardTitleSm)
                Text(
                    unchosenRoles.joinToString(", ") { it.label },
                    style = NocturneType.CardBody,
                    color = NocturneColors.Accent200,
                )
                Text(
                    "Each of these fits this model and more than one file could fill it, so the " +
                        "choice is yours: pick one per role under All Parameters and it appears " +
                        "here as a switch.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            }
        }
        return
    }

    SectionKicker(
        "Components · $armedCount of ${available.size} on",
        Modifier.padding(top = 20.dp, bottom = 8.dp),
    )

    // Said here, next to the switches that fix it, rather than after Generate
    // has spent a minute finding out.
    MissingCards(missing)

    // Grouped by what the thing is *for*, in a fixed order, so the four ways of
    // encoding a prompt sit under one heading instead of reading as four
    // unrelated files scattered through the list.
    available
        .groupBy { it.role.family }
        .toList()
        .sortedBy { (family, _) -> family.ordinal }
        .forEach { (family, inFamily) ->
            NHelp(family.label, Modifier.padding(top = 10.dp, bottom = 4.dp))
            inFamily.forEach { attachment ->
                Column(
                    Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp)
                        .background(
                            if (attachment.enabled) NocturneColors.Accent900 else NocturneColors.Surface,
                            Radius.Md,
                        )
                        .ring(
                            if (attachment.enabled) NocturneColors.Accent else NocturneColors.Divider,
                            Radius.Md,
                        )
                        .then(
                            if (attachment.applicable) {
                                Modifier.nClickableFlat { onToggle(attachment.modelId) }
                            } else {
                                Modifier
                            },
                        )
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                ) {
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(9.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        // The role leads, because the role is the slot; the
                        // file is which one is in it.
                        Text(
                            attachment.role.label,
                            style = NocturneType.Row,
                            color = if (attachment.enabled) NocturneColors.Accent200 else NocturneColors.Text,
                            modifier = Modifier.weight(1f),
                        )
                        Text(
                            when {
                                !attachment.applicable -> "n/a"
                                attachment.enabled -> "on"
                                else -> "off"
                            },
                            style = NocturneType.Mono2Xs,
                            color = if (attachment.enabled) NocturneColors.Accent else NocturneColors.TextMuted,
                        )
                    }
                    // Chosen, but this model has no use for it — so it is not
                    // armed, not passed to the loader, and says which.
                    if (!attachment.applicable) {
                        Text(
                            "Not used by ${architectureLabel ?: "this model"}",
                            style = NocturneType.Help,
                            color = NocturneColors.TextMuted,
                            modifier = Modifier.padding(top = 2.dp),
                        )
                    }
                    // Which file is in the slot — by the name it is known by,
                    // which is the one given by hand where there is one.
                    Text(
                        attachment.displayName,
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMuted,
                        modifier = Modifier.padding(top = 2.dp),
                    )
                    // The role is right and the file is wrong, which nothing
                    // else on this row can show: both T5-XXLs are called
                    // T5-XXL. Said rather than corrected — the choice stays
                    // the user's, and the loader will pass whatever is here.
                    attachment.mismatch?.let { note ->
                        Text(
                            note,
                            style = NocturneType.Help,
                            color = NocturneColors.Accent200,
                            modifier = Modifier.padding(top = 3.dp),
                        )
                    }
                    // Only the roles the runtime actually weights get a dial,
                    // and weight is a per-run thought, so it lives here rather
                    // than beside the choice of file.
                    if (attachment.enabled && attachment.role.weighted) {
                        LabeledSlider(
                            label = "Weight",
                            value = attachment.weight,
                            display = String.format("%.2f", attachment.weight),
                            range = 0f..2f,
                            onChange = { onWeight(attachment.modelId, it) },
                            modifier = Modifier.padding(top = 6.dp),
                        )
                    }
                    if (attachment.enabled) {
                        strengthFor(attachment.role)?.let { strength ->
                            LabeledSlider(
                                label = "Strength",
                                value = strength,
                                display = String.format("%.2f", strength),
                                range = 0f..1f,
                                onChange = { onStrength(attachment.role, it) },
                                modifier = Modifier.padding(top = 6.dp),
                            )
                        }
                    }
                }
            }
        }
}

@Composable
private fun MissingCards(missing: List<MissingComponent>) {
    missing.forEach { entry ->
        NCard(Modifier.padding(bottom = 8.dp)) {
            // A substitution is a choice, not a fault, and reads as one.
            // Neither is a part that is on its way: there is nothing to do
            // about it and nothing has gone wrong, so it does not take the
            // colour the other findings do.
            Text(
                entry.what,
                style = NocturneType.CardTitleSm,
                color = when (entry.state) {
                    MissingComponent.State.SUBSTITUTES,
                    MissingComponent.State.ARRIVING,
                    -> NocturneColors.TextMuted
                    else -> NocturneColors.Accent200
                },
            )
            Text(
                "${entry.because}. ${entry.remedy}.",
                style = NocturneType.CardBody,
                color = NocturneColors.Text.copy(alpha = 0.8f),
            )
        }
    }
}
