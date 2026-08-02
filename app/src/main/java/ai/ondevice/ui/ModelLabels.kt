package ai.ondevice.ui

import ai.ondevice.core.FileLabels
import ai.ondevice.core.Labels
import ai.ondevice.data.db.ModelEntity

/**
 * How an installed model is named in a list you pick from.
 *
 * `displayName` is the repo, and a repo can hold several models: the SD 3.5
 * encoder repo gives CLIP-L, CLIP-G and a T5 the same name, and the diffusers
 * layout calls every component's weights `diffusion_pytorch_model.safetensors`,
 * so neither the name nor the filename is reliably a distinguishing one.
 *
 * That is a display problem in some places and a correctness problem in others:
 * a dropdown hands back the label rather than the row, so two rows sharing a
 * label means choosing the second selects the first.
 *
 * Qualifiers are added only where they are needed and in the order that tells
 * a person the most: what slot it fills, at what precision, and finally the
 * filename — trimmed to the shortest tail that is unique among these paths, so
 * a folder appears only when the filename alone cannot separate them.
 *
 * A label set by hand on the model's own screen replaces the repo name and is
 * normally the end of it: the app qualifies what it derived, and a person who
 * has named something has said what they meant. Two models given the *same*
 * hand-written name are qualified like any other namesake, because the point is
 * to be able to tell them apart.
 */
fun List<ModelEntity>.pickerLabels(): List<String> {
    val fileLabels = FileLabels.distinguish(map { it.localPath })
    return Labels.unique(
        map { model ->
            Labels.Item(
                // A name typed by hand is the answer, not a starting point.
                name = model.label,
                // The quant is not a tie-breaker, it is part of what this is:
                // it appears in the model's own id, and two installs of one
                // repo at different precisions are two different things to run.
                always = listOf(model.quant),
                qualifiers = listOf(
                    model.attachmentRole?.label,
                    model.architecture,
                    fileLabels[model.localPath],
                ),
            )
        },
    )
}

/** The label for one model within its list, for showing the current choice. */
fun List<ModelEntity>.labelFor(model: ModelEntity?): String? {
    if (model == null) return null
    val index = indexOfFirst { it.id == model.id }
    return if (index >= 0) pickerLabels()[index] else model.label
}
