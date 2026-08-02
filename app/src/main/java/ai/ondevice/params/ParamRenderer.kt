package ai.ondevice.params

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.foundation.layout.Box
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import ai.ondevice.core.SparseParams
import ai.ondevice.ui.components.NChipRow
import ai.ondevice.ui.components.NDot
import ai.ondevice.ui.components.NEnumRow
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NSlider
import ai.ondevice.ui.components.NSwitch
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.NTextArea
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.theme.ruleBelow
import kotlinx.serialization.json.JsonPrimitive

/** SPEC §16.4 — the renderer. */
/** One installed file a `path` parameter can point at. */
data class PathChoice(
    val label: String,
    val detail: String,
    val path: String,
    /** What the app detected this file to be, or null if it could not tell. */
    val role: ai.ondevice.core.AttachmentRole? = null,
)

@Composable
fun ParamRow(
    spec: ParamSpec,
    values: SparseParams,
    onChange: (String, Any?) -> Unit,
    modifier: Modifier = Modifier,
    showKeyLine: Boolean = true,
    /** Installed files a `path` parameter may be pointed at. */
    pathChoices: List<PathChoice> = emptyList(),
    /**
     * Why this row cannot be edited, or null when it can.
     *
     * A row that does not apply is shown rather than dropped: what a model
     * cannot do is worth as much as what it can, and a count of hidden things
     * is not something anyone can act on.
     */
    disabledBecause: String? = null,
) {
    val current = values[spec.key] ?: spec.default
    val modified = spec.key in values &&
        renderJson(values[spec.key]!!) != spec.default?.let { renderJson(it) }

    Column(
        modifier
            .fillMaxWidth()
            .ruleBelow()
            .padding(vertical = 11.dp)
            .alpha(if (disabledBecause == null) 1f else 0.45f),
    ) {
        Row(
            Modifier.fillMaxWidth().padding(bottom = 3.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(spec.label, style = NocturneType.Row)
            if (modified) NDot()
            if (spec.requiresReload) {
                NTag(
                    "reload",
                    style = NTagStyle.Neutral,
                    textStyle = NocturneType.Meta.copy(fontSize = androidx.compose.ui.unit.TextUnit(9f, androidx.compose.ui.unit.TextUnitType.Sp)),
                    contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 5.dp, vertical = 0.dp),
                )
            }
            Text(
                summarise(spec, current),
                style = NocturneType.MonoValue,
                color = NocturneColors.Accent300,
                modifier = Modifier.weight(1f),
                textAlign = androidx.compose.ui.text.style.TextAlign.End,
            )
        }

        if (showKeyLine) {
            Text(
                "${spec.key} · ${spec.tier.label.lowercase()}",
                style = NocturneType.Mono2Xs,
                color = NocturneColors.TextMuted,
                modifier = Modifier.padding(bottom = 5.dp),
            )
        }

        // Widget selection is table-driven off the type. No parameter names here.
        //
        // A disabled row still draws its control, so the value it holds stays
        // legible; the overlay eats the touches rather than every widget in the
        // design system having to grow an `enabled` flag.
        Box {
            ParamControl(spec, values, onChange, pathChoices)
            if (disabledBecause != null) {
                Box(
                    Modifier
                        .matchParentSize()
                        .pointerInput(Unit) {
                            awaitPointerEventScope {
                                while (true) {
                                    awaitPointerEvent().changes.forEach { it.consume() }
                                }
                            }
                        },
                )
            }
        }

        disabledBecause?.let {
            Text(
                it,
                style = NocturneType.Help,
                color = NocturneColors.Accent300,
                modifier = Modifier.padding(top = 5.dp),
            )
        }
        if (spec.help.isNotBlank()) {
            NHelp(spec.help, Modifier.padding(top = 5.dp))
        }
        if (modified && disabledBecause == null) {
            Text(
                "reset to ${spec.defaultDisplay}",
                style = NocturneType.Help,
                color = NocturneColors.Accent,
                modifier = Modifier
                    .padding(top = 3.dp)
                    .nClickableFlat { onChange(spec.key, null) },
            )
        }
    }
}

/** One branch per [ParamType]. This `when` is the whole of §16.4's table. */
@Composable
private fun ParamControl(
    spec: ParamSpec,
    values: SparseParams,
    onChange: (String, Any?) -> Unit,
    pathChoices: List<PathChoice>,
) {
    when (spec.type) {
        ParamType.FLOAT -> if (spec.isRange) {
            val v = values.float(spec.key) ?: (spec.default as? JsonPrimitive)?.content?.toFloatOrNull() ?: 0f
            NSlider(
                value = v,
                onValueChange = { onChange(spec.key, it) },
                valueRange = spec.min!!.toFloat()..spec.max!!.toFloat(),
                steps = discreteSteps(spec),
            )
        } else {
            NumericField(spec, values, onChange, decimal = true)
        }

        ParamType.INT -> if (spec.isRange) {
            val v = values.int(spec.key)?.toFloat()
                ?: (spec.default as? JsonPrimitive)?.content?.toFloatOrNull() ?: 0f
            NSlider(
                value = v,
                onValueChange = { onChange(spec.key, it.toInt()) },
                valueRange = spec.min!!.toFloat()..spec.max!!.toFloat(),
                steps = discreteSteps(spec),
            )
        } else {
            NumericField(spec, values, onChange, decimal = false)
        }

        ParamType.BOOL -> {
            val checked = values.bool(spec.key)
                ?: ((spec.default as? JsonPrimitive)?.content?.toBooleanStrictOrNull() ?: false)
            NSwitch(checked = checked, onCheckedChange = { onChange(spec.key, it) })
        }

        ParamType.ENUM -> {
            val selected = values.string(spec.key) ?: (spec.default as? JsonPrimitive)?.content
            NEnumRow(
                options = spec.values,
                selected = selected,
                onSelect = { onChange(spec.key, it) },
            )
        }

        ParamType.PATH -> {
            val v = values.string(spec.key) ?: (spec.default as? JsonPrimitive)?.content.orEmpty()
            val choices = choicesFor(spec, pathChoices)
            if (choices.isEmpty()) {
                NothingInstalled()
            } else {
                // Same shape as the chat model picker: choose from what is installed.
                val labels = listOf("None") + uniqueLabels(choices)
                val selected = choices.indexOfFirst { it.path == v }
                    .let { if (it < 0) 0 else it + 1 }
                ai.ondevice.ui.components.NDropdown(
                    options = labels,
                    selected = labels[selected],
                    onSelect = { label ->
                        val index = labels.indexOf(label)
                        onChange(spec.key, if (index <= 0) null else choices[index - 1].path)
                    },
                )
                choices.getOrNull(selected - 1)?.let { chosen ->
                    NHelp(chosen.detail, Modifier.padding(top = 4.dp))
                }
            }
        }

        ParamType.WEIGHTED_PATHS -> {
            val choices = choicesFor(spec, pathChoices)
            if (choices.isEmpty()) {
                NothingInstalled()
            } else {
                WeightedPathStack(
                    entries = ai.ondevice.core.WeightedPaths.parse(values[spec.key]),
                    choices = choices,
                    max = spec.max?.toFloat() ?: 2f,
                    onChange = { entries ->
                        onChange(spec.key, ai.ondevice.core.WeightedPaths.toJson(entries))
                    },
                )
            }
        }

        ParamType.STRING -> {
            val v = values.string(spec.key) ?: (spec.default as? JsonPrimitive)?.content.orEmpty()
            NInput(
                value = v,
                onValueChange = { onChange(spec.key, it) },
                textStyle = NocturneType.MonoSm,
            )
        }

        ParamType.TEXT -> {
            val v = values.string(spec.key) ?: (spec.default as? JsonPrimitive)?.content.orEmpty()
            NTextArea(
                value = v,
                onValueChange = { onChange(spec.key, it) },
                minHeight = 66.dp,
                textStyle = NocturneType.Row,
            )
        }

        ParamType.STRING_ARRAY, ParamType.INT_ARRAY -> {
            val chips = values.stringList(spec.key)
                ?: (spec.default as? kotlinx.serialization.json.JsonArray)
                    ?.mapNotNull { (it as? JsonPrimitive)?.content }
                ?: emptyList()
            NChipRow(
                chips = chips.map { it.replace("\n", "\\n") },
                onRemove = { i -> onChange(spec.key, chips.filterIndexed { idx, _ -> idx != i }) },
                onAdd = { onChange(spec.key, chips + "") },
            )
        }

        ParamType.MAP -> {
            // logit_bias and friends. The token picker lives in the chat screen,
            // where a tokenizer is loaded; here the raw pairs are editable.
            val raw = values[spec.key]?.toString() ?: "{}"
            NInput(
                value = raw,
                onValueChange = { onChange(spec.key, it) },
                textStyle = NocturneType.MonoSm,
                placeholder = "{ \"token_id\": bias }",
            )
        }

        ParamType.ORDERED_LIST -> {
            // The sampler chain gets its own full screen (S9) because ordering materially changes output and a row here can't carry a drag handle.
            val order = values.stringList(spec.key)
                ?: (spec.default as? kotlinx.serialization.json.JsonArray)
                    ?.mapNotNull { (it as? JsonPrimitive)?.content }
                ?: emptyList()
            Text(
                order.joinToString(" → "),
                style = NocturneType.MonoSm,
                color = NocturneColors.Accent300,
            )
        }
    }
}

/**
 * The installed files this parameter may name.
 *
 * When the key belongs to a role and nothing installed fills that role, the
 * answer is "nothing" — not "everything". Offering a T5 encoder to the VAE slot
 * is offering a run that fails inside the runtime.
 */
private fun choicesFor(spec: ParamSpec, pathChoices: List<PathChoice>): List<PathChoice> {
    val wanted = ai.ondevice.core.AttachmentRole.entries.firstOrNull { it.paramKey == spec.key }
        ?: return pathChoices
    return pathChoices.filter { it.role == wanted }
}

/** Two files of the same name are two rows that read as one; number them. */
private fun uniqueLabels(choices: List<PathChoice>): List<String> =
    choices.mapIndexed { index, choice ->
        if (choices.count { it.label == choice.label } > 1) {
            "${choice.label}  (${index + 1})"
        } else {
            choice.label
        }
    }

@Composable
private fun NothingInstalled() {
    NHelp(
        "Nothing installed that this could point at. Download one on the Add model screen and " +
            "it appears here — a path typed by hand would only fail later, inside the runtime.",
    )
}

/**
 * Several files under one key, each with its own strength.
 *
 * The stack is ordered and duplicates are allowed to be prevented rather than
 * policed: the same LoRA twice is the same LoRA at whichever multiplier came
 * last, so the picker offers only files not already in the stack.
 */
@Composable
private fun WeightedPathStack(
    entries: List<ai.ondevice.core.WeightedPath>,
    choices: List<PathChoice>,
    max: Float,
    onChange: (List<ai.ondevice.core.WeightedPath>) -> Unit,
) {
    val labels = uniqueLabels(choices)

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        entries.forEachIndexed { index, entry ->
            val at = choices.indexOfFirst { it.path == entry.path }
            Column(
                Modifier
                    .fillMaxWidth()
                    .ring(NocturneColors.Divider, ai.ondevice.ui.theme.Radius.Sm)
                    .padding(horizontal = 10.dp, vertical = 9.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                ai.ondevice.ui.components.NDropdown(
                    options = labels,
                    selected = labels.getOrNull(at),
                    placeholder = entry.path.substringAfterLast('/'),
                    onSelect = { label ->
                        val picked = choices.getOrNull(labels.indexOf(label)) ?: return@NDropdown
                        onChange(entries.replaceAt(index, entry.copy(path = picked.path)))
                    },
                )
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("Strength", style = NocturneType.Help, color = NocturneColors.TextMuted)
                    NSlider(
                        value = entry.weight,
                        onValueChange = { onChange(entries.replaceAt(index, entry.copy(weight = it))) },
                        valueRange = 0f..max,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        String.format("%.2f", entry.weight),
                        style = NocturneType.MonoValue,
                        color = NocturneColors.Accent300,
                    )
                    Text(
                        "Remove",
                        style = NocturneType.Help,
                        color = NocturneColors.Accent,
                        modifier = Modifier.nClickableFlat {
                            onChange(entries.filterIndexed { i, _ -> i != index })
                        },
                    )
                }
            }
        }

        val unused = choices.filter { choice -> entries.none { it.path == choice.path } }
        if (unused.isNotEmpty()) {
            ai.ondevice.ui.components.NButton(
                if (entries.isEmpty()) "Add one" else "Add another",
                onClick = { onChange(entries + ai.ondevice.core.WeightedPath(unused.first().path)) },
                block = true,
            )
        }
    }
}

private fun <T> List<T>.replaceAt(index: Int, value: T): List<T> =
    mapIndexed { i, existing -> if (i == index) value else existing }

/** What the value line at the top of a row says. */
private fun summarise(spec: ParamSpec, current: kotlinx.serialization.json.JsonElement?): String {
    if (spec.type == ParamType.WEIGHTED_PATHS) {
        val count = ai.ondevice.core.WeightedPaths.parse(current).size
        return if (count == 0) "none" else "$count attached"
    }
    return current?.let { renderJson(it) } ?: "—"
}

@Composable
private fun NumericField(
    spec: ParamSpec,
    values: SparseParams,
    onChange: (String, Any?) -> Unit,
    decimal: Boolean,
) {
    val text = values[spec.key]?.let { renderJson(it) }
        ?: (spec.default as? JsonPrimitive)?.content.orEmpty()
    NInput(
        value = text,
        onValueChange = { raw ->
            if (decimal) raw.toFloatOrNull()?.let { onChange(spec.key, it) }
            else raw.toIntOrNull()?.let { onChange(spec.key, it) }
        },
        keyboardType = if (decimal) {
            androidx.compose.ui.text.input.KeyboardType.Decimal
        } else {
            androidx.compose.ui.text.input.KeyboardType.Number
        },
        textStyle = NocturneType.MonoSm,
    )
}

/** Sliders are step-snapped when the manifest gives a step, so a value the user drags to is one the runtime will actually accept. */
private fun discreteSteps(spec: ParamSpec): Int {
    val min = spec.min ?: return 0
    val max = spec.max ?: return 0
    val step = spec.step ?: return 0
    if (step <= 0.0) return 0
    val count = ((max - min) / step).toInt() - 1
    return count.coerceIn(0, 200)
}
