package ai.ondevice.ui.components

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import ai.ondevice.engine.RuntimeBuffer
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType

/**
 * What the runtime reserved, and what it is saying.
 *
 * One implementation because it is one card. The still screen and the clip
 * screen each had their own copy and they had already drifted — only one of
 * them said "Unloaded".
 *
 * What this card no longer does is report memory, and that is the point of it.
 * It used to list every loaded file with its size, sum those sizes, and print
 * the sum under a heading reading "In memory" — a statement about storage
 * dressed as one about memory, and wrong by a factor of three: the weights are
 * memory-mapped, so a bundle whose files come to 10.7 GB was measured holding
 * 3.94 GB once its prompt encoder was finished with and the kernel had taken
 * the pages back.
 *
 * Replacing that sum with a measured figure only moved the problem. The
 * resource line directly above this card already reports RAM, sampled from the
 * same place — so the card's own figure was a second number for one quantity,
 * a few hundred megabytes off because it was sampled a moment apart, and two
 * nearly-equal numbers on one screen read as a disagreement rather than as
 * agreement. One measurement, in one place.
 *
 * What is left is the thing nothing else says: the working memory the runtime
 * reserves per module, in its own figures. Two of them, because ggml reserves
 * two and reports each exactly once — a graph buffer for a graph's intermediate
 * tensors, and a module's own persistent cache. The app used to keep only
 * whichever line arrived last, which is how a decode observed taking three
 * gigabytes came to be described by a lone "851.60 MB". The cache is the one
 * that matters for video: Wan's decoder is causal in time and carries its
 * feature maps from one frame to the next, so it grows with frame count while
 * the graph does not.
 */
@Composable
fun ResidentCard(
    loadingNow: Boolean,
    loadingWhat: List<String>,
    buffers: List<RuntimeBuffer>,
    modifier: Modifier = Modifier,
    /** Whether anything is in the context at all, which decides the heading. */
    loaded: Boolean = true,
    unloadReason: String? = null,
    /** The loader's progress, or the sampler's — whichever is running. */
    stage: String? = null,
) {
    NCard(
        // NCard wraps its content, and this one holds short monospace lines, so
        // it drew narrower than everything around it and read as a different
        // kind of thing.
        modifier.fillMaxWidth(),
        ring = if (loadingNow) NocturneColors.Accent800 else NocturneColors.Neutral700,
    ) {
        // A heading only where there is something to head.
        //
        // "Reserved by the runtime" sat above two short rows that already read
        // as what they are — a module and a size — and cost more height than
        // the rows it introduced. Loading and unloaded still take one, because
        // in both cases the card is saying something about a state rather than
        // listing figures.
        when {
            loadingNow -> Text("Loading into memory", style = NocturneType.CardTitleSm)
            !loaded -> Text("Unloaded", style = NocturneType.CardTitleSm)
        }

        // Naming what is going in, while it goes in. Loading four gigabytes is
        // minutes of one opaque call, and this is the only thing that says what
        // the wait is for.
        if (loadingNow) {
            loadingWhat.forEach {
                Text(it, style = NocturneType.MonoXs, color = NocturneColors.Accent300)
            }
        } else {
            // Two groups, each labelled and totalled.
            //
            // This card used to list graph reservations alone: "t5 graph 297
            // MB, Wan2.2-TI2V-5B graph 56 MB" for a process holding 10.7 GB of
            // weights. That is not a small error but the wrong quantity — the
            // graph is the smallest of the terms and was the only one counted,
            // so the card answered "what is this run holding" with 3% of the
            // answer.
            //
            // Kept split rather than summed to one number, because the split is
            // the surprising part: the text encoder is nearly twice the
            // diffusion model, and the diffusion model is the file the user
            // chose. Totals on the group rows so the two scales can be compared
            // without adding up the children.
            //
            // Working memory is reserved as each module first builds a graph,
            // so the decoder's does not exist until the decode — an empty
            // group here means not yet asked for rather than nothing.
            MemoryGroup(
                label = "in memory",
                totalMb = buffers.filter { it.isResident }.sumOf { it.residentMb },
                rows = buffers.filter { it.isResident }.map { it.what to megabytes(it.residentMb) },
            )
            MemoryGroup(
                label = "working",
                totalMb = buffers.filterNot { it.isResident }.sumOf { it.computeMb + it.cacheMb },
                rows = buffers.filterNot { it.isResident }.map { buffer ->
                    buffer.what to buildString {
                        if (buffer.computeMb > 0) append("graph ${megabytes(buffer.computeMb)}")
                        if (buffer.cacheMb > 0) {
                            if (isNotEmpty()) append(" · ")
                            append("cache ${megabytes(buffer.cacheMb)}")
                        }
                    }
                },
            )
            if (!loaded) {
                unloadReason?.let {
                    Text(it, style = NocturneType.Help, color = NocturneColors.TextMuted)
                }
            }
        }

        stage?.takeIf { it.isNotBlank() }?.let {
            Text(
                readable(it),
                style = NocturneType.Help,
                color = NocturneColors.TextMuted,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

/**
 * One labelled, totalled group of the resident card.
 *
 * The module name is left as the runtime spells it — `wan_vae`, `t5`, and
 * `text_encoders` — because it is an identifier, and prettying an identifier
 * only makes it harder to match against a log.
 */
@Composable
private fun MemoryGroup(label: String, totalMb: Double, rows: List<Pair<String, String>>) {
    if (rows.isEmpty()) return
    Row(Modifier.fillMaxWidth()) {
        Text(label, style = NocturneType.MonoXs, color = NocturneColors.Text, modifier = Modifier.weight(1f))
        Text(megabytes(totalMb), style = NocturneType.MonoXs, color = NocturneColors.Text)
    }
    rows.forEach { (name, figure) ->
        Row(Modifier.fillMaxWidth()) {
            Text(
                "  $name",
                style = NocturneType.MonoXs,
                color = NocturneColors.Accent300,
                modifier = Modifier.weight(1f),
            )
            Text(figure, style = NocturneType.MonoXs, color = NocturneColors.TextMuted)
        }
    }
}

private fun megabytes(mb: Double): String =
    if (mb >= 1024.0) String.format("%.2f GB", mb / 1024.0) else String.format("%.0f MB", mb)

/**
 * The runtime's own sentence, tidied — not translated.
 *
 * These lines are written for a developer reading a log: internal function
 * names with underscores, seconds to two decimal places, and a parenthesised
 * breakdown of where they went. On a phone, under a picture, they were the one
 * thing on the screen that looked like it had escaped from somewhere.
 *
 * The tidying is deliberately mechanical — underscores out, the developer's
 * breakdown dropped, a duration written the way durations are written
 * everywhere else, a capital at the front. No table mapping the runtime's
 * words to nicer ones: that would be a list to keep in step with a dependency
 * that is re-cloned rather than patched, and a line it did not recognise would
 * come through worse than untouched.
 */
private fun readable(raw: String): String {
    var line = raw.trim()
    // "(read: 0.07s, memcpy: 0.00s, …)", "(VRAM 0.00MB, RAM 10713.69MB)" — a
    // breakdown of a number that has already been given.
    line = line.substringBefore(" (read:").substringBefore(" (VRAM").trim()
    // Upstream says "taking 78.94s" where the rest of the app says "1m 19s".
    line = TAKING.replace(line) { match ->
        "in " + seconds(match.groupValues[1].toDoubleOrNull() ?: 0.0)
    }
    line = line.replace('_', ' ')
    return line.replaceFirstChar { it.uppercase() }
}

private val TAKING = Regex("""taking (\d+(?:\.\d+)?)s""")

private fun seconds(value: Double): String = when {
    value >= 60.0 -> "${(value / 60).toInt()}m ${(value % 60).toInt()}s"
    value >= 10.0 -> "${value.toInt()}s"
    else -> String.format("%.1fs", value)
}
