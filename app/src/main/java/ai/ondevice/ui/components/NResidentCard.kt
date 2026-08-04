package ai.ondevice.ui.components

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import ai.ondevice.engine.RuntimeBuffer
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType

/**
 * What the runtime is holding, in the runtime's own figures.
 *
 * One implementation because it is one card. The still screen and the clip
 * screen each had their own copy, and they had already drifted — only one of
 * them said "Unloaded", only one showed the weight total.
 *
 * The weights and the buffers are shown as two groups rather than one list,
 * which is the correction this card exists for. They used to run together: a
 * list of file sizes under a heading that says "In memory", with the runtime's
 * most recent buffer line dropped on the end. Read down it, and 851.60 MB
 * looks like the last row of a sum. It is not — it is one module's graph
 * allocator reservation, and a decode observed taking three gigabytes was
 * holding that *plus* the decoder's weights, plus a cache buffer several times
 * larger, plus the frames themselves.
 *
 * So the buffers are labelled as reservations, kept apart from the weights,
 * and never added to them.
 */
@Composable
fun ResidentCard(
    loadingNow: Boolean,
    loadingWhat: List<String>,
    resident: List<String>,
    buffers: List<RuntimeBuffer>,
    modifier: Modifier = Modifier,
    /** What these files weigh on disk — a fact about storage, not about memory. */
    weightsTotal: String? = null,
    /**
     * What the process is actually holding, sampled from RSS.
     *
     * The only honest total on this card, and the reason nothing else here is
     * presented as one. Adding file sizes overstates it — the weights are
     * memory-mapped, so a bundle reporting 10.7 GB of params sampled at
     * 3.94 GB once its encoder was finished with — and adding the runtime's
     * buffer reservations would double-count and could never come back down,
     * because the runtime announces every allocation and no free.
     */
    measured: String? = null,
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
        Text(
            when {
                loadingNow -> "Loading into memory"
                resident.isEmpty() -> "Unloaded"
                else -> "In memory"
            },
            style = NocturneType.CardTitleSm,
        )
        (if (loadingNow) loadingWhat else resident).forEach {
            Text(it, style = NocturneType.MonoXs, color = NocturneColors.Accent300)
        }
        // The measured figure leads, and the file sizes follow it as context.
        //
        // It used to be the other way round and the measurement was not here at
        // all: a sum of file sizes, under a heading that says "In memory",
        // reading as a claim about memory. It is not one. These files are
        // memory-mapped, so what is resident is whatever has been touched and
        // not yet reclaimed — which on this device meant a bundle whose files
        // come to 10.7 GB sampling at 3.94 GB, because the encoder had done its
        // work and the kernel had taken the pages back.
        if (!loadingNow) {
            measured?.let {
                Text(it, style = NocturneType.MonoXs, color = NocturneColors.Accent200)
            }
            weightsTotal?.let {
                Text(it, style = NocturneType.MonoXs, color = NocturneColors.TextMuted)
            }
        }

        // The runtime's reservations, which are a different kind of number and
        // now say so — in the heading, once, rather than in a paragraph under
        // each list. Only after a load, because they are reserved as each
        // module first builds a graph: the decoder's do not exist until the
        // decode, and showing an empty heading before then would suggest they
        // were nothing rather than not yet asked for.
        if (!loadingNow && buffers.isNotEmpty()) {
            NHelp("Reserved by the runtime", Modifier.padding(top = 8.dp))
            buffers.forEach { buffer ->
                Text(
                    buildString {
                        append(buffer.what)
                        if (buffer.computeMb > 0) append(" · graph ${megabytes(buffer.computeMb)}")
                        if (buffer.cacheMb > 0) append(" · cache ${megabytes(buffer.cacheMb)}")
                    },
                    style = NocturneType.MonoXs,
                    color = NocturneColors.Accent300,
                )
            }
        }

        if (resident.isEmpty()) {
            unloadReason?.let {
                Text(it, style = NocturneType.Help, color = NocturneColors.TextMuted)
            }
        }
        stage?.let {
            Text(
                it,
                style = NocturneType.Help,
                color = NocturneColors.TextMuted,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

private fun megabytes(mb: Double): String =
    if (mb >= 1024.0) String.format("%.2f GB", mb / 1024.0) else String.format("%.0f MB", mb)
