package ai.ondevice.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import coil3.compose.AsyncImage

/**
 * A picture this run will be given, picked or not yet.
 *
 * One implementation because it is one idea, and because the two screens that
 * ask for a picture had drifted into asking differently. The still screen used
 * this: a button until something is chosen, then a 56dp thumbnail with the
 * filename beside it and Replace and Remove as words. The clip screen used a
 * full-width square that stayed square whether or not it held anything — three
 * of them, for the first frame, the last frame and the control frame, which is
 * most of a screen's height spent on empty boxes saying "Choose…".
 *
 * The row won on the merits rather than by seniority: it names the file it
 * holds, which a thumbnail alone does not, and it distinguishes replacing from
 * removing, which the square could only do by growing a second control under
 * it.
 */
@Composable
fun PickedImageField(
    label: String,
    uri: String?,
    emptyLabel: String,
    onPick: () -> Unit,
    onClear: () -> Unit,
    modifier: Modifier = Modifier,
) {
    NField(label, modifier.padding(top = 12.dp)) {
        if (uri == null) {
            NButton(
                emptyLabel,
                onClick = onPick,
                style = NButtonStyle.Secondary,
                block = true,
            )
        } else {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                AsyncImage(
                    model = uri,
                    contentDescription = label,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier
                        .size(56.dp)
                        .clip(Radius.Sm)
                        .background(NocturneColors.Neutral900)
                        .ring(NocturneColors.Divider, Radius.Sm),
                )
                Column(Modifier.weight(1f)) {
                    Text("Picked", style = NocturneType.Row)
                    Text(
                        uri.substringAfterLast('/'),
                        style = NocturneType.MonoXs,
                        color = NocturneColors.TextMuted,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Text(
                    "Replace",
                    style = NocturneType.Meta,
                    color = NocturneColors.Accent,
                    modifier = Modifier.nClickableFlat(onClick = onPick).padding(6.dp),
                )
                Text(
                    "Remove",
                    style = NocturneType.Meta,
                    color = NocturneColors.TextMuted,
                    modifier = Modifier.nClickableFlat(onClick = onClear).padding(6.dp),
                )
            }
        }
    }
}
