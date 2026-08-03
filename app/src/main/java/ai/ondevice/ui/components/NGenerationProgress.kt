package ai.ondevice.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import ai.ondevice.core.Fmt
import ai.ondevice.engine.DiffusionPhase
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.bottomScrim

/**
 * Where a run has got to, over the picture it is making.
 *
 * Inside the viewport rather than under it, because the progress belongs to
 * the thing being rendered and putting it below costs a line of layout that
 * moves everything else down the moment a run starts.
 *
 * The two screens had this in different places — the still screen overlaid it
 * on the preview, the clip screen put a bar and a line of text underneath —
 * so the same run reported itself in two shapes depending on which tab you
 * were looking at. One implementation, called from both.
 *
 * Meant to be placed inside a [Box] that holds the preview; it aligns itself
 * to the bottom.
 */
@Composable
fun BoxScope.GenerationProgress(
    phase: DiffusionPhase,
    step: Int,
    steps: Int,
    secondsPerStep: Float,
    etaSeconds: Long,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier
            .align(Alignment.BottomCenter)
            .fillMaxWidth()
            .bottomScrim()
            .padding(horizontal = 11.dp, vertical = 9.dp),
    ) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                // Only the sampling phase has steps worth counting.
                if (phase == DiffusionPhase.SAMPLING && steps > 0) {
                    "step $step/$steps"
                } else {
                    phase.label
                },
                style = NocturneType.MonoSm,
                color = NocturneColors.Accent200,
            )
            if (secondsPerStep > 0f) {
                Text("·", style = NocturneType.MonoSm, color = Color.White.copy(alpha = 0.7f))
                // The rate, not the device: there is one device.
                Text(
                    "${String.format("%.1f", secondsPerStep)} s/it",
                    style = NocturneType.MonoSm,
                    color = Color.White.copy(alpha = 0.7f),
                )
            }
            Box(Modifier.weight(1f))
            // An ETA computed from a rate we do not have yet is a guess dressed
            // as a number. Better to show nothing.
            if (secondsPerStep > 0f && etaSeconds > 0) {
                Text(
                    Fmt.eta(etaSeconds),
                    style = NocturneType.MonoSm,
                    color = Color.White.copy(alpha = 0.7f),
                )
            }
        }
        NProgressBar(
            fraction = if (steps > 0) (step.toFloat() / steps).coerceIn(0f, 1f) else 0f,
            modifier = Modifier.padding(top = 6.dp),
            height = 4.dp,
            fill = NocturneColors.Accent400,
            track = Color.Black.copy(alpha = 0.45f),
        )
    }
}
