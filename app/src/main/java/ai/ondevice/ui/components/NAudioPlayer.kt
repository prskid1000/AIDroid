package ai.ondevice.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.dp
import ai.ondevice.core.Fmt
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.ring
import java.io.File

/**
 * Play a finished audio file, with a position you can move.
 *
 * There was no way to hear anything the app had produced without leaving it.
 * Speak played once, as it synthesised, and if you missed it the only replay
 * was to synthesise again — which for OmniVoice is minutes. A rendered WAV in
 * the library could not be played at all, only shared to some other app that
 * could. For a screen whose entire output is audio, that is the wrong shape.
 *
 * `MediaPlayer` rather than ExoPlayer: these are local WAVs of a few seconds to
 * a few minutes, already decoded, and ExoPlayer is several megabytes of
 * dependency to solve a problem this does not have.
 */
@Composable
fun NAudioPlayer(
    file: File,
    modifier: Modifier = Modifier,
    autoPlay: Boolean = false,
    onAutoPlayed: () -> Unit = {},
    label: String? = null,
) {
    if (!file.isFile) return

    val player = remember(file.path) {
        runCatching {
            android.media.MediaPlayer().apply {
                setDataSource(file.path)
                prepare()
            }
        }.getOrNull()
    }
    var playing by remember(file.path) { mutableStateOf(false) }
    var position by remember(file.path) { mutableIntStateOf(0) }
    val duration = player?.duration ?: 0

    // Released on the way out, and on the way to a *different* file: a
    // MediaPlayer left open holds an audio focus request and a file descriptor,
    // and a screen that renders twice would leak one per render.
    DisposableEffect(file.path) {
        onDispose { runCatching { player?.release() } }
    }

    // A one-shot, acknowledged back to the caller. Without that, every
    // recomposition would restart the clip from the top — and a Speak result
    // recomposes as soon as you touch the scrubber.
    LaunchedEffect(file.path, autoPlay) {
        if (autoPlay && player != null) {
            runCatching { player.start() }
            playing = player.isPlaying
            onAutoPlayed()
        }
    }

    // Polled rather than driven by a listener, because MediaPlayer has no
    // position callback at all — `setOnSeekCompleteListener` fires for seeks and
    // nothing fires as it plays.
    LaunchedEffect(playing, file.path) {
        while (playing && player != null) {
            position = runCatching { player.currentPosition }.getOrDefault(0)
            if (!player.isPlaying) {
                playing = false
                // Rewind at the end so the button means "play again" rather
                // than doing nothing on a player parked at its last frame.
                position = 0
                runCatching { player.seekTo(0) }
            }
            kotlinx.coroutines.delay(POLL_MILLIS)
        }
    }

    val fraction = if (duration > 0) (position.toFloat() / duration).coerceIn(0f, 1f) else 0f

    NCard(modifier, gap = 8.dp) {
        if (label != null) {
            Text(label, style = NocturneType.CardTitleSm, maxLines = 1)
        }
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier
                    .size(38.dp)
                    .clip(CircleShape)
                    .background(NocturneColors.Accent900)
                    .ring(NocturneColors.Accent700, CircleShape)
                    .nClickableFlat {
                        val active = player ?: return@nClickableFlat
                        if (active.isPlaying) {
                            runCatching { active.pause() }
                            playing = false
                        } else {
                            runCatching { active.start() }
                            playing = true
                        }
                    },
                contentAlignment = Alignment.Center,
            ) {
                Icon(
                    if (playing) NIcons.Pause else NIcons.Play,
                    contentDescription = if (playing) "Pause" else "Play",
                    tint = NocturneColors.Accent200,
                    modifier = Modifier.size(16.dp),
                )
            }

            Scrubber(
                fraction = fraction,
                onSeek = { target ->
                    val active = player ?: return@Scrubber
                    val to = (target * duration).toInt().coerceIn(0, duration)
                    runCatching { active.seekTo(to) }
                    position = to
                },
                modifier = Modifier.weight(1f),
            )

            Text(
                "${Fmt.duration(position.toLong())} / ${Fmt.duration(duration.toLong())}",
                style = NocturneType.MonoXs,
                color = NocturneColors.TextMuted,
            )
        }
    }
}

/**
 * The track. Dragged as well as tapped, because a three-minute clip you can
 * only jump around by tapping is one you cannot find a word in.
 */
@Composable
private fun Scrubber(
    fraction: Float,
    onSeek: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    var width by remember { mutableIntStateOf(1) }
    var dragging by remember { mutableFloatStateOf(-1f) }
    val shown = if (dragging >= 0f) dragging else fraction

    Box(
        modifier
            .height(26.dp)
            .onSizeChanged { width = it.width.coerceAtLeast(1) }
            .pointerInput(Unit) {
                detectHorizontalDragGestures(
                    onDragStart = { offset -> dragging = (offset.x / width).coerceIn(0f, 1f) },
                    onHorizontalDrag = { change, _ ->
                        dragging = (change.position.x / width).coerceIn(0f, 1f)
                    },
                    onDragEnd = {
                        if (dragging >= 0f) onSeek(dragging)
                        dragging = -1f
                    },
                    onDragCancel = { dragging = -1f },
                )
            }
            .pointerInput(Unit) {
                detectTapGestures { offset ->
                    onSeek((offset.x / width).coerceIn(0f, 1f))
                }
            },
        contentAlignment = Alignment.CenterStart,
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .height(4.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(NocturneColors.Neutral700),
        )
        Box(
            Modifier
                .fillMaxWidth(shown)
                .height(4.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(NocturneColors.Accent500),
        )
    }
}

private const val POLL_MILLIS = 120L
