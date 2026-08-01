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

/** Play a finished audio file, with a position you can move. */
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
    var speed by remember { mutableFloatStateOf(1f) }
    val duration = player?.duration ?: 0

    DisposableEffect(file.path) {
        onDispose { runCatching { player?.release() } }
    }

    // A one-shot, acknowledged back to the caller.
    LaunchedEffect(file.path, autoPlay) {
        if (autoPlay && player != null) {
            runCatching { player.start() }
            playing = player.isPlaying
            onAutoPlayed()
        }
    }

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

            Skip(-SKIP_MILLIS, "Back 10 seconds") {
                val active = player ?: return@Skip
                val to = (position - SKIP_MILLIS).coerceIn(0, duration)
                runCatching { active.seekTo(to) }
                position = to
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

            Skip(SKIP_MILLIS, "Forward 10 seconds") {
                val active = player ?: return@Skip
                val to = (position + SKIP_MILLIS).coerceIn(0, duration)
                runCatching { active.seekTo(to) }
                position = to
            }
        }

        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "${Fmt.duration(position.toLong())} / ${Fmt.duration(duration.toLong())}",
                style = NocturneType.MonoXs,
                color = NocturneColors.TextMuted,
                modifier = Modifier.padding(end = 4.dp),
            )
            SPEEDS.forEach { option ->
                SpeedChip(
                    speed = option,
                    selected = option == speed,
                    onClick = {
                        speed = option
                        val active = player ?: return@SpeedChip
                        // Assigning playbackParams *starts* a paused player on several Android versions, so the pause is reasserted rather than assumed.
                        runCatching {
                            active.playbackParams = active.playbackParams.setSpeed(option)
                            if (!playing) active.pause()
                        }
                    },
                )
            }
        }
    }
}

/** ±10 s, the one seek worth a dedicated control on speech. */
@Composable
private fun Skip(millis: Int, description: String, onClick: () -> Unit) {
    Box(
        Modifier.size(30.dp).clip(CircleShape).nClickableFlat(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            if (millis < 0) NIcons.RotateBack else NIcons.Rotate,
            contentDescription = description,
            tint = NocturneColors.TextMuted,
            modifier = Modifier.size(15.dp),
        )
    }
}

/** Playback rate. */
@Composable
private fun SpeedChip(speed: Float, selected: Boolean, onClick: () -> Unit) {
    Text(
        if (speed == 1f) "1x" else "${speed}x".replace(".0x", "x"),
        style = NocturneType.MonoXs,
        color = if (selected) NocturneColors.Accent200 else NocturneColors.TextMuted,
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(if (selected) NocturneColors.Accent900 else NocturneColors.Neutral900)
            .nClickableFlat(onClick = onClick)
            .padding(horizontal = 7.dp, vertical = 3.dp),
    )
}

/** The track. */
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
private const val SKIP_MILLIS = 10_000
private val SPEEDS = listOf(0.75f, 1f, 1.25f, 1.5f, 2f)
