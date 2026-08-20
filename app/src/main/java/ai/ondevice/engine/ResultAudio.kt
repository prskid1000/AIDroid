package ai.ondevice.engine

import android.content.Context
import android.media.MediaMetadata
import android.media.MediaPlayer
import android.media.session.MediaSession
import android.media.session.PlaybackState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

/**
 * A real player for a finished spoken line, not two buttons that look like one.
 *
 * `Notification.MediaStyle` on its own gets you a row of plain actions and
 * nothing else. The transport controls, the scrubber, the elapsed and total
 * time, the lock-screen and Quick Settings surfaces — all of that is drawn by
 * the platform from a [MediaSession], and a MediaStyle notification with no
 * session token attached is simply a notification with buttons on it. This is
 * the session.
 *
 * The scrubber needs two things and both are easy to leave out. `MediaMetadata`
 * has to carry `METADATA_KEY_DURATION`, or the bar has no length and does not
 * appear. And `PlaybackState` has to carry the position *with* a playback speed,
 * because the system extrapolates the moving part itself from the position, the
 * speed and the timestamp — which is why nothing here ticks once a second to
 * push updates that would be wrong between them anyway.
 *
 * One player, and one session. Two spoken lines over each other is not
 * something anybody asked for, and two sessions would have the platform pick
 * one to show and quietly drop the other.
 */
object ResultAudio {

    /** What the notification draws, so it can be rebuilt on a state change. */
    data class State(
        val path: String? = null,
        val playing: Boolean = false,
        val positionMillis: Long = 0,
        val durationMillis: Long = 0,
    ) {
        val active: Boolean get() = path != null
    }

    private val _state = MutableStateFlow(State())
    val state: StateFlow<State> = _state.asStateFlow()

    private var player: MediaPlayer? = null
    private var session: MediaSession? = null

    /**
     * What the card says, which is not what the file is called.
     *
     * The media carousel draws the session's own metadata, not the
     * notification's title and text — so a card built from the file name read
     * `…6385715.wav`, which names the one thing about the clip nobody needs.
     * Carried from the result that produced it instead.
     */
    private var heading: String = ""
    private var byline: String = ""

    /** The session's token, for `MediaStyle.setMediaSession`. Null before first use. */
    val token: MediaSession.Token? get() = session?.sessionToken

    /**
     * Make the session exist, without starting anything.
     *
     * The scrubber, the times and the transport row are drawn by the platform
     * from the session, and the session was created on first `play()` — so the
     * notification arrived with no token attached and showed two bare buttons
     * until something was tapped. By then the player it was meant to offer had
     * already been the thing you were looking for.
     *
     * Called when the notification is posted, so it is a player from the first
     * frame: parked at zero, with the length of the clip already known.
     */
    @Synchronized
    fun prepare(context: Context, path: String, title: String, subtitle: String) {
        heading = title
        byline = subtitle
        if (path == _state.value.path && player != null) {
            // Same clip, new wording: the card still has to be redrawn.
            session?.setMetadata(metadata(path, _state.value.durationMillis))
            return
        }
        if (!File(path).isFile) return
        release()
        ensureSession(context)
        // Opened only to read the length, and released immediately: holding a
        // prepared player for every result would keep a codec alive per
        // notification, on the process that is already holding the weights.
        val duration = runCatching {
            val probe = MediaPlayer()
            try {
                probe.setDataSource(path)
                probe.prepare()
                probe.duration.toLong()
            } finally {
                probe.release()
            }
        }.getOrDefault(0L)
        session?.setMetadata(metadata(path, duration))
        session?.isActive = true
        _state.value = State(path, playing = false, positionMillis = 0, durationMillis = duration)
        publish()
    }

    @Synchronized
    fun play(context: Context, path: String) {
        if (!File(path).isFile) {
            EngineLog.w(TAG, "nothing to play at $path")
            return
        }
        // Same file, merely paused: resume rather than start again. Tapping
        // play after pause should not go back to the beginning.
        if (path == _state.value.path && player != null) {
            resume()
            return
        }
        release()
        ensureSession(context)

        runCatching {
            player = MediaPlayer().apply {
                setDataSource(path)
                setOnCompletionListener { finished() }
                prepare()
                start()
            }
        }.onFailure {
            EngineLog.w(TAG, "could not play $path: ${it.message}")
            release()
            return
        }

        val duration = player?.duration?.toLong() ?: 0L
        session?.setMetadata(metadata(path, duration))
        session?.isActive = true
        _state.value = State(path, playing = true, positionMillis = 0, durationMillis = duration)
        publish()
    }

    @Synchronized
    fun pause() {
        val active = player ?: return
        runCatching { if (active.isPlaying) active.pause() }
        _state.value = _state.value.copy(
            playing = false,
            positionMillis = runCatching { active.currentPosition.toLong() }.getOrDefault(0L),
        )
        publish()
    }

    @Synchronized
    fun resume() {
        val active = player ?: return
        runCatching { active.start() }
        _state.value = _state.value.copy(playing = true)
        publish()
    }

    @Synchronized
    fun seekTo(millis: Long) {
        val active = player ?: return
        runCatching { active.seekTo(millis.toInt()) }
        _state.value = _state.value.copy(positionMillis = millis)
        publish()
    }

    @Synchronized
    fun stop() {
        release()
        _state.value = State()
        session?.isActive = false
        publish()
    }

    private fun finished() {
        // Kept, rather than torn down: the notification stays, showing a player
        // parked at the end, which is what every other media app does and what
        // makes "play it again" one tap instead of one regeneration.
        runCatching { player?.seekTo(0) }
        _state.value = _state.value.copy(playing = false, positionMillis = 0)
        publish()
    }

    private fun release() {
        runCatching { player?.takeIf { it.isPlaying }?.stop() }
        runCatching { player?.release() }
        player = null
    }

    private fun metadata(path: String, duration: Long): MediaMetadata =
        MediaMetadata.Builder()
            .putString(
                MediaMetadata.METADATA_KEY_TITLE,
                heading.ifBlank { File(path).name },
            )
            .putString(
                MediaMetadata.METADATA_KEY_ARTIST,
                byline.ifBlank { "On-Device AI" },
            )
            // Without this the scrubber has no length and is not drawn at all.
            .putLong(MediaMetadata.METADATA_KEY_DURATION, duration)
            .build()

    private fun ensureSession(context: Context) {
        if (session != null) return
        session = MediaSession(context.applicationContext, TAG).apply {
            setCallback(
                object : MediaSession.Callback() {
                    override fun onPlay() = resume()
                    override fun onPause() = pause()
                    override fun onStop() = stop()
                    override fun onSeekTo(pos: Long) = seekTo(pos)
                },
            )
        }
    }

    /**
     * Tell the platform where playback is and how fast it is moving.
     *
     * The speed is what makes the scrubber move on its own. Publishing a
     * position with a speed of zero gives a bar that only advances when
     * something else happens to republish, which reads as playback stuttering.
     */
    private fun publish() {
        val current = _state.value
        val position = player?.let { runCatching { it.currentPosition.toLong() }.getOrNull() }
            ?: current.positionMillis
        session?.setPlaybackState(
            PlaybackState.Builder()
                .setActions(
                    PlaybackState.ACTION_PLAY or
                        PlaybackState.ACTION_PAUSE or
                        PlaybackState.ACTION_PLAY_PAUSE or
                        PlaybackState.ACTION_STOP or
                        PlaybackState.ACTION_SEEK_TO,
                )
                .setState(
                    when {
                        current.playing -> PlaybackState.STATE_PLAYING
                        current.active -> PlaybackState.STATE_PAUSED
                        else -> PlaybackState.STATE_STOPPED
                    },
                    position,
                    if (current.playing) 1.0f else 0f,
                )
                .build(),
        )
    }

    private const val TAG = "ResultAudio"
}
