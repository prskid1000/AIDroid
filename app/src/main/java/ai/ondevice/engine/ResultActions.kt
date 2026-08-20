package ai.ondevice.engine

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.media.MediaPlayer
import android.os.Bundle
import android.widget.Toast

/**
 * The buttons on a finished-run notification.
 *
 * A result notification that only says "Speech ready" is telling you about
 * something you then have to go and find. The two results worth acting on
 * from the shade are the two the app can act on without a screen: a spoken
 * line can be played, and a transcript can be copied.
 *
 * An activity rather than a broadcast receiver, and the reason is the
 * clipboard. Since Android 10 a background process may not reliably write it —
 * the write is dropped rather than refused, which is the worst of both — and
 * an activity is a foreground context by definition. It draws nothing, lives
 * for one `onCreate`, and carries `taskAffinity=""` so it leaves no card in
 * recents, which is the reasoning already written on `HandoffActivity` and
 * `OAuthCallbackActivity`.
 */
class ResultActionActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        when (intent?.action) {
            ACTION_PLAY -> intent.getStringExtra(EXTRA_PATH)?.let(ResultAudio::play)
            ACTION_STOP -> ResultAudio.stop()
            ACTION_COPY -> intent.getStringExtra(EXTRA_TEXT)?.let { text ->
                val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                clipboard.setPrimaryClip(ClipData.newPlainText("transcript", text))
                // Android 13 and later show their own clipboard confirmation,
                // and a second one on top of it is noise.
                if (android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.TIRAMISU) {
                    Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show()
                }
            }
        }
        finish()
    }

    companion object {
        const val ACTION_PLAY = "ai.ondevice.result.PLAY"
        const val ACTION_STOP = "ai.ondevice.result.STOP"
        const val ACTION_COPY = "ai.ondevice.result.COPY"
        const val EXTRA_PATH = "path"
        const val EXTRA_TEXT = "text"
    }
}

/**
 * One player, for the shade.
 *
 * Deliberately not the Voice tab's player: that one belongs to a screen and
 * ends with it, which is exactly wrong for a notification that exists because
 * there is no screen. One at a time, because two spoken lines over each other
 * is not a feature anybody asked for.
 */
object ResultAudio {

    private var player: MediaPlayer? = null

    @Synchronized
    fun play(path: String) {
        stop()
        runCatching {
            player = MediaPlayer().apply {
                setDataSource(path)
                setOnCompletionListener { stop() }
                prepare()
                start()
            }
        }.onFailure {
            EngineLog.w("ResultAudio", "could not play $path: ${it.message}")
            stop()
        }
    }

    @Synchronized
    fun stop() {
        runCatching { player?.takeIf { it.isPlaying }?.stop() }
        runCatching { player?.release() }
        player = null
    }
}
