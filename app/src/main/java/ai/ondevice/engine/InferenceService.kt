package ai.ondevice.engine

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.PowerManager
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import ai.ondevice.MainActivity
import ai.ondevice.R
import ai.ondevice.core.Fmt
import ai.ondevice.data.hf.DeviceCapabilities
import ai.ondevice.data.prefs.AppPrefs
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject

/** SPEC §2.1 — the inference host. */
@AndroidEntryPoint
class InferenceService : LifecycleService() {

    @Inject lateinit var engines: EngineManager

    @Inject lateinit var capabilities: DeviceCapabilities

    @Inject lateinit var prefs: AppPrefs

    // The sessions own the runs now, so they are the only things that know
    // what is happening. Injected rather than guessed at from the engine's
    // loaded model, which was all this service used to have and which says
    // nothing at all during an image, a clip or a transcription.
    @Inject lateinit var video: ai.ondevice.ui.vm.VideoSession

    @Inject lateinit var image: ai.ondevice.ui.vm.ImageSession

    // Chat and Voice were not here, and their absence is why the notification
    // said "Model in memory" through a transcription and had no token rate
    // during an answer: the service was reading `EngineState.tokensPerSecond`,
    // which nothing has ever written. The sessions hold the real numbers.
    @Inject lateinit var chat: ai.ondevice.ui.vm.ChatSession

    @Inject lateinit var voice: ai.ondevice.ui.vm.VoiceSession

    /**
     * The HTTP surface.
     *
     * Here rather than in a service of its own, because this one already
     * declares the `specialUse` foreground type and already exists to stop the
     * system reclaiming a process holding gigabytes. A socket is exactly that
     * kind of thing to hold.
     */
    @Inject lateinit var proxy: ai.ondevice.proxy.ProxyServer

    private var wakeLock: PowerManager.WakeLock? = null

    @Inject lateinit var foreground: ai.ondevice.engine.workflow.ForegroundWatcher

    /** Fires the finished-run banners; see [RunResultNotifier]. */
    private val results by lazy { RunResultNotifier(this, foreground) }

    /** The last snapshot, so a finish can be seen as the edge it is. */
    private var previous: RunSnapshot? = null

    /** Half a snapshot each, so neither combine exceeds the typed overloads. */
    private data class LocalRuns(
        val engine: EngineState,
        val count: Int,
        val clip: ai.ondevice.ui.vm.VideoState,
        val still: ai.ondevice.ui.vm.ImageState,
    )

    private data class OtherRuns(
        val chat: ai.ondevice.ui.vm.ChatState,
        val voice: ai.ondevice.ui.vm.VoiceState,
        val served: ai.ondevice.proxy.ProxyServer.Status,
        val remote: ai.ondevice.proxy.ProxyActivity?,
    )

    override fun onCreate() {
        super.onCreate()
        createChannel()
        startForeground(NOTIFICATION_ID, buildNotification(null))

        // Alive while anything is loaded *or* anything is running.
        //
        // It used to stop the moment llama held nothing, which is every moment
        // of an image, a clip, a transcription or a spoken line: those run on
        // the other four engines, and this service's only notion of "busy" was
        // the chat model's. So the one service that keeps the process alive
        // when you leave the app switched itself off at the start of every run
        // that was not a conversation — and a forty-five minute clip, in a
        // process holding ten gigabytes, is the first thing Android reclaims.
        lifecycleScope.launch {
            // The socket first, then the watcher that decides whether to stop.
            //
            // These were two independent coroutines and the order between them
            // was a race this lost every time: the collector fires immediately
            // with the initial state — nothing loaded, nothing running, nothing
            // listening — decides there is no reason to exist, and calls
            // `stopSelf()` before `sync()` has opened anything. The service was
            // started and stopped inside the same eighty milliseconds, which in
            // logcat reads as "Background started FGS: Allowed" followed by an
            // FGS stop, and on the device reads as a proxy that is switched on
            // and listening on nothing.
            //
            // Sequenced rather than guarded with a flag, because "look before
            // deciding there is nothing to do" is the actual requirement and a
            // flag would be a second way of saying it.
            runCatching { proxy.sync() }

            // Two typed combines folded into one, because the overloads stop at
            // five flows and there are eight things that can be happening at
            // once. Each half is a real tuple rather than an array of Any — the
            // casts that would need are exactly the sort of thing that survives
            // a refactor by silently reading the wrong field.
            val local: kotlinx.coroutines.flow.Flow<LocalRuns> = combine(
                engines.state,
                running,
                video.state,
                image.state,
            ) { engine, count, clip, still -> LocalRuns(engine, count, clip, still) }

            val elsewhere: kotlinx.coroutines.flow.Flow<OtherRuns> = combine(
                chat.state,
                voice.state,
                proxy.status,
                proxy.activity,
            ) { conversation, spoken, served, remote ->
                OtherRuns(conversation, spoken, served, remote)
            }

            combine(local, elsewhere) { l, o ->
                RunSnapshot(
                    engine = l.engine,
                    count = l.count,
                    clip = l.clip,
                    still = l.still,
                    chat = o.chat,
                    voice = o.voice,
                    served = o.served,
                    remote = o.remote,
                )
            }
                .collectLatest { snapshot ->
                    notificationManager().notify(NOTIFICATION_ID, buildNotification(snapshot))
                    announceFinished(snapshot)
                    // The third term is load-bearing. An idle proxy has nothing
                    // loaded and nothing running, so without it this service
                    // stopped itself the moment the last generation finished —
                    // and took the listening socket with it. The server was then
                    // up for exactly as long as the last request, which is
                    // indistinguishable from it never having worked.
                    if (RunStatus.shouldStop(snapshot)) stopSelf()
                }
        }

    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        when (intent?.action) {
            ACTION_ACQUIRE_WAKELOCK -> acquireWakeLock()
            // Only when the last runner has finished. Two runs can overlap —
            // a clip rendering while a conversation answers — and the first to
            // finish must not take the lock out from under the second.
            ACTION_RELEASE_WAKELOCK -> if (running.value == 0) releaseWakeLock()
        }
        return START_NOT_STICKY
    }

    private fun acquireWakeLock() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        val lock = wakeLock ?: pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKELOCK_TAG)
            .apply { setReferenceCounted(false) }
            .also { wakeLock = it }
        // Re-arms the timeout rather than returning early when already held:
        // the timeout is a backstop against a leaked lock, and a run that
        // outlives it would otherwise have the CPU taken away mid-generation.
        lock.acquire(WAKELOCK_TIMEOUT_MS)
    }

    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    override fun onDestroy() {
        releaseWakeLock()
        super.onDestroy()
    }



    /**
     * What is happening, in the shape a status bar can show.
     *
     * The deciding is in [RunStatus] rather than here: it is a pure function of
     * a snapshot, so it can be tested without a device — and it used to be a
     * `when` in this file that knew about two of the five kinds of run and said
     * "Model in memory" through the other three.
     */
    private fun buildNotification(snapshot: RunSnapshot? = null): Notification {
        val open = PendingIntent.getActivity(
            this,
            0,
            // Reuse the running instance rather than stacking a new one. With
            // the activity's default launch mode a tap built a second
            // MainActivity and destroyed the first, which cleared the
            // activity-scoped view models and cancelled the generation they
            // were holding — so opening the app from the notification that said
            // it was generating is what stopped it generating.
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )

        val line = snapshot?.let { RunStatus.describe(it) }
            ?: RunLine("Model in memory", "")

        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notify_generate)
            .setContentIntent(open)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setContentTitle(line.title)
            .setContentText(line.detail)
            .apply {
                // Determinate while there are steps to count, indeterminate
                // while there are not. A load and a VAE decode take minutes and
                // report no step, and a bar frozen at zero for that long reads
                // as a run that has stalled rather than one that is working.
                if (line.determinate) {
                    setProgress(line.steps, line.step, false)
                } else if (line.title !in RESTING) {
                    setProgress(0, 0, true)
                }
            }
            .build()
    }

    /**
     * Say what a run left behind, once it has finished.
     *
     * Edge-triggered against the previous snapshot rather than polled, because
     * "is finished" is not a state anything here holds — only the transition
     * out of running is observable, and it passes once.
     */
    private fun announceFinished(now: RunSnapshot) {
        val was = previous
        previous = now
        was ?: return

        // A picture.
        if (was.still.generating && !now.still.generating) {
            now.still.lastImage
                ?.takeIf { it.path != was.still.lastImage?.path }
                ?.let { image ->
                    results.notify(
                        RunResultNotifier.Result.Picture(
                            title = "Picture ready",
                            path = image.path,
                            caption = RunResultNotifier.summary(
                                0f,
                                now.still.elapsedMillis,
                                "${image.width}x${image.height}",
                            ),
                        ),
                    )
                }
        }

        // A clip. Its first frame stands for it — there is no muxer here, and a
        // directory of PNGs has nothing else to show.
        if (was.clip.generating && !now.clip.generating) {
            now.clip.clip
                ?.takeIf { it.directory != was.clip.clip?.directory }
                ?.let { clip ->
                    results.notify(
                        RunResultNotifier.Result.Clip(
                            title = "Clip ready",
                            firstFrame = clip.frames.firstOrNull(),
                            caption = RunResultNotifier.summary(
                                0f,
                                now.clip.elapsedMillis,
                                "${clip.frames.size} frames",
                            ),
                        ),
                    )
                }
        }

        // An answer, local or remote. The text is worth showing: it is the
        // whole result, and a notification that says only "done" makes you open
        // the app to find out what it said.
        if (was.chat.generating && !now.chat.generating) {
            now.chat.messages.lastOrNull()
                ?.takeIf { it.role == ai.ondevice.core.MessageRole.ASSISTANT }
                ?.content
                ?.takeIf { it.isNotBlank() }
                ?.let { text ->
                    results.notify(
                        RunResultNotifier.Result.Words(
                            title = "Answer ready",
                            body = text,
                            caption = RunResultNotifier.summary(
                                was.chat.tokensPerSecond,
                                0L,
                                now.chat.model?.label,
                            ),
                        ),
                    )
                }
        }

        // A spoken line.
        if (was.voice.speaking && !now.voice.speaking) {
            now.voice.lastAudioPath
                ?.takeIf { it != was.voice.lastAudioPath }
                ?.let { path ->
                    results.notify(
                        RunResultNotifier.Result.Sound(
                            title = "Speech ready",
                            path = path,
                            caption = RunResultNotifier.summary(
                                0f,
                                now.voice.elapsedMillis,
                                now.voice.ttsModel?.label,
                            ),
                        ),
                    )
                }
        }

        // A transcript, which is words and should be read as words.
        if (was.voice.fileProgress in 0.0001f..0.9999f && now.voice.fileProgress >= 1f) {
            now.voice.segments
                .joinToString(" ") { it.text }
                .trim()
                .takeIf { it.isNotBlank() }
                ?.let { text ->
                    results.notify(
                        RunResultNotifier.Result.Words(
                            title = "Transcript ready",
                            body = text,
                            caption = RunResultNotifier.summary(
                                0f,
                                now.voice.elapsedMillis,
                                now.voice.sttModel?.label,
                            ),
                        ),
                    )
                }
        }
    }

    private fun notificationManager() =
        getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            notificationManager().createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    getString(R.string.inference_channel_name),
                    NotificationManager.IMPORTANCE_LOW,
                ).apply { setShowBadge(false) },
            )
        }
    }

    companion object {
        const val CHANNEL_ID = "inference"
        const val NOTIFICATION_ID = 1002
        const val ACTION_ACQUIRE_WAKELOCK = "ai.ondevice.inference.WAKE"
        const val ACTION_RELEASE_WAKELOCK = "ai.ondevice.inference.SLEEP"
        private const val WAKELOCK_TAG = "OnDeviceAI::generation"

        /** Titles that describe a state rather than work, so they get no spinner. */
        private val RESTING = setOf(
            "Model in memory", "Serving the API", "Proxy not listening", "Idle",
        )

        /**
         * A backstop against a leaked lock, not a budget for a run.
         *
         * It was thirty minutes, which is shorter than a clip: a Wan run
         * measured on this device takes three quarters of an hour at 384
         * square, so the CPU was taken away with a third of the sampling left
         * to do. Every acquire re-arms it, so what it actually bounds is how
         * long a lock can outlive the process that forgot to release it.
         */
        private const val WAKELOCK_TIMEOUT_MS = 4 * 60 * 60 * 1000L

        /**
         * How many generations are in flight, across every engine.
         *
         * The service used to infer this from the chat model being loaded,
         * which is true of a conversation and false of every other kind of
         * run. Counted rather than a flag because two can overlap — a clip
         * rendering while a conversation answers — and the first to finish
         * must not stop the service under the second.
         */
        private val running = MutableStateFlow(0)

        fun holdWakeLock(context: Context) {
            running.update { it + 1 }
            runCatching {
                context.startForegroundService(
                    Intent(context, InferenceService::class.java)
                        .setAction(ACTION_ACQUIRE_WAKELOCK),
                )
            }.onFailure {
                // Logged rather than swallowed. Since Android 12 this start is
                // refused outright when the app is in the background, and the
                // run then proceeds with nothing holding the CPU and nothing in
                // the shade to say it is happening — which is indistinguishable
                // from the request never having arrived, and was.
                EngineLog.w(
                    "InferenceService",
                    "could not start the foreground service for a run: ${it.message}",
                    it,
                )
            }
        }

        fun releaseWakeLock(context: Context) {
            running.update { (it - 1).coerceAtLeast(0) }
            runCatching {
                context.startService(
                    Intent(context, InferenceService::class.java)
                        .setAction(ACTION_RELEASE_WAKELOCK),
                )
            }
        }

        /**
         * Run [block] with the process held awake and in the foreground.
         *
         * The bracket rather than the two calls, because the release has to
         * happen on every way out — finished, cancelled, or thrown — and four
         * of the five generation paths simply did not make it. Only the
         * conversation ever held the lock; an image, a clip, a transcription
         * and a spoken line all ran with nothing keeping the process alive.
         */
        suspend fun <T> holdingWakeLock(context: Context, block: suspend () -> T): T {
            holdWakeLock(context)
            return try {
                block()
            } finally {
                releaseWakeLock(context)
            }
        }
    }
}
