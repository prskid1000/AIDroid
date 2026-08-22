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

    /**
     * What brings the socket back when this service does not.
     *
     * Injected here rather than left to the app, because this class is the one
     * that finds out — it is the only place that learns the platform refused to
     * let it be foreground, and the only place that sees itself being destroyed.
     */
    @Inject lateinit var watchdog: ai.ondevice.proxy.ProxyWatchdog

    @Inject lateinit var runner: ModelRunner

    /**
     * What the proxy made, said by the proxy.
     *
     * The three in-app sessions are watched for an edge; anything without a
     * screen behind it has no edge to watch and reports here instead.
     */
    @Inject lateinit var results: RunResults

    private var wakeLock: PowerManager.WakeLock? = null

    @Inject lateinit var foreground: ai.ondevice.engine.workflow.ForegroundWatcher

    /** Fires the finished-run banners; see [RunResultNotifier]. */
    private val notifier by lazy { RunResultNotifier(this, foreground) }

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
        val clip: ai.ondevice.proxy.VideoJobs.Job?,
    )

    override fun onCreate() {
        super.onCreate()
        createChannel()
        /*
         * Guarded, because this is now also reached by a restart nobody asked
         * for. With `START_STICKY` the system brings this service back after it
         * kills the process, and a `startForeground` that the platform refuses
         * throws — in `onCreate`, which would take the whole process down and
         * turn one lost socket into a crash loop. A refusal here means the
         * service may not run, and the honest response is to stop rather than
         * to die: the next activity start asks again, and that one is allowed.
         */
        runCatching { startForeground(NOTIFICATION_ID, buildNotification(null)) }
            .onFailure {
                EngineLog.w("InferenceService", "not allowed to be foreground: ${it.message}")
                /*
                 * Refused, and this is where it used to end.
                 *
                 * Measured on this device: the sticky restart happened, this
                 * line was logged, the service stopped, and the proxy stayed
                 * down for fourteen hours while the process sat frozen with
                 * nothing in it. One refusal was final, because the only thing
                 * that ever asked again was somebody opening the app.
                 *
                 * So the refusal now arms the one mechanism that can ask from
                 * outside a process with no standing. See [ProxyWatchdog].
                 */
                refusedForeground = true
                // Armed here and now, on this thread. It was a `lifecycleScope`
                // launch, which is a coroutine scope that `stopSelf()` on the
                // next line cancels — so the single most important call in this
                // fix was racing the thing that makes it necessary.
                //
                // Armed unconditionally, without asking whether the proxy is on:
                // that answer is a suspending DataStore read, and `check` asks
                // it anyway and disarms itself when it is no.
                runCatching { watchdog.arm(RETRY_MILLIS) }
                stopSelf()
                return
            }

        // Armed on the way up, not only on the way down. Whatever kills this
        // service next may not give it the chance to notice — a process killed
        // outright runs no `onDestroy` — so the check has to already be pending
        // before anything goes wrong.
        lifecycleScope.launch { runCatching { watchdog.sync() } }

        // What the proxy made, once it has made it.
        //
        // Separate from the status notification above, and that separation is
        // the point: the ongoing one says what is happening and goes away with
        // it, so a picture that finished left "Serving the API" on screen and
        // nothing to say a picture had been made at all.
        //
        // Its own coroutine because a result is an event rather than a state —
        // there is no snapshot to fold it into.
        lifecycleScope.launch {
            results.produced.collect { produced -> notifier.notify(produced) }
        }

        // Play, pause and stop change what the buttons should say, and the
        // notification is the only place they are said. Redrawn from the
        // player's own state rather than from whichever button was tapped,
        // because the lock screen and Quick Settings can drive it too.
        lifecycleScope.launch {
            ResultAudio.state.collect { notifier.refresh() }
        }

        // Alive while anything is loaded, anything is running, or the proxy is
        // listening.
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
                proxy.videoJob,
            ) { conversation, spoken, served, remote, clip ->
                OtherRuns(conversation, spoken, served, remote, clip)
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
                    videoJob = o.clip,
                    resident = runner.residentRuntime,
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
        /*
         * Sticky, so a system kill is survivable.
         *
         * This was `START_NOT_STICKY`, which is the right answer for a service
         * that exists for the duration of one job — and the wrong one for a
         * service that also holds a listening socket. The device kills this
         * process under memory pressure like any other: logcat says
         * `Process ai.ondevice has died: prcp FGS` with `isKilledByAm=false`,
         * every other app on the phone dies in the same second, and with
         * `NOT_STICKY` there is no line after it about restarting anything. The
         * proxy was then off until somebody next opened the app, which from the
         * outside is a server that stops on its own after a while.
         *
         * Sticky brings the service back with a null intent, `onCreate` runs
         * `proxy.sync()` again, and the socket is back. When the proxy is *not*
         * switched on there is nothing to come back for — the watcher in
         * `onCreate` sees nothing loaded, nothing running and nothing listening
         * and calls `stopSelf()`, and a service stopped that way is not
         * restarted again. So this costs one short-lived process after a kill
         * and buys back the only feature that needs the process to outlive the
         * screen.
         */
        return START_STICKY
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

    /**
     * The app swiped out of recents.
     *
     * Not the same event as being stopped, and on this hardware not a harmless
     * one: the platform is free to kill the process, and the OEM layer takes it
     * as permission rather than as a suggestion. The service is not stopped
     * here — a proxy that is switched on has not been switched off by somebody
     * tidying their recents — but the check is re-armed, because whether this
     * process is still alive in a minute is now out of our hands.
     */
    override fun onTaskRemoved(rootIntent: Intent?) {
        armWatchdogIfServing()
        super.onTaskRemoved(rootIntent)
    }

    override fun onDestroy() {
        releaseWakeLock()
        // Every way out ends here, including the ones nobody chose: the system
        // stopping the service, the OEM's battery manager doing it, a crash on
        // another thread. The only one this cannot cover is a process killed
        // outright, which is why `onCreate` arms as well.
        armWatchdogIfServing()
        super.onDestroy()
    }

    /**
     * Arm from a place that cannot suspend.
     *
     * Read off the status flow rather than the stored document, because
     * `onDestroy` runs after `lifecycleScope` is cancelled and a DataStore read
     * is suspending. The flow's `enabled` is the same answer one step fresher —
     * and when the service is stopping *because* the proxy was switched off it
     * is already false, so this disarms instead, which is what should happen.
     */
    private fun armWatchdogIfServing() {
        // The refusal path already armed, and it is the one case where the test
        // below gets the wrong answer: `onCreate` returned before `proxy.sync()`
        // ran, so the status still says disabled, and this would cancel the
        // retry that the refusal exists to schedule. That is the whole bug,
        // restored by the code meant to fix it.
        if (refusedForeground) return
        runCatching {
            if (proxy.status.value.enabled) watchdog.arm() else watchdog.disarm()
        }
    }

    /** Set when the platform refused foreground standing; see [armWatchdogIfServing]. */
    private var refusedForeground = false



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
                    notifier.notify(
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
                    notifier.notify(
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
                    notifier.notify(
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
                    notifier.notify(
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
                    notifier.notify(
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

        /**
         * How long to wait before trying again after the platform said no.
         *
         * Shorter than the watchdog's own interval, because a refusal is
         * usually a moment rather than a state — the app was mid-transition,
         * or the process had just been rebuilt — and the ordinary fifteen
         * minutes would be a long time to be off for a condition that has
         * already passed. If it has not passed, the next check is fifteen
         * minutes behind this one anyway.
         */
        private const val RETRY_MILLIS = 60 * 1000L

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
