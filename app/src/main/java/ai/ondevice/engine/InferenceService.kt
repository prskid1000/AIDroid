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
            combine(
                engines.state,
                running,
                video.state,
                image.state,
                proxy.status,
            ) { engine, count, clip, still, served ->
                Progress(engine, count, clip, still, served)
            }
                .collectLatest { progress ->
                    notificationManager().notify(NOTIFICATION_ID, buildNotification(progress))
                    // The third term is new and load-bearing. An idle proxy has
                    // nothing loaded and nothing running, so without it this
                    // service stopped itself the moment the last generation
                    // finished — and took the listening socket with it. The
                    // server was then up for exactly as long as the last
                    // request, which is indistinguishable from it never working.
                    if (progress.engine.loaded == null &&
                        progress.count == 0 &&
                        !progress.served.listening
                    ) {
                        stopSelf()
                    }
                }
        }

        // Match the stored configuration on every start, so enabling the proxy
        // and starting this service are one action rather than two that have to
        // happen in the right order.
        lifecycleScope.launch { runCatching { proxy.sync() } }

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

    /** What the three things that can be running are each doing. */
    private data class Progress(
        val engine: EngineState,
        val count: Int,
        val clip: ai.ondevice.ui.vm.VideoState,
        val still: ai.ondevice.ui.vm.ImageState,
        val served: ai.ondevice.proxy.ProxyServer.Status,
    )

    /**
     * What is happening, said the way the screen says it.
     *
     * This used to read "Model loaded" with a token rate under it and nothing
     * else, whatever was actually going on — the service knew only whether the
     * chat model was resident, so a clip four minutes into a five minute run
     * was described as a loaded model. It now names the work, counts the steps
     * and carries the same bar the screen draws, because the notification is
     * the only view of a run once you have left the app.
     */
    private fun buildNotification(progress: Progress? = null): Notification {
        val open = PendingIntent.getActivity(
            this,
            0,
            // Reuse the running instance rather than stacking a new one.
            // With the activity's default launch mode a tap built a second
            // MainActivity and destroyed the first, which cleared the
            // activity-scoped view models and cancelled the generation they
            // were holding — so opening the app from the notification that
            // said it was generating is what stopped it generating.
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val builder = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notify_generate)
            .setContentIntent(open)
            .setOnlyAlertOnce(true)
            .setOngoing(true)

        // Whichever run is actually going. Only one can sample at a time —
        // the diffusion engine holds a load lock — so the first match is the
        // whole story rather than an arbitrary pick.
        val clip = progress?.clip?.takeIf { it.generating || it.loadingModel }
        val still = progress?.still?.takeIf { it.generating || it.loadingModel }

        when {
            clip != null -> builder.describe(
                what = "Making a clip",
                model = clip.model?.label,
                phase = if (clip.loadingModel) "loading weights" else clip.phase.label,
                step = clip.step,
                steps = clip.progressSteps,
                rate = clip.secondsPerStep,
            )
            still != null -> builder.describe(
                what = "Making a picture",
                model = still.model?.label,
                phase = if (still.loadingModel) "loading weights" else still.phase.label,
                step = still.step,
                steps = still.progressSteps,
                rate = still.secondsPerStep,
            )
            progress != null && progress.engine.tokensPerSecond > 0 -> builder
                .setContentTitle("Answering")
                .setContentText(
                    listOfNotNull(
                        progress.engine.loaded?.modelId,
                        Fmt.tokensPerSecond(progress.engine.tokensPerSecond),
                    ).joinToString(" · "),
                )
                .setProgress(0, 0, true)
            progress?.count.let { it != null && it > 0 } -> builder
                .setContentTitle("Working")
                .setContentText(progress?.engine?.loaded?.modelId.orEmpty())
                .setProgress(0, 0, true)
            // Nothing running, but the port is open. Worth its own line: this
            // is the one resting state the person did not start by tapping
            // something, and the address is what makes it recognisable rather
            // than alarming.
            progress?.served?.listening == true -> builder
                .setContentTitle("Serving the API")
                .setContentText(
                    listOfNotNull(
                        progress.served.url,
                        progress.engine.loaded?.modelId,
                    ).joinToString(" · "),
                )
            // Nothing running: the model is resident and that is all there is
            // to say. Worth saying, because several gigabytes are being held.
            else -> builder
                .setContentTitle("Model in memory")
                .setContentText(progress?.engine?.loaded?.modelId.orEmpty())
        }
        return builder.build()
    }

    /**
     * One run, in the shape a status bar can show: what, on what, how far.
     *
     * The bar is determinate while there are steps to count and indeterminate
     * while there are not — a load and a decode take minutes and report no
     * step, and a bar frozen at zero for that long reads as a run that has
     * stalled rather than one that is working.
     */
    private fun Notification.Builder.describe(
        what: String,
        model: String?,
        phase: String,
        step: Int,
        steps: Int,
        rate: Float,
    ): Notification.Builder {
        setContentTitle(what)
        setContentText(
            listOfNotNull(
                model,
                if (steps > 0 && step > 0) "step $step/$steps" else phase,
                rate.takeIf { it > 0f }?.let { String.format("%.0f s/it", it) },
            ).joinToString(" · "),
        )
        if (steps > 0 && step > 0) setProgress(steps, step, false) else setProgress(0, 0, true)
        return this
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
