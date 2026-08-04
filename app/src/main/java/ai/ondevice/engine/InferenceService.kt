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
            combine(engines.state, running) { state, count -> state to count }
                .collectLatest { (state, count) ->
                    notificationManager().notify(NOTIFICATION_ID, buildNotification(state, count))
                    if (state.loaded == null && count == 0) stopSelf()
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

    private fun buildNotification(state: EngineState?, active: Int = 0): Notification {
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
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setContentTitle(
                state?.loaded?.modelId
                    ?: if (active > 0) "Generating" else "Model loaded",
            )
            .setContentText(
                if (state != null && state.tokensPerSecond > 0) {
                    Fmt.tokensPerSecond(state.tokensPerSecond)
                } else {
                    ""
                },
            )
            .setContentIntent(open)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .build()
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
