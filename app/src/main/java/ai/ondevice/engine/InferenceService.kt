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
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * SPEC §2.1 — the inference host. It survives backgrounding and owns
 * memory-pressure negotiation and the wake-lock.
 *
 * Two rules from the spec are enforced here rather than in the UI, because the
 * UI is not guaranteed to exist while generation runs:
 *  - the wake-lock is held **only** while generating, and released on
 *    completion;
 *  - generation is cancellable at every stage, and cancelling frees native
 *    memory rather than merely detaching the callback.
 *
 * There is deliberately no thermal policy. The kernel governor already throttles
 * a hot SoC, and the app's own layer over it did not work: three of the four
 * settings wrote `n_threads` into a struct that a live llama.cpp context never
 * re-reads (it is reload-only), so "reduce threads" and "downshift to CPU"
 * changed nothing at all, and the policy was sampled once with `first()` so
 * editing it did nothing until the service restarted. A control that reads as
 * protection and provides none is worse than admitting the platform handles it.
 */
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

        lifecycleScope.launch {
            engines.state.collectLatest { state ->
                notificationManager().notify(NOTIFICATION_ID, buildNotification(state))
                if (state.loaded == null) stopSelf()
            }
        }

    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        when (intent?.action) {
            ACTION_ACQUIRE_WAKELOCK -> acquireWakeLock()
            ACTION_RELEASE_WAKELOCK -> releaseWakeLock()
        }
        return START_NOT_STICKY
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKELOCK_TAG).apply {
            setReferenceCounted(false)
            acquire(WAKELOCK_TIMEOUT_MS)
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    override fun onDestroy() {
        releaseWakeLock()
        super.onDestroy()
    }

    private fun buildNotification(state: EngineState?): Notification {
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setContentTitle(state?.loaded?.modelId ?: "Model loaded")
            .setContentText(
                buildString {
                    state?.backend?.let { append(it.label) }
                    if (state != null && state.tokensPerSecond > 0) {
                        append(" · ${Fmt.tokensPerSecond(state.tokensPerSecond)}")
                    }
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
        private const val WAKELOCK_TIMEOUT_MS = 30 * 60 * 1000L

        fun holdWakeLock(context: Context) {
            context.startForegroundService(
                Intent(context, InferenceService::class.java).setAction(ACTION_ACQUIRE_WAKELOCK),
            )
        }

        fun releaseWakeLock(context: Context) {
            context.startService(
                Intent(context, InferenceService::class.java).setAction(ACTION_RELEASE_WAKELOCK),
            )
        }
    }
}
