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
import ai.ondevice.core.ThermalPolicy
import ai.ondevice.data.hf.DeviceCapabilities
import ai.ondevice.data.prefs.AppPrefs
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * SPEC §2.1 — the inference host. It survives backgrounding and owns
 * memory-pressure negotiation; §8.3 adds thermal and wake-lock policy.
 *
 * Three rules from the spec are enforced here rather than in the UI, because
 * the UI is not guaranteed to exist while generation runs:
 *  - the wake-lock is held **only** while generating, and released on
 *    completion;
 *  - the thermal status is read live and the configured policy applied at
 *    `THERMAL_STATUS_SEVERE`;
 *  - generation is cancellable at every stage, and cancelling frees native
 *    memory rather than merely detaching the callback.
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

        lifecycleScope.launch { watchThermal() }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        when (intent?.action) {
            ACTION_ACQUIRE_WAKELOCK -> acquireWakeLock()
            ACTION_RELEASE_WAKELOCK -> releaseWakeLock()
        }
        return START_NOT_STICKY
    }

    /**
     * §8.3 — read `PowerManager.getCurrentThermalStatus()` and act on the
     * configured policy. The user's choice is honoured literally: "continue
     * regardless" means the app does not quietly throttle behind their back.
     */
    private suspend fun watchThermal() {
        val policy = prefs.thermalPolicy.first()
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        pm.addThermalStatusListener { status ->
            if (status < PowerManager.THERMAL_STATUS_SEVERE) return@addThermalStatusListener
            when (policy) {
                ThermalPolicy.CONTINUE -> Unit
                ThermalPolicy.REDUCE_THREADS,
                ThermalPolicy.DOWNSHIFT_CPU,
                -> lifecycleScope.launch {
                    // Both are applied through the same string-keyed boundary as
                    // any other parameter change — there is no special path.
                    engines.llama?.applyParams(
                        ai.ondevice.core.SparseParams.of(
                            "n_threads" to (capabilities.performanceCores / 2).coerceAtLeast(1),
                        ),
                    )
                }
                ThermalPolicy.PAUSE -> lifecycleScope.launch { engines.unload() }
            }
        }
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
                    append(" · thermal ${capabilities.thermalLabel}")
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
