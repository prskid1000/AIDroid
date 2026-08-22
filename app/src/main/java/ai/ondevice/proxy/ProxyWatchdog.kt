package ai.ondevice.proxy

import ai.ondevice.R
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.engine.EngineLog
import ai.ondevice.engine.InferenceService
import android.app.AlarmManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.PowerManager
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.first
import javax.inject.Inject
import javax.inject.Singleton

/**
 * The thing that notices the proxy is gone and brings it back.
 *
 * **Why this exists at all, measured rather than assumed.** `START_STICKY` was
 * supposed to be the answer: the system kills the process, brings the service
 * back, `onCreate` re-opens the socket. It does bring the service back. What it
 * does not do is let it be a foreground service, and the persisted engine log
 * on this device caught the exact sentence — *"not allowed to be foreground:
 * Service.startForeground() not allowed due to mAllowStartForeground false"*.
 *
 * That was the last line written for the next fourteen hours. The restarted
 * service stopped itself — correctly, because a service that may not go
 * foreground cannot hold a socket — and nothing ever asked again. The process
 * stayed alive and frozen (*"unfreezing … reason = moto_freezer"*, adj 199)
 * with no service in it, so from the outside the phone was on, the app was
 * running, and the port answered nobody. Opening the app by hand was the only
 * thing that fixed it, which is precisely the report.
 *
 * So the restart cannot come from inside a process that has already lost its
 * standing. It has to come from something the platform lets start a foreground
 * service from the background, and this app already knows what that is: an
 * **exact alarm**. It is the same exemption [ai.ondevice.engine.workflow.Scheduler]
 * relies on to run a workflow at seven in the morning, chosen there for the
 * same reason and written up in the manifest beside the permission.
 *
 * **The cost, stated rather than buried.** A wake-up every fifteen minutes for
 * as long as the proxy is switched on. When the service is up the check is a
 * broadcast, a boolean and an idempotent start that AMS turns into a no-op —
 * but it is still a wake-up, and this is the honest price of the alternative
 * being fourteen hours of silence. Nothing is armed while the proxy is off.
 *
 * **When even this is refused**, `SCHEDULE_EXACT_ALARM` having been denied, the
 * alarm still fires — it simply may not start a service when it does. Then the
 * answer is a notification and a tap, the same answer `ScheduleReceiver` gives
 * a run it was not allowed to start. SPEC §1.2: a thing that cannot work says
 * so, and says what would fix it.
 */
@Singleton
class ProxyWatchdog @Inject constructor(
    @ApplicationContext private val context: Context,
    private val prefs: AppPrefs,
) {

    private val alarms: AlarmManager
        get() = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager

    private fun notifications(): NotificationManager =
        context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    /**
     * Whether the proxy is switched on — or null when that could not be read.
     *
     * **Three answers, not two, and the third one is the whole point.** This was
     * a plain `Boolean` folding every failure into `false`, and it was caught
     * doing exactly what this class exists to prevent: the watchdog woke, read
     * nothing, concluded the proxy was off, cancelled its own alarm and never
     * ran again. A silent failure inside the thing written to end silent
     * failures.
     *
     * The empty document is the case that matters, and it is not hypothetical —
     * it was observed three times in a row on this device, straight after an
     * install, with no exception thrown anywhere. `ProxyDocument.parse("")`
     * returns `EMPTY` and `EMPTY.enabled` is `false`, so a read that came back
     * with nothing was indistinguishable from a proxy somebody had switched
     * off. `AppPrefs` already says what a blank actually means, on the key
     * itself: *"Empty means never configured, not 'off'."* This is that comment,
     * enforced.
     */
    suspend fun enabled(): Boolean? = runCatching {
        val raw = prefs.proxyDocument.first()
        // Blank is "could not tell", never "no". A configured proxy is at least
        // a JSON object, so there is no reading of an empty string that is a
        // real answer to this question.
        if (raw.isBlank()) null else ProxyConfig(ProxyDocument.parse(raw)).enabled
    }.onFailure {
        EngineLog.w("ProxyWatchdog", "could not read the proxy configuration: ${it.message}", it)
    }.getOrNull()

    /**
     * Whether a restart can happen without anybody present.
     *
     * The exact-alarm permission and nothing else: an inexact alarm wakes us
     * with no way to do the one thing we were woken for.
     */
    val canRestartUnattended: Boolean
        get() = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            runCatching { alarms.canScheduleExactAlarms() }.getOrDefault(false)
        } else {
            true
        }

    /**
     * Whether the system has been told to stop economising on this app.
     *
     * Not something this app can grant itself, and not something it should
     * pretend does not matter. This phone freezes the process the moment it is
     * not in front — the `moto_freezer` line above — and battery optimisation
     * is the switch that governs how readily. The proxy is a server: frozen and
     * off are the same thing to whoever is calling it.
     */
    val exemptFromBatteryOptimisation: Boolean
        get() = runCatching {
            (context.getSystemService(Context.POWER_SERVICE) as PowerManager)
                .isIgnoringBatteryOptimizations(context.packageName)
        }.getOrDefault(false)

    /** Where to send somebody to grant exact alarms, or null when it is already granted. */
    fun exactAlarmSettings(): Intent? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && !canRestartUnattended) {
            Intent(android.provider.Settings.ACTION_REQUEST_SCHEDULE_EXACT_ALARM)
                .setData(android.net.Uri.parse("package:" + context.packageName))
        } else {
            null
        }

    /**
     * Where to send somebody to stop the system economising, or null when it already is not.
     *
     * The list screen rather than the request dialog. `ACTION_REQUEST_IGNORE_
     * BATTERY_OPTIMIZATIONS` puts up a system prompt and is a Play-policy
     * restricted action that this app would have to justify; the list is always
     * allowed, and it is one tap further for something nobody should be nagged
     * into.
     */
    fun batterySettings(): Intent? =
        if (exemptFromBatteryOptimisation) {
            null
        } else {
            Intent(android.provider.Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
        }

    // ── arming ───────────────────────────────────────────────────────────

    /**
     * Make sure a check is pending, [inMillis] from now.
     *
     * Idempotent by construction: one `PendingIntent`, one request code, so
     * arming twice moves the alarm rather than stacking a second one. Called
     * from every place that has just learned something about the service's fate
     * — it started, it stopped, the task was swiped away, the app was updated —
     * because none of those know which of them will turn out to be the last.
     */
    fun arm(inMillis: Long = INTERVAL_MILLIS) {
        val at = System.currentTimeMillis() + inMillis
        runCatching {
            if (canRestartUnattended) {
                alarms.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, at, pending())
            } else {
                // Still armed, because being woken late is worth more than not
                // being woken at all. What happens when it lands is in [check].
                alarms.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, at, pending())
            }
        }.onFailure {
            EngineLog.w("ProxyWatchdog", "could not arm the restart check: ${it.message}")
        }
    }

    fun disarm() {
        runCatching { alarms.cancel(pending()) }
        runCatching { notifications().cancel(NOTIFICATION_ID) }
    }

    /**
     * Arm or disarm to match the stored configuration.
     *
     * Three outcomes rather than two, because [enabled] has three answers and
     * only one of them — a document that was read and says off — is a reason to
     * stop watching.
     */
    suspend fun sync() {
        // Said out loud, because "the watchdog quietly decided there was nothing
        // to watch" is the same shape of failure it exists to catch — and it is
        // the one this file hit first: a check that runs, decides no, and leaves
        // no trace is indistinguishable from one that never ran.
        when (enabled()) {
            true -> {
                EngineLog.i(
                    "ProxyWatchdog",
                    "the proxy is on; a restart check is armed for " +
                        "${INTERVAL_MILLIS / 60_000} minutes from now",
                )
                arm()
            }
            false -> {
                EngineLog.i("ProxyWatchdog", "the proxy is off; no restart check armed")
                disarm()
            }
            // Keep watching. Disarming on an answer this uncertain is how the
            // watchdog switched itself off, and staying armed costs one wake-up
            // against a proxy that never comes back.
            null -> {
                EngineLog.w(
                    "ProxyWatchdog",
                    "could not tell whether the proxy is on; staying armed rather than assuming",
                )
                arm()
            }
        }
    }

    // ── the check ────────────────────────────────────────────────────────

    /**
     * Is the proxy meant to be up? Then put it back.
     *
     * There is deliberately no test for "is the service already running". It
     * would need `getRunningServices`, which has returned only the caller's own
     * services since Android 8 and is documented as being for the running-apps
     * UI, or a flag held in a process that may not exist to be read. Starting a
     * service that is already started is a no-op landing in `onStartCommand`
     * with a null action — so asking is both harder than doing and worse.
     */
    suspend fun check() {
        when (enabled()) {
            false -> {
                EngineLog.i("ProxyWatchdog", "woke, the proxy is switched off, standing down")
                disarm()
                return
            }
            // Woken, and the configuration would not read. Nothing is started —
            // there is no evidence anybody asked for a server — but the next
            // check is armed, because the alternative is never asking again.
            null -> {
                EngineLog.w(
                    "ProxyWatchdog",
                    "woke and could not read the configuration; will look again",
                )
                arm()
                return
            }
            true -> Unit
        }

        val failure = runCatching {
            context.startForegroundService(Intent(context, InferenceService::class.java))
        }.exceptionOrNull()

        if (failure == null) {
            EngineLog.i("ProxyWatchdog", "woke and asked for the server; it was allowed")
            runCatching { notifications().cancel(NOTIFICATION_ID) }
        } else {
            // Refused: the alarm woke us and the platform still would not let
            // this app hold a socket. There is nothing left to do but say so.
            EngineLog.w(
                "ProxyWatchdog",
                "woke to restart the proxy and was refused: ${failure.message}",
            )
            notifyDown()
        }

        // Re-armed last and unconditionally. An alarm that fires once and does
        // not set the next one is a watchdog that watches for fifteen minutes,
        // and that failure looks exactly like one that was never armed.
        arm()
    }

    /**
     * The tap that starts a server this app was not allowed to start itself.
     *
     * Deliberately not silent, and deliberately not repeated: one id, so each
     * check replaces the last rather than stacking a column of them, and it is
     * cancelled the moment a start succeeds.
     */
    private fun notifyDown() {
        val manager = notifications()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            runCatching {
                manager.createNotificationChannel(
                    NotificationChannel(
                        CHANNEL_ID,
                        context.getString(R.string.proxy_channel_name),
                        NotificationManager.IMPORTANCE_DEFAULT,
                    ).apply { setShowBadge(false) },
                )
            }
        }
        val open = PendingIntent.getActivity(
            context,
            REQUEST_OPEN,
            Intent(context, ai.ondevice.MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val body = "This device stopped it and will not let it start again on its own. " +
            "Tap to open the app, which is allowed to. Settings → Proxy says what to " +
            "change so this stops happening."
        runCatching {
            manager.notify(
                NOTIFICATION_ID,
                Notification.Builder(context, CHANNEL_ID)
                    .setSmallIcon(R.drawable.ic_notify_generate)
                    .setContentTitle("The API is not being served")
                    .setContentText(body)
                    .setStyle(Notification.BigTextStyle().bigText(body))
                    .setContentIntent(open)
                    .setAutoCancel(true)
                    .build(),
            )
        }
    }

    private fun pending(): PendingIntent = PendingIntent.getBroadcast(
        context,
        REQUEST_CHECK,
        Intent(context, ProxyWatchdogReceiver::class.java).setAction(ACTION_CHECK),
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
    )

    companion object {
        const val ACTION_CHECK = "ai.ondevice.proxy.WATCHDOG_CHECK"

        /**
         * How long the proxy may be down before anything notices.
         *
         * Fifteen minutes is not a tuning choice so much as the floor: Doze
         * throttles `setExactAndAllowWhileIdle` to roughly one firing every
         * nine minutes per app, so anything shorter is a wake-up the system
         * quietly declines to honour. The number it replaces is fourteen hours.
         */
        const val INTERVAL_MILLIS = 15 * 60 * 1000L

        private const val CHANNEL_ID = "proxy"
        private const val NOTIFICATION_ID = 1004
        private const val REQUEST_CHECK = 1
        private const val REQUEST_OPEN = 2
    }
}
