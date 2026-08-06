package ai.ondevice.engine.workflow

import ai.ondevice.core.workflow.Schedule
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.hf.DeviceCapabilities
import ai.ondevice.workflow.ScheduleReceiver
import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import dagger.hilt.android.qualifiers.ApplicationContext
import java.time.ZonedDateTime
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Waking up to run a workflow.
 *
 * **An exact alarm, and not `WorkManager`.** Every run needs a foreground
 * service, and Android 12 and later forbid starting one from the background —
 * but exact alarms are explicitly exempt from that restriction, which is the
 * whole reason this shape works. `WorkManager` would have run the graph inside
 * *its* foreground service, which since Android 14 must declare a service type,
 * so `specialUse` and its justification would have to be merged onto a service
 * this app does not own, beside the `InferenceService` that already does exactly
 * this job. Two foreground services for one run, to avoid one permission
 * prompt, is a bad trade.
 *
 * The cost is that `SCHEDULE_EXACT_ALARM` is denied by default from Android 13
 * onward and this app targets 35, so it has to be asked for — and what happens
 * when it is refused is in [armInexact], which is the same answer a Send step
 * gives when the app is in the background: the platform will not do this
 * unattended, so say so and offer the tap.
 */
@Singleton
class Scheduler @Inject constructor(
    @ApplicationContext private val context: Context,
    private val db: OnDeviceDatabase,
    private val capabilities: DeviceCapabilities,
) {

    private val alarms: AlarmManager
        get() = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager

    /** Whether a run can start without anybody present. */
    val canRunUnattended: Boolean
        get() = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            alarms.canScheduleExactAlarms()
        } else {
            true
        }

    /** Where to send somebody to grant it, or null when there is nothing to grant. */
    fun permissionIntent(): Intent? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && !alarms.canScheduleExactAlarms()) {
            Intent(android.provider.Settings.ACTION_REQUEST_SCHEDULE_EXACT_ALARM)
                .setData(android.net.Uri.parse("package:${context.packageName}"))
        } else {
            null
        }

    // ── arming ───────────────────────────────────────────────────────────

    /**
     * Set, move or clear this workflow's alarm to match its schedule.
     *
     * Keyed on the workflow id so re-arming replaces rather than stacks, which
     * matters because this is called on every edit.
     */
    fun arm(workflowId: String, schedule: Schedule, now: ZonedDateTime = ZonedDateTime.now()) {
        val at = schedule.nextOccurrence(now)
        if (at == null) {
            cancel(workflowId)
            return
        }
        val millis = at.toInstant().toEpochMilli()
        val pending = pendingFor(workflowId)
        runCatching {
            if (canRunUnattended) {
                alarms.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, millis, pending)
            } else {
                armInexact(millis, pending)
            }
        }
    }

    /**
     * The fallback, which is a notification rather than nothing.
     *
     * Without the exact-alarm permission this app may still be woken — it simply
     * may not start a foreground service when it is, and every run needs one. So
     * the alarm still fires, and the receiver posts *"this is due, tap to start
     * it"*. One tap, and the ordinary foreground path takes over.
     *
     * `setAndAllowWhileIdle` rather than `set`, so Doze delays it rather than
     * holding it until the next maintenance window — a morning briefing that
     * arrives at lunchtime is not one.
     */
    private fun armInexact(millis: Long, pending: PendingIntent) {
        alarms.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, millis, pending)
    }

    fun cancel(workflowId: String) {
        runCatching { alarms.cancel(pendingFor(workflowId)) }
    }

    /**
     * Re-arm everything.
     *
     * Alarms do not survive a reboot, and a wall-clock schedule has to be
     * recomputed when the clock or the zone moves under it — so this is called
     * from boot, from a time change, and after a run.
     */
    suspend fun rearmAll() {
        db.workflows().mostRecent(LIMIT).forEach { workflow ->
            arm(workflow.id, Schedule.decode(workflow.scheduleJson))
        }
    }

    private fun pendingFor(workflowId: String): PendingIntent = PendingIntent.getBroadcast(
        context,
        workflowId.hashCode(),
        ScheduleReceiver.intentFor(context, workflowId),
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
    )

    // ── the guards ───────────────────────────────────────────────────────

    /**
     * Whether it is reasonable to start this right now, and why not when it is not.
     *
     * The part of scheduling that is actually design rather than plumbing.
     * Waking up is easy; deciding that three quarters of an hour of GPU at full
     * tilt is a good idea at three in the morning, unattended, is not — a hot
     * phone on a bedside table with nobody watching is a worse outcome than a
     * run that did not happen. Returns null when it may proceed.
     */
    fun refuseReason(schedule: Schedule): String? {
        val battery = context.registerReceiver(
            null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED),
        )
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val percent = if (level >= 0 && scale > 0) level * 100 / scale else 100
        val plugged = (battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0

        return when {
            schedule.requireCharging && !plugged ->
                "the phone was not charging, and this run is set to need it"
            percent < schedule.minBatteryPercent ->
                "the battery was $percent% and this run is set to need " +
                    "${schedule.minBatteryPercent}%"
            /*
             * A throttled run is slower *and* hotter, so starting one while the
             * device is already complaining makes both worse.
             *
             * The app's thermal *policy* was removed because three of its four
             * settings could not do what they claimed — but reading the status
             * to decide whether to start something unattended is a different
             * question, and one the reading answers honestly.
             */
            capabilities.thermalStatus >= android.os.PowerManager.THERMAL_STATUS_SEVERE ->
                "the phone was already too warm (${capabilities.thermalLabel}) to run it well"
            else -> null
        }
    }

    private companion object {
        /** How many workflows to consider when re-arming. */
        const val LIMIT = 60
    }
}
