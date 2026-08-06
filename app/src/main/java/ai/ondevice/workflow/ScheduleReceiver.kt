package ai.ondevice.workflow

import ai.ondevice.R
import ai.ondevice.core.workflow.Schedule
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.engine.workflow.Scheduler
import ai.ondevice.engine.workflow.WorkflowLauncher
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.time.ZonedDateTime
import javax.inject.Inject

/**
 * Where a scheduled run begins.
 *
 * A receiver rather than a service, and reached by an exact alarm rather than by
 * `WorkManager`, because an app woken by an exact alarm is exempt from the rule
 * that forbids starting a foreground service from the background — see
 * [Scheduler]. Everything after that is the ordinary run path.
 *
 * It also re-arms. A schedule that fires once and does not set the next alarm is
 * a schedule that runs once, and the failure looks exactly like a schedule that
 * was never saved.
 */
@AndroidEntryPoint
class ScheduleReceiver : BroadcastReceiver() {

    @Inject lateinit var db: OnDeviceDatabase

    @Inject lateinit var launcher: WorkflowLauncher

    @Inject lateinit var scheduler: Scheduler

    override fun onReceive(context: Context, intent: Intent) {
        // Re-arm everything on the clock moving, then stop: no run is due.
        if (intent.action in RECOMPUTE) {
            val pending = goAsync()
            CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
                try { scheduler.rearmAll() } finally { pending.finish() }
            }
            return
        }

        val workflowId = intent.getStringExtra(EXTRA_WORKFLOW) ?: return
        val pending = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                fire(context, workflowId)
            } finally {
                pending.finish()
            }
        }
    }

    private suspend fun fire(context: Context, workflowId: String) {
        val workflow = db.workflows().get(workflowId) ?: return
        val schedule = Schedule.decode(workflow.scheduleJson)
        if (!schedule.enabled) return

        val now = System.currentTimeMillis()

        /*
         * Refused, recorded, and said — never silently dropped.
         *
         * A schedule that quietly does nothing is indistinguishable from a
         * broken one, and this is the only place that can tell the difference.
         * SPEC 1.2 applies as much to a run that did not happen as to one that
         * failed.
         */
        val refusal = scheduler.refuseReason(schedule)
            ?: "something else was already running".takeIf { launcher.busy }

        if (refusal != null) {
            save(workflowId, schedule.copy(lastSkippedAt = now, lastSkipReason = refusal))
            scheduler.arm(workflowId, schedule, ZonedDateTime.now())
            return
        }

        if (!scheduler.canRunUnattended) {
            // Woken, but not allowed to start a service. The same answer a Send
            // step gives from the background: offer the tap rather than fail.
            notifyDue(context, workflow.id, workflow.name)
        } else {
            launcher.launch(workflowId)
        }

        // A "once" schedule disarms itself in the same write that records the
        // firing; a recurring one is armed for its next occurrence.
        val fired = schedule.copy(
            lastFiredAt = now,
            enabled = schedule.isRecurring,
            lastSkipReason = null,
        )
        save(workflowId, fired)
        scheduler.arm(workflowId, fired, ZonedDateTime.now())
    }

    private suspend fun save(workflowId: String, schedule: Schedule) {
        val row = db.workflows().get(workflowId) ?: return
        db.workflows().upsert(row.copy(scheduleJson = schedule.encode()))
    }

    /** The tap that starts a run this app was not allowed to start itself. */
    private fun notifyDue(context: Context, workflowId: String, name: String) {
        val manager =
            context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            manager.createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    context.getString(R.string.schedule_channel_name),
                    NotificationManager.IMPORTANCE_DEFAULT,
                ),
            )
        }
        val open = PendingIntent.getActivity(
            context,
            workflowId.hashCode(),
            TriggerActivity.intentFor(context, workflowId),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        manager.notify(
            NOTIFICATION_BASE + (workflowId.hashCode() and 0xFFFF),
            Notification.Builder(context, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_notify_generate)
                .setContentTitle("$name is due")
                .setContentText(
                    "This app has not been allowed to start runs on its own. Tap to run it now.",
                )
                .setContentIntent(open)
                .setAutoCancel(true)
                .build(),
        )
    }

    companion object {
        const val ACTION_FIRE = "ai.ondevice.action.SCHEDULE_FIRE"
        private const val EXTRA_WORKFLOW = "ai.ondevice.extra.SCHEDULED_WORKFLOW"
        private const val CHANNEL_ID = "schedule"
        private const val NOTIFICATION_BASE = 1300

        /**
         * Broadcasts that invalidate every armed alarm.
         *
         * A schedule is a wall-clock time, so moving the clock or the zone moves
         * every one of them. All three are themselves exempt from the
         * foreground-service start restriction, which is why re-arming from here
         * is safe.
         */
        private val RECOMPUTE = setOf(
            Intent.ACTION_TIME_CHANGED,
            Intent.ACTION_TIMEZONE_CHANGED,
            Intent.ACTION_BOOT_COMPLETED,
        )

        fun intentFor(context: Context, workflowId: String): Intent =
            Intent(context, ScheduleReceiver::class.java)
                .setAction(ACTION_FIRE)
                .putExtra(EXTRA_WORKFLOW, workflowId)
    }
}
