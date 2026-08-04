package ai.ondevice.data.download

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.LifecycleService
import ai.ondevice.MainActivity
import ai.ondevice.R
import ai.ondevice.core.Fmt
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import javax.inject.Inject

/** SPEC §3.4 — downloads run in a foreground service and survive app kill. */
@AndroidEntryPoint
class DownloadService : LifecycleService() {

    @Inject lateinit var downloader: Downloader

    override fun onCreate() {
        super.onCreate()
        createChannel()
        startForeground(NOTIFICATION_ID, buildNotification(null))

        lifecycleScope.launch {
            downloader.observeJobs().collectLatest { jobs ->
                val active = jobs.firstOrNull {
                    it.state == ai.ondevice.core.DownloadState.RUNNING ||
                        it.state == ai.ondevice.core.DownloadState.VERIFYING
                }
                if (active == null && jobs.none { it.state == ai.ondevice.core.DownloadState.QUEUED }) {
                    stopSelf()
                } else {
                    notificationManager().notify(NOTIFICATION_ID, buildNotification(active))
                }
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        when (intent?.action) {
            ACTION_START -> intent.getStringExtra(EXTRA_JOB_ID)?.let { downloader.start(it) }
            ACTION_PAUSE -> intent.getStringExtra(EXTRA_JOB_ID)?.let { downloader.pause(it) }
            ACTION_CANCEL -> intent.getStringExtra(EXTRA_JOB_ID)?.let { downloader.cancel(it) }
        }
        return START_STICKY
    }

    private fun buildNotification(job: DownloadJob?): Notification {
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
                .putExtra(MainActivity.EXTRA_DESTINATION, MainActivity.DEST_DOWNLOADS),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )

        val builder = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentIntent(open)
            .setOnlyAlertOnce(true)
            .setOngoing(true)

        if (job == null) {
            builder.setContentTitle("Preparing download")
        } else {
            builder
                .setContentTitle(job.displayName)
                .setContentText(
                    "${Fmt.percent(job.fraction)} · ${Fmt.bytes(job.bytesDone)} of ${Fmt.bytes(job.bytesTotal)}",
                )
                .setProgress(100, (job.fraction * 100).toInt(), false)
                .addAction(
                    Notification.Action.Builder(
                        null,
                        "Pause",
                        PendingIntent.getService(
                            this,
                            1,
                            Intent(this, DownloadService::class.java)
                                .setAction(ACTION_PAUSE)
                                .putExtra(EXTRA_JOB_ID, job.id),
                            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
                        ),
                    ).build(),
                )
        }
        return builder.build()
    }

    private fun notificationManager() =
        getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            notificationManager().createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    getString(R.string.download_channel_name),
                    NotificationManager.IMPORTANCE_LOW,
                ).apply { setShowBadge(false) },
            )
        }
    }

    companion object {
        const val CHANNEL_ID = "downloads"
        const val NOTIFICATION_ID = 1001
        const val ACTION_START = "ai.ondevice.download.START"
        const val ACTION_PAUSE = "ai.ondevice.download.PAUSE"
        const val ACTION_CANCEL = "ai.ondevice.download.CANCEL"
        const val EXTRA_JOB_ID = "job_id"

        fun start(context: Context, jobId: String) {
            context.startForegroundService(
                Intent(context, DownloadService::class.java)
                    .setAction(ACTION_START)
                    .putExtra(EXTRA_JOB_ID, jobId),
            )
        }

        /**
         * Bring the service up for a transfer that has already been started.
         *
         * [start] tells the service to begin a job; this says a job is under
         * way and the process needs to stay alive for it. The Downloader calls
         * it as it launches, which is the one place every download passes
         * through — and the reason there is no job id here is that the service
         * reads the queue itself and stops when nothing is left in it.
         */
        fun ensureRunning(context: Context) {
            runCatching {
                context.startForegroundService(
                    Intent(context, DownloadService::class.java),
                )
            }
        }
    }
}
