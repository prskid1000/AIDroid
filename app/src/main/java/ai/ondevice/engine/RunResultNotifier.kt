package ai.ondevice.engine

import ai.ondevice.MainActivity
import ai.ondevice.R
import ai.ondevice.core.Fmt
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File

/**
 * What a run left behind, once it has finished.
 *
 * The ongoing notification says what is happening and then goes away with the
 * thing it was describing, which leaves nothing at all to mark that a
 * forty-minute clip actually landed. Anything long enough to be worth leaving
 * the app for is long enough to be worth telling you about when it is done —
 * and a picture is worth showing rather than describing, so it is shown.
 *
 * Its own notification id, and deliberately not ongoing: this one is dismissed
 * by looking at it, which is the whole difference between a status and a
 * result.
 */
class RunResultNotifier(
    private val context: Context,
    /**
     * Whether any screen of this app is in front of the user.
     *
     * The app already counts this, from the activity callbacks, for the Send
     * step that has to know whether it may open a chooser or must park itself
     * in a notification. Asked rather than answered again here: a second count
     * is a second thing that can disagree, and this one would disagree exactly
     * when the first is right.
     */
    private val foreground: ai.ondevice.engine.workflow.ForegroundWatcher,
) {

    /** What finished, and what it produced. */
    sealed interface Result {
        val title: String

        data class Picture(
            override val title: String,
            val path: String,
            val caption: String,
        ) : Result

        data class Clip(
            override val title: String,
            val firstFrame: String?,
            val caption: String,
        ) : Result

        data class Words(
            override val title: String,
            val body: String,
            val caption: String,
        ) : Result

        data class Sound(
            override val title: String,
            val path: String,
            val caption: String,
        ) : Result
    }

    fun notify(result: Result) {
        // Nothing to say to somebody who is looking at the screen it happened
        // on. The ongoing notification is different — it exists for the case
        // where you have already left — but a result banner over the very tab
        // that just drew the result is noise.
        if (foreground.isForeground) return

        ensureChannel()
        val builder = Notification.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notify_generate)
            .setContentTitle(result.title)
            .setAutoCancel(true)
            .setOnlyAlertOnce(false)
            .setContentIntent(open())

        when (result) {
            is Result.Picture -> {
                builder.setContentText(result.caption)
                // Down-sampled before it goes anywhere near the notification.
                // A 1024-square PNG is four megabytes as a Bitmap, and the
                // system rejects a notification whose extras exceed its
                // transaction limit — silently, by showing nothing at all.
                thumbnail(result.path)?.let { bitmap ->
                    builder.setLargeIcon(bitmap)
                    builder.style = Notification.BigPictureStyle()
                        .bigPicture(bitmap)
                        .bigLargeIcon(null as Bitmap?)
                }
            }
            is Result.Clip -> {
                builder.setContentText(result.caption)
                result.firstFrame?.let { path ->
                    thumbnail(path)?.let { bitmap ->
                        builder.setLargeIcon(bitmap)
                        builder.style = Notification.BigPictureStyle()
                            .bigPicture(bitmap)
                            .bigLargeIcon(null as Bitmap?)
                    }
                }
            }
            is Result.Words -> {
                val preview = result.body.trim().replace(Regex("\\s+"), " ")
                builder.setContentText(preview.take(SUMMARY_CHARS))
                builder.style = Notification.BigTextStyle()
                    .bigText(preview.take(BIG_TEXT_CHARS))
                    .setSummaryText(result.caption)
            }
            is Result.Sound -> builder.setContentText(result.caption)
        }

        manager().notify(NOTIFICATION_ID, builder.build())
    }

    /**
     * A bitmap small enough to survive the binder.
     *
     * `inSampleSize` rather than a scaled copy, so the full-size image is never
     * decoded in the first place — this runs in the process that is already
     * holding several gigabytes of model weights, and it is the last one that
     * should be allocating a spare four megabytes to make a thumbnail.
     */
    private fun thumbnail(path: String): Bitmap? = runCatching {
        val file = File(path)
        if (!file.isFile) return null

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        val largest = maxOf(bounds.outWidth, bounds.outHeight)
        var sample = 1
        while (largest / sample > MAX_EDGE) sample *= 2

        BitmapFactory.decodeFile(
            path,
            BitmapFactory.Options().apply { inSampleSize = sample },
        )
    }.getOrNull()

    private fun open(): PendingIntent = PendingIntent.getActivity(
        context,
        1,
        Intent(context, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
    )

    private fun ensureChannel() {
        // Separate from the ongoing channel, and at DEFAULT rather than LOW.
        // The status channel is deliberately silent because it updates several
        // times a second; this one fires once, when something you asked for is
        // ready, and being silent would defeat it.
        manager().createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                context.getString(R.string.results_channel_name),
                NotificationManager.IMPORTANCE_DEFAULT,
            ).apply { setShowBadge(true) },
        )
    }

    private fun manager() =
        context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    companion object {
        const val CHANNEL_ID = "results"
        const val NOTIFICATION_ID = 1003

        private const val MAX_EDGE = 1024
        private const val SUMMARY_CHARS = 120
        private const val BIG_TEXT_CHARS = 900

        /** "12.4 tok/s · 1m 3s", for the line under a finished answer. */
        fun summary(tokensPerSecond: Float, millis: Long, extra: String? = null): String =
            listOfNotNull(
                tokensPerSecond.takeIf { it > 0f }?.let { Fmt.tokensPerSecond(it) },
                millis.takeIf { it > 1000L }?.let { it / 1000 }
                    ?.let { if (it >= 60) "${it / 60}m ${it % 60}s" else "${it}s" },
                extra,
            ).joinToString(" · ")
    }
}
