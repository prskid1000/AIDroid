package ai.ondevice.engine.workflow

import ai.ondevice.R
import ai.ondevice.workflow.HandoffActivity
import android.app.ActivityOptions
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.content.FileProvider
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Getting a finished result out of the app, against the platform's rules.
 *
 * There are three routes and only one of them is automatic in the sense people
 * mean by it.
 *
 * - **The clipboard** needs nobody and works from anywhere.
 * - **A share, with the app on screen** opens the chooser now.
 * - **A share, with the app in the background** cannot open anything. Android
 *   refuses activity starts from a backgrounded app, and for a run that is
 *   minutes to the better part of an hour that is the ordinary case rather than
 *   the edge one. It becomes a notification, and the tap is what launches it.
 *
 * And no intent *sends* a mail. Handing text to Gmail opens Gmail's composer
 * filled in; a person taps send. Delivery with nobody present is an API call,
 * which in this app is a Tool step against a connected server.
 */
@Singleton
class HandoffDispatcher @Inject constructor(
    @ApplicationContext private val context: Context,
    private val foreground: ForegroundWatcher,
) {

    /**
     * Do it now if it can be done now.
     *
     * Returns false when it could not — the caller parks it and shows it on the
     * run screen. Never throws: a hand-off that cannot go out must not fail a
     * run whose expensive half already succeeded.
     */
    fun dispatch(handoff: Handoff): Boolean = runCatching {
        when (handoff.target) {
            HandoffTarget.CLIPBOARD -> {
                copyToClipboard(handoff)
                true
            }
            /*
             * The check decides it, and `startActivity` is not consulted.
             *
             * A background activity launch does not throw. The platform logs
             * "Background activity launch blocked!" and the call returns
             * normally — so a runCatching around it reports success for a
             * launch that never happened, and the result is lost with nothing
             * anywhere to say so. That is precisely what it did.
             *
             * So the foreground check is the whole decision, it is read *here*
             * rather than earlier, and anything that is not certainly on screen
             * is parked. Being wrong towards parking costs a tap on a
             * notification; being wrong the other way costs the result.
             */
            HandoffTarget.APP -> {
                if (!foreground.isForeground) {
                    notify(handoff)
                    return@runCatching false
                }
                context.startActivity(
                    chooserFor(handoff).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                )
                true
            }
        }
    }.getOrElse {
        // Threw for some other reason — no app resolves it, most likely a
        // named package that has since been uninstalled. Still a result
        // somebody is waiting for, so it is parked rather than dropped.
        runCatching { notify(handoff) }
        false
    }

    private fun copyToClipboard(handoff: Handoff) {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val label = handoff.subject.ifBlank { "Workflow result" }
        val clip = if (handoff.export != null) {
            ClipData.newUri(context.contentResolver, label, uriFor(handoff.export))
        } else {
            ClipData.newPlainText(label, handoff.text)
        }
        clipboard.setPrimaryClip(clip)
    }

    /**
     * The intent this hand-off is, before anyone decides how to fire it.
     *
     * A named package narrows which app resolves it; it never names an
     * activity, because the receiving activity's class is that app's business
     * and would break the first time they refactored.
     */
    fun intentFor(handoff: Handoff): Intent = Intent(Intent.ACTION_SEND).apply {
        val export = handoff.export
        if (export != null) {
            type = export.mime
            putExtra(Intent.EXTRA_STREAM, uriFor(export))
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } else {
            type = "text/plain"
        }
        if (handoff.text.isNotBlank()) putExtra(Intent.EXTRA_TEXT, handoff.text)
        if (handoff.subject.isNotBlank()) {
            putExtra(Intent.EXTRA_SUBJECT, handoff.subject)
            putExtra(Intent.EXTRA_TITLE, handoff.subject)
        }
        handoff.packageName?.let { setPackage(it) }
    }

    fun chooserFor(handoff: Handoff): Intent {
        val send = intentFor(handoff)
        // A package was named, so there is nothing to choose between. Wrapping
        // it in a chooser anyway shows a one-row sheet, which reads as the app
        // failing to do what was asked.
        return if (handoff.packageName != null) {
            send
        } else {
            Intent.createChooser(send, handoff.subject.ifBlank { "Send" })
        }
    }

    private fun uriFor(export: ai.ondevice.core.Export) = FileProvider.getUriForFile(
        context,
        "${context.packageName}.fileprovider",
        export.staged,
    )

    /**
     * Park it in the shade, with the launch privilege it will need on the tap.
     *
     * Apps targeting Android 15 and above no longer implicitly grant background
     * activity start privileges to the pending intents they create, so a
     * notification built the ordinary way is tapped and nothing happens —
     * silently, with no log unless you go looking. The creator has to opt in.
     *
     * `getActivity` and not a receiver or a service: since Android 12 a
     * notification may not trampoline an activity start through either.
     */
    private fun notify(handoff: Handoff) {
        createChannel()
        val intent = HandoffActivity.intentFor(context, handoff.nodeId)
        val pending = PendingIntent.getActivity(
            context,
            handoff.nodeId.hashCode(),
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            backgroundStartOptions(),
        )
        manager().notify(
            NOTIFICATION_BASE + (handoff.nodeId.hashCode() and 0xFFFF),
            Notification.Builder(context, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_notify_generate)
                .setContentTitle(handoff.describe)
                .setContentText("The workflow finished while the app was away. Tap to send it.")
                .setContentIntent(pending)
                .setAutoCancel(true)
                .build(),
        )
    }

    private fun backgroundStartOptions() = ActivityOptions.makeBasic()
        .setPendingIntentCreatorBackgroundActivityStartMode(
            ActivityOptions.MODE_BACKGROUND_ACTIVITY_START_ALLOWED,
        )
        .toBundle()

    private fun manager() =
        context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            manager().createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    context.getString(R.string.handoff_channel_name),
                    NotificationManager.IMPORTANCE_DEFAULT,
                ),
            )
        }
    }

    private companion object {
        const val CHANNEL_ID = "handoff"
        const val NOTIFICATION_BASE = 1100
    }
}

/**
 * Whether any of this app's screens is in front of the user.
 *
 * Counted from the activity callbacks rather than read from a lifecycle
 * observer, so it needs no dependency this project does not already have. The
 * question it answers is exactly the one the platform asks before allowing an
 * activity start.
 */
@Singleton
class ForegroundWatcher @Inject constructor() {

    private var started = 0

    val isForeground: Boolean get() = started > 0

    fun onStart() { started++ }

    fun onStop() { started = (started - 1).coerceAtLeast(0) }
}
