package ai.ondevice

import android.app.Activity
import android.content.Intent
import android.os.Bundle

/**
 * Brings the app back after a run that could not be stopped any other way.
 *
 * The obvious approach does not work. Setting an alarm and killing the process
 * leaves nothing with foreground standing when the alarm fires, and Android 10
 * onward drops an activity start from the background — measured: the process
 * came back and the screen never did. No permission grants it either;
 * SCHEDULE_EXACT_ALARM governs when an alarm fires, not what it may then do.
 *
 * So the launch happens while the app is still in front, which is always
 * allowed, and the dying is delegated. This activity lives in its own process
 * (`android:process=":restart"`), so killing the main one does not take it with
 * them: it kills the caller, starts a fresh MainActivity from its own
 * foreground, and then ends itself. The new process is a genuinely new process
 * — the models are re-read from disk, which is the point, because the whole
 * reason to be here is that a native call would not let go of them.
 */
class RestartActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val doomed = intent.getIntExtra(EXTRA_PID, 0)
        if (doomed > 0 && doomed != android.os.Process.myPid()) {
            android.os.Process.killProcess(doomed)
        }

        packageManager.getLaunchIntentForPackage(packageName)?.let { launch ->
            // A new task, and the old one cleared: the activity that asked for
            // this is in a process that no longer exists, and restoring its
            // stack would restore a screen describing a run that is gone.
            launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            startActivity(launch)
        }

        finish()
        // This process has done its one job. Left alive it would sit in the
        // task list as a second, invisible copy of the app.
        android.os.Process.killProcess(android.os.Process.myPid())
    }

    companion object {
        const val EXTRA_PID = "ai.ondevice.restart.pid"
    }
}
