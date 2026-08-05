package ai.ondevice.core

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context

/**
 * Kill this process and come back, for the work that cannot be asked to stop.
 *
 * Cancel is honoured everywhere the runtime offers somewhere to honour it, and
 * that is most of a run: sampling and the VAE decode abort inside the current
 * ggml graph. Two phases offer nothing. `new_sd_ctx` reads a checkpoint off
 * storage with no callback and no flag, and the prompt encode is one graph
 * whose abandonment hands sd.cpp an empty result it asserts on rather than
 * checks — so abandoning it is a crash, not a cancel. In both, a press is
 * recorded and applied whenever the runtime next reaches a point that can take
 * it, which on this hardware has been measured at three and a half minutes.
 *
 * A spinner that means "wait, possibly for minutes, possibly not" is not a
 * stop. This is: the process dies, the work dies with it, and the memory is
 * returned by the kernel rather than by asking politely.
 *
 * It costs less than it sounds. Checkpoints are on disk, settings and history
 * are in Room and already committed, and the run being killed is one somebody
 * has just said they no longer want. What is lost is the partial result, which
 * is exactly what Cancel discards anyway.
 *
 * The alarm is set before the kill because a dead process cannot start
 * anything. `RTC` rather than `RTC_WAKEUP`: the screen is on — somebody just
 * pressed a button — and this does not deserve a wake lock.
 */
fun forceStopAndRestart(context: Context) {
    val app = context.applicationContext
    val launch = app.packageManager.getLaunchIntentForPackage(app.packageName)
    if (launch != null) {
        val pending = PendingIntent.getActivity(
            app,
            RESTART_REQUEST,
            launch,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_CANCEL_CURRENT,
        )
        // A moment, not none: the alarm has to outlive the process that set it.
        app.getSystemService(AlarmManager::class.java)
            ?.set(AlarmManager.RTC, System.currentTimeMillis() + RESTART_DELAY_MILLIS, pending)
    }
    android.os.Process.killProcess(android.os.Process.myPid())
}

private const val RESTART_REQUEST = 0xA1D2
private const val RESTART_DELAY_MILLIS = 400L
