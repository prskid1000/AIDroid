package ai.ondevice.core

import android.content.Context
import android.content.Intent

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
 * The kill is delegated rather than done here, and that is the whole trick.
 * An alarm set before dying does not work: when it fires there is nothing left
 * with foreground standing, and Android 10 onward drops the activity start —
 * measured, the process returned and the screen did not. So a second process
 * does the killing and the launching, from a foreground it still has.
 */
fun forceStopAndRestart(context: Context) {
    val app = context.applicationContext
    val handover = Intent(app, Class.forName("ai.ondevice.RestartActivity")).apply {
        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        putExtra("ai.ondevice.restart.pid", android.os.Process.myPid())
    }
    // Started from the foreground, which is the only moment this is allowed.
    // It kills this process from its own, then launches a fresh one.
    app.startActivity(handover)
}
