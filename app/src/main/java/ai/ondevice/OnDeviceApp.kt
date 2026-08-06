package ai.ondevice

import android.app.Activity
import android.app.Application
import android.os.Bundle
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.di.ApplicationScope
import ai.ondevice.engine.RuntimeRegistry
import dagger.hilt.android.HiltAndroidApp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import javax.inject.Inject

/** SPEC §13 — offline-first, no telemetry, no account, no crash reporting that transmits content. */
@HiltAndroidApp
class OnDeviceApp : Application() {

    @Inject lateinit var db: OnDeviceDatabase

    @Inject lateinit var registry: RuntimeRegistry

    @Inject @ApplicationScope lateinit var scope: CoroutineScope

    @Inject lateinit var downloader: ai.ondevice.data.download.Downloader

    @Inject lateinit var foreground: ai.ondevice.engine.workflow.ForegroundWatcher

    @Inject lateinit var shortcuts: ai.ondevice.workflow.ShortcutPublisher

    @Inject lateinit var scheduler: ai.ondevice.engine.workflow.Scheduler

    override fun onCreate() {
        super.onCreate()
        scope.launch {
            // A download interrupted by a crash, a force-stop or a reinstall leaves a row saying RUNNING with nothing behind it.
            downloader.resumeInterrupted()
        }
        scope.launch {
            // Shortcuts do not survive a reinstall or a "clear data", and which
            // workflows belong in a share sheet is derived from graphs that only
            // exist once the database is open. Republished once at startup so the
            // rows are there before anybody goes looking for them.
            runCatching { shortcuts.republish() }
            /*
             * Alarms are lost more often than a reboot.
             *
             * BOOT_COMPLETED covers a restart, but the system also drops an
             * app's alarms when it is force-stopped, and a reinstall clears them
             * outright — after which a saved schedule would sit in the database
             * looking armed and never fire again until somebody happened to edit
             * it. Re-arming at startup is cheap and makes the row the truth.
             */
            runCatching { scheduler.rearmAll() }
        }
        watchForeground()
    }

    /**
     * Whether any screen of this app is in front of the user.
     *
     * The question the platform asks before allowing an activity start, and the
     * one a finished Send step has to answer to know whether it may open a
     * chooser or must park itself in a notification. Counted from the activity
     * callbacks rather than through a lifecycle observer, so it needs no
     * dependency this project does not already have.
     */
    private fun watchForeground() {
        registerActivityLifecycleCallbacks(
            object : ActivityLifecycleCallbacks {
                override fun onActivityStarted(activity: Activity) = foreground.onStart()
                override fun onActivityStopped(activity: Activity) = foreground.onStop()
                override fun onActivityCreated(activity: Activity, state: Bundle?) = Unit
                override fun onActivityResumed(activity: Activity) = Unit
                override fun onActivityPaused(activity: Activity) = Unit
                override fun onActivitySaveInstanceState(activity: Activity, out: Bundle) = Unit
                override fun onActivityDestroyed(activity: Activity) = Unit
            },
        )
    }
}
