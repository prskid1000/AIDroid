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

    @Inject lateinit var requests: ai.ondevice.proxy.RequestLog

    @Inject lateinit var watchdog: ai.ondevice.proxy.ProxyWatchdog

    override fun onCreate() {
        super.onCreate()
        /*
         * Both logs get their file before anything else runs.
         *
         * First, because the lines worth having are the ones from startup —
         * a runtime that would not load, a bundle that was the wrong ABI — and
         * a log installed after the thing it was meant to record is a log of
         * the quiet part. Neither call touches the disk on this thread — both
         * read their tail and write theirs on [scope] — and both keep buffering
         * from the moment the class is first touched, so the lines from before
         * this ran are not the ones that go missing.
         */
        val diagnostics = java.io.File(filesDir, "logs")
        ai.ondevice.engine.EngineLog.persistTo(java.io.File(diagnostics, "engine.jsonl"), scope)
        requests.persistTo(java.io.File(diagnostics, "requests.jsonl"), scope)
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
            /*
             * And the proxy's own alarm, for the same reason as the two above.
             *
             * It is armed by the service too, but the service is exactly the
             * thing that may not be running: if the last one was refused its
             * foreground standing and stopped, the alarm it armed is the only
             * thing left — and a force-stop or a reinstall drops that alarm as
             * surely as it drops a schedule. This is the one place that runs
             * whatever else has happened.
             */
            runCatching { watchdog.sync() }
        }
        watchForeground()
    }

    /**
     * Start the inference service when the proxy is configured on.
     *
     * The service is what holds the socket, so starting it is the only thing
     * that makes "enabled" mean "listening". Only when it was already switched
     * on — an app launch is not consent.
     *
     * **Called from the first activity start, and that is the whole point.**
     * This lived in `onCreate` alongside the download sweep and the alarm
     * re-arm, and it did not work: those two are ordinary background work, and
     * this is a foreground-service start. Since Android 12 the platform refuses
     * one from the background, `Application.onCreate` *is* the background as far
     * as that check is concerned, and the refusal is an exception that a
     * `runCatching` swallowed without a word. The symptom was a proxy that read
     * as enabled on its own screen and was listening on nothing until somebody
     * toggled it off and on again.
     *
     * An activity having started is the platform's own definition of the
     * exemption, so by here it is allowed. Failure is logged rather than
     * swallowed, because the last one cost an afternoon.
     *
     * **Asked on every activity start, not once per process.** It used to be
     * latched by a flag set *before* the attempt, so a single refusal — the
     * platform's, or a `first()` on a store that was not open yet — was final
     * for the life of the process, and the only way back was the Proxy screen,
     * where an edit calls `sync()` by another route. That is exactly the report:
     * enabled on launch, listening only after you go and look at it. Starting an
     * already-running service is a no-op, so the cheap fix is to stop
     * remembering and just ask again.
     */
    private suspend fun startProxyIfEnabled() {
        /*
         * Asked of the watchdog rather than re-derived here.
         *
         * This was the same three lines it holds, with one difference that
         * turned out to matter: it folded a blank document into `false`, and a
         * blank document means *never read*, not *switched off* — the comment on
         * the key in `AppPrefs` says exactly that. A cold read that came back
         * empty was therefore silently read as consent withheld, and the proxy
         * did not start. One answer to the question, in one place, with a third
         * value for "could not tell".
         */
        if (watchdog.enabled() != true) return
        runCatching {
            startForegroundService(
                android.content.Intent(this, ai.ondevice.engine.InferenceService::class.java),
            )
        }.onFailure {
            android.util.Log.w(
                "OnDeviceApp",
                "proxy is enabled but the service would not start: ${it.message}",
                it,
            )
        }
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
                override fun onActivityStarted(activity: Activity) {
                    foreground.onStart()
                    // The platform counts us as foreground from here, which is
                    // what a foreground-service start requires.
                    scope.launch { startProxyIfEnabled() }
                }
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
