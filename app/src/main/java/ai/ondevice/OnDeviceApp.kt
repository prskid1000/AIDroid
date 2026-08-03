package ai.ondevice

import android.app.Application
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

    override fun onCreate() {
        super.onCreate()
        scope.launch {
            // A download interrupted by a crash, a force-stop or a reinstall leaves a row saying RUNNING with nothing behind it.
            downloader.resumeInterrupted()
        }
    }

}
