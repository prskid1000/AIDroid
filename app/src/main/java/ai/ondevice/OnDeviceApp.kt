package ai.ondevice

import android.app.Application
import ai.ondevice.core.RuntimeState
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.RuntimeBundleEntity
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
            seed()
            // A download interrupted by a crash, a force-stop or a reinstall leaves a row saying RUNNING with nothing behind it.
            downloader.resumeInterrupted()
        }
    }

    /** One row per engine this build carries, written once. */
    private suspend fun seed() {
        if (db.runtimes().count() == 0) {
            db.runtimes().insertAll(
                registry.descriptors.map { descriptor ->
                    RuntimeBundleEntity(
                        engine = descriptor.id,
                        buildTag = descriptor.version.takeIf { descriptor.installed },
                        upstreamCommit = descriptor.upstreamCommit,
                        jniContract = descriptor.jniContract,
                        installedAt = if (descriptor.installed) System.currentTimeMillis() else null,
                        sizeBytes = descriptor.sizeBytes,
                        state = if (descriptor.installed) RuntimeState.INSTALLED else RuntimeState.NOT_INSTALLED,
                        architectureCount = descriptor.architectures.size,
                        backendsJson = descriptor.backends.joinToString(","),
                    )
                },
            )
        }
    }
}
