package ai.ondevice.data.download

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import ai.ondevice.data.ModelStorage
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * SPEC §3.4 — "Cleanup: remove partial files on cancel; orphan sweep on boot."
 *
 * Downloads resume across reboot, so the reboot is exactly when a `.part` with
 * no surviving job row becomes garbage worth reclaiming.
 */
@AndroidEntryPoint
class BootSweepReceiver : BroadcastReceiver() {

    @Inject lateinit var downloader: Downloader

    @Inject lateinit var storage: ModelStorage

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED) return
        val pending = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                downloader.sweepOrphans(storage.modelsDir())
            } finally {
                pending.finish()
            }
        }
    }
}
