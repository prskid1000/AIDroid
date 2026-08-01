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
