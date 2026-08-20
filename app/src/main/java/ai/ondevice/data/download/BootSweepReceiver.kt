package ai.ondevice.data.download

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import ai.ondevice.data.ModelStorage
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject

@AndroidEntryPoint
class BootSweepReceiver : BroadcastReceiver() {

    @Inject lateinit var downloader: Downloader

    @Inject lateinit var storage: ModelStorage

    @Inject lateinit var prefs: ai.ondevice.data.prefs.AppPrefs

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED) return
        val pending = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                downloader.sweepOrphans(storage.modelsDir())
                restartProxy(context)
            } finally {
                pending.finish()
            }
        }
    }

    /**
     * Bring the HTTP surface back up after a reboot.
     *
     * Without this the server is up only once somebody has opened the app since
     * the phone last restarted -- a server that works until the battery runs
     * out and then silently does not, with nothing on the far end to say why.
     *
     * Safe here and not elsewhere: BOOT_COMPLETED is itself exempt from the
     * rule forbidding a foreground-service start from the background, which is
     * the same fact that decides how the workflow scheduler re-arms.
     *
     * Only when it was already switched on. A reboot is not consent.
     */
    private suspend fun restartProxy(context: Context) {
        val enabled = runCatching {
            ai.ondevice.proxy.ProxyConfig(
                ai.ondevice.proxy.ProxyDocument.parse(prefs.proxyDocument.first()),
            ).enabled
        }.getOrDefault(false)
        if (!enabled) return
        runCatching {
            context.startForegroundService(
                Intent(context, ai.ondevice.engine.InferenceService::class.java),
            )
        }
    }
}
