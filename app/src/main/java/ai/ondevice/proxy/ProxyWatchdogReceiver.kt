package ai.ondevice.proxy

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Where the watchdog alarm lands, and where an app update is noticed.
 *
 * A receiver rather than a service for the reason [ProxyWatchdog] exists: this
 * has to run in a process that may have just been created for it, and it has to
 * be allowed to start a foreground service from there. Both of the ways in here
 * carry that exemption — an exact alarm, and `ACTION_MY_PACKAGE_REPLACED`.
 *
 * **The package-replaced filter is not decoration.** Installing a new version
 * stops the old process and every service in it, and nothing brings the proxy
 * back: sideloading an update meant the API was down until somebody opened the
 * app, which on this project is the same afternoon the update was built. It is
 * the same gap `BootSweepReceiver` closes for a reboot, from the other
 * direction.
 *
 * `goAsync` because reading the stored configuration is a DataStore read and
 * therefore suspending, and a receiver that returns before its work is done is
 * a process the system is free to kill mid-way.
 */
@AndroidEntryPoint
class ProxyWatchdogReceiver : BroadcastReceiver() {

    @Inject lateinit var watchdog: ProxyWatchdog

    override fun onReceive(context: Context, intent: Intent) {
        val pending = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                watchdog.check()
            } finally {
                pending.finish()
            }
        }
    }
}
