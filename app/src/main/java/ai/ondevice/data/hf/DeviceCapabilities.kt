package ai.ondevice.data.hf

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.Environment
import android.os.PowerManager
import android.os.StatFs

/** What this device can actually do — measured, not assumed. */
class DeviceCapabilities(private val context: Context) {

    private val activityManager: ActivityManager
        get() = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

    private val powerManager: PowerManager
        get() = context.getSystemService(Context.POWER_SERVICE) as PowerManager

    /** Total physical RAM. */
    val totalRamBytes: Long
        get() = ActivityManager.MemoryInfo().also { activityManager.getMemoryInfo(it) }.totalMem

    /** What is free right now — the number the fit estimate compares against. */
    val availableRamBytes: Long
        get() = ActivityManager.MemoryInfo().also { activityManager.getMemoryInfo(it) }.availMem

    val isLowMemory: Boolean
        get() = ActivityManager.MemoryInfo().also { activityManager.getMemoryInfo(it) }.lowMemory

    val freeStorageBytes: Long
        get() = runCatching {
            val dir = context.getExternalFilesDir(null) ?: Environment.getDataDirectory()
            StatFs(dir.absolutePath).availableBytes
        }.getOrDefault(0L)

    val totalCores: Int get() = Runtime.getRuntime().availableProcessors()

    val performanceCores: Int
        get() = runCatching {
            val freqs = (0 until totalCores).mapNotNull { cpu ->
                java.io.File("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq")
                    .takeIf { it.canRead() }
                    ?.readText()
                    ?.trim()
                    ?.toLongOrNull()
            }
            if (freqs.isEmpty()) return@runCatching fallbackPerformanceCores()
            val max = freqs.max()
            // Anything within 15% of the fastest core counts as a big core; that
            // groups prime + performance clusters together and excludes little.
            freqs.count { it >= max * 0.85 }.coerceAtLeast(1)
        }.getOrElse { fallbackPerformanceCores() }

    private fun fallbackPerformanceCores() = (totalCores / 2).coerceIn(1, 8)

    /** How many threads an inference run should take: every core but one. */
    val inferenceThreads: Int get() = (totalCores - 1).coerceAtLeast(1)

    /** Read, but no longer acted on: the app's thermal policy is gone because three of its four settings could not do what they claimed. */
    val thermalStatus: Int
        get() = runCatching { powerManager.currentThermalStatus }.getOrDefault(PowerManager.THERMAL_STATUS_NONE)

    val thermalLabel: String
        get() = when (thermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "none"
            PowerManager.THERMAL_STATUS_LIGHT -> "light"
            PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
            PowerManager.THERMAL_STATUS_SEVERE -> "severe"
            PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
            else -> "unknown"
        }

    val batteryPercent: Int
        get() = runCatching {
            val bm = context.getSystemService(Context.BATTERY_SERVICE) as android.os.BatteryManager
            bm.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)
        }.getOrDefault(100)

    val socModel: String
        get() = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL.takeIf { it.isNotBlank() && it != Build.UNKNOWN } ?: Build.HARDWARE
        } else {
            Build.HARDWARE
        }
}
