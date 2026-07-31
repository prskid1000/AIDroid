package ai.ondevice.data.hf

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.Environment
import android.os.PowerManager
import android.os.StatFs
import ai.ondevice.core.BackendId

/**
 * What this device can actually do — measured, not assumed.
 *
 * SPEC §8.2 is blunt about the principle: "Do not assume backend performance on
 * this hardware — measure it." The numbers here are the ones that don't need a
 * benchmark (RAM, storage, cores, thermal state); throughput comes from
 * [ai.ondevice.engine.Benchmarker].
 */
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

    /**
     * Default thread count is the *performance*-core count, not total cores
     * (SPEC §8.1) — saturating the little cores costs more in scheduling than
     * it returns in throughput.
     *
     * Android exposes no direct performance-core count, so this reads the
     * per-core max frequency from sysfs and counts the cores in the top
     * frequency band, falling back to a conservative fraction of the total.
     */
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

    /**
     * How many threads an inference run should take: every core but one.
     *
     * Stated once, here, because until now every engine had its own answer and
     * none of them was this device's. llama.cpp took
     * `hardware_concurrency() / 2` in its JNI layer — half the machine, on a
     * constant nothing could see or change. sd.cpp took the physical core count.
     * Kokoro and OmniVoice took a `threads` parameter that every caller left at
     * zero, so ONNX Runtime chose for itself, and the recorded traces show that
     * choice using under a quarter of an eight-core phone. Three engines, three
     * policies, no way to tell what any of them did.
     *
     * One core is left free rather than all of them being claimed. Inference is
     * not the only thing running: the UI thread still has to draw the progress
     * it is being asked to show, and a device with nothing left to schedule the
     * compositor on reads as hung rather than as busy. On a single-core device
     * the arithmetic would hand back zero, so it is floored at one.
     *
     * This supersedes [performanceCores] as the *inference* thread count. That
     * property survives because the Settings screen reports it, and because the
     * distinction it draws is real — it is simply not the choice being made here.
     */
    val inferenceThreads: Int get() = (totalCores - 1).coerceAtLeast(1)

    /**
     * Read, but no longer acted on: the app's thermal policy is gone because
     * three of its four settings could not do what they claimed. Kept because
     * the benchmark's own numbers mean less from a device that was already hot,
     * so a reading is worth having even when nothing throttles on it.
     */
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

    /**
     * Backends the build could plausibly use. Which one is actually *fastest*
     * is a measurement, not a guess — §8.2 — so this only reports availability.
     *
     * Hexagon is v2 (its SDK is Qualcomm-account-gated, §15), so it is reported
     * only when a Hexagon runtime bundle is installed; that check lives in the
     * runtime registry rather than here.
     */
    fun candidateBackends(hasOpenCl: Boolean, hasHexagon: Boolean): List<BackendId> = buildList {
        if (hasOpenCl) add(BackendId.OPENCL)
        if (hasHexagon) add(BackendId.HEXAGON)
        add(BackendId.CPU)
    }

    companion object {
        /**
         * SPEC §3.3 / §15: a single Hexagon HTP session tops out near 3.5 GB.
         * Past that the model must be layer-split across HTP0..HTP3 or fall
         * back to OpenCL — silently failing is not an option.
         */
        const val HEXAGON_SESSION_CAP_BYTES = 3_758_096_384L // 3.5 GiB
    }
}
