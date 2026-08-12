package ai.ondevice.engine

import ai.ondevice.core.PredictionKind
import ai.ondevice.core.SparseParams
import ai.ondevice.data.db.PredictionRunDao
import ai.ondevice.data.db.PredictionRunEntity
import ai.ondevice.data.hf.DeviceCapabilities
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** What a prediction cost, sampled while it ran. */
@Serializable
data class ResourceTrace(
    /** Milliseconds between samples. Doubles each time the trace is decimated. */
    val intervalMillis: Int,
    val elapsedMillis: Long,
    /** Process CPU as a percentage of *one* core-second per wall-second, 0..100. */
    val cpuPercent: List<Int>,
    /** Resident set size of this process, in MB — the mmap'd model included. */
    val rssMb: List<Int>,
    /** What the whole device had free, in MB. */
    val availMb: List<Int>,
    /** GPU busy, 0..100, or empty when this device does not expose it. */
    val gpuPercent: List<Int> = emptyList(),
    /**
     * Mean CPU clock across the cores that reported one, in MHz. Empty when
     * `/sys` will not say.
     *
     * **The number that separates "the model is slow" from "the platform never
     * clocked up", and it was the one thing this trace could not answer.** On
     * this phone a run has been observed sitting at 1713 MHz against a
     * 3321/3801 MHz ceiling — under 60% of the clock that was available — with
     * CPU busy reading high the whole time, because busy is a *time* measure
     * and says nothing about the rate the work was done at. Two runs can report
     * the same 95% and differ twofold in throughput.
     *
     * A mean rather than every core: the question here is what the run got, not
     * which core got it. `cpuPercent` is already normalised across cores for the
     * same reason.
     *
     * Defaulted so every trace already in the database still parses.
     */
    val clockMhz: List<Int> = emptyList(),
    /** RSS before the run started, so the model's own footprint is legible. */
    val baselineRssMb: Int,
    val totalRamMb: Int,
    val cores: Int,
) {
    val isEmpty: Boolean get() = cpuPercent.isEmpty()

    val peakCpuPercent: Int get() = cpuPercent.maxOrNull() ?: 0
    val meanCpuPercent: Int get() = if (cpuPercent.isEmpty()) 0 else cpuPercent.average().toInt()
    val peakRssMb: Int get() = rssMb.maxOrNull() ?: 0

    /** Null when unmeasured, so a caption can say so rather than print 0%. */
    val peakGpuPercent: Int? get() = gpuPercent.maxOrNull()
    val meanGpuPercent: Int? get() = if (gpuPercent.isEmpty()) null else gpuPercent.average().toInt()

    /** Null when unmeasured, for the same reason the GPU pair is. */
    val peakClockMhz: Int? get() = clockMhz.maxOrNull()
    val meanClockMhz: Int? get() = if (clockMhz.isEmpty()) null else clockMhz.average().toInt()

    /** The lowest memory reading of the run, and the bottom of the graph's RAM axis. */
    val floorRssMb: Int get() = rssMb.minOrNull() ?: 0
    val minAvailMb: Int get() = availMb.minOrNull() ?: 0

    /** How much the run added on top of what was already resident. */
    val deltaRssMb: Int get() = (peakRssMb - baselineRssMb).coerceAtLeast(0)


    val peakRssBytes: Long get() = peakRssMb.toLong() * BYTES_PER_MB

    fun toJson(): String = JSON.encodeToString(serializer(), this)

    /** The same run at half the resolution. */
    fun halved(): ResourceTrace = copy(
        intervalMillis = intervalMillis * 2,
        cpuPercent = cpuPercent.mergePairs { a, b -> (a + b) / 2 },
        gpuPercent = gpuPercent.mergePairs { a, b -> (a + b) / 2 },
        clockMhz = clockMhz.mergePairs { a, b -> (a + b) / 2 },
        rssMb = rssMb.mergePairs { a, b -> maxOf(a, b) },
        availMb = availMb.mergePairs { a, b -> minOf(a, b) },
    )

    companion object {
        const val BYTES_PER_MB = 1024L * 1024L

        val EMPTY = ResourceTrace(
            intervalMillis = SAMPLE_INTERVAL_MILLIS,
            elapsedMillis = 0,
            cpuPercent = emptyList(),
            rssMb = emptyList(),
            availMb = emptyList(),
            gpuPercent = emptyList(),
            clockMhz = emptyList(),
            baselineRssMb = 0,
            totalRamMb = 0,
            cores = 1,
        )

        private val JSON = Json { ignoreUnknownKeys = true; isLenient = true }

        /** A malformed or absent trace is no trace, never a crash. */
        fun parse(json: String?): ResourceTrace? {
            if (json.isNullOrBlank()) return null
            return runCatching { JSON.decodeFromString(serializer(), json) }
                .getOrNull()
                ?.takeIf { !it.isEmpty }
        }
    }
}

private fun List<Int>.mergePairs(combine: (Int, Int) -> Int): List<Int> {
    val merged = ArrayList<Int>(size / 2 + 1)
    var i = 0
    while (i < size) {
        merged += if (i + 1 < size) combine(this[i], this[i + 1]) else this[i]
        i += 2
    }
    return merged
}

/** 500 ms — fine enough to show a load ramp, coarse enough to cost nothing. */
const val SAMPLE_INTERVAL_MILLIS = 500

/** The most points a trace will ever hold. */
const val MAX_TRACE_POINTS = 180

/** File a finished run. */
suspend fun PredictionRunDao.record(
    kind: PredictionKind,
    artifactId: String,
    modelId: String?,
    startedAt: Long,
    trace: ResourceTrace,
    stats: SparseParams = SparseParams.EMPTY,
) {
    if (trace.isEmpty) return
    upsert(
        PredictionRunEntity(
            id = java.util.UUID.randomUUID().toString(),
            kind = kind,
            artifactId = artifactId,
            modelId = modelId,
            startedAt = startedAt,
            elapsedMillis = trace.elapsedMillis,
            peakCpuPercent = trace.peakCpuPercent,
            meanCpuPercent = trace.meanCpuPercent,
            peakRssBytes = trace.peakRssBytes,
            traceJson = trace.toJson(),
            statsJson = stats.toJsonString(),
        ),
    )
}

/** Samples the process while something is generating. */
class ResourceRecorder(private val capabilities: DeviceCapabilities) {

    /** A run in progress. */
    interface Handle {
        /** Updated every tick, for the graph drawn while the run is still going. */
        val live: StateFlow<ResourceTrace>
        fun stop(): ResourceTrace
    }

    fun start(scope: CoroutineScope): Handle = Recording(capabilities, scope)

    private class Recording(
        private val capabilities: DeviceCapabilities,
        scope: CoroutineScope,
    ) : Handle {

        private val cores = capabilities.totalCores.coerceAtLeast(1)
        private val totalRamMb = (capabilities.totalRamBytes / ResourceTrace.BYTES_PER_MB).toInt()
        private val baselineRssMb = (readRssBytes() / ResourceTrace.BYTES_PER_MB).toInt()

        private val startedWallMillis = android.os.SystemClock.elapsedRealtime()
        private var lastWallMillis = startedWallMillis
        private var lastCpuMillis = android.os.Process.getElapsedCpuTime()

        private var intervalMillis = SAMPLE_INTERVAL_MILLIS
        private val cpu = mutableListOf<Int>()
        private val rss = mutableListOf<Int>()
        private val avail = mutableListOf<Int>()
        private val gpu = mutableListOf<Int>()
        private val clock = mutableListOf<Int>()

        private val _live = MutableStateFlow(ResourceTrace.EMPTY)
        override val live: StateFlow<ResourceTrace> = _live.asStateFlow()

        // Default, not Main and not the caller's dispatcher: the point of a
        // sampler is to describe the work, never to compete with it.
        private val job: Job = scope.launch(Dispatchers.Default) {
            while (isActive) {
                delay(SAMPLE_INTERVAL_MILLIS.toLong())
                sample()
                _live.value = snapshot()
            }
        }

        override fun stop(): ResourceTrace {
            job.cancel()
            // One last reading, so a run shorter than a single tick is still
            // described by something rather than by an empty graph.
            if (cpu.isEmpty()) sample()
            return snapshot()
        }

        @Synchronized
        private fun sample() {
            val wall = android.os.SystemClock.elapsedRealtime()
            val cpuMillis = android.os.Process.getElapsedCpuTime()
            val wallDelta = (wall - lastWallMillis).coerceAtLeast(1)
            val cpuDelta = (cpuMillis - lastCpuMillis).coerceAtLeast(0)
            lastWallMillis = wall
            lastCpuMillis = cpuMillis

            // Divided by the core count, so 100% means "every core busy" rather
            // than the 800% a raw process figure would report on an octa-core.
            cpu += (cpuDelta * 100 / wallDelta / cores).toInt().coerceIn(0, 100)
            rss += (readRssBytes() / ResourceTrace.BYTES_PER_MB).toInt()
            avail += (capabilities.availableRamBytes / ResourceTrace.BYTES_PER_MB).toInt()
            readGpuBusyPercent()?.let { gpu += it }
            readMeanClockMhz(cores)?.let { clock += it }

            if (cpu.size > MAX_TRACE_POINTS) decimate()
        }

        private fun decimate() {
            val halved = snapshot().halved()
            cpu.replaceWith(halved.cpuPercent)
            rss.replaceWith(halved.rssMb)
            avail.replaceWith(halved.availMb)
            gpu.replaceWith(halved.gpuPercent)
            clock.replaceWith(halved.clockMhz)
            intervalMillis = halved.intervalMillis
        }

        private fun MutableList<Int>.replaceWith(values: List<Int>) {
            clear()
            addAll(values)
        }

        @Synchronized
        private fun snapshot() = ResourceTrace(
            intervalMillis = intervalMillis,
            elapsedMillis = android.os.SystemClock.elapsedRealtime() - startedWallMillis,
            cpuPercent = cpu.toList(),
            rssMb = rss.toList(),
            availMb = avail.toList(),
            gpuPercent = gpu.toList(),
            clockMhz = clock.toList(),
            baselineRssMb = baselineRssMb,
            totalRamMb = totalRamMb,
            cores = cores,
        )
    }

    private companion object {
        /** Resident set size, from this process' own `/proc` entry. */
        fun readRssBytes(): Long = runCatching {
            val fields = java.io.File("/proc/self/statm").readText().trim().split(' ')
            fields[1].toLong() * PAGE_SIZE
        }.getOrDefault(0L)

        val PAGE_SIZE: Long = runCatching { android.system.Os.sysconf(android.system.OsConstants._SC_PAGESIZE) }
            .getOrDefault(4096L)
            .takeIf { it > 0 } ?: 4096L

        /** Adreno's own busy counter, in the only place an app can read it. */
        fun readGpuBusyPercent(): Int? = runCatching {
            val parts = java.io.File(GPU_BUSY_PATH).readText().trim().split(WHITESPACE)
            val busy = parts[0].toLong()
            val total = parts[1].toLong()
            if (total <= 0L) null else (busy * 100 / total).toInt().coerceIn(0, 100)
        }.getOrNull()

        const val GPU_BUSY_PATH = "/sys/class/kgsl/kgsl-3d0/gpubusy"

        /**
         * Mean current clock across the cores that will say, in MHz.
         *
         * `scaling_cur_freq` per cpu, in kHz. Cores that are offline or that
         * deny the read are skipped rather than counted as zero: a parked core
         * is not running at 0 MHz, it is not running, and averaging it in would
         * report a throttle that never happened. Null when none of them answered,
         * which is a device that does not expose cpufreq to apps at all.
         *
         * Read every tick, and cheap enough to be: eight small `/sys` reads at
         * 2 Hz. `scaling_cur_freq` is the governor's *requested* rate rather than
         * a hardware measurement — `cpuinfo_cur_freq` is the measured one and is
         * root-only on Android — so this tracks what the platform decided to give
         * the run, which is exactly the question being asked of it.
         */
        fun readMeanClockMhz(cores: Int): Int? {
            var total = 0L
            var counted = 0
            for (cpu in 0 until cores) {
                val khz = runCatching {
                    java.io.File("/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq")
                        .readText().trim().toLong()
                }.getOrNull() ?: continue
                if (khz <= 0L) continue
                total += khz
                counted++
            }
            return if (counted == 0) null else (total / counted / 1000L).toInt()
        }

        private val WHITESPACE = Regex("""\s+""")
    }
}

    /**
     * Resident memory split into the half that can be given back and the
     * half that cannot.
     *
     * `total params memory size` is what the runtime *mapped*, not what is
     * in RAM — `enable_mmap` defaults on, so a 10.46 GB model is a promise
     * about address space and the resident figure is usually far below it.
     * Reporting the mapped number as memory overstates it, and reporting
     * one RSS figure hides the distinction that decides whether this
     * process survives pressure.
     *
     * `Anonymous` is everything allocated rather than mapped from a file:
     * compute buffers, the VAE's feature cache, the heap. It cannot be
     * evicted, so it is the number the kernel's killer effectively reads.
     * The remainder of Rss is weight pages faulted in from the checkpoint —
     * genuinely in memory, and genuinely reclaimable, because the file they
     * came from is still on disk.
     */
    fun residentMemorySplit(): Pair<Long, Long> = runCatching {
        var rss = 0L
        var anon = 0L
        java.io.File("/proc/self/smaps_rollup").forEachLine { line ->
            when {
                line.startsWith("Rss:") -> rss = kilobytesIn(line)
                line.startsWith("Anonymous:") -> anon = kilobytesIn(line)
            }
        }
        // Clamped rather than trusted: the two fields are sampled as the
        // kernel walks the maps, so a growing process can report an
        // Anonymous larger than the Rss read a moment earlier.
        val allocated = anon.coerceAtMost(rss)
        allocated to (rss - allocated)
    }.getOrDefault(0L to 0L)

    private fun kilobytesIn(line: String): Long =
        line.split(Regex("\\s+")).getOrNull(1)?.toLongOrNull()?.times(1024L) ?: 0L
