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

    /** The lowest memory reading of the run, and the bottom of the graph's RAM axis. */
    val floorRssMb: Int get() = rssMb.minOrNull() ?: 0
    val minAvailMb: Int get() = availMb.minOrNull() ?: 0

    /** How much the run added on top of what was already resident. */
    val deltaRssMb: Int get() = (peakRssMb - baselineRssMb).coerceAtLeast(0)

    /** The latest reading, as against the highest one. */
    val currentRssMb: Int get() = rssMb.lastOrNull() ?: 0

    /**
     * Where the run stands against where it started — signed, unlike
     * [deltaRssMb], which is a high-water mark and cannot be negative.
     *
     * Negative is the interesting direction and it happens often: the weights
     * are memory-mapped, so once a prompt encoder has done its work the kernel
     * takes its pages back and the process is holding less part-way through a
     * run than it was at the start of one.
     */
    val netRssMb: Int get() = currentRssMb - baselineRssMb

    /**
     * What the process is holding, in one line — measured, not added up.
     *
     * The app's only honest memory total. Summing the model files overstates
     * it, because they are memory-mapped and the kernel takes back whatever is
     * no longer being read: a diffusion bundle whose files come to 10.7 GB was
     * measured sampling at 3.94 GB once its prompt encoder had finished. And
     * the runtime's own buffer figures cannot be summed into a total either —
     * it announces every allocation and never a free, so any running total
     * built from them could only climb.
     *
     * Peak is worth saying beside the current figure because the peak is what
     * decides whether a run fits, and it usually happens somewhere in the
     * middle where nobody is looking.
     */
    val heldSummary: String
        get() = buildString {
            append("Holding ${gb(currentRssMb)}")
            if (peakRssMb > currentRssMb) append(" · Peak ${gb(peakRssMb)}")
            // Which way it is going, named rather than signed: a minus in front
            // of a memory figure reads as an error more often than as a fall.
            //
            // A rise is only worth saying when there was something to rise
            // from. Said unconditionally it produced "Holding 6.63 GB · Adding
            // 6.40 GB" — two large numbers a fifth of a gigabyte apart, which
            // is the same fact twice and reads as the two disagreeing. When the
            // app was idle before the run, "Holding" has already said it.
            //
            // A fall always earns its place. It is the one thing about this
            // screen that surprises people: the process holding less part-way
            // through a run than at the start of it, because the weights are
            // memory-mapped and the encoder's pages went back to the kernel.
            when {
                netRssMb < 0 -> append(" · Removing ${gb(-netRssMb)}")
                netRssMb > 0 && baselineRssMb * 5 >= currentRssMb ->
                    append(" · Adding ${gb(netRssMb)}")
            }
        }

    private fun gb(mb: Int): String =
        if (mb >= 1024) String.format("%.2f GB", mb / 1024.0) else "$mb MB"

    val peakRssBytes: Long get() = peakRssMb.toLong() * BYTES_PER_MB

    fun toJson(): String = JSON.encodeToString(serializer(), this)

    /** The same run at half the resolution. */
    fun halved(): ResourceTrace = copy(
        intervalMillis = intervalMillis * 2,
        cpuPercent = cpuPercent.mergePairs { a, b -> (a + b) / 2 },
        gpuPercent = gpuPercent.mergePairs { a, b -> (a + b) / 2 },
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

            if (cpu.size > MAX_TRACE_POINTS) decimate()
        }

        private fun decimate() {
            val halved = snapshot().halved()
            cpu.replaceWith(halved.cpuPercent)
            rss.replaceWith(halved.rssMb)
            avail.replaceWith(halved.availMb)
            gpu.replaceWith(halved.gpuPercent)
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

        private val WHITESPACE = Regex("""\s+""")
    }
}
