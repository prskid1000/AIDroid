package ai.ondevice.engine

import ai.ondevice.core.BackendId
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

/**
 * What a prediction cost, sampled while it ran.
 *
 * The app has always reported *throughput* — tokens per second, seconds per
 * step, an audio-seconds-per-wall-second realtime factor — and never what the
 * device paid to reach it. On a phone those are the questions that decide
 * whether a model is usable at all: did it saturate the cores, and how close did
 * it come to the memory ceiling. Both are recorded here, per run, so the answer
 * survives the run instead of being a thing you had to be watching for.
 *
 * Three parallel arrays rather than a list of point objects, and megabytes and
 * whole percent rather than bytes and floats. A trace is stored as JSON in every
 * row, so its size is a per-artifact cost paid forever; this shape is about four
 * times smaller than the obvious one and loses nothing anybody reads.
 */
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
    /**
     * GPU busy, 0..100, or empty when this device does not expose it.
     *
     * Empty and zero are different answers and are kept different: a run with
     * no GPU series means "nobody could tell you", a run of zeroes means "the
     * GPU sat idle". Defaulted so traces written before this existed still
     * parse — they are the first case, and correctly so.
     *
     * Device-wide, not this process': the counter is the kernel's, and it
     * counts every client of the GPU including the compositor drawing this
     * screen. On a phone running one heavy job that is close enough to be
     * useful and dishonest to present as exact, which is why the caption says
     * "device" and not "app".
     */
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

    /**
     * The lowest memory reading of the run, and the bottom of the graph's RAM
     * axis. Not [baselineRssMb]: the baseline is taken before the run starts and
     * a run that *frees* memory would draw below the floor.
     */
    val floorRssMb: Int get() = rssMb.minOrNull() ?: 0
    val minAvailMb: Int get() = availMb.minOrNull() ?: 0

    /** How much the run added on top of what was already resident. */
    val deltaRssMb: Int get() = (peakRssMb - baselineRssMb).coerceAtLeast(0)

    val peakRssBytes: Long get() = peakRssMb.toLong() * BYTES_PER_MB

    fun toJson(): String = JSON.encodeToString(serializer(), this)

    /**
     * The same run at half the resolution.
     *
     * Adjacent samples merge and the interval doubles, so the window the trace
     * covers is unchanged — only how finely it is described. Which of the three
     * series takes which reduction is the part that matters: CPU averages,
     * because a percentage over a longer window *is* a mean; RSS takes the
     * maximum and free RAM the minimum, because those two are read for their
     * worst moment. Averaging a 4 GB spike between two 1 GB samples would report
     * 2.5 GB and erase the near-miss on the memory ceiling, which is the single
     * most useful thing a trace can record.
     *
     * A trailing unpaired sample is carried through rather than dropped, so the
     * end of a run never disappears.
     */
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

/**
 * The most points a trace will ever hold.
 *
 * Without a cap, a long image batch or a twenty-minute transcription writes
 * thousands of points into a database row that is read on a list screen. With a
 * naive cap — stop sampling at N — the graph would silently describe only the
 * first ninety seconds of a run and look complete. So the trace halves its own
 * resolution instead: adjacent samples merge and the interval doubles, which
 * keeps the whole run in view at a bounded size.
 */
const val MAX_TRACE_POINTS = 180

/**
 * File a finished run.
 *
 * One definition rather than a private copy in each of the four view models that
 * calls it: the summary columns are derived from the trace, and four places
 * deriving them independently is four chances for a peak on a list screen to
 * disagree with the peak on the graph beside it.
 *
 * An empty trace is dropped. A run shorter than a single sample tick has nothing
 * to say, and a row of zeroes would read as "this used no CPU" rather than "this
 * was too quick to measure".
 */
suspend fun PredictionRunDao.record(
    kind: PredictionKind,
    artifactId: String,
    modelId: String?,
    backend: BackendId?,
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
            backend = backend,
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

/**
 * Samples the process while something is generating.
 *
 * Everything in this app runs in one process — there is no `android:process` on
 * any component in the manifest — so llama.cpp, sd.cpp, whisper and ONNX Runtime
 * all land in the same counters and one recorder covers all four.
 */
class ResourceRecorder(private val capabilities: DeviceCapabilities) {

    /**
     * A run in progress.
     *
     * [stop] deliberately does not suspend. Chat's teardown runs inside a
     * `withContext(NonCancellable)` after a Stop press and Image's runs in a
     * `finally` right after cancelling the native loop; a suspending stop in
     * either place would be skipped exactly when the run was most interesting.
     */
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

        /**
         * Halve the resolution in place, by the rules [ResourceTrace.halved]
         * states — one definition, so what the recorder does and what the tests
         * check cannot drift apart.
         */
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
        /**
         * Resident set size, from this process' own `/proc` entry.
         *
         * RSS rather than Java heap or native heap on purpose: llama.cpp mmaps
         * the GGUF, so the gigabytes that actually decide whether a model runs
         * appear in neither heap figure. `Debug.getPss()` would be more precise
         * about shared pages and walks `smaps` to get there, which is tens of
         * milliseconds — far too expensive twice a second. This is one short
         * read of a single line.
         *
         * Field 2 of `statm` is resident pages.
         */
        fun readRssBytes(): Long = runCatching {
            val fields = java.io.File("/proc/self/statm").readText().trim().split(' ')
            fields[1].toLong() * PAGE_SIZE
        }.getOrDefault(0L)

        val PAGE_SIZE: Long = runCatching { android.system.Os.sysconf(android.system.OsConstants._SC_PAGESIZE) }
            .getOrDefault(4096L)
            .takeIf { it > 0 } ?: 4096L

        /**
         * Adreno's own busy counter, in the only place an app can read it.
         *
         * `gpubusy` holds two microsecond figures — busy and total — accumulated
         * *since the last read*, and reading resets them. That makes it exactly
         * a utilisation over the sampling interval, and also means two readers
         * would steal each other's numbers; there is one sampler in this process
         * and it is the only thing here that opens the file.
         *
         * `/sys/class/kgsl` is unreadable to `adb shell` on this device and
         * readable to the app, which is the reverse of the usual arrangement and
         * worth writing down — the check that matters is the one run as the app.
         *
         * Null on a device with no Adreno, no counter, or no permission: the
         * series stays empty and the graph says the GPU was not measured rather
         * than drawing a flat zero.
         */
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
