package ai.ondevice.engine

import android.content.Context
import android.os.Build
import android.os.PerformanceHintManager
import android.os.Process

/**
 * Tells the platform how long a unit of inference is supposed to take (ADPF).
 *
 * **What this is for.** Nothing in this app ever told Android that a run is
 * performance-sensitive, so the scheduler had no way to tell a decode loop from
 * a text field. EAS reads a thread that is never blocked and never late as one
 * that is keeping up, and keeps its clocks where they are: on this phone
 * (SM8845, Android 16) a sustained native arm64 workload was measured holding
 * **1713–1977 MHz against a 3321/3801 MHz ceiling** — under 60% of the clock
 * that was sitting there. That measurement is not of this app; it is of the
 * hardware, and it is the reason this class exists rather than evidence that it
 * works. See the closing note.
 *
 * `PerformanceHintManager` is the one lever an unprivileged app has for this.
 * The two obvious alternatives are both closed:
 *
 * - **uclamp.** `/proc/sys/kernel/sched_util_clamp_min` is there, but raising a
 *   task's `uclamp_min` above the system default needs `CAP_SYS_NICE`, which we
 *   do not have and cannot ask for.
 * - **CPU affinity.** Pinning to the prime cores was measured on this same phone
 *   at **19% slower** than letting the scheduler place threads. A mask *removes*
 *   runqueues; ggml's pool is barrier-synchronised, so the contention costs more
 *   than the extra megahertz return. Do not add a `taskset` here.
 *
 * ADPF takes nothing away. It names threads and a deadline, reports how long the
 * work took, and leaves placement and frequency to the platform — so the failure
 * mode that produced the 19% cannot happen. The platform is listening on this
 * device: `performance_hint: [android.os.IHintManager]` is a registered service
 * and `debug.sf.enable_adpf_cpu_hint` is `true`.
 *
 * ## The target is learned, never invented
 *
 * A hint session needs a deadline, and inference has no natural one — a token is
 * not a frame and nothing is dropped if it is late. Declaring an arbitrary
 * "should take 30 ms" would be a number this app made up, which is the kind of
 * claim [SPEC §1.2] exists to forbid.
 *
 * So the target is **the fastest this device has been observed doing one unit of
 * this exact work, over a sliding window of the last [WINDOW] units**. The hint
 * then says: *hold the pace you have already shown you can hold.* It never asks
 * for a speed the hardware has not demonstrated, and it does produce a real
 * overrun signal — the thing the platform acts on — whenever a unit falls behind
 * that pace, which is what DVFS sag and thermal drift look like.
 *
 * The window is short on purpose. Work that legitimately gets more expensive as
 * a run proceeds — a decode with a growing KV cache is the obvious one — would
 * overrun an all-time best forever, and a session permanently pinned to "boost"
 * is a session that has stopped saying anything. Sixteen units is long enough to
 * survive a single slow one and short enough to follow a cost curve.
 *
 * A "unit" is whatever the caller is counting: one decoded token, one sampler
 * step, one phoneme token, one audio sample. Passing the count lets a workload
 * with uneven pieces (a TTS chunk is as long as its sentence) be compared
 * against a per-unit rate rather than against a flat duration.
 *
 * ## What is deliberately not covered
 *
 * **ggml's worker threads.** A session names threads, and llama.cpp decodes on a
 * pool of them — sixteen on this phone. The pool is created inside libggml with
 * no thread names and no API that hands the tids back, so from Kotlin there is
 * nothing to name. What a session does cover is the thread that called in, which
 * is ggml worker 0: a full participant that runs the graph in barrier lockstep
 * with the other fifteen, so its duration *is* the graph's duration even though
 * the hint reaches one runqueue of sixteen. Confirmed on the device — the armed
 * line reads `1 thread(s)` for a decode with `threads=16`. Widening it means
 * exporting the pool's tids through the JNI layer, which is a real change and
 * has not been made.
 *
 * **Work with no unit boundary the app can see.** llama.cpp's prompt eval,
 * whisper's decode of a whole take and sd.cpp's VAE pass are each a single
 * uninterruptible JNI call. A session that reports once, after the call it was
 * meant to steer has already returned, steers nothing. Where the same shape of
 * call recurs — whisper transcribes take after take — [carryOver] keeps the rate
 * across calls so a later one could be hinted; where it does not, no session is
 * opened at all.
 *
 * **Work that is too slow to have a deadline.** See [MAX_UNIT_NANOS]. Every
 * caller here is wired the same way and each one decides at runtime, from its own
 * measured cost, whether it is in range — nothing is hardcoded about which
 * runtime is fast. On this phone the expectation is that sd.cpp's sampler steps
 * and whisper's whole-take decodes are *never* in range and will log a refusal
 * with their measured cost rather than hint, which is the point of measuring
 * instead of asserting.
 *
 * **GPU durations.** `reportActualWorkDuration` takes a GPU figure from Android
 * 15, and nothing here measures one. Deriving it from the CPU time and handing
 * that to a system that will clock the GPU from it is worse than reporting none.
 *
 * ## Unmeasured
 *
 * No before/after exists for this app. The argument above is from the clock
 * figures and from what the API is documented to do, not from a run of this
 * code. The instruments to settle it are already here — `ResourceTrace`'s
 * `cpuPercent`, and the `t/s` each engine already logs — so quote those, not
 * this comment.
 */
class CpuHints private constructor(
    private val tag: String,
    private val manager: PerformanceHintManager?,
    private val window: ArrayDeque<Long>,
    tids: IntArray,
) : AutoCloseable {

    private val threads = LinkedHashSet<Int>().apply { tids.forEach(::add) }

    private var session: PerformanceHintManager.Session? = null
    private var targetNanos = 0L
    private var reports = 0
    private var declined = false
    private var saidThreadMoved = false

    /**
     * Run one unit of work on this thread, time it, and report what it cost.
     *
     * The session is armed *before* [block] so the platform has the deadline in
     * hand while the work runs; the first unit of a run is therefore unhinted
     * unless a [carryOver] rate was already known, because until something has
     * been timed there is no honest target to declare.
     */
    fun <T> unit(units: Long = 1L, block: () -> T): T {
        follow()
        arm(units)
        val hinted = session != null
        val started = System.nanoTime()
        val result = block()
        val took = System.nanoTime() - started
        learn(units, took)
        if (hinted) report(took)
        return result
    }

    /**
     * The same, for work this thread did not run.
     *
     * sd.cpp's sampler is one blocking call with a step counter read from
     * outside it, so the durations come from watching that counter move rather
     * than from bracketing the work. They are quantised by the poll interval,
     * which is worth remembering before reading much into a single figure.
     */
    fun observed(units: Long, nanos: Long) {
        val hinted = session != null
        learn(units, nanos)
        if (hinted) report(nanos)
        // Armed after, not before: the caller learns a unit is over by seeing
        // the next one begin, so this is the arming for that next one.
        arm(units)
    }

    override fun close() {
        val open = session ?: return
        session = null
        runCatching { open.close() }
        EngineLog.i(tag, "cpu hints closed after $reports report(s), last target ${millis(targetNanos)}")
    }

    // — the session —

    /** Open or retarget, once there is a measured rate to target. */
    private fun arm(units: Long) {
        val hinter = manager
        if (hinter == null || declined) return
        val perUnit = rate() ?: return

        // Out of range, so nothing is said at all — see [MAX_UNIT_NANOS].
        if (perUnit > MAX_UNIT_NANOS) {
            declined = true
            EngineLog.i(
                tag,
                "a unit costs ${millis(perUnit)}, which is not a deadline this API is for; " +
                    "not hinting",
            )
            // A run that was frame-scale and has drifted out of range gives the
            // session back rather than holding one with a target nobody can act on.
            session?.let { runCatching { it.close() } }
            session = null
            return
        }

        val target = (perUnit * units.coerceAtLeast(1L)).coerceAtLeast(FLOOR_NANOS)

        val existing = session
        if (existing == null) {
            val opened = runCatching {
                hinter.createHintSession(threads.toIntArray(), target)
            }.getOrNull()
            if (opened == null) {
                // Documented behaviour, not a fault: a device whose power HAL
                // has no hint support answers null, and inference then runs
                // exactly as it did before this class existed.
                declined = true
                EngineLog.i(tag, "the platform declined a hint session; this run is not hinted")
                return
            }
            session = opened
            targetNanos = target
            EngineLog.i(
                tag,
                "cpu hints armed on ${threads.size} thread(s), " +
                    "target ${millis(target)} for $units unit(s)",
            )
            return
        }

        // Hysteresis, because the target moves every time the window slides and
        // a binder call per token to say "a shade faster" is noise on both ends.
        if (targetNanos <= 0L || kotlin.math.abs(target - targetNanos) * 8 > targetNanos) {
            runCatching { existing.updateTargetWorkDuration(target) }
            targetNanos = target
        }
    }

    private fun report(nanos: Long) {
        if (nanos <= 0L) return
        val open = session ?: return
        reports++
        runCatching { open.reportActualWorkDuration(nanos) }
    }

    /**
     * Keep the session pointed at the thread actually doing the work.
     *
     * A flow that suspends between units — `flowOn(Dispatchers.Default)` does,
     * whenever its buffer fills — can resume on a different pool thread, and a
     * session left hinting the thread that ran the *first* token is hinting
     * nothing. `setThreads` is API 34, so below that the drift is recorded and
     * lived with rather than papered over.
     *
     * Threads accumulate rather than replace, and are capped: a pool that
     * bounces between two workers should end up with both covered, and a hint
     * spread over everything is not a hint.
     */
    private fun follow() {
        val tid = Process.myTid()
        if (tid in threads) return

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            if (!saidThreadMoved) {
                saidThreadMoved = true
                EngineLog.i(tag, "the work moved to tid $tid; setThreads needs API 34, so hints stay put")
            }
            return
        }
        if (threads.size >= MAX_THREADS) return

        threads += tid
        val open = session ?: return
        runCatching { open.setThreads(threads.toIntArray()) }
            .onFailure { threads -= tid }
    }

    // — the learned rate —

    private fun learn(units: Long, nanos: Long) {
        if (units <= 0L || nanos <= 0L) return
        val perUnit = nanos / units
        if (perUnit <= 0L) return
        synchronized(window) {
            window.addLast(perUnit)
            while (window.size > WINDOW) window.removeFirst()
        }
    }

    /**
     * The fastest unit in the window — see the class comment on why the fastest.
     *
     * Null until [MIN_SAMPLES] of them exist, because the first unit of a run is
     * routinely the slowest thing that will happen and a target built from it
     * alone is nonsense. Measured on the device: the first `nativeNextToken` of a
     * Qwen3.5-9B Q4_K_M turn at a 262144 context took **12.87 s** against tenths
     * of a second for the tokens after it, and armed the session at a 12866 ms
     * deadline for one token. A target that generous tells the platform it has
     * twelve seconds of slack, which is the opposite of the thing being asked
     * for. The minimum-of-window pulls it back as soon as a second unit lands, so
     * the only cost of waiting is one more unhinted unit at the start of a run.
     */
    private fun rate(): Long? = synchronized(window) {
        if (window.size < MIN_SAMPLES) null else window.min()
    }

    private fun millis(nanos: Long): String = "%.1f ms".format(nanos / 1_000_000.0)

    companion object {
        /** Units of history the target is drawn from. */
        private const val WINDOW = 16

        /** How many of them must exist before a target is honest. See [rate]. */
        private const val MIN_SAMPLES = 2

        /** See [follow]. */
        private const val MAX_THREADS = 4

        /** Below a millisecond a deadline says nothing, and the API needs it positive. */
        private const val FLOOR_NANOS = 1_000_000L

        /**
         * The largest per-unit cost worth declaring a deadline for.
         *
         * ADPF is a frame API. Every implementation behind it — AOSP's default
         * `PowerHintSession`, and the `IPowerModule`/`IMdpfExt` pair this phone
         * actually runs — is written for targets between about 8 ms (120 Hz) and
         * 33 ms (30 Hz). 250 ms is already an order of magnitude past the slow end
         * of that, and past it the HAL's response is not defined anywhere and was
         * not measured here.
         *
         * This exists because the first version of this class did not have it and
         * the device said so. A Qwen3.5-9B Q4_K_M turn at a 64000 context, on a
         * phone in `Thermal Status: 3` with a Windows container running beside it,
         * decoded at roughly ten seconds a token — and armed a session at a
         * **13084 ms** target for one token, which it then held for the whole run.
         * A thirteen-second deadline is four hundred times outside the range the
         * API is built for, and the most likely reading of it at the other end is
         * "this thread has thirteen seconds, it needs almost nothing".
         *
         * So work this slow is not hinted. That is [SPEC §1.2]: refuse, say the
         * number that caused the refusal, and let the run proceed exactly as it
         * did before rather than hand the power HAL a figure nobody can act on.
         * It takes sd.cpp's sampler steps, whisper's whole-take decodes and any
         * model slow enough to spend a quarter-second a token out of scope, and
         * leaves what is genuinely frame-scale: token generation that is fast
         * enough for the deadline to mean something.
         */
        private const val MAX_UNIT_NANOS = 250_000_000L

        /** Rates that outlive one run, by [carryOver] key. */
        private val carried = HashMap<String, ArrayDeque<Long>>()

        /**
         * A hint session for the work about to start on this thread.
         *
         * Never null: a device without the service, one whose HAL declines, and
         * a caller with no [context] to give all get an instance that reports
         * nothing, so no caller needs a second code path for the unhinted case.
         *
         * @param carryOver keeps the learned rate for later runs of the same
         *   work — the key has to name whatever makes the cost what it is,
         *   normally the model, or two different models will target each other's
         *   pace. Leave it null for anything with enough units in one run to
         *   learn its own rate, which is everything except a whole-take decode.
         * @param tids the threads to hint, defaulting to the caller's. Pass it
         *   explicitly when the work runs somewhere else — sd.cpp's sampler is
         *   watched from a poller, not from the thread inside the JNI call.
         */
        fun open(
            context: Context?,
            tag: String,
            carryOver: String? = null,
            tids: IntArray = intArrayOf(Process.myTid()),
        ): CpuHints {
            val manager = runCatching {
                context?.getSystemService(PerformanceHintManager::class.java)
            }.getOrNull()
            val window = when (carryOver) {
                null -> ArrayDeque()
                else -> synchronized(carried) { carried.getOrPut(carryOver) { ArrayDeque() } }
            }
            return CpuHints(tag, manager, window, tids)
        }
    }
}
