package ai.ondevice.speech

import ai.ondevice.core.BackendId
import ai.onnxruntime.OrtSession
import android.util.Log

/**
 * Which piece of silicon ONNX Runtime should use, applied to a session.
 *
 * ONNX Runtime is not ggml, and the difference matters here. It has no OpenCL
 * provider and no Vulkan provider on Android at all, so "GPU" — the option that
 * means Adreno-through-OpenCL everywhere else in this app — has nothing behind
 * it for Kokoro or OmniVoice. The only accelerator it can reach on this phone is
 * the same Hexagon NPU that llama.cpp reaches, and it reaches it through
 * Qualcomm's QNN library rather than through ggml's HTP backend.
 *
 * That is why the build depends on `onnxruntime-android-qnn`. It is the ordinary
 * runtime plus Qualcomm's shared objects, and it is expensive: about 175 MB of
 * `libQnn*.so` in the APK, most of it `libQnnHtpPrepare.so`, which is what
 * compiles a graph for the DSP on the device.
 *
 * **The NPU is not free, and for these two models may not be a win at all.** The
 * HTP backend wants static shapes. Kokoro and OmniVoice are both variable-length
 * — the text is as long as the text is — so QNN will refuse the dynamic parts and
 * hand them back, and a graph split between two devices pays a copy at every
 * boundary. This is wired so the question can be answered by measuring rather
 * than by arguing, and [describe] exists so the answer is visible in logcat
 * rather than inferred from a stopwatch.
 */
object OrtProviders {

    /**
     * Open a session on [backend], falling back to the CPU if it will not take
     * the graph.
     *
     * The fallback is not defensive tidiness, it is the whole contract. QNN does
     * not partition around what it cannot do — it refuses the session:
     *
     *     ORT_FAIL - qnn_model.cc:73 ParseGraphInputOrOutput
     *     Dynamic shape is not supported yet,
     *     for output: /encoder/text_encoder/lstm/Transpose_output_0
     *
     * That is Kokoro's full-precision export, whose text encoder is an LSTM over
     * a sentence, so its shapes are as variable as the sentence is. The
     * quantised export happens to partition and load; the fp32 one does not
     * load at all. Without this, turning the NPU on in Settings would stop a
     * voice that had been working — an accelerator making the app worse than not
     * having it.
     *
     * So the device choice is a preference, exactly as it is everywhere else in
     * this app: asked for, and clamped to what actually works. The retry builds
     * fresh options because `SessionOptions` cannot have a provider removed.
     */
    fun createSession(
        env: ai.onnxruntime.OrtEnvironment,
        path: String,
        backend: BackendId,
        tag: String,
        configure: OrtSession.SessionOptions.() -> Unit,
    ): OrtSession {
        val wanted = OrtSession.SessionOptions().apply {
            configure()
            apply(this, backend, tag)
        }
        if (backend == BackendId.CPU) return env.createSession(path, wanted)

        return runCatching { env.createSession(path, wanted) }.getOrElse { failure ->
            Log.w(
                tag,
                "${backend.label} would not take ${path.substringAfterLast('/')} — " +
                    "loading on the CPU instead: ${failure.message?.lineSequence()?.firstOrNull()}",
            )
            runCatching { wanted.close() }
            env.createSession(path, OrtSession.SessionOptions().apply(configure))
        }
    }

    /**
     * Add the execution provider for [backend], if this build and this phone
     * have one.
     *
     * CPU is not "no provider" — it is ORT's default and needs nothing added, so
     * it is the honest floor here exactly as it is in [ai.ondevice.engine.ComputeDevice].
     * A provider that fails to register is logged and skipped: the session still
     * runs, on the CPU, which is a slower answer rather than no answer.
     */
    fun apply(options: OrtSession.SessionOptions, backend: BackendId, tag: String) {
        when (backend) {
            BackendId.HEXAGON -> addQnn(options, tag)
            // ORT has no provider for the Adreno on Android. Saying so once, in
            // the log, beats a session that quietly runs on the CPU while the
            // screen says GPU.
            BackendId.OPENCL -> Log.i(
                tag,
                "ONNX Runtime has no GPU provider on Android — this session runs on the CPU",
            )
            BackendId.CPU -> Unit
        }
    }

    private fun addQnn(options: OrtSession.SessionOptions, tag: String) {
        if (!qnnAvailable) {
            Log.i(tag, "no QNN provider in this build — running on the CPU")
            return
        }
        runCatching {
            options.addQnn(
                mapOf(
                    // The DSP. "libQnnCpu.so" and "libQnnGpu.so" also ship in the
                    // AAR and neither is worth choosing: the first is slower than
                    // ORT's own CPU provider and the second is Qualcomm's GPU
                    // path, which is not the one Adreno is tuned for here.
                    "backend_path" to "libQnnHtp.so",
                    // Sustained high performance rather than burst. A voice line
                    // is a second or two of work and the burst profile spends
                    // most of it ramping.
                    "htp_performance_mode" to "high_performance",
                    "htp_graph_finalization_optimization_mode" to "3",
                ),
            )
            Log.i(tag, "QNN HTP provider added")
        }.onFailure {
            // Reaching here means the provider exists but refused this device or
            // this graph. The session is still usable on the CPU.
            Log.w(tag, "QNN provider unavailable — running on the CPU", it)
        }
    }

    /**
     * Whether this build has the QNN provider, asked of the class rather than
     * assumed from the dependency.
     *
     * `addQnn` only exists on the QNN artifact. Testing for the method means the
     * plain `onnxruntime-android` can be swapped back in — if 175 MB turns out
     * to buy nothing — without touching any of the call sites.
     */
    val qnnAvailable: Boolean by lazy {
        runCatching {
            OrtSession.SessionOptions::class.java.getMethod("addQnn", Map::class.java)
            true
        }.getOrDefault(false)
    }

    /** What an ONNX session could run on here, in [BackendId] terms. */
    fun available(): List<BackendId> =
        if (qnnAvailable) listOf(BackendId.CPU, BackendId.HEXAGON) else listOf(BackendId.CPU)

    /** For the log line at load: which device was asked for and what it got. */
    fun describe(backend: BackendId): String = when {
        backend == BackendId.HEXAGON && qnnAvailable -> "HEXAGON (QNN HTP)"
        backend == BackendId.HEXAGON -> "CPU (no QNN in this build)"
        backend == BackendId.OPENCL -> "CPU (ONNX Runtime has no Android GPU provider)"
        else -> "CPU"
    }
}
