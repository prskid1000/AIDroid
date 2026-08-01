package ai.ondevice.speech

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.ondevice.engine.codeSummary
import ai.ondevice.engine.signalSummary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * OmniVoice, the second voice.
 *
 * It is not a replacement for Kokoro and is not offered as one. Measured on one
 * desktop at four threads, Kokoro synthesises at 0.46 s of compute per second of
 * speech and OmniVoice at 2.7–3.5 s. Classifier-free guidance has since doubled
 * the forward passes — 64 rather than 32 for a default run — so the honest
 * figure is around 5.5–7 s, and that part is derived from the pass count rather
 * than re-measured. Either way a phone multiplies it again, and Kokoro remains
 * the default because most of the time you want a sentence read aloud, not
 * composed.
 *
 * What it buys, and Kokoro genuinely cannot do:
 *
 *  - **Emotion and non-verbal sound.** `[laughter]`, `[sigh]`, `[breath]` are
 *    tokens it was trained on. Kokoro's vocabulary is 115 IPA symbols; there is
 *    nowhere to put a laugh.
 *  - **Voice cloning** from a few seconds of reference audio, and voice *design*
 *    from a written description.
 *  - **Any language.** Kokoro is capped at the six espeak-ng phonemisers this
 *    build ships; OmniVoice reads text directly and needs no phonemiser at all.
 *
 * ### How it generates, and why that shapes the API
 *
 * This is **not** an autoregressive model. It fills a fixed-length grid of audio
 * tokens by iterative unmasking: every one of the `frames × 8` slots starts as
 * MASK, and across 32 steps the most confident are committed until none are
 * left. Two consequences run all the way to the UI:
 *
 *  1. **The duration is chosen before the first forward pass.** Nothing predicts
 *     it. Ask for too few frames and the sentence is cut off mid-word; too many
 *     and the tail fills with noise. [estimateFrames] guesses from the text and
 *     deliberately over-allocates, because trailing silence can be trimmed and a
 *     truncated word cannot be recovered.
 *  2. **There is no KV cache to add.** Unmasking rewrites earlier positions, so
 *     every step must re-read the whole sequence. The exported graph takes its
 *     `past_key_values` as empty tensors for exactly this reason. The cost is
 *     structural, not a missing optimisation.
 */
class OmniVoiceEngine {

    private val mutex = Mutex()

    private var environment: OrtEnvironment? = null
    private var embeddings: OrtSession? = null
    private var decoder: OrtSession? = null
    private var heads: OrtSession? = null
    private var vocoder: OrtSession? = null

    /**
     * The encode half of the Higgs audio tokenizer, for voice cloning.
     *
     * Optional rather than required: they are 328 MB and only a clone needs
     * them, so an install without them still speaks. [supportsCloning] is what
     * a screen asks before offering the feature.
     */
    private var acousticEncoder: OrtSession? = null
    private var semanticEncoder: OrtSession? = null
    private var quantizer: OrtSession? = null
    private var tokenizer: QwenTokenizer? = null
    private var pastNames: List<String> = emptyList()
    private var describedGraphs = false
    private var describedShapes = false

    /** Reset per synthesis; see the trace guard in [forward]. */
    private var tracedForwards = 0

    /**
     * Whether the backbone declares `attention_mask` with four dimensions.
     *
     * Read off the graph rather than assumed, so both exports load: the
     * published causal one wants a 2-D mask, the corrected one a 4-D square.
     * The bidirectional probe then decides whether it can actually be used.
     */
    private var fourDimensionalMask = false

    /** Null until measured; see [attentionIsBidirectional]. Reset on unload. */
    private var bidirectional: Boolean? = null

    /** Null until measured; see [vocoderProducesAudio]. Reset on unload. */
    private var vocoderWorks: Boolean? = null
    private var llmEmbedType: OnnxJavaType = OnnxJavaType.FLOAT
    private var headsInputType: OnnxJavaType = OnnxJavaType.FLOAT

    @Volatile
    private var loadedPath: String? = null

    @Volatile
    var lastError: String? = null
        private set

    val isLoaded: Boolean get() = decoder != null

    /** Whether this install carries the encoders a reference clip needs. */
    val supportsCloning: Boolean
        get() = acousticEncoder != null && semanticEncoder != null && quantizer != null

    /**
     * A reference clip, as the eight codebooks the backbone reads.
     *
     * [samples] is mono at [SAMPLE_RATE]. The two encoders disagree about rate —
     * acoustic wants 24 kHz and semantic 16 kHz — so the resample happens here
     * rather than being the caller's problem, and they can disagree by a frame
     * at the tail, which the quantizer will not accept.
     */
    private fun encodeReference(samples: FloatArray): Array<LongArray> {
        val env = environment ?: OrtEnvironment.getEnvironment()
        val acoustic = acousticEncoder!!
        val semantic = semanticEncoder!!
        val quant = quantizer!!

        // Whole frames only; a partial one has no codes to be.
        val usable = samples.size - samples.size % SAMPLES_PER_FRAME
        check(usable >= SAMPLES_PER_FRAME) {
            "The reference clip is shorter than one ${SAMPLES_PER_FRAME * 1000 / SAMPLE_RATE} ms frame."
        }
        val trimmed = samples.copyOf(usable)
        val resampled = resample(trimmed, SAMPLE_RATE, SEMANTIC_SAMPLE_RATE)

        val closeables = mutableListOf<OnnxTensor>()
        try {
            fun feed(session: OrtSession, name: String, data: FloatArray, shape: LongArray): OnnxTensor {
                val type = typeOf(session, name)
                return if (type == OnnxJavaType.FLOAT16) {
                    val shorts = java.nio.ShortBuffer.allocate(data.size)
                    data.forEach { shorts.put(floatToHalf(it)) }
                    shorts.rewind()
                    OnnxTensor.createTensor(env, shorts, shape, OnnxJavaType.FLOAT16)
                } else {
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
                }.also(closeables::add)
            }

            val acousticIn = feed(
                acoustic, "waveform_24k", trimmed, longArrayOf(1, 1, usable.toLong()),
            )
            val semanticIn = feed(
                semantic, "waveform_16k", resampled, longArrayOf(1, resampled.size.toLong()),
            )

            acoustic.run(mapOf("waveform_24k" to acousticIn)).use { acousticOut ->
                semantic.run(mapOf("waveform_16k" to semanticIn)).use { semanticOut ->
                    val acousticFeatures = acousticOut.pick("acoustic_features")
                    val semanticFeatures = semanticOut.pick("semantic_features")
                    val frames = minOf(
                        acousticFeatures.info.shape.last(), semanticFeatures.info.shape.last(),
                    ).toInt()

                    val quantType = typeOf(quant, "semantic_features")
                    val inputs = mapOf(
                        "acoustic_features" to trimFrames(
                            env, acousticFeatures, frames, typeOf(quant, "acoustic_features"),
                        ).also(closeables::add),
                        "semantic_features" to trimFrames(
                            env, semanticFeatures, frames, quantType,
                        ).also(closeables::add),
                    )
                    quant.run(inputs).use { coded ->
                        val codes = coded.pick("codes")
                        // [codebooks, batch, frames], batch of one.
                        val flat = codes.longBuffer
                        val values = LongArray(flat.remaining()).also(flat::get)
                        return Array(CODEBOOKS) { cb ->
                            LongArray(frames) { f -> values[cb * frames + f] }
                        }
                    }
                }
            }
        } finally {
            closeables.forEach { runCatching { it.close() } }
        }
    }

    /**
     * `[batch, channels, frames]` cut to [frames] and converted to [target].
     *
     * The cut is why this exists: at 25 frames a second from two different
     * sample rates the encoders can differ by one frame, and the quantizer
     * multiplies them together.
     */
    private fun trimFrames(
        env: OrtEnvironment,
        tensor: OnnxTensor,
        frames: Int,
        target: OnnxJavaType,
    ): OnnxTensor {
        val shape = tensor.info.shape
        val channels = shape[1].toInt()
        val sourceFrames = shape[2].toInt()
        val source = readFloats(tensor)
        val cut = FloatArray(channels * frames)
        for (c in 0 until channels) {
            System.arraycopy(source, c * sourceFrames, cut, c * frames, frames)
        }
        val cutShape = longArrayOf(1, channels.toLong(), frames.toLong())
        return if (target == OnnxJavaType.FLOAT16) {
            val shorts = java.nio.ShortBuffer.allocate(cut.size)
            cut.forEach { shorts.put(floatToHalf(it)) }
            shorts.rewind()
            OnnxTensor.createTensor(env, shorts, cutShape, OnnxJavaType.FLOAT16)
        } else {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(cut), cutShape)
        }
    }

    /**
     * Linear resampling.
     *
     * Good enough because of what it feeds: the semantic encoder is a speech
     * model reading 16 kHz, and the artefacts linear interpolation adds sit
     * above 8 kHz where it has nothing to hear. A polyphase filter would be
     * more correct and no more accurate here.
     */
    private fun resample(samples: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to || samples.isEmpty()) return samples
        val length = (samples.size.toLong() * to / from).toInt()
        val ratio = from.toDouble() / to
        return FloatArray(length) { i ->
            val position = i * ratio
            val left = position.toInt()
            val right = (left + 1).coerceAtMost(samples.size - 1)
            val fraction = (position - left).toFloat()
            samples[left] * (1f - fraction) + samples[right] * fraction
        }
    }

    val runtimeAvailable: Boolean get() = ONNX_AVAILABLE

    /**
     * Whether a directory looks like an OmniVoice install.
     *
     * Checked by parts rather than by name: the publisher lays the backbone out
     * under a precision folder (`int4/`, `fp16/`) and the vocoder under
     * `audio_tokenizer/`, and a user who reorganised the folders should still
     * get a working model rather than a lecture about layout.
     */
    fun looksInstalled(directory: File): Boolean =
        REQUIRED.all { find(directory, it) != null }

    /**
     * Whether a directory also carries the encoders a voice clone needs.
     *
     * Asked of the files rather than of [supportsCloning], because a screen has
     * to decide whether to offer the feature before anything is loaded, and
     * loading is a minute of work.
     */
    fun cloningLooksInstalled(directory: File): Boolean =
        CLONING.all { find(directory, it) != null }

    suspend fun load(
        directory: File,
        threads: Int = 0,
        backend: ai.ondevice.core.BackendId = ai.ondevice.core.BackendId.CPU,
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            check(ONNX_AVAILABLE) { "The ONNX Runtime is not installed in this build." }

            val files = REQUIRED.associateWith { name ->
                find(directory, name) ?: error(
                    "${directory.name} is missing $name. An OmniVoice install needs the three " +
                        "backbone graphs and the Higgs decoder.",
                )
            }
            val tokenizerFile = find(directory, "tokenizer.json")
                ?: error("${directory.name} has no tokenizer.json, and OmniVoice reads text directly.")

            mutex.withLock {
                if (loadedPath == directory.absolutePath) return@withLock
                unlockedUnload()

                val env = OrtEnvironment.getEnvironment()
                val options = {
                    OrtSession.SessionOptions().apply {
                        if (threads > 0) setIntraOpNumThreads(threads)
                        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                        // Settings -> Compute device; see [OrtProviders]. Applied
                        // per session because OmniVoice is five graphs, not one.
                        OrtProviders.apply(this, backend, TAG)
                    }
                }
                android.util.Log.i(TAG, "loading OmniVoice on ${OrtProviders.describe(backend)}")

                val loadedTokenizer = QwenTokenizer.load(tokenizerFile).getOrThrow()
                fun open(file: java.io.File): OrtSession =
                    runCatching { env.createSession(file.absolutePath, options()) }
                        .getOrElse { error(explainLoadFailure(file, it)) }

                val emb = open(files.getValue(EMBEDDINGS))
                val llm = open(files.getValue(DECODER))
                val head = open(files.getValue(HEADS))
                val voc = open(files.getValue(VOCODER))

                // Cloning is all-or-nothing — one encoder short and the
                // quantizer has nothing to pair — so the three are opened
                // together or not at all. A failure here is not a failure to
                // load the model; it costs the clone, not the voice.
                val cloningGraphs = runCatching {
                    CLONING.map { open(find(directory, it) ?: error("$it is missing")) }
                }.getOrElse {
                    android.util.Log.i(TAG, "no voice cloning: ${it.message}")
                    emptyList()
                }

                environment = env
                embeddings = emb
                decoder = llm
                this@OmniVoiceEngine.heads = head
                vocoder = voc
                acousticEncoder = cloningGraphs.getOrNull(0)
                semanticEncoder = cloningGraphs.getOrNull(1)
                quantizer = cloningGraphs.getOrNull(2)
                tokenizer = loadedTokenizer
                // 28 layers × key and value. Read from the graph rather than
                // assumed, so a re-export with a different depth still runs.
                pastNames = llm.inputNames.filter { it.contains("past") }
                fourDimensionalMask =
                    (llm.inputInfo["attention_mask"]?.info as? ai.onnxruntime.TensorInfo)
                        ?.shape?.size == 4
                llmEmbedType = typeOf(llm, "inputs_embeds")
                headsInputType = typeOf(head, "hidden_states")
                loadedPath = directory.absolutePath

                // Which four files were picked, and how big each one really is.
                // `find` chooses the largest copy of every name and the copies
                // live in sibling folders, so the set that loads is assembled
                // from several places and no screen shows which. When the voice
                // comes out wrong the precision of these files is the first
                // thing worth ruling out, and it is unreadable from anywhere
                // else on the device.
                files.forEach { (name, file) ->
                    android.util.Log.i(TAG, "graph $name ← ${describeFile(directory, file)}")
                }
                android.util.Log.i(
                    TAG,
                    "loaded threads=$threads embeds=$llmEmbedType heads=$headsInputType " +
                        "vocoder=${
                            (voc.outputInfo.values.firstOrNull()?.info as? ai.onnxruntime.TensorInfo)
                                ?.type ?: "unknown"
                        } " +
                        "pastInputs=${pastNames.size} fourDimensionalMask=$fourDimensionalMask " +
                        "cloning=$supportsCloning vocab=${loadedTokenizer.size}",
                )
            }
        }.onFailure { lastError = it.message }
    }

    /**
     * Turn a graph-load failure into something the user can act on.
     *
     * The int4 export quantises its embedding table four bits to a weight and
     * says so with a `bits` attribute on `com.microsoft.GatherBlockQuantized`,
     * which ONNX Runtime gained in 1.23. This build ships 1.28, so it loads. The
     * message survives for anyone running an older runtime, where the failure is
     * a node dump that reads like a corrupt download and is nothing of the kind.
     */
    private fun explainLoadFailure(file: java.io.File, cause: Throwable): String {
        val message = cause.message.orEmpty()
        val unsupportedQuant = "GatherBlockQuantized" in message ||
            ("bits" in message && "Unrecognized attribute" in message)
        return if (unsupportedQuant) {
            "${file.name} is a 4-bit export that needs the `bits` attribute on " +
                "GatherBlockQuantized, added in ONNX Runtime 1.23. This build has a runtime " +
                "older than that. Nothing is wrong with the download."
        } else {
            "${file.name} could not be loaded. $message"
        }
    }

    /**
     * Whether the backbone can see the whole sequence, which this model needs.
     *
     * OmniVoice unmasks a grid: a frame at position 5 has to attend to a frame
     * committed at position 50, or ordering the commits by confidence buys
     * nothing. That requires bidirectional attention.
     *
     * `onnx-community/OmniVoice-Onnx` was exported with onnxruntime-genai's
     * ModelBuilder, which builds *autoregressive* decoders — attention is
     * masked to the past. Measured on the int4 export: change the last token of
     * a twelve-token sequence and every earlier hidden state is bit-identical,
     * max|diff| exactly 0. The repo's own eval.py never catches this because it
     * checks the embeddings encoder and the heads decoder against PyTorch and
     * calls the LLM "a black-box genai model".
     *
     * The result is not subtly worse audio, it is a buzz: unmasking degenerates
     * to a handful of repeated codes. Reproduced outside the app, in Python,
     * with the reference algorithm — 42 distinct codes across 1024 slots.
     *
     * So this is measured rather than assumed, and per model directory. Nothing
     * here names OmniVoice: any export whose backbone cannot see forwards fails
     * the same way and earns the same refusal.
     */
    private fun attentionIsBidirectional(): Boolean = runCatching {
        val probe = 12
        val ids = Array(CODEBOOKS) { LongArray(probe) { i -> (5 + i).toLong() } }
        val mask = BooleanArray(probe)
        val attention = LongArray(probe) { 1L }

        val first = forward(ids, mask, attention, probe)
        for (cb in 0 until CODEBOOKS) ids[cb][probe - 1] = 900L
        val second = forward(ids, mask, attention, probe)

        // Compare only what precedes the changed position. If the backbone is
        // causal those logits cannot have moved at all.
        val upto = (probe - 1) * AUDIO_VOCAB
        var largest = 0f
        for (cb in 0 until CODEBOOKS) {
            val base = cb * probe * AUDIO_VOCAB
            for (i in 0 until upto) {
                val d = kotlin.math.abs(first[base + i] - second[base + i])
                if (d > largest) largest = d
            }
        }
        android.util.Log.i(TAG, "bidirectional probe: max|diff| before the change = $largest")
        largest > 1e-6f
    }.getOrDefault(true) // Could not ask; do not refuse on a failed measurement.

    /**
     * Whether the vocoder can turn codes into numbers at all, asked in about a
     * tenth of a second before three minutes are spent filling a grid for it.
     *
     * The shipped `higgs_decoder.onnx` is float16 from its weights to its output
     * with no Cast anywhere in it — 136 initialisers, 601 activations, all fp16 —
     * and on arm64 the CPU has real ARMv8.2 fp16 arithmetic, which stops at
     * 65504. It overflows, and every one of its 48 000 samples comes back NaN.
     * The same file is fine on an x86 emulator, where ORT has no fp16 kernels
     * and wraps the whole graph in fp32 casts instead, so a model can be
     * verified on one and be incapable of speech on the other.
     *
     * Measured rather than assumed, and per device: nothing here refuses a graph
     * for being fp16, only for returning arithmetic that is not a number. A
     * probe that cannot run at all is not evidence of anything and does not
     * refuse.
     */
    private fun vocoderProducesAudio(): Boolean = runCatching {
        val codes = Array(CODEBOOKS) { cb ->
            LongArray(PROBE_FRAMES) { f -> ((cb * 131 + f * 17) % CODEBOOK_SIZE).toLong() }
        }
        val probe = decodeToWaveform(codes, PROBE_FRAMES)
        probe.isNotEmpty() && probe.all { it.isFinite() }
    }.getOrDefault(true)

    /** Why an all-NaN waveform happens, and the one thing that fixes it. */
    private fun fp16Overflow(detail: String): String =
        "This OmniVoice install cannot produce audio on this device: $detail. Its " +
            "$VOCODER computes in float16 throughout, and arm64 has real fp16 arithmetic " +
            "which stops at 65504, so the vocoder overflows. An x86 emulator has no fp16 " +
            "kernels and quietly computes the same graph in fp32, which is why a model can " +
            "pass testing there and be silent here. A copy of $VOCODER exported at full " +
            "precision fixes it — put it in a subfolder of the model and it will be picked " +
            "up, because the largest copy of each graph is the one that loads."

    suspend fun unload() = mutex.withLock { unlockedUnload() }

    private fun unlockedUnload() {
        listOf(embeddings, decoder, heads, vocoder, acousticEncoder, semanticEncoder, quantizer)
            .forEach { runCatching { it?.close() } }
        embeddings = null
        decoder = null
        heads = null
        vocoder = null
        acousticEncoder = null
        semanticEncoder = null
        quantizer = null
        tokenizer = null
        pastNames = emptyList()
        fourDimensionalMask = false
        loadedPath = null
        bidirectional = null
        vocoderWorks = null
        describedGraphs = false
        describedShapes = false
        tracedForwards = 0
    }

    suspend fun synthesize(request: OmniVoiceRequest): Result<KokoroAudio> =
        withContext(Dispatchers.Default) {
            runCatching {
                val tok = tokenizer ?: error("No OmniVoice model is loaded.")
                val text = request.text.trim()
                check(text.isNotEmpty()) { "There is nothing to say." }

                val reference = request.reference
                if (reference != null) {
                    check(supportsCloning) {
                        "This OmniVoice install cannot clone a voice: it is missing the three " +
                            "encoder graphs that turn a recording into audio tokens. Reinstalling " +
                            "the model adds them; the voice you describe in words still works."
                    }
                }

                val textTokens = buildPrompt(
                    tok, text, request.language, request.instruction,
                    referenceText = reference?.transcript,
                    cloning = reference != null,
                )
                check(textTokens.isNotEmpty()) { "That text produced no tokens." }

                val frames = request.frames ?: estimateFrames(text, request.speed)
                tracedForwards = 0
                android.util.Log.i(
                    TAG,
                    "synthesising chars=${text.length} tokens=${textTokens.size} frames=$frames " +
                        "(${"%.2f".format(frames.toFloat() / FRAMES_PER_SECOND)}s grid) " +
                        "steps=${request.steps} guidance=${request.guidance} " +
                        "tShift=${request.timestepShift} layerPenalty=${request.layerPenalty} " +
                        "posTemp=${request.positionTemperature} " +
                        "classTemp=${request.classTemperature} " +
                        "language=${request.language ?: "None"} " +
                        "instruction=${if (request.instruction.isNullOrBlank()) "None" else "set"} " +
                        "cloning=${reference != null}",
                )

                // Lift a quiet reference to a working level before encoding, and
                // remember by how much so the generated audio can be put back
                // down to match. Doing only the first half makes a clone of a
                // softly-spoken voice come out loud.
                var referenceLoudness = 0f
                val referenceCodes = reference?.let {
                    val prepared = if (it.sampleRate == SAMPLE_RATE) it.samples
                    else resample(it.samples, it.sampleRate, SAMPLE_RATE)
                    check(prepared.isNotEmpty()) { "The reference recording is empty." }
                    referenceLoudness = kotlin.math.sqrt(
                        prepared.fold(0.0) { sum, s -> sum + s.toDouble() * s } / prepared.size,
                    ).toFloat()
                    val levelled = if (referenceLoudness > 0f && referenceLoudness < QUIET_RMS) {
                        FloatArray(prepared.size) { i -> prepared[i] * QUIET_RMS / referenceLoudness }
                    } else {
                        prepared
                    }
                    mutex.withLock { encodeReference(levelled) }
                }

                // Measured once per load, before spending a minute of compute on
                // a grid the backbone cannot unmask.
                if (bidirectional == null) {
                    bidirectional = mutex.withLock { attentionIsBidirectional() }
                }
                check(bidirectional == true) {
                    "This OmniVoice export cannot produce speech. Its backbone was built as an " +
                        "autoregressive decoder — each position sees only the ones before it — " +
                        "and OmniVoice unmasks a grid, so a frame has to see the frames committed " +
                        "after it. Measured here, not guessed: changing the last token leaves " +
                        "every earlier output bit-identical. The audio would be a buzz, so the " +
                        "app stops rather than playing one. A re-export with full attention " +
                        "would work; nothing is wrong with your download or this device."
                }

                // Also once per load, and before the grid rather than after it:
                // the failure it catches costs three minutes to reach otherwise,
                // and arrives as a refusal either way.
                if (vocoderWorks == null) {
                    vocoderWorks = mutex.withLock { vocoderProducesAudio() }
                }
                check(vocoderWorks == true) {
                    fp16Overflow("a $PROBE_FRAMES-frame probe of its vocoder returned no finite sample")
                }

                val codes = mutex.withLock { unmask(textTokens, referenceCodes, frames, request) }
                val decoded = mutex.withLock { decodeToWaveform(codes, frames) }
                // Put the clone back to the reference's own level.
                val samples = if (referenceLoudness > 0f && referenceLoudness < QUIET_RMS) {
                    FloatArray(decoded.size) { i -> decoded[i] * referenceLoudness / QUIET_RMS }
                } else {
                    decoded
                }

                checkFinite(samples)
                val kept = if (request.trimSilence) trimTail(samples) else samples
                android.util.Log.i(
                    TAG,
                    "synthesised raw=${samples.size} kept=${kept.size} " +
                        "(${"%.2f".format(kept.size.toFloat() / SAMPLE_RATE)}s)",
                )
                KokoroAudio(
                    samples = kept,
                    sampleRate = SAMPLE_RATE,
                    phonemes = "",
                    chunks = 1,
                )
            }.onFailure {
                lastError = it.message
                android.util.Log.e(TAG, "synthesis failed", it)
            }
        }

    /**
     * How the text is framed before the model sees it.
     *
     * There are two candidate answers and they disagree, which is worth stating
     * plainly because this is the difference between speech and noise.
     *
     * `modeling_omnivoice.py` — the PyTorch model code — wraps every request as
     * a style block followed by a delimited text block:
     *
     *     <|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>
     *     <|text_start|>…<|text_end|>
     *
     * All six of those specials are real entries in this repo's tokenizer, and
     * "None" is upstream's own literal for "nothing specified", not a
     * placeholder invented here. That is a good argument, and it is the one
     * this engine followed first.
     *
     * `inference.py` — shipped in the ONNX repo — does none of it and hands the
     * model bare prose. Following it was a mistake: the framing above is what
     * `generate()` builds, verified against the source, and dropping it cost
     * 33 dB of dynamic range in a measured A/B. Bare prose gave 25.9 dB and an
     * invented word before the sentence; the framing gave 58.8 dB against the
     * PyTorch reference's 54.7, and the invented word disappeared. Without
     * `<|text_start|>` the model has no marker for where the text begins, so it
     * makes up a lead-in and then runs out of grid.
     *
     * `inference.py` was wrong in three separate ways — this, the unmasking
     * schedule, and no classifier-free guidance. It is not a reliable witness
     * for how these graphs want to be driven; `models/omnivoice.py` is.
     *
     * `<|denoise|>` stays absent: upstream emits it only with a reference clip.
     */
    private fun buildPrompt(
        tok: QwenTokenizer,
        text: String,
        language: String?,
        instruction: String?,
        referenceText: String?,
        cloning: Boolean,
    ): IntArray {
        val style = buildString {
            // Only ever with a reference clip — upstream emits it nowhere else.
            if (cloning) append("<|denoise|>")
            append("<|lang_start|>").append(language ?: "None").append("<|lang_end|>")
            append("<|instruct_start|>").append(instruction ?: "None").append("<|instruct_end|>")
        }
        // Cloning is continuation rather than conditioning: the reference
        // transcript goes in front of the text as one sentence, so the model
        // reads what the reference said and carries straight on into the new
        // words. That is also why a clone needs a transcript and not just audio.
        val combined = if (!referenceText.isNullOrBlank()) {
            referenceText.trim() + " " + text.trim()
        } else {
            text.trim()
        }
        return tok.encodeWithMarkup(style) +
            tok.encodeWithMarkup("<|text_start|>$combined<|text_end|>")
    }

    // — the unmasking loop —

    /**
     * Confidence-ordered iterative unmasking, as the model's own `generate()`
     * does it.
     *
     * The version this replaces was translated from the ONNX repo's
     * `inference.py`, and that script is a cruder algorithm than the model
     * expects — it would degrade the audio even on a correct graph. Four things
     * differ, and each is upstream's, not invented here:
     *
     *  1. **The unit is a slot, not a frame.** The grid has `frames × 8`
     *     independently maskable (codebook, frame) cells, and a step commits the
     *     most confident *cells*. Committing all eight codebooks of a frame
     *     together throws away most of the ordering the model is scored on.
     *  2. **The schedule is a shifted diffusion curve**, not `remaining /
     *     stepsLeft`. With `t_shift` 0.1 it is convex: the opening steps commit a
     *     handful of slots and the last commits a fifth of the grid, so the
     *     cheap early decisions are made on the least evidence and the rest
     *     follow from them.
     *  3. **The codebook bias is subtractive** — `score − codebook × 5.0` — so
     *     codebook 0 is committed well before codebook 7. The 8,8,6,6,4,4,2,2
     *     multiplicative weighting that used to be here appears nowhere upstream;
     *     it was guessed.
     *  4. **Classifier-free guidance.** Every step runs the grid a second time
     *     with no text at all and pushes the prediction away from what the model
     *     would say unprompted. This is what doubles the cost, and it is also
     *     what the corrected export bought: an unconditional branch is only
     *     meaningful if attention can see the whole grid.
     *
     * Measured against the PyTorch reference on the same sentence: 54.4 dB
     * dynamic range against 54.7, spectral centroid 1390 Hz against 1431.
     *
     * The schedule sums to exactly `frames × 8`, so the grid is always full at
     * the end. That matters because MASK (1024) is outside the codec's 0..1023
     * range and the vocoder does not degrade gracefully when it sees one — it
     * fails. No safety net is needed for it; arithmetic covers it.
     */
    private suspend fun unmask(
        textTokens: IntArray,
        referenceCodes: Array<LongArray>?,
        frames: Int,
        request: OmniVoiceRequest,
    ): Array<LongArray> {
        val steps = request.steps.coerceIn(1, MAX_STEPS)
        val textLength = textTokens.size
        // A reference clip sits between the text and the grid as real codes the
        // model can see. It is context, never a target, so the grid it fills is
        // the same size either way and only the sequence in front of it grows.
        val referenceLength = referenceCodes?.firstOrNull()?.size ?: 0
        val gridStart = textLength + referenceLength
        val condSequence = gridStart + frames
        val slots = CODEBOOKS * frames

        // The grid being filled, [codebook][frame]. Everything starts masked.
        val tokens = Array(CODEBOOKS) { LongArray(frames) { MASK_ID.toLong() } }
        val schedule = buildSchedule(frames, steps, request.timestepShift)
        // A fixed seed makes a run reproducible, which matters because the
        // position temperature genuinely randomises which slots go first: the
        // same text twice is otherwise two different takes.
        val random = java.util.Random(
            if (request.seed != 0L) request.seed else System.nanoTime(),
        )

        // The conditional branch: text, the reference if there is one, then the
        // grid. Only the grid is rewritten between steps, so everything in front
        // of it is laid in once.
        val condIds = Array(CODEBOOKS) { LongArray(condSequence) }
        for (cb in 0 until CODEBOOKS) {
            for (i in 0 until textLength) condIds[cb][i] = textTokens[i].toLong()
            referenceCodes?.get(cb)?.copyInto(condIds[cb], textLength)
        }
        // Audio starts where the text ends, which puts the reference on the
        // audio side of the line — it is sound the model hears, not words it
        // reads, and the transcript in the text is what tells it what was said.
        val condAudio = BooleanArray(condSequence) { it >= textLength }
        val condAttention = LongArray(condSequence) { 1L }
        // The unconditional branch is the grid alone — not the same sequence
        // with the text masked out, but a shorter one with no text in it.
        val uncondAudio = BooleanArray(frames) { true }
        val uncondAttention = LongArray(frames) { 1L }

        var committed = 0
        val scores = FloatArray(slots)
        val predicted = IntArray(slots)
        val probabilities = FloatArray(AUDIO_VOCAB)
        val unguided = FloatArray(AUDIO_VOCAB)
        val ranked = FloatArray(AUDIO_VOCAB)

        for (step in 0 until steps) {
            val take = schedule[step]
            if (take <= 0) continue
            currentCoroutineContext().ensureActive()

            for (cb in 0 until CODEBOOKS) tokens[cb].copyInto(condIds[cb], gridStart)
            val conditional = forward(condIds, condAudio, condAttention, condSequence)
            val unconditional = if (request.guidance != 0f) {
                forward(tokens, uncondAudio, uncondAttention, frames)
            } else {
                null
            }

            for (cb in 0 until CODEBOOKS) {
                val penalty = cb * request.layerPenalty
                for (f in 0 until frames) {
                    val slot = cb * frames + f
                    if (tokens[cb][f] != MASK_ID.toLong()) {
                        // Already committed, and a slot is only decided once.
                        scores[slot] = Float.NEGATIVE_INFINITY
                        continue
                    }
                    predicted[slot] = scoreSlot(
                        conditional = conditional,
                        condBase = ((cb.toLong() * condSequence + (gridStart + f)) * AUDIO_VOCAB).toInt(),
                        unconditional = unconditional,
                        uncondBase = ((cb.toLong() * frames + f) * AUDIO_VOCAB).toInt(),
                        guidance = request.guidance,
                        classTemperature = request.classTemperature,
                        probabilities = probabilities,
                        unguided = unguided,
                        ranked = ranked,
                        random = random,
                    )
                    var best = Float.NEGATIVE_INFINITY
                    for (v in 0 until AUDIO_VOCAB) {
                        if (probabilities[v] > best) best = probabilities[v]
                    }
                    scores[slot] = perturb(best - penalty, request.positionTemperature, random)
                }
            }

            // The `take` most confident still-masked slots, across the whole
            // grid rather than per frame.
            val order = (0 until slots)
                .filter { scores[it] > Float.NEGATIVE_INFINITY }
                .sortedByDescending { scores[it] }
                .take(take)
            order.forEach { slot ->
                tokens[slot / frames][slot % frames] = predicted[slot].toLong()
            }
            committed += order.size

            // What this step actually decided. Degenerate unmasking is not an
            // error at any layer — the codes are in range, the vocoder decodes
            // them and the result is a buzz — so the count of distinct codes
            // committed is the number that separates speech from a tone, and it
            // is free here because `predicted` is already in hand.
            android.util.Log.i(
                TAG,
                "step ${step + 1}/$steps take=$take committed=$committed/$slots " +
                    "chose ${LongArray(order.size) { predicted[order[it]].toLong() }.codeSummary()}",
            )
        }

        for (cb in 0 until CODEBOOKS) {
            android.util.Log.i(TAG, "grid codebook $cb ${tokens[cb].codeSummary()}")
        }
        return tokens
    }

    /**
     * How many slots to commit at each step.
     *
     * `t' = shift·t / (1 + (shift−1)·t)` over `linspace(0, 1, steps+1)`, and the
     * step's share is the gap between consecutive points. At the upstream
     * default of 0.1 the curve is convex, so the first step of thirty-two takes
     * around 0.5% of the grid and the last takes a fifth. Rounding up every
     * step and giving the remainder to the last guarantees the total is exact.
     */
    internal fun buildSchedule(frames: Int, steps: Int, shift: Float): IntArray {
        val safeShift = shift.coerceIn(MIN_T_SHIFT, 1f)
        val curve = FloatArray(steps + 1) {
            val t = it.toFloat() / steps
            safeShift * t / (1f + (safeShift - 1f) * t)
        }
        val total = frames * CODEBOOKS
        var remaining = total
        return IntArray(steps) { step ->
            val take = if (step == steps - 1) {
                remaining
            } else {
                minOf(kotlin.math.ceil(total * (curve[step + 1] - curve[step])).toInt(), remaining)
            }
            remaining -= take
            take
        }
    }

    /**
     * Guided log-probabilities for one slot, left in [probabilities]; returns
     * the code to commit there.
     *
     * The caller reads the confidence back off [probabilities] rather than
     * getting it returned, because with a class temperature the sampled code
     * and the most probable one are different codes — and it is the *most
     * probable* one whose probability orders the commits.
     */
    private fun scoreSlot(
        conditional: FloatArray,
        condBase: Int,
        unconditional: FloatArray?,
        uncondBase: Int,
        guidance: Float,
        classTemperature: Float,
        probabilities: FloatArray,
        unguided: FloatArray,
        ranked: FloatArray,
        random: java.util.Random,
    ): Int {
        logSoftmax(conditional, condBase, probabilities)
        if (unconditional != null) {
            logSoftmax(unconditional, uncondBase, unguided)
            // c + g·(c − u), then renormalised. Pushing away from what the model
            // says with no text is what sharpens it towards this text.
            for (v in 0 until AUDIO_VOCAB) {
                probabilities[v] += guidance * (probabilities[v] - unguided[v])
            }
            logSoftmax(probabilities, 0, probabilities)
        }
        // MASK is in the vocabulary and in the softmax denominator, but it can
        // never be committed — a MASK reaching the vocoder is a hard failure.
        probabilities[MASK_ID] = Float.NEGATIVE_INFINITY

        if (classTemperature <= 0f) {
            var argmax = 0
            var best = Float.NEGATIVE_INFINITY
            for (v in 0 until AUDIO_VOCAB) {
                if (probabilities[v] > best) {
                    best = probabilities[v]
                    argmax = v
                }
            }
            return argmax
        }

        // Sample instead of taking the best, from the top tenth of the
        // vocabulary. Off by default; upstream's own default is 0.
        //
        // Sorted through a scratch array rather than `sortedDescending()`, which
        // returns a boxed List<Float> — a thousand allocations per slot, and
        // there are frames × 8 slots on every one of the steps.
        val keep = kotlin.math.ceil(TOP_K_RATIO * AUDIO_VOCAB).toInt()
        probabilities.copyInto(ranked)
        java.util.Arrays.sort(ranked)
        val threshold = ranked[AUDIO_VOCAB - keep]
        var argmax = 0
        var best = Float.NEGATIVE_INFINITY
        for (v in 0 until AUDIO_VOCAB) {
            if (probabilities[v] < threshold) continue
            val sampled = perturb(probabilities[v], classTemperature, random)
            if (sampled > best) {
                best = sampled
                argmax = v
            }
        }
        return argmax
    }

    /**
     * `omnivoice._gumbel_sample`: divide by the temperature, then add Gumbel
     * noise. Dividing rather than multiplying is the direction that makes a
     * *higher* temperature mean *more* randomness, since it shrinks the real
     * differences the fixed-scale noise then swamps.
     */
    private fun perturb(value: Float, temperature: Float, random: java.util.Random): Float {
        if (temperature <= 0f) return value
        val uniform = random.nextFloat()
        val gumbel = -kotlin.math.ln(-kotlin.math.ln(uniform + 1e-10f) + 1e-10f)
        return value / temperature + gumbel
    }

    /** Log-softmax of `source[base until base + AUDIO_VOCAB]` into [destination]. */
    private fun logSoftmax(source: FloatArray, base: Int, destination: FloatArray) {
        var max = Float.NEGATIVE_INFINITY
        for (v in 0 until AUDIO_VOCAB) {
            val value = source[base + v]
            if (value > max) max = value
        }
        var sum = 0f
        for (v in 0 until AUDIO_VOCAB) sum += kotlin.math.exp(source[base + v] - max)
        val normaliser = max + kotlin.math.ln(sum)
        for (v in 0 until AUDIO_VOCAB) destination[v] = source[base + v] - normaliser
    }

    /** One pass: embeddings → 28-layer backbone → per-codebook logits, flattened. */
    private fun forward(
        ids: Array<LongArray>,
        audioMask: BooleanArray,
        attention: LongArray,
        sequence: Int,
    ): FloatArray {
        val env = environment ?: OrtEnvironment.getEnvironment()
        val emb = embeddings!!
        val llm = decoder!!
        val head = heads!!

        // Once per load. Reading these graphs positionally is what produced
        // noise; if it ever does again, this says in one line whether the names
        // are being found or whether `pick` is silently falling back to output
        // zero, which is the failure it was written to end.
        if (!describedGraphs) {
            describedGraphs = true
            android.util.Log.i(
                TAG,
                "llm outputs=${llm.outputNames.take(4)} (${llm.outputNames.size} total) " +
                    "heads outputs=${head.outputNames} " +
                    "embeddings outputs=${emb.outputNames}",
            )
        }
        // The first two passes are the conditional and unconditional branches of
        // step one, which is where a broken graph is already broken. After that
        // the numbers repeat and the log would be 64 copies of the same line.
        val trace = tracedForwards < 2
        if (trace) tracedForwards++

        val flat = LongArray(CODEBOOKS * sequence)
        for (cb in 0 until CODEBOOKS) ids[cb].copyInto(flat, cb * sequence)

        val idsTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(flat), longArrayOf(1, CODEBOOKS.toLong(), sequence.toLong()),
        )
        // ORT's Java API takes booleans as one byte each.
        val maskBytes = ByteBuffer.allocateDirect(sequence).order(ByteOrder.nativeOrder())
        audioMask.forEach { maskBytes.put(if (it) 1 else 0) }
        maskBytes.rewind()
        val maskTensor = OnnxTensor.createTensor(
            env, maskBytes, longArrayOf(1, sequence.toLong()), OnnxJavaType.BOOL,
        )

        val closeables = mutableListOf<OnnxTensor>(idsTensor, maskTensor)
        try {
            val embeds = emb.run(mapOf("input_ids" to idsTensor, "audio_mask" to maskTensor))
            embeds.use { embedded ->
                // The graphs disagree on precision, and the *shipped* pairing is
                // exactly the one that disagrees: an fp16 audio embedding table
                // feeding an int4 backbone whose activations are fp32. Adapt to
                // whatever each declares rather than assuming they match — an
                // unadapted feed is rejected outright, which at least fails
                // loudly.
                val hidden = adapt(env, embedded.pick("inputs_embeds"), llmEmbedType)
                    .also(closeables::add)
                // A full square mask, every position visible to every other.
                //
                // The genai-built graph took a 2-D mask and reduced it to
                // `seqlens_k` for fused GroupQueryAttention, which is causal and
                // accepts no arbitrary mask — the reason it could only buzz. The
                // corrected export takes the 4-D boolean mask that Transformers
                // passes through verbatim, which is what makes the attention
                // bidirectional. Supplying it is not optional: it is the fix.
                val squareMask = if (fourDimensionalMask) {
                    val cells = ByteBuffer.allocateDirect(sequence * sequence)
                        .order(ByteOrder.nativeOrder())
                    repeat(sequence * sequence) { cells.put(1) }
                    cells.rewind()
                    OnnxTensor.createTensor(
                        env, cells,
                        longArrayOf(1, 1, sequence.toLong(), sequence.toLong()),
                        OnnxJavaType.BOOL,
                    )
                } else {
                    OnnxTensor.createTensor(
                        env, LongBuffer.wrap(attention), longArrayOf(1, sequence.toLong()),
                    )
                }
                closeables += squareMask

                val inputs = HashMap<String, OnnxTensor>(pastNames.size + 2)
                inputs["inputs_embeds"] = hidden
                inputs["attention_mask"] = squareMask
                // The corrected export has no cache inputs — unmasking runs a
                // full-sequence forward every step and never reuses one.
                pastNames.forEach { name ->
                    val empty = emptyPast(env, llmEmbedType)
                    closeables += empty
                    inputs[name] = empty
                }

                llm.run(inputs).use { decoded ->
                    val states = adapt(env, decoded.pick("hidden_states"), headsInputType)
                        .also(closeables::add)
                    head.run(mapOf("hidden_states" to states)).use { scored ->
                        val logits = scored.pick("logits")
                        if (!describedShapes) {
                            describedShapes = true
                            android.util.Log.i(
                                TAG,
                                "seq=$sequence embeds=${hidden.info.shape.toList()}/${hidden.info.type} " +
                                    "hidden=${states.info.shape.toList()} " +
                                    "logits=${logits.info.shape.toList()}/${logits.info.type} " +
                                    "expected logits=[1, $CODEBOOKS, $sequence, $AUDIO_VOCAB]",
                            )
                        }
                        val values = readFloats(logits)
                        // Three stages, three lines, once. Each of them can hand
                        // the next one finite numbers that are not the signal —
                        // an embedding table read at the wrong precision, a
                        // backbone whose hidden states have collapsed, a heads
                        // graph scoring an attention cache it was handed by
                        // mistake — and none of the three throws when it does.
                        if (trace) {
                            android.util.Log.i(
                                TAG,
                                "pass seq=$sequence embeds ${readFloats(hidden).signalSummary()}",
                            )
                            android.util.Log.i(
                                TAG,
                                "pass seq=$sequence hidden ${readFloats(states).signalSummary()}",
                            )
                            android.util.Log.i(
                                TAG,
                                "pass seq=$sequence logits ${values.signalSummary()}",
                            )
                        }
                        return values
                    }
                }
            }
        } finally {
            closeables.forEach { runCatching { it.close() } }
        }
    }

    /**
     * Return [tensor] in [target] precision, reusing it untouched when it
     * already matches. The caller closes whatever comes back; when nothing
     * changed that is the original, which the surrounding `use` also closes —
     * harmless, since closing an ORT tensor twice is a no-op.
     */
    private fun adapt(env: OrtEnvironment, tensor: OnnxTensor, target: OnnxJavaType): OnnxTensor {
        if (tensor.info.type == target) return tensor
        val shape = tensor.info.shape
        return when (target) {
            OnnxJavaType.FLOAT -> {
                val floats = readFloats(tensor)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), shape)
            }
            OnnxJavaType.FLOAT16 -> {
                val floats = readFloats(tensor)
                val shorts = java.nio.ShortBuffer.allocate(floats.size)
                floats.forEach { shorts.put(floatToHalf(it)) }
                shorts.rewind()
                OnnxTensor.createTensor(env, shorts, shape, OnnxJavaType.FLOAT16)
            }
            else -> tensor
        }
    }

    private fun emptyPast(env: OrtEnvironment, type: OnnxJavaType): OnnxTensor {
        val shape = longArrayOf(1, KV_HEADS, 0, HEAD_DIM)
        return if (type == OnnxJavaType.FLOAT16) {
            OnnxTensor.createTensor(env, java.nio.ShortBuffer.allocate(0), shape, OnnxJavaType.FLOAT16)
        } else {
            OnnxTensor.createTensor(env, FloatBuffer.allocate(0), shape)
        }
    }

    private fun readFloats(tensor: OnnxTensor): FloatArray =
        if (tensor.info.type == OnnxJavaType.FLOAT16) {
            val shorts = tensor.shortBuffer
            FloatArray(shorts.remaining()) { halfToFloat(shorts.get(it)) }
        } else {
            val floats = tensor.floatBuffer
            FloatArray(floats.remaining()).also(floats::get)
        }

    private fun floatToHalf(value: Float): Short {
        val bits = java.lang.Float.floatToIntBits(value)
        val sign = (bits ushr 16) and 0x8000
        var exponent = ((bits ushr 23) and 0xFF) - 127 + 15
        var mantissa = bits and 0x7FFFFF
        return when {
            exponent >= 0x1F -> (sign or 0x7C00).toShort()          // overflow to infinity
            exponent <= 0 -> {
                if (exponent < -10) return sign.toShort()            // underflow to zero
                mantissa = mantissa or 0x800000
                val shift = 14 - exponent
                (sign or (mantissa ushr shift)).toShort()
            }
            else -> (sign or (exponent shl 10) or (mantissa ushr 13)).toShort()
        }
    }

    /** Codes to 24 kHz mono. The vocoder wants `[codebooks, batch, frames]`. */
    private fun decodeToWaveform(codes: Array<LongArray>, frames: Int): FloatArray {
        val env = environment ?: OrtEnvironment.getEnvironment()
        val voc = vocoder!!

        val flat = LongArray(CODEBOOKS * frames)
        for (cb in 0 until CODEBOOKS) codes[cb].copyInto(flat, cb * frames)
        val tensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(flat), longArrayOf(CODEBOOKS.toLong(), 1, frames.toLong()),
        )
        return try {
            voc.run(mapOf(voc.inputNames.first() to tensor)).use { result ->
                readAudio(result.pick("waveform_24k")).also { waveform ->
                    android.util.Log.i(
                        TAG,
                        "vocoder frames=$frames ${waveform.signalSummary()} " +
                            "(${"%.2f".format(waveform.size.toFloat() / SAMPLE_RATE)}s)",
                    )
                }
            }
        } finally {
            runCatching { tensor.close() }
        }
    }

    /**
     * The named output, or the first one if the graph does not use that name.
     *
     * Every one of these graphs was read positionally — `result[0]` — and for
     * `llm_decoder` that is a coin flip it was losing. The graph returns
     * `hidden_states` *and* a `present.N.key`/`present.N.value` pair for all
     * twenty-eight layers; `genai_config.json` lists only the cache tensors. If
     * a cache tensor comes first, the heads decoder was being handed twenty-odd
     * megabytes of attention memory and asked to score it as if it were the
     * backbone's output. It scores it perfectly happily, and every code that
     * falls out is noise.
     *
     * The repository's own `inference.py` asks by name at all three stages —
     * `["hidden_states"]`, `["logits"]`, `["waveform_24k"]` — which is the
     * detail this was translated past. Falling back to the first output keeps a
     * differently-named export working rather than failing outright.
     */
    private fun OrtSession.Result.pick(name: String): OnnxTensor =
        (get(name).orElse(null) ?: get(0)) as OnnxTensor

    /**
     * The vocoder is exported at fp16, so its output is half-precision. ORT's
     * Java binding hands that back as shorts; widening is ours to do. The fp32
     * branch exists because the same class should keep working if a full
     * precision build is installed instead.
     */
    private fun readAudio(tensor: OnnxTensor): FloatArray =
        if (tensor.info.type == OnnxJavaType.FLOAT16) {
            val shorts = tensor.shortBuffer
            FloatArray(shorts.remaining()) { halfToFloat(shorts.get(it)) }
        } else {
            val floats = tensor.floatBuffer
            FloatArray(floats.remaining()).also(floats::get)
        }

    private fun halfToFloat(half: Short): Float {
        val bits = half.toInt() and 0xFFFF
        val sign = bits ushr 15
        val exponent = (bits ushr 10) and 0x1F
        val mantissa = bits and 0x3FF
        val value = when (exponent) {
            0 -> if (mantissa == 0) 0f else mantissa * TWO_POW_NEG_24
            0x1F -> if (mantissa == 0) Float.POSITIVE_INFINITY else Float.NaN
            else -> Math.scalb((1024 + mantissa).toFloat(), exponent - 25)
        }
        return if (sign == 1) -value else value
    }

    /**
     * Refuse a waveform that is not a waveform.
     *
     * The backstop behind [vocoderProducesAudio], for the overflow a short probe
     * does not reach. The same refusal Kokoro makes and for the same reason:
     * [trimTail] reads a peak, `abs(NaN) > peak` is false, so an all-NaN buffer
     * looks exactly like a quiet one and passes straight through to a WAV file
     * the app reports as saved.
     */
    private fun checkFinite(samples: FloatArray) {
        val nonFinite = samples.count { !it.isFinite() }
        if (nonFinite == 0) return
        error(
            fp16Overflow(
                "the vocoder returned $nonFinite non-finite samples out of ${samples.size}",
            ),
        )
    }

    /**
     * Trim the tail only.
     *
     * The grid is deliberately over-allocated, so the end of a clip is usually
     * silence or the model idling. The *start* is left alone: cutting there
     * would clip an opening plosive, and there is nothing to gain since
     * generation begins at the first frame.
     */
    private fun trimTail(samples: FloatArray): FloatArray {
        if (samples.isEmpty()) return samples
        var peak = 0f
        for (sample in samples) {
            val magnitude = kotlin.math.abs(sample)
            if (magnitude > peak) peak = magnitude
        }
        if (peak <= 0f) return samples
        val floor = peak * SILENCE_FRACTION
        var end = samples.size
        while (end > 0 && kotlin.math.abs(samples[end - 1]) < floor) end--
        if (end == 0) return samples
        val pad = (SAMPLE_RATE / 10).coerceAtMost(samples.size - end) // keep 100 ms
        return samples.copyOfRange(0, end + pad)
    }

    private fun typeOf(session: OrtSession, input: String): OnnxJavaType =
        (session.inputInfo[input]?.info as? ai.onnxruntime.TensorInfo)?.type ?: OnnxJavaType.FLOAT

    /**
     * Where a graph came from and what it weighs, counting its external weights.
     *
     * The `.onnx` file of a graph with external data is a few kilobytes of
     * structure, so its own size says nothing about precision — the 87 MB beside
     * `audio_embeddings_encoder.onnx` is the part that distinguishes an int4
     * export from an fp16 one. The path is relative to the model directory
     * because which *folder* a graph came from is the whole question when the
     * publisher ships one name at several precisions.
     */
    private fun describeFile(directory: File, file: File): String {
        val data = File(file.parentFile, "${file.name}.data")
        val bytes = file.length() + if (data.isFile) data.length() else 0L
        val relative = file.absolutePath.removePrefix(directory.absolutePath).trimStart('/', '\\')
        return "$relative (${bytes / 1024 / 1024} MB${if (data.isFile) " incl. .data" else ""})"
    }

    /**
     * Locate a graph, preferring the *least* quantised copy available.
     *
     * The publisher ships the same four graphs at several precisions in sibling
     * folders, and for two of them the choice is not a trade-off but a
     * correctness question. `audio_embeddings_encoder` holds the audio codebook
     * embedding table and `audio_heads_decoder` produces codebook logits;
     * quantising those to int4 does not degrade the voice, it destroys it —
     * measured on the same sentence, the int4 pair produced steady noise
     * (envelope variation 0.16, every frame above threshold) while the fp16
     * pair produced speech (0.68, 72%). The backbone itself survives int4
     * happily, which is why the largest file is not the one that matters.
     *
     * So: same name, biggest file wins.
     */
    private fun find(directory: File, name: String): File? =
        directory.walkTopDown()
            .filter { it.isFile && it.name == name }
            .maxByOrNull { candidate ->
                // ONNX keeps weights in a sibling .data file, so the graph's own
                // size says nothing about its precision.
                candidate.length() + File(candidate.parentFile, "$name.data").let {
                    if (it.isFile) it.length() else 0L
                }
            }

    companion object {
        private const val TAG = "OmniVoice"

        /**
         * The parameter keys this engine reads. See [KokoroEngine.PARAM_KEYS]
         * for why these are declared rather than enumerated; the reader is the
         * same one.
         */
        val PARAM_KEYS = setOf(
            "voice_design", "lang_code", "speed", "steps", "guidance_scale", "frames",
            "trim_silence", "volume",
            // The rest of upstream's generation config. They were absent while
            // the loop was a translation of `inference.py`, which has no
            // equivalent of any of them; the ported loop reads all five.
            "t_shift", "layer_penalty", "position_temperature", "class_temperature", "seed",
        )

        const val SAMPLE_RATE = 24_000

        /**
         * 960 samples per frame — 25 frames a second.
         *
         * Worth stating because the manifest's `downsample_factor: 320` invites
         * the wrong answer. The reference implementation prints
         * `frames * 960 / 24000` seconds and the vocoder agrees: 375 frames in,
         * 360 000 samples out. Reading 320 as the hop makes every duration and
         * every speed estimate wrong by a factor of three.
         */
        const val SAMPLES_PER_FRAME = 960
        const val FRAMES_PER_SECOND = SAMPLE_RATE / SAMPLES_PER_FRAME

        const val CODEBOOKS = 8
        const val CODEBOOK_SIZE = 1024
        const val MASK_ID = 1024
        const val AUDIO_VOCAB = CODEBOOK_SIZE + 1

        /**
         * Upstream's `OmniVoiceGenerationConfig` defaults, verbatim.
         *
         * Named here rather than written into the manifest so the engine is the
         * one source: a screen that wants to show a default asks for it, the
         * same way the native runtimes report theirs.
         */
        const val DEFAULT_STEPS = 32
        const val DEFAULT_GUIDANCE = 2.0f
        const val DEFAULT_T_SHIFT = 0.1f
        const val DEFAULT_LAYER_PENALTY = 5.0f
        const val DEFAULT_POSITION_TEMPERATURE = 5.0f
        const val DEFAULT_CLASS_TEMPERATURE = 0.0f

        /** Long enough for the decoder's dilated stack to reach its full depth. */
        private const val PROBE_FRAMES = 8

        private const val MAX_STEPS = 256
        /** Below this the schedule's denominator collapses towards zero. */
        private const val MIN_T_SHIFT = 0.01f
        /** `_filter_top_k`'s ratio: the top tenth of the vocabulary. */
        private const val TOP_K_RATIO = 0.1f

        private const val KV_HEADS = 8L
        private const val HEAD_DIM = 128L
        private const val SILENCE_FRACTION = 0.02f
        private const val TWO_POW_NEG_24 = 5.9604645e-8f

        /**
         * The encode half of the audio tokenizer, in the order they chain:
         * waveform to acoustic features, waveform to semantic features, both to
         * codes. Not in [REQUIRED] — an install without them still speaks.
         */
        private val CLONING = listOf(
            "acoustic_encoder.onnx", "semantic_encoder.onnx", "quantizer_encoder.onnx",
        )

        /** What the semantic encoder wants, where everything else is at 24 kHz. */
        const val SEMANTIC_SAMPLE_RATE = 16_000

        /**
         * Below this, upstream lifts a reference clip to a working level and
         * puts the generated audio back down to match afterwards. Doing only
         * the first half makes every clone of a quiet voice come out loud.
         */
        private const val QUIET_RMS = 0.1f

        private const val EMBEDDINGS = "audio_embeddings_encoder.onnx"
        private const val DECODER = "llm_decoder.onnx"
        private const val HEADS = "audio_heads_decoder.onnx"
        private const val VOCODER = "higgs_decoder.onnx"
        private val REQUIRED = listOf(EMBEDDINGS, DECODER, HEADS, VOCODER)

        private val ONNX_AVAILABLE: Boolean = runCatching {
            Class.forName("ai.onnxruntime.OrtEnvironment")
            true
        }.getOrDefault(false)

        /**
         * How long to make the grid.
         *
         * Nothing in the model predicts duration, so this is a guess and it is
         * biased long on purpose: an over-long grid ends in silence that
         * [trimTail] removes, while an under-long one truncates mid-word and
         * cannot be repaired. Roughly fourteen characters of English per second
         * of speech, divided by the speed the user asked for, plus a quarter
         * again for headroom.
         */
        fun estimateFrames(text: String, speed: Float): Int {
            val seconds = (text.length / CHARS_PER_SECOND) / speed.coerceIn(0.5f, 2f)
            val withHeadroom = seconds * 1.25f + 0.5f
            return (withHeadroom * FRAMES_PER_SECOND).toInt().coerceIn(MIN_FRAMES, MAX_FRAMES)
        }

        private const val CHARS_PER_SECOND = 14f
        private const val MIN_FRAMES = 50      // 2 s
        private const val MAX_FRAMES = 750     // 30 s, and already minutes of compute
    }
}

data class OmniVoiceRequest(
    val text: String,
    val speed: Float = 1.0f,
    /** Override the estimated grid length, in 40 ms frames. */
    val frames: Int? = null,
    val steps: Int = OmniVoiceEngine.DEFAULT_STEPS,
    val trimSilence: Boolean = true,
    /** Goes in `<|lang_start|>…<|lang_end|>`; null means the literal "None". */
    val language: String? = null,
    /** Voice design: a written description of how it should sound. */
    val instruction: String? = null,
    /**
     * How hard to push away from what the model would say with no text.
     *
     * Zero switches the unconditional branch off entirely and halves the
     * synthesis time, which is the only large speed dial this model has. It also
     * costs most of the quality, so it is a trade rather than a free win.
     */
    val guidance: Float = OmniVoiceEngine.DEFAULT_GUIDANCE,
    /** Shape of the commit schedule; 1.0 is linear, lower is back-loaded. */
    val timestepShift: Float = OmniVoiceEngine.DEFAULT_T_SHIFT,
    /** How strongly later codebooks are held back until earlier ones are set. */
    val layerPenalty: Float = OmniVoiceEngine.DEFAULT_LAYER_PENALTY,
    /** Randomness in *which slot* is committed next. */
    val positionTemperature: Float = OmniVoiceEngine.DEFAULT_POSITION_TEMPERATURE,
    /** Randomness in *which code* a slot gets. Zero takes the most likely. */
    val classTemperature: Float = OmniVoiceEngine.DEFAULT_CLASS_TEMPERATURE,
    /** Zero picks a fresh one per run; anything else makes the run repeatable. */
    val seed: Long = 0L,
    /** A voice to copy, or null for the one the instruction describes. */
    val reference: VoiceReference? = null,
)

/**
 * A recording to copy the voice from, and what was said in it.
 *
 * The transcript is not optional in the way it looks. Cloning is continuation:
 * the model is given the reference's words *and* its sound, and asked to keep
 * going. Without the words it has audio it cannot account for, and the
 * generated speech drifts to fit whatever it assumes was said. Three to ten
 * seconds is upstream's recommendation; beyond twenty it warns that quality
 * falls off as well as speed.
 */
data class VoiceReference(
    /** Mono, any rate — resampled to the model's 24 kHz on the way in. */
    val samples: FloatArray,
    val sampleRate: Int,
    val transcript: String?,
) {
    // FloatArray gives data classes reference equality, which for a few seconds
    // of audio is both what we want and cheap. Spelled out because the generated
    // versions would compare arrays by identity silently.
    override fun equals(other: Any?): Boolean =
        this === other || (other is VoiceReference &&
            samples === other.samples &&
            sampleRate == other.sampleRate &&
            transcript == other.transcript)

    override fun hashCode(): Int =
        (System.identityHashCode(samples) * 31 + sampleRate) * 31 + (transcript?.hashCode() ?: 0)
}
