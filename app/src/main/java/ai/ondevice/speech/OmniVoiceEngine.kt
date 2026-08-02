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

/** OmniVoice, the second voice. */
class OmniVoiceEngine {

    private val mutex = Mutex()

    private var environment: OrtEnvironment? = null
    private var embeddings: OrtSession? = null
    private var decoder: OrtSession? = null
    private var heads: OrtSession? = null
    private var vocoder: OrtSession? = null

    /** The encode half of the Higgs audio tokenizer, for voice cloning. */
    private var acousticEncoder: OrtSession? = null
    private var semanticEncoder: OrtSession? = null
    private var quantizer: OrtSession? = null
    private var tokenizer: QwenTokenizer? = null
    private var pastNames: List<String> = emptyList()
    private var describedGraphs = false
    private var describedShapes = false

    /** Reset per synthesis; see the trace guard in [forward]. */
    private var tracedForwards = 0

    /** Whether the backbone declares `attention_mask` with four dimensions. */
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

    /** A reference clip, as the eight codebooks the backbone reads. */
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

    /** `[batch, channels, frames]` cut to [frames] and converted to [target]. */
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

    /** Linear resampling. */
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

    /** Whether a directory looks like an OmniVoice install. */
    fun looksInstalled(directory: File): Boolean =
        REQUIRED.all { find(directory, it) != null }

    /**
     * Any one of OmniVoice's graphs is here, so this folder is its.
     *
     * Weaker than [looksInstalled] on purpose: it answers "whose folder is
     * this", not "can this run". Kokoro's half-installed check accepts any
     * folder holding an `.onnx` and no voice packs, which a partial OmniVoice
     * also satisfies — this is what keeps the two from claiming each other.
     */
    fun partlyInstalled(directory: File): Boolean =
        REQUIRED.any { find(directory, it) != null }

    /** Whether a directory also carries the encoders a voice clone needs. */
    fun cloningLooksInstalled(directory: File): Boolean =
        CLONING.all { find(directory, it) != null }

    suspend fun load(directory: File, threads: Int = 0): Result<Unit> = withContext(Dispatchers.IO) {
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
                    }
                }

                val loadedTokenizer = QwenTokenizer.load(tokenizerFile).getOrThrow()
                fun open(file: java.io.File): OrtSession =
                    runCatching { env.createSession(file.absolutePath, options()) }
                        .getOrElse { error(explainLoadFailure(file, it)) }

                val emb = open(files.getValue(EMBEDDINGS))
                val llm = open(files.getValue(DECODER))
                val head = open(files.getValue(HEADS))
                val voc = open(files.getValue(VOCODER))

                // Cloning is all-or-nothing — one encoder short and the quantizer has nothing to pair — so the three are opened together or not at all.
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

    /** Turn a graph-load failure into something the user can act on. */
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

    /** Whether the backbone can see the whole sequence, which this model needs. */
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

    /** Whether the vocoder can turn codes into numbers at all, asked in about a tenth of a second before three minutes are spent filling a grid for it. */
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

                // Lift a quiet reference to a working level before encoding, and remember by how much so the generated audio can be put back down to match.
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

    /** How the text is framed before the model sees it. */
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
        val combined = if (!referenceText.isNullOrBlank()) {
            referenceText.trim() + " " + text.trim()
        } else {
            text.trim()
        }
        return tok.encodeWithMarkup(style) +
            tok.encodeWithMarkup("<|text_start|>$combined<|text_end|>")
    }

    // — the unmasking loop —

    /** Confidence-ordered iterative unmasking, as the model's own `generate()` does it. */
    private suspend fun unmask(
        textTokens: IntArray,
        referenceCodes: Array<LongArray>?,
        frames: Int,
        request: OmniVoiceRequest,
    ): Array<LongArray> {
        val steps = request.steps.coerceIn(1, MAX_STEPS)
        val textLength = textTokens.size
        // A reference clip sits between the text and the grid as real codes the model can see.
        val referenceLength = referenceCodes?.firstOrNull()?.size ?: 0
        val gridStart = textLength + referenceLength
        val condSequence = gridStart + frames
        val slots = CODEBOOKS * frames

        // The grid being filled, [codebook][frame]. Everything starts masked.
        val tokens = Array(CODEBOOKS) { LongArray(frames) { MASK_ID.toLong() } }
        val schedule = buildSchedule(frames, steps, request.timestepShift)
        val random = java.util.Random(
            if (request.seed != 0L) request.seed else System.nanoTime(),
        )

        // The conditional branch: text, the reference if there is one, then the grid.
        val condIds = Array(CODEBOOKS) { LongArray(condSequence) }
        for (cb in 0 until CODEBOOKS) {
            for (i in 0 until textLength) condIds[cb][i] = textTokens[i].toLong()
            referenceCodes?.get(cb)?.copyInto(condIds[cb], textLength)
        }
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

            // What this step actually decided.
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

    /** How many slots to commit at each step. */
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

    /** Guided log-probabilities for one slot, left in [probabilities]; returns the code to commit there. */
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

        // Sample instead of taking the best, from the top tenth of the vocabulary.
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

    /** `omnivoice._gumbel_sample`: divide by the temperature, then add Gumbel noise. */
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

        // Once per load.
        if (!describedGraphs) {
            describedGraphs = true
            android.util.Log.i(
                TAG,
                "llm outputs=${llm.outputNames.take(4)} (${llm.outputNames.size} total) " +
                    "heads outputs=${head.outputNames} " +
                    "embeddings outputs=${emb.outputNames}",
            )
        }
        // The first two passes are the conditional and unconditional branches of step one, which is where a broken graph is already broken.
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
                val hidden = adapt(env, embedded.pick("inputs_embeds"), llmEmbedType)
                    .also(closeables::add)
                // A full square mask, every position visible to every other.
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
                        // Three stages, three lines, once.
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

    /** Return [tensor] in [target] precision, reusing it untouched when it already matches. */
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

    /** The named output, or the first one if the graph does not use that name. */
    private fun OrtSession.Result.pick(name: String): OnnxTensor =
        (get(name).orElse(null) ?: get(0)) as OnnxTensor

    /** The vocoder is exported at fp16, so its output is half-precision. */
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

    /** Refuse a waveform that is not a waveform. */
    private fun checkFinite(samples: FloatArray) {
        val nonFinite = samples.count { !it.isFinite() }
        if (nonFinite == 0) return
        error(
            fp16Overflow(
                "the vocoder returned $nonFinite non-finite samples out of ${samples.size}",
            ),
        )
    }

    /** Trim the tail only. */
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

    /** Where a graph came from and what it weighs, counting its external weights. */
    private fun describeFile(directory: File, file: File): String {
        val data = File(file.parentFile, "${file.name}.data")
        val bytes = file.length() + if (data.isFile) data.length() else 0L
        val relative = file.absolutePath.removePrefix(directory.absolutePath).trimStart('/', '\\')
        return "$relative (${bytes / 1024 / 1024} MB${if (data.isFile) " incl. .data" else ""})"
    }

    /** Locate a graph, preferring the *least* quantised copy available. */
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

        /** The parameter keys this engine reads. */
        val PARAM_KEYS = setOf(
            "voice_design", "lang_code", "speed", "steps", "guidance_scale", "frames",
            "trim_silence", "volume",
            // The rest of upstream's generation config.
            "t_shift", "layer_penalty", "position_temperature", "class_temperature", "seed",
        )

        const val SAMPLE_RATE = 24_000

        /** 960 samples per frame — 25 frames a second. */
        const val SAMPLES_PER_FRAME = 960
        const val FRAMES_PER_SECOND = SAMPLE_RATE / SAMPLES_PER_FRAME

        const val CODEBOOKS = 8
        const val CODEBOOK_SIZE = 1024
        const val MASK_ID = 1024
        const val AUDIO_VOCAB = CODEBOOK_SIZE + 1

        /** Upstream's `OmniVoiceGenerationConfig` defaults, verbatim. */
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

        /** The encode half of the audio tokenizer, in the order they chain: waveform to acoustic features, waveform to semantic features, both to codes. */
        private val CLONING = listOf(
            "acoustic_encoder.onnx", "semantic_encoder.onnx", "quantizer_encoder.onnx",
        )

        /** What the semantic encoder wants, where everything else is at 24 kHz. */
        const val SEMANTIC_SAMPLE_RATE = 16_000

        /** Below this, upstream lifts a reference clip to a working level and puts the generated audio back down to match afterwards. */
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

        /** How long to make the grid. */
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
    val language: String? = null,
    /** Voice design: a written description of how it should sound. */
    val instruction: String? = null,
    /** How hard to push away from what the model would say with no text. */
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

/** A recording to copy the voice from, and what was said in it. */
data class VoiceReference(
    /** Mono, any rate — resampled to the model's 24 kHz on the way in. */
    val samples: FloatArray,
    val sampleRate: Int,
    val transcript: String?,
) {
    // FloatArray gives data classes reference equality, which for a few seconds of audio is both what we want and cheap.
    override fun equals(other: Any?): Boolean =
        this === other || (other is VoiceReference &&
            samples === other.samples &&
            sampleRate == other.sampleRate &&
            transcript == other.transcript)

    override fun hashCode(): Int =
        (System.identityHashCode(samples) * 31 + sampleRate) * 31 + (transcript?.hashCode() ?: 0)
}
