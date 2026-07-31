package ai.ondevice.speech

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
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
    private var tokenizer: QwenTokenizer? = null
    private var pastNames: List<String> = emptyList()
    private var describedGraphs = false
    private var describedShapes = false

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
    private var llmEmbedType: OnnxJavaType = OnnxJavaType.FLOAT
    private var headsInputType: OnnxJavaType = OnnxJavaType.FLOAT

    @Volatile
    private var loadedPath: String? = null

    @Volatile
    var lastError: String? = null
        private set

    val isLoaded: Boolean get() = decoder != null

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

                environment = env
                embeddings = emb
                decoder = llm
                this@OmniVoiceEngine.heads = head
                vocoder = voc
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
        android.util.Log.i("OmniVoice", "bidirectional probe: max|diff| before the change = $largest")
        largest > 1e-6f
    }.getOrDefault(true) // Could not ask; do not refuse on a failed measurement.

    suspend fun unload() = mutex.withLock { unlockedUnload() }

    private fun unlockedUnload() {
        listOf(embeddings, decoder, heads, vocoder).forEach { runCatching { it?.close() } }
        embeddings = null
        decoder = null
        heads = null
        vocoder = null
        tokenizer = null
        pastNames = emptyList()
        fourDimensionalMask = false
        loadedPath = null
        bidirectional = null
        describedGraphs = false
        describedShapes = false
    }

    suspend fun synthesize(request: OmniVoiceRequest): Result<KokoroAudio> =
        withContext(Dispatchers.Default) {
            runCatching {
                val tok = tokenizer ?: error("No OmniVoice model is loaded.")
                val text = request.text.trim()
                check(text.isNotEmpty()) { "There is nothing to say." }

                val textTokens = buildPrompt(tok, text, request.language, request.instruction)
                check(textTokens.isNotEmpty()) { "That text produced no tokens." }

                val frames = request.frames ?: estimateFrames(text, request.speed)

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

                val codes = mutex.withLock { unmask(textTokens, frames, request) }
                val samples = mutex.withLock { decodeToWaveform(codes, frames) }

                KokoroAudio(
                    samples = if (request.trimSilence) trimTail(samples) else samples,
                    sampleRate = SAMPLE_RATE,
                    phonemes = "",
                    chunks = 1,
                )
            }.onFailure { lastError = it.message }
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
    ): IntArray {
        val style = buildString {
            append("<|lang_start|>").append(language ?: "None").append("<|lang_end|>")
            append("<|instruct_start|>").append(instruction ?: "None").append("<|instruct_end|>")
        }
        return tok.encodeWithMarkup(style) + tok.encodeWithMarkup("<|text_start|>$text<|text_end|>")
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
        frames: Int,
        request: OmniVoiceRequest,
    ): Array<LongArray> {
        val steps = request.steps.coerceIn(1, MAX_STEPS)
        val textLength = textTokens.size
        val condSequence = textLength + frames
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

        // The conditional branch: text, then the grid. Only the grid half is
        // rewritten between steps, so the text is laid in once.
        val condIds = Array(CODEBOOKS) { LongArray(condSequence) }
        for (cb in 0 until CODEBOOKS) {
            for (i in 0 until textLength) condIds[cb][i] = textTokens[i].toLong()
        }
        val condAudio = BooleanArray(condSequence) { it >= textLength }
        val condAttention = LongArray(condSequence) { 1L }
        // The unconditional branch is the grid alone — not the same sequence
        // with the text masked out, but a shorter one with no text in it.
        val uncondAudio = BooleanArray(frames) { true }
        val uncondAttention = LongArray(frames) { 1L }

        val scores = FloatArray(slots)
        val predicted = IntArray(slots)
        val probabilities = FloatArray(AUDIO_VOCAB)
        val unguided = FloatArray(AUDIO_VOCAB)
        val ranked = FloatArray(AUDIO_VOCAB)

        for (step in 0 until steps) {
            val take = schedule[step]
            if (take <= 0) continue
            currentCoroutineContext().ensureActive()

            for (cb in 0 until CODEBOOKS) tokens[cb].copyInto(condIds[cb], textLength)
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
                        condBase = ((cb.toLong() * condSequence + (textLength + f)) * AUDIO_VOCAB).toInt(),
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
                "OmniVoice",
                "llm outputs=${llm.outputNames.take(4)} (${llm.outputNames.size} total) " +
                    "heads outputs=${head.outputNames} " +
                    "embeddings outputs=${emb.outputNames}",
            )
        }

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
                                "OmniVoice",
                                "seq=$sequence embeds=${hidden.info.shape.toList()}/${hidden.info.type} " +
                                    "hidden=${states.info.shape.toList()} " +
                                    "logits=${logits.info.shape.toList()}/${logits.info.type} " +
                                    "expected logits=[1, $CODEBOOKS, $sequence, $AUDIO_VOCAB]",
                            )
                        }
                        return readFloats(logits)
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
                readAudio(result.pick("waveform_24k"))
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

        private const val MAX_STEPS = 256
        /** Below this the schedule's denominator collapses towards zero. */
        private const val MIN_T_SHIFT = 0.01f
        /** `_filter_top_k`'s ratio: the top tenth of the vocabulary. */
        private const val TOP_K_RATIO = 0.1f

        private const val KV_HEADS = 8L
        private const val HEAD_DIM = 128L
        private const val SILENCE_FRACTION = 0.02f
        private const val TWO_POW_NEG_24 = 5.9604645e-8f

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
)
