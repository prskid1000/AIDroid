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
 * speech and OmniVoice at 2.7–3.5 s — six or seven times slower, and a phone
 * multiplies that again. Kokoro remains the default because most of the time you
 * want a sentence read aloud, not composed.
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
 * frames by iterative unmasking: every position starts as MASK, and across 32
 * steps the most confident positions are committed until none are left. Two
 * consequences run all the way to the UI:
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
     * says so with a `bits` attribute on `com.microsoft.GatherBlockQuantized`.
     * That attribute is on ONNX Runtime's main branch and is in no released
     * version — not 1.22, which is the newest Android build published — so the
     * graph is refused outright with a node dump that reads like a corrupt
     * download. It is not corrupt and no newer runtime we can depend on will
     * read it; the answer is the other export, whose embeddings encoder is
     * 327 MB rather than 87 MB and uses no such op. Everything else in the two
     * is the same file, `llm_decoder` included.
     */
    private fun explainLoadFailure(file: java.io.File, cause: Throwable): String {
        val message = cause.message.orEmpty()
        val unsupportedQuant = "GatherBlockQuantized" in message ||
            ("bits" in message && "Unrecognized attribute" in message)
        return if (unsupportedQuant) {
            "${file.name} is a 4-bit export this ONNX Runtime cannot read: it needs the `bits` " +
                "attribute on GatherBlockQuantized, which no released ONNX Runtime has yet. " +
                "Install OmniVoice's other variant — the one whose audio_embeddings_encoder is " +
                "around 327 MB rather than 87 MB. Nothing is wrong with the download."
        } else {
            "${file.name} could not be loaded. $message"
        }
    }

    suspend fun unload() = mutex.withLock { unlockedUnload() }

    private fun unlockedUnload() {
        listOf(embeddings, decoder, heads, vocoder).forEach { runCatching { it?.close() } }
        embeddings = null
        decoder = null
        heads = null
        vocoder = null
        tokenizer = null
        pastNames = emptyList()
        loadedPath = null
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
                val codes = mutex.withLock { unmask(textTokens, frames, request.steps) }
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
     * `inference.py` — shipped in the ONNX repo, written against these exact
     * exported graphs — does none of it. It calls `tokenizer.encode(text,
     * add_special_tokens=True)` and hands the model bare prose.
     *
     * The wrapped form produced noise on device. So the default is now the
     * reference script's, on the reasoning that the script distributed with an
     * export describes that export better than the model code it was converted
     * from: conversion is where a training-time framing gets baked into the
     * graph, and if it were baked in, applying it again outside would double it.
     *
     * The wrapper survives for the case it is the only expression of — a
     * language or a written voice description. There is no other way to say
     * those, and upstream's form is the only documented one. That path is
     * unverified; if it also comes back as noise, the wrapper is simply wrong
     * for this export and `lang_code`/`voice_design` cannot be honoured by it.
     *
     * `<|denoise|>` stays absent: upstream emits it only with a reference clip.
     */
    private fun buildPrompt(
        tok: QwenTokenizer,
        text: String,
        language: String?,
        instruction: String?,
    ): IntArray {
        val styled = !language.isNullOrBlank() || !instruction.isNullOrBlank()
        if (!styled) return tok.encode(text)

        val style = buildString {
            append("<|lang_start|>").append(language ?: "None").append("<|lang_end|>")
            append("<|instruct_start|>").append(instruction ?: "None").append("<|instruct_end|>")
        }
        return tok.encodeWithMarkup(style) + tok.encodeWithMarkup("<|text_start|>$text<|text_end|>")
    }

    // — the 32-step loop —

    /**
     * Confidence-ordered iterative unmasking.
     *
     * Each step scores every still-masked frame by how sure the model is of its
     * best code, weighted across the eight codebooks — the earlier codebooks
     * carry most of the signal, hence the 8,8,6,6,4,4,2,2 weighting — and
     * commits the most confident ones.
     *
     * The number committed per step is ceil(remaining / stepsLeft) rather than a
     * fixed fraction. That matters: with a fixed fraction a handful of frames can
     * still be masked when the steps run out, and MASK (1024) is outside the
     * codec's 0..1023 range, so the vocoder does not degrade gracefully — it
     * fails. Dividing by the steps remaining guarantees the grid is full.
     */
    private suspend fun unmask(textTokens: IntArray, frames: Int, steps: Int): Array<LongArray> {
        val textLength = textTokens.size
        val sequence = textLength + frames

        // (codebooks, sequence), flattened per codebook for cheap column reads.
        val ids = Array(CODEBOOKS) { LongArray(sequence) }
        for (cb in 0 until CODEBOOKS) {
            for (i in 0 until textLength) ids[cb][i] = textTokens[i].toLong()
            for (i in textLength until sequence) ids[cb][i] = MASK_ID.toLong()
        }

        val audioMask = BooleanArray(sequence) { it >= textLength }
        val attention = LongArray(sequence) { 1L }
        var remaining = frames

        for (step in 0 until steps) {
            if (remaining == 0) break
            currentCoroutineContext().ensureActive()

            val logits = forward(ids, audioMask, attention, sequence)

            // Confidence per generation-region frame.
            val confidence = FloatArray(frames)
            val best = Array(CODEBOOKS) { IntArray(frames) }
            for (cb in 0 until CODEBOOKS) {
                val weight = CODEBOOK_WEIGHTS[cb] / WEIGHT_SUM
                for (f in 0 until frames) {
                    val base = ((cb.toLong() * sequence + (textLength + f)) * AUDIO_VOCAB).toInt()
                    var maxLogit = Float.NEGATIVE_INFINITY
                    var argmax = 0
                    // Real codes only — MASK at 1024 must never be selected.
                    for (v in 0 until CODEBOOK_SIZE) {
                        val value = logits[base + v]
                        if (value > maxLogit) {
                            maxLogit = value
                            argmax = v
                        }
                    }
                    var sum = 0f
                    for (v in 0 until CODEBOOK_SIZE) sum += kotlin.math.exp(logits[base + v] - maxLogit)
                    best[cb][f] = argmax
                    confidence[f] += weight / sum // exp(max - max) == 1, so p_max = 1/sum
                }
            }

            val masked = (0 until frames).filter { ids[0][textLength + it] == MASK_ID.toLong() }
            if (masked.isEmpty()) break

            val stepsLeft = (steps - step).coerceAtLeast(1)
            val take = ((masked.size + stepsLeft - 1) / stepsLeft).coerceAtLeast(1)
            val order = masked.sortedByDescending { confidence[it] }.take(take)
            order.forEach { f ->
                for (cb in 0 until CODEBOOKS) ids[cb][textLength + f] = best[cb][f].toLong()
            }
            remaining -= order.size
        }

        // Safety net: anything still masked is filled greedily from one more
        // pass, because a MASK reaching the vocoder is a hard failure.
        val leftover = (0 until frames).filter { ids[0][textLength + it] == MASK_ID.toLong() }
        if (leftover.isNotEmpty()) {
            val logits = forward(ids, audioMask, attention, sequence)
            leftover.forEach { f ->
                for (cb in 0 until CODEBOOKS) {
                    val base = ((cb.toLong() * sequence + (textLength + f)) * AUDIO_VOCAB).toInt()
                    var maxLogit = Float.NEGATIVE_INFINITY
                    var argmax = 0
                    for (v in 0 until CODEBOOK_SIZE) {
                        if (logits[base + v] > maxLogit) {
                            maxLogit = logits[base + v]
                            argmax = v
                        }
                    }
                    ids[cb][textLength + f] = argmax.toLong()
                }
            }
        }

        return Array(CODEBOOKS) { cb -> ids[cb].copyOfRange(textLength, sequence) }
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
                val hidden = adapt(env, embedded[0] as OnnxTensor, llmEmbedType).also(closeables::add)
                val attentionTensor = OnnxTensor.createTensor(
                    env, LongBuffer.wrap(attention), longArrayOf(1, sequence.toLong()),
                )
                closeables += attentionTensor

                // Empty past for every layer: the graph is exported to run a
                // full-sequence forward, which is what unmasking needs.
                val inputs = HashMap<String, OnnxTensor>(pastNames.size + 2)
                inputs["inputs_embeds"] = hidden
                inputs["attention_mask"] = attentionTensor
                pastNames.forEach { name ->
                    val empty = emptyPast(env, llmEmbedType)
                    closeables += empty
                    inputs[name] = empty
                }

                llm.run(inputs).use { decoded ->
                    val states = adapt(env, decoded[0] as OnnxTensor, headsInputType)
                        .also(closeables::add)
                    head.run(mapOf("hidden_states" to states)).use { scored ->
                        return readFloats(scored[0] as OnnxTensor)
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
                readAudio(result[0] as OnnxTensor)
            }
        } finally {
            runCatching { tensor.close() }
        }
    }

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
            "voice_design", "lang_code", "speed", "steps", "frames",
            "trim_silence", "volume",
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
        const val DEFAULT_STEPS = 32

        private const val KV_HEADS = 8L
        private const val HEAD_DIM = 128L
        private const val SILENCE_FRACTION = 0.02f
        private const val TWO_POW_NEG_24 = 5.9604645e-8f

        private val CODEBOOK_WEIGHTS = floatArrayOf(8f, 8f, 6f, 6f, 4f, 4f, 2f, 2f)
        private val WEIGHT_SUM = CODEBOOK_WEIGHTS.sum()

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
)
