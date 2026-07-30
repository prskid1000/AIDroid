package ai.ondevice.speech

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Kokoro-82M, actually running.
 *
 * The pipeline has three stages and this class owns the last two:
 *
 *  1. text → IPA, by espeak-ng ([Phonemizer]).
 *  2. IPA → token ids, through Kokoro's fixed 115-symbol vocabulary.
 *  3. ids + a style vector → a 24 kHz waveform, by the ONNX graph.
 *
 * The style vector is the part that is easy to get wrong. A Kokoro voice pack
 * is not one embedding but **510 of them**, one per possible input length, and
 * the model expects the row matching the token count it is about to be given.
 * Using row 0 for everything — which is what happens if you read the file as a
 * flat 256-float vector — produces audio that is recognisably the right voice
 * and subtly wrong in its pacing for every utterance but the shortest.
 */
class KokoroEngine(private val phonemizer: Phonemizer) {

    private val mutex = Mutex()
    private val json = Json { ignoreUnknownKeys = true }

    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputNames: InputNames? = null

    @Volatile
    private var loadedPath: String? = null

    @Volatile
    private var vocabulary: Map<Char, Long> = DEFAULT_VOCABULARY

    @Volatile
    var lastError: String? = null
        private set

    val isLoaded: Boolean get() = session != null

    /**
     * Whether this device could speak with Kokoro at all: the ONNX runtime has
     * to be present *and* the phonemiser, because a voice with no front end is
     * a model that can only be handed silence.
     */
    val runtimeAvailable: Boolean
        get() = ONNX_AVAILABLE && phonemizer.available

    val unavailableReason: String?
        get() = when {
            !ONNX_AVAILABLE -> "The ONNX Runtime is not installed in this build."
            !phonemizer.available -> phonemizer.unavailableReason
            else -> null
        }

    /**
     * Load a model directory.
     *
     * [directory] is what the downloader produced: an `.onnx` graph, a `voices/`
     * folder of `.bin` packs, and optionally the repo's `tokenizer.json`.
     */
    suspend fun load(directory: File, threads: Int = 0): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            check(ONNX_AVAILABLE) { unavailableReason!! }

            val model = findModel(directory)
                ?: error(
                    "No .onnx file in ${directory.name}. A Kokoro install needs the graph " +
                        "itself, not only its voice packs.",
                )

            mutex.withLock {
                if (loadedPath == model.absolutePath) return@withLock

                unlockedUnload()
                val env = OrtEnvironment.getEnvironment()
                val options = OrtSession.SessionOptions().apply {
                    // 0 lets ORT pick; anything else is the user's thread budget.
                    if (threads > 0) setIntraOpNumThreads(threads)
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

                    // Kokoro's text encoder is ALBERT, and ALBERT's whole idea
                    // is that every layer *shares* one set of weights. In this
                    // graph a single fp16 initialiser —
                    // encoder.bert.encoder.albert_layer_groups.0.albert_layers
                    //   .0.full_layer_layer_norm.weight
                    // — is read by 12 SkipLayerNormalization nodes at once.
                    //
                    // ORT pre-packs constant initialisers into a kernel-private
                    // layout at session creation and then drops the original
                    // tensor, which is sound when one kernel owns it. With a
                    // shared initialiser the first node packs it away and every
                    // later node asking for the same input gets a null back:
                    //
                    //   Non-zero status code returned while running
                    //   SkipLayerNormalization node ... Missing Input: <that name>
                    //
                    // It reads like a truncated download and is not one. The
                    // file is byte-identical to the repo, sha256-verified, and
                    // the initialiser is present in the graph — it is released
                    // out from under the node that needs it. Turning pre-packing
                    // off costs a little matmul throughput and is the difference
                    // between Kokoro speaking and not.
                    addConfigEntry("session.disable_prepacking", "1")
                }
                val created = env.createSession(model.absolutePath, options)

                // Bind by *name*, not position. Upstream has shipped these
                // inputs in more than one order across exports, and a silently
                // transposed style/speed pair produces audio rather than an
                // error — the worst kind of mismatch to debug.
                val names = InputNames.from(created.inputNames)
                    ?: error(
                        "This ONNX graph does not look like Kokoro. It takes " +
                            "${created.inputNames.joinToString()}, and Kokoro takes token ids, " +
                            "a style vector and a speed.",
                    )

                environment = env
                session = created
                inputNames = names
                loadedPath = model.absolutePath
                vocabulary = readVocabulary(directory) ?: DEFAULT_VOCABULARY
            }
        }.onFailure { lastError = it.message }
    }

    suspend fun unload() = mutex.withLock { unlockedUnload() }

    private fun unlockedUnload() {
        runCatching { session?.close() }
        session = null
        inputNames = null
        loadedPath = null
    }

    /**
     * Speak [text] in [voiceId], returning 24 kHz mono samples in [-1, 1].
     *
     * Long passages are split rather than truncated. Kokoro's positional
     * encoding stops at 510 tokens and a longer input does not fail — it comes
     * back clipped mid-word, which is the sort of quiet data loss §1.2 rules
     * out. So the text is broken at sentence boundaries, each piece is
     * synthesised on its own, and the pieces are joined with a short silence.
     */
    suspend fun synthesize(request: KokoroRequest): Result<KokoroAudio> =
        withContext(Dispatchers.Default) {
            runCatching {
                val active = session ?: error("No Kokoro model is loaded.")
                val names = inputNames!!

                val chunks = splitForContext(request.text, request)
                check(chunks.isNotEmpty()) { "There is nothing to say." }

                val pieces = mutableListOf<FloatArray>()
                val phonemesUsed = StringBuilder()

                mutex.withLock {
                    chunks.forEach { chunk ->
                        currentCoroutineContext().ensureActive()
                        if (phonemesUsed.isNotEmpty()) phonemesUsed.append(' ')
                        phonemesUsed.append(chunk.phonemes)
                        // Per chunk, not per request: the style row is chosen by
                        // the length of the tokens about to be fed in, and the
                        // chunks deliberately differ in length.
                        val style = styleFor(request, chunk.tokens.size)
                        val piece = runGraph(active, names, chunk.tokens, style, request.speed)
                        // Trimmed per chunk rather than once at the end: the
                        // silence that matters is the padding *between* two
                        // separately-synthesised sentences, and trimming the
                        // join afterwards cannot reach it.
                        pieces += if (request.trimSilence) trimSilence(piece) else piece
                    }
                }

                KokoroAudio(
                    samples = amplify(join(pieces), request.volume),
                    sampleRate = SAMPLE_RATE,
                    phonemes = phonemesUsed.toString(),
                    chunks = chunks.size,
                )
            }.onFailure { lastError = it.message }
        }

    // — the graph —

    private fun runGraph(
        active: OrtSession,
        names: InputNames,
        tokens: LongArray,
        style: FloatArray,
        speed: Float,
    ): FloatArray {
        val env = environment ?: OrtEnvironment.getEnvironment()

        // The model is trained with a leading and trailing pad, and the style
        // row is chosen by the *unpadded* length — see the class comment.
        val padded = LongArray(tokens.size + 2)
        tokens.copyInto(padded, destinationOffset = 1)

        val idsTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(padded),
            longArrayOf(1, padded.size.toLong()),
        )
        val styleTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(style),
            longArrayOf(1, style.size.toLong()),
        )
        val speedTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(floatArrayOf(speed)),
            longArrayOf(1),
        )

        return try {
            active.run(
                mapOf(
                    names.ids to idsTensor,
                    names.style to styleTensor,
                    names.speed to speedTensor,
                ),
            ).use { results ->
                flatten(results[0].value)
            }
        } finally {
            idsTensor.close()
            styleTensor.close()
            speedTensor.close()
        }
    }

    /** ORT hands back `float[n]` or `float[1][n]` depending on the export. */
    private fun flatten(value: Any?): FloatArray = when (value) {
        is FloatArray -> value
        is Array<*> -> {
            @Suppress("UNCHECKED_CAST")
            val rows = value as Array<FloatArray>
            if (rows.size == 1) rows[0] else rows.reduce { a, b -> a + b }
        }
        else -> error("The model returned ${value?.javaClass?.simpleName ?: "nothing"} where a waveform was expected.")
    }

    // — voices —

    /**
     * The style row for this request, blended if a second voice is set.
     *
     * Blending is a genuine Kokoro capability rather than a UI flourish: the
     * style space is linear enough that interpolating two packs gives a voice
     * between them. Both packs are indexed at the *same* row so the blend is
     * between two voices at one length, not two lengths of one voice.
     */
    private suspend fun styleFor(request: KokoroRequest, tokenCount: Int): FloatArray {
        val row = tokenCount.coerceIn(0, STYLE_ROWS - 1)
        val primary = readStyleRow(request.voicePack, row)
        val blendPack = request.blendPack ?: return primary
        val secondary = readStyleRow(blendPack, row)
        val ratio = request.blendRatio.coerceIn(0f, 1f)
        return FloatArray(primary.size) { i -> primary[i] * (1f - ratio) + secondary[i] * ratio }
    }

    private suspend fun readStyleRow(pack: File, row: Int): FloatArray = withContext(Dispatchers.IO) {
        val offset = row.toLong() * STYLE_DIMENSIONS * Float.SIZE_BYTES
        val bytes = ByteArray(STYLE_DIMENSIONS * Float.SIZE_BYTES)
        java.io.RandomAccessFile(pack, "r").use { file ->
            require(file.length() >= offset + bytes.size) {
                "${pack.name} is ${file.length()} bytes; a Kokoro voice pack is " +
                    "${STYLE_ROWS * STYLE_DIMENSIONS * Float.SIZE_BYTES}."
            }
            file.seek(offset)
            file.readFully(bytes)
        }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        FloatArray(STYLE_DIMENSIONS).also(buffer::get)
    }

    // — tokenisation —

    /**
     * Phonemise and split so no piece exceeds the model's 510-token limit.
     *
     * The split is on sentence boundaries first, because that is where a seam
     * between two separately-synthesised pieces is inaudible. Only a single
     * sentence that is *itself* too long falls back to splitting on any space,
     * which is audible but still better than truncation.
     */
    private suspend fun splitForContext(text: String, request: KokoroRequest): List<Chunk> {
        val sentences = text
            .split(splitRegex(request.splitPattern))
            .map(String::trim)
            .filter(String::isNotEmpty)
            .ifEmpty { listOf(text.trim()) }

        val out = mutableListOf<Chunk>()
        val pending = StringBuilder()

        suspend fun flush() {
            if (pending.isEmpty()) return
            out += chunkOf(pending.toString(), request)
            pending.clear()
        }

        for (sentence in sentences) {
            val candidate = if (pending.isEmpty()) sentence else "$pending $sentence"
            val tokens = tokenize(phonemize(candidate, request))
            if (tokens.size <= MAX_TOKENS) {
                pending.clear()
                pending.append(candidate)
                continue
            }
            flush()
            val alone = tokenize(phonemize(sentence, request))
            if (alone.size <= MAX_TOKENS) {
                pending.append(sentence)
            } else {
                out += splitLongSentence(sentence, request)
            }
        }
        flush()
        return out
    }

    /**
     * The user's pattern, or the default if theirs does not compile.
     *
     * `split_pattern` is a free-text Expert parameter, so a half-typed regex is
     * an ordinary state to be in rather than an exceptional one. Falling back
     * keeps the passage speakable; throwing would turn a typo in a field the
     * user may not even remember setting into "Kokoro could not speak that".
     */
    private fun splitRegex(pattern: String): Regex =
        runCatching { Regex(pattern) }.getOrElse {
            lastError = "Chunk pattern is not a valid regular expression; using the default."
            Regex(KokoroRequest.DEFAULT_SPLIT_PATTERN)
        }

    private suspend fun splitLongSentence(sentence: String, request: KokoroRequest): List<Chunk> {
        val out = mutableListOf<Chunk>()
        val pending = StringBuilder()
        for (word in sentence.split(' ').filter(String::isNotEmpty)) {
            val candidate = if (pending.isEmpty()) word else "$pending $word"
            if (tokenize(phonemize(candidate, request)).size <= MAX_TOKENS) {
                pending.clear()
                pending.append(candidate)
            } else {
                if (pending.isNotEmpty()) out += chunkOf(pending.toString(), request)
                pending.clear()
                pending.append(word)
            }
        }
        if (pending.isNotEmpty()) out += chunkOf(pending.toString(), request)
        return out
    }

    private suspend fun chunkOf(text: String, request: KokoroRequest): Chunk {
        val phonemes = phonemize(text, request)
        return Chunk(phonemes = phonemes, tokens = tokenize(phonemes))
    }

    private suspend fun phonemize(text: String, request: KokoroRequest): String =
        phonemizer.phonemize(text, request.voiceId, request.languageOverride).getOrThrow()

    // — output shaping —

    /**
     * Drop leading and trailing near-silence.
     *
     * The threshold is relative to the chunk's own peak rather than absolute,
     * because a quiet voice at 1.0 gain and a loud one are both legitimate and
     * a fixed floor would eat the start of the first. Everything between the
     * first and last loud-enough sample is kept, including internal pauses —
     * those are the sentence's own rhythm, not padding.
     */
    private fun trimSilence(samples: FloatArray): FloatArray {
        if (samples.isEmpty()) return samples
        var peak = 0f
        for (sample in samples) {
            val magnitude = kotlin.math.abs(sample)
            if (magnitude > peak) peak = magnitude
        }
        if (peak <= 0f) return FloatArray(0)
        val floor = peak * SILENCE_FRACTION

        var start = 0
        while (start < samples.size && kotlin.math.abs(samples[start]) < floor) start++
        var end = samples.size
        while (end > start && kotlin.math.abs(samples[end - 1]) < floor) end--
        if (start >= end) return FloatArray(0)

        // Leave a few milliseconds either side. Cutting exactly at the first
        // sample above the floor clips the attack of a plosive and makes the
        // word start with a click.
        val pad = SAMPLE_RATE / 200 // 5 ms
        val from = (start - pad).coerceAtLeast(0)
        val to = (end + pad).coerceAtMost(samples.size)
        return samples.copyOfRange(from, to)
    }

    /** Gain, hard-limited. Above 1.0 the user asked for clipping; give them clipping, not wrap-around. */
    private fun amplify(samples: FloatArray, volume: Float): FloatArray {
        if (volume == 1.0f) return samples
        val gain = volume.coerceIn(0f, 2f)
        for (i in samples.indices) {
            samples[i] = (samples[i] * gain).coerceIn(-1f, 1f)
        }
        return samples
    }

    /**
     * IPA to ids, one symbol at a time.
     *
     * Symbols outside the vocabulary are dropped. That is not silent damage:
     * espeak is configured for a language Kokoro was trained on, so anything it
     * emits that the model has no id for is a diacritic the model never saw,
     * and passing an unknown id would be worse than omitting it.
     */
    private fun tokenize(phonemes: String): LongArray =
        phonemes.mapNotNull { vocabulary[it] }.toLongArray()

    private fun readVocabulary(directory: File): Map<Char, Long>? = runCatching {
        val file = File(directory, "tokenizer.json").takeIf { it.isFile } ?: return null
        val vocab = json.parseToJsonElement(file.readText())
            .jsonObject["model"]!!.jsonObject["vocab"]!!.jsonObject
        vocab.entries
            .mapNotNull { (symbol, id) ->
                symbol.singleOrNull()?.let { it to id.jsonPrimitive.content.toLong() }
            }
            .toMap()
            .takeIf { it.isNotEmpty() }
    }.getOrNull()

    /**
     * Whether a directory holds a Kokoro install: one graph and at least one
     * style pack.
     *
     * Tested by shape, not by name. A pack is exactly [STYLE_ROWS] ×
     * [STYLE_DIMENSIONS] floats, which is a far more specific claim than "there
     * is a `.bin` in here somewhere" — and specificity is the point, because the
     * library can hold two ONNX voice models at once and each engine has to be
     * able to recognise its own without either of them being identified by the
     * repository it came from.
     */
    fun looksInstalled(directory: File): Boolean =
        findModel(directory) != null && voicePacks(directory).isNotEmpty()

    /** The style packs in [directory], identified by their exact byte length. */
    fun voicePacks(directory: File): List<File> =
        directory.walkTopDown().filter { it.isFile && it.length() == PACK_BYTES }.toList()

    private fun findModel(directory: File): File? =
        directory.walkTopDown()
            .filter { it.isFile && it.extension.equals("onnx", ignoreCase = true) }
            // Prefer the smallest graph present: a repo often carries several
            // quantisations and the small one is the one a phone should run.
            .minByOrNull { it.length() }

    private fun join(pieces: List<FloatArray>): FloatArray {
        if (pieces.size == 1) return pieces.first()
        val gap = SAMPLE_RATE / 20 // 50 ms between sentences
        val total = pieces.sumOf { it.size } + gap * (pieces.size - 1)
        val out = FloatArray(total)
        var at = 0
        pieces.forEachIndexed { index, piece ->
            piece.copyInto(out, at)
            at += piece.size + if (index == pieces.lastIndex) 0 else gap
        }
        return out
    }

    private data class Chunk(val phonemes: String, val tokens: LongArray)

    /** Which of the graph's inputs is which, resolved once at load. */
    private data class InputNames(val ids: String, val style: String, val speed: String) {
        companion object {
            fun from(names: Set<String>): InputNames? {
                val ids = names.firstOrNull { it.contains("input", true) || it.contains("token", true) }
                val style = names.firstOrNull { it.contains("style", true) || it.contains("ref", true) }
                val speed = names.firstOrNull { it.contains("speed", true) }
                return if (ids != null && style != null && speed != null) {
                    InputNames(ids, style, speed)
                } else {
                    null
                }
            }
        }
    }

    companion object {
        const val SAMPLE_RATE = 24_000
        const val STYLE_ROWS = 510
        const val STYLE_DIMENSIONS = 256

        /** The model's positional limit, less the two pad tokens. */
        const val MAX_TOKENS = STYLE_ROWS - 2

        /** A voice pack's exact size, which is how one is recognised on disk. */
        const val PACK_BYTES: Long = STYLE_ROWS.toLong() * STYLE_DIMENSIONS * Float.SIZE_BYTES

        /** What counts as silence, as a fraction of the chunk's own peak. */
        const val SILENCE_FRACTION = 0.02f

        /** Whether the ORT classes are actually in the APK. */
        private val ONNX_AVAILABLE: Boolean = runCatching {
            Class.forName("ai.onnxruntime.OrtEnvironment")
            true
        }.getOrDefault(false)

        /**
         * Kokoro v1.0's symbol table.
         *
         * This is part of the *model*, not a choice this app makes, and it is
         * embedded so a hand-placed `model.onnx` works without its repo's
         * `tokenizer.json` alongside. When that file is present it wins.
         */
        private val DEFAULT_VOCABULARY: Map<Char, Long> = buildMap {
            fun put(symbols: String, from: Long) =
                symbols.forEachIndexed { index, c -> put(c, from + index) }

            put("$", 0)
            put(";:,.!?", 1)
            put("—…\"()“” ", 9)
            put("̃ʣʥʦʨᵝꭧ", 17)
            put("AI", 24)
            put('O', 31); put('Q', 33); put('S', 35); put('T', 36)
            put('W', 39); put('Y', 41); put('ᵊ', 42)
            put("abcdef", 43)
            put("hijklmnopqrstuvwxyz", 50)
            put("ɑɐɒæ", 69)
            put('β', 75); put("ɔɕç", 76)
            put("ɖðʤə", 80); put("ɚɛɜ", 85)
            put('ɟ', 90); put('ɡ', 92); put('ɥ', 99)
            put("ɨɪʝ", 101)
            put("ɯɰŋɳɲɴø", 110)
            put('ɸ', 118); put("θœ", 119)
            put('ɹ', 123); put("ɾɻ", 125)
            put('ʁ', 128); put("ɽʂʃʈʧ", 129)
            put("ʊʋ", 135); put("ʌɣɤ", 138)
            put("χʎ", 142); put("ʒʔ", 147)
            put("ˈˌː", 156)
            put('ʰ', 162); put('ʲ', 164)
            put('↓', 169); put("→", 171); put("↗↘", 172)
            put('ᵻ', 177)
        }
    }
}

data class KokoroRequest(
    val text: String,
    val voiceId: String,
    val voicePack: File,
    val speed: Float = 1.0f,
    val blendPack: File? = null,
    val blendRatio: Float = 0f,
    /**
     * Where long text is cut. Kokoro stops at 510 tokens and returns clipped
     * audio rather than an error, so this is what keeps a long passage whole.
     */
    val splitPattern: String = DEFAULT_SPLIT_PATTERN,
    /** Trim near-silence from each chunk before the pieces are joined. */
    val trimSilence: Boolean = true,
    /** Gain on the finished waveform. Hard-limited, so >1 clips rather than wraps. */
    val volume: Float = 1.0f,
    /**
     * Force a particular espeak voice instead of deriving it from [voiceId].
     * Null means derive — which is right almost always, and wrong exactly when
     * you want an American-trained voice reading British spellings.
     */
    val languageOverride: String? = null,
) {
    companion object {
        const val DEFAULT_SPLIT_PATTERN = "(?<=[.!?…])\\s+"
    }
}

data class KokoroAudio(
    val samples: FloatArray,
    val sampleRate: Int,
    val phonemes: String,
    val chunks: Int,
) {
    val durationSeconds: Float get() = samples.size.toFloat() / sampleRate
}
