package ai.ondevice.speech

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.ondevice.engine.signalSummary
import android.util.Log
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

/** Kokoro-82M, actually running. */
class KokoroEngine(
    private val context: android.content.Context,
    private val phonemizer: Phonemizer,
) {

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

    val runtimeAvailable: Boolean
        get() = ONNX_AVAILABLE && phonemizer.available

    val unavailableReason: String?
        get() = when {
            !ONNX_AVAILABLE -> "The ONNX Runtime is not installed in this build."
            !phonemizer.available -> phonemizer.unavailableReason
            else -> null
        }

    /** Load a model directory. */
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

                    // Kokoro's text encoder is ALBERT, and ALBERT's whole idea is that every layer *shares* one set of weights.
                    addConfigEntry("session.disable_prepacking", "1")
                }
                val created = env.createSession(model.absolutePath, options)

                // Bind by *name*, not position.
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
                Log.i(
                    TAG,
                    "loaded ${model.name} (${model.length() / 1024 / 1024} MB) " +
                        "inputs=${created.inputNames.joinToString()} " +
                        "vocab=${vocabulary.size} threads=$threads",
                )
            }
        }.onFailure {
            lastError = it.message
            Log.e(TAG, "load failed", it)
        }
    }

    suspend fun unload() = mutex.withLock { unlockedUnload() }

    private fun unlockedUnload() {
        runCatching { session?.close() }
        session = null
        inputNames = null
        loadedPath = null
    }

    /** Speak [text] in [voiceId], returning 24 kHz mono samples in [-1, 1]. */
    suspend fun synthesize(request: KokoroRequest): Result<KokoroAudio> =
        withContext(Dispatchers.Default) {
            runCatching {
                val active = session ?: error("No Kokoro model is loaded.")
                val names = inputNames!!

                val chunks = splitForContext(request.text, request)
                check(chunks.isNotEmpty()) { "There is nothing to say." }

                val pieces = mutableListOf<FloatArray>()
                val phonemesUsed = StringBuilder()

                // A chunk is one graph run, and the chunks deliberately differ in
                // length, so the unit counted is the token rather than the chunk
                // — a target set from one sentence would be wrong for the next
                // one by however much longer it is. No carry-over key: the rate
                // is per token, so a request with more than one chunk learns its
                // own, and a single-chunk request is one graph run with nothing
                // to steer anyway.
                val hints = ai.ondevice.engine.CpuHints.open(context, TAG)
                try {
                    mutex.withLock {
                        chunks.forEach { chunk ->
                            currentCoroutineContext().ensureActive()
                            if (phonemesUsed.isNotEmpty()) phonemesUsed.append(' ')
                            phonemesUsed.append(chunk.phonemes)
                            // Per chunk, not per request: the style row is chosen by the length of the tokens about to be fed in, and the chunks deliberately differ in length.
                            val style = styleFor(request, chunk.tokens.size)
                            val piece = hints.unit(chunk.tokens.size.toLong()) {
                                runGraph(active, names, chunk.tokens, style, request.speed)
                            }
                            checkFinite(piece)
                            val kept = if (request.trimSilence) trimSilence(piece) else piece
                            // Every stage that can silently produce nothing, on one line.
                            Log.i(
                                TAG,
                                "chunk phonemes=${chunk.phonemes.length} tokens=${chunk.tokens.size} " +
                                    "raw=${piece.size} kept=${kept.size} ${piece.signalSummary()}",
                            )
                            pieces += kept
                        }
                    }
                } finally {
                    hints.close()
                }

                val joined = amplify(join(pieces), request.volume)
                Log.i(
                    TAG,
                    "synthesised chunks=${chunks.size} samples=${joined.size} " +
                        "(${"%.2f".format(joined.size.toFloat() / SAMPLE_RATE)}s)",
                )
                KokoroAudio(
                    samples = joined,
                    sampleRate = SAMPLE_RATE,
                    phonemes = phonemesUsed.toString(),
                    chunks = chunks.size,
                )
            }.onFailure {
                lastError = it.message
                Log.e(TAG, "synthesis failed", it)
            }
        }

    /** Refuse a waveform that is not a waveform. */
    private fun checkFinite(samples: FloatArray) {
        val nonFinite = samples.count { !it.isFinite() }
        if (nonFinite == 0) return
        val graph = loadedPath?.substringAfterLast('/') ?: "the loaded graph"
        error(
            "$graph produced $nonFinite non-finite samples out of ${samples.size}: it computed " +
                "its way to NaN rather than to audio. Graphs whose activations are float16 " +
                "overflow on arm64, where the CPU has real fp16 arithmetic and float16 stops at " +
                "65504 — the same file works on an x86 emulator, which has no fp16 kernels and " +
                "quietly computes in fp32. Install a variant without f16 in its name (q8, q4, " +
                "quantized or the full-precision graph) and this voice will speak.",
        )
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

    /** The style row for this request, blended if a second voice is set. */
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

    /** Phonemise and split so no piece exceeds the model's 510-token limit. */
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

    /** The user's pattern, or the default if theirs does not compile. */
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

    /** Drop leading and trailing near-silence. */
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

        // Leave a few milliseconds either side.
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

    /** IPA to ids, one symbol at a time. */
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

    /** Whether a directory holds a Kokoro install: one graph and at least one style pack. */
    fun looksInstalled(directory: File): Boolean =
        findModel(directory) != null && voicePacks(directory).isNotEmpty()

    /**
     * The graph is here and not one speaker vector is.
     *
     * Half-installed, which [looksInstalled] cannot distinguish from absent
     * because it needs both — and the two want opposite advice. This state was
     * reachable for as long as the resolver could not classify the packs in
     * `voices/` at all: the download completed, reported success, and left a
     * folder that cannot speak.
     */
    fun graphOnly(directory: File): Boolean =
        findModel(directory) != null && voicePacks(directory).isEmpty()

    /** The style packs in [directory], identified by their exact byte length. */
    fun voicePacks(directory: File): List<File> =
        directory.walkTopDown().filter { it.isFile && it.length() == PACK_BYTES }.toList()

    /** The graph to load, preferring one this device can actually compute. */
    /** The graph to load: the smallest non-fp16 export in the folder. */
    private fun findModel(directory: File): File? =
        directory.walkTopDown()
            .filter { it.isFile && it.extension.equals("onnx", ignoreCase = true) }
            .sortedWith(compareBy({ if (usesFloat16Activations(it)) 1 else 0 }, { it.length() }))
            .firstOrNull()

    /** Whether a graph's name marks it as float16-activation. */
    private fun usesFloat16Activations(file: File): Boolean =
        Regex("(^|[_-])(fp16|f16)([_-]|\\.)", RegexOption.IGNORE_CASE)
            .containsMatchIn(file.name)

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
        private const val TAG = "KokoroEngine"

        /** The parameter keys this engine reads, for [ai.ondevice.params.EngineParams]. */
        val PARAM_KEYS = setOf(
            "voice", "speed", "lang_code", "voice_blend", "split_pattern",
            "trim_silence", "volume",
        )

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

        /** Kokoro v1.0's symbol table. */
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
    /** Where long text is cut. */
    val splitPattern: String = DEFAULT_SPLIT_PATTERN,
    /** Trim near-silence from each chunk before the pieces are joined. */
    val trimSilence: Boolean = true,
    /** Gain on the finished waveform. Hard-limited, so >1 clips rather than wraps. */
    val volume: Float = 1.0f,
    /** Force a particular espeak voice instead of deriving it from [voiceId]. */
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
