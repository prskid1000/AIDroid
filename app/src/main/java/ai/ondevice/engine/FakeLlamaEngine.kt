package ai.ondevice.engine

import ai.ondevice.core.BackendId
import ai.ondevice.core.SparseParams
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.onCompletion
import kotlin.random.Random

/**
 * A stand-in for `libllama_jni.so`, sitting behind the real
 * [InferenceEngine] contract.
 *
 * This exists so the app is a complete, running program before the native layer
 * lands — SPEC §14 sequences the build by risk and this is what "the app
 * already standing" looks like at step 3. Everything above it (the chat loop,
 * the streaming Flow, cancellation, the parameter report, the prompt inspector)
 * is the real implementation and does not change when the JNI arrives; only
 * this file is replaced.
 *
 * It deliberately keeps the parts that are easy to get wrong honest:
 *  - Parameters arrive as an opaque string-keyed map and it reports back which
 *    keys it recognised, exactly as the native side must (§16.7).
 *  - Unknown keys are reported, never fatal (§16.6).
 *  - Cancellation runs teardown in `onCompletion`, which is where the native
 *    implementation must free its buffers rather than merely dropping the
 *    callback (Appendix A #7).
 */
class FakeLlamaEngine(
    override val descriptor: RuntimeDescriptor,
) : InferenceEngine {

    private var loaded: LoadedModel? = null
    private var appliedParams: SparseParams = SparseParams.EMPTY

    override val isLoaded: Boolean get() = loaded != null
    override val loadedModelId: String? get() = loaded?.modelId

    override suspend fun load(request: LoadRequest): Result<LoadedModel> = runCatching {
        // Warm-swap: unload the old model before loading the new one. The app
        // never holds two at once (SPEC §3.5).
        if (loaded != null) unload()
        delay(SIMULATED_LOAD_MILLIS)

        val ctx = request.params.int("n_ctx") ?: 8192
        val model = LoadedModel(
            modelId = request.modelId,
            backend = request.backend,
            contextLength = ctx,
            layers = 36,
            embeddingLength = 2560,
            embeddingLengthKv = 1024,
            chatTemplate = request.chatTemplate,
            stopSequences = deriveStops(request.chatTemplate),
            loadMillis = SIMULATED_LOAD_MILLIS,
        )
        loaded = model
        appliedParams = request.params
        model
    }

    override suspend fun unload() {
        loaded = null
        appliedParams = SparseParams.EMPTY
    }

    /**
     * The native side maps keys onto `common_params` and the sampler chain via
     * a generated dispatch table. Here the recognised set is the manifest's own
     * key list; the shape of the answer is what matters.
     */
    override suspend fun applyParams(params: SparseParams): ParamReport {
        appliedParams = appliedParams.overlaidWith(params)
        val applied = mutableListOf<String>()
        val rejected = mutableListOf<String>()
        params.keys.forEach { key ->
            if (key in RECOGNISED_KEYS) applied += key else rejected += key
        }
        return ParamReport(applied = applied, rejected = rejected)
    }

    override fun generate(request: GenerateRequest): Flow<GenerationEvent> = flow {
        val model = loaded ?: run {
            emit(GenerationEvent.Failed("No model is loaded.", "Pick a model in chat settings."))
            return@flow
        }

        val params = appliedParams.overlaidWith(request.params)
        val promptTokens = estimatePromptTokens(request)
        val cached = if (params.bool("cache_prompt") != false) (promptTokens * 0.9).toInt() else 0

        emit(GenerationEvent.PromptProcessed(promptTokens, cached, promptTokensPerSecond(model.backend)))

        val started = System.currentTimeMillis()

        // Reasoning blocks: models with a `<think>` pair get their thinking
        // parsed out and rendered collapsed (SPEC §4.4).
        if (request.messages.lastOrNull()?.content?.length?.let { it > 24 } == true) {
            val thinkStarted = System.currentTimeMillis()
            var thinkTokens = 0
            THINKING_SAMPLE.chunkedWords().forEach { word ->
                currentCoroutineContext().ensureActive()
                delay(tokenDelay(model.backend) / 2)
                thinkTokens++
                emit(GenerationEvent.ThinkingDelta(word))
            }
            emit(GenerationEvent.ThinkingDone(thinkTokens, System.currentTimeMillis() - thinkStarted))
        }

        val maxTokens = params.int("n_predict")?.takeIf { it > 0 } ?: Int.MAX_VALUE
        val reply = replyFor(request)
        var index = 0

        for (word in reply.chunkedWords()) {
            currentCoroutineContext().ensureActive()
            if (index >= maxTokens) {
                emit(GenerationEvent.Done(StopReason.MAX_TOKENS, index, System.currentTimeMillis() - started))
                return@flow
            }
            delay(tokenDelay(model.backend))
            emit(GenerationEvent.Token(word, index))
            index++
            if (index % 8 == 0) {
                val elapsed = (System.currentTimeMillis() - started).coerceAtLeast(1)
                emit(
                    GenerationEvent.Stats(
                        tokensPerSecond = index * 1000f / elapsed,
                        generatedTokens = index,
                        contextUsed = promptTokens + index,
                        backend = model.backend,
                    ),
                )
            }
        }
        emit(GenerationEvent.Done(StopReason.EOS, index, System.currentTimeMillis() - started))
    }.onCompletion {
        // Where the native implementation frees its sampler chain and batch
        // buffers. Detaching the callback is not enough — the memory has to go.
        releaseGenerationResources()
    }

    private fun releaseGenerationResources() = Unit

    override suspend fun renderPrompt(request: GenerateRequest): RenderedPrompt {
        val model = loaded
        val template = model?.chatTemplate
        val text = buildString {
            request.systemPrompt?.let {
                append("<|im_start|>system\n").append(it).append("\n<|im_end|>\n")
            }
            request.messages.forEach { m ->
                append("<|im_start|>").append(m.role).append('\n')
                if (m.imagePaths.isNotEmpty()) {
                    append("<|vision_start|>…${m.imagePaths.size * IMAGE_TOKENS} image tokens…<|vision_end|>")
                }
                append(m.content).append("\n<|im_end|>\n")
            }
            append("<|im_start|>assistant\n")
        }
        val imageTokens = request.messages.sumOf { it.imagePaths.size } * IMAGE_TOKENS
        return RenderedPrompt(
            text = text,
            tokens = tokenize(text),
            template = template,
            templateSource = if (template != null) "from GGUF metadata" else "runtime default",
            totalTokens = estimatePromptTokens(request),
            imageTokens = imageTokens,
            cachedTokens = (estimatePromptTokens(request) * 0.9).toInt(),
            contextLimit = model?.contextLength ?: 8192,
            stopSequences = model?.stopSequences ?: emptyList(),
        )
    }

    override suspend fun tokenCount(text: String): Int = (text.length / 3.6f).toInt().coerceAtLeast(1)

    // — helpers —

    private fun tokenize(text: String): List<PromptToken> {
        val out = mutableListOf<PromptToken>()
        val specialPattern = Regex("""<\|[a-z_]+\|>""")
        var cursor = 0
        specialPattern.findAll(text).forEach { match ->
            if (match.range.first > cursor) {
                text.substring(cursor, match.range.first).chunkedWords().forEach {
                    out += PromptToken(it, out.size, special = false)
                }
            }
            out += PromptToken(match.value, out.size, special = true)
            cursor = match.range.last + 1
        }
        if (cursor < text.length) {
            text.substring(cursor).chunkedWords().forEach { out += PromptToken(it, out.size, special = false) }
        }
        return out
    }

    private fun estimatePromptTokens(request: GenerateRequest): Int {
        val textTokens = request.messages.sumOf { (it.content.length / 3.6f).toInt() } +
            (request.systemPrompt?.length ?: 0) / 4
        val imageTokens = request.messages.sumOf { it.imagePaths.size } * IMAGE_TOKENS
        return textTokens + imageTokens + 24
    }

    private fun deriveStops(template: String?): List<String> = when {
        template == null -> listOf("</s>")
        template.contains("im_end") -> listOf("<|im_end|>", "<|endoftext|>")
        template.contains("end_of_turn") -> listOf("<end_of_turn>")
        else -> listOf("</s>")
    }

    private fun tokenDelay(backend: BackendId): Long = when (backend) {
        BackendId.OPENCL -> 71 // ≈14.1 t/s, the figure the canvas shows
        BackendId.HEXAGON -> 85
        BackendId.CPU -> 120
    }

    private fun promptTokensPerSecond(backend: BackendId): Float = when (backend) {
        BackendId.OPENCL -> 184f
        BackendId.HEXAGON -> 211f
        BackendId.CPU -> 42f
    }

    private fun replyFor(request: GenerateRequest): String {
        val last = request.messages.lastOrNull()?.content.orEmpty()
        return when {
            last.contains("KV", ignoreCase = true) || last.contains("cache", ignoreCase = true) -> KV_REPLY
            last.contains("?") -> GENERIC_QUESTION_REPLY
            else -> GENERIC_REPLY
        }
    }

    private companion object {
        const val SIMULATED_LOAD_MILLIS = 850L
        const val IMAGE_TOKENS = 1456

        val RECOGNISED_KEYS: Set<String> = setOf(
            "n_ctx", "n_batch", "n_ubatch", "n_gpu_layers", "n_threads", "n_threads_batch",
            "use_mmap", "use_mlock", "flash_attn", "cache_type_k", "cache_type_v",
            "no_kv_offload", "defrag_thold", "n_parallel", "rope_freq_base", "rope_freq_scale",
            "rope_scaling_type", "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_fast",
            "yarn_beta_slow", "yarn_orig_ctx", "split_mode", "main_gpu", "check_tensors",
            "pooling_type", "mmproj_use_gpu",
            "temp", "top_k", "top_p", "min_p", "typical_p", "top_n_sigma", "min_keep",
            "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
            "penalize_nl", "dry_multiplier", "dry_base", "dry_allowed_length",
            "dry_penalty_last_n", "dry_sequence_breakers", "xtc_probability", "xtc_threshold",
            "mirostat", "mirostat_tau", "mirostat_eta", "dynatemp_range", "dynatemp_exponent",
            "samplers", "seed", "ignore_eos", "n_probs", "grammar", "json_schema", "logit_bias",
            "n_predict", "stop", "n_keep", "cache_prompt", "context_shift",
        )

        const val THINKING_SAMPLE =
            "The user wants 2 × n_layer × n_ctx × n_embd_kv × bytes. For this model n_layer=36 " +
                "and n_embd_kv=1024 (8 KV heads × 128). At 32768 and f16 that's 4.83 GB, plus " +
                "2.5 GB weights — so around 7.6 GB resident. Free RAM is 10.4 GB, so it fits, " +
                "but headroom drops under 3 GB…"

        const val KV_REPLY =
            "Your arithmetic checks out. At 32K with an f16 cache you're looking at 4.83 GB of KV " +
                "alone — nearly double the weights. Two ways down:\n\nSet cache_type_k and " +
                "cache_type_v to q8_0 and the cache halves to 2.4 GB with almost no quality cost. " +
                "Enabling flash attention on top saves the attention scratch buffer as well, which " +
                "is another few hundred megabytes at this context."

        const val GENERIC_QUESTION_REPLY =
            "Running locally on this device, so the answer is bounded by what's loaded rather than " +
                "by a network round trip. Ask a follow-up and the KV cache is reused — the prompt " +
                "won't be reprocessed from scratch."

        const val GENERIC_REPLY =
            "Loaded and generating on-device. Nothing in this exchange left the handset: the only " +
                "outbound calls this app makes are to the Hugging Face API and the downloads you " +
                "start yourself."
    }
}

/** Split into token-ish chunks that keep their trailing space, as a tokenizer would. */
private fun String.chunkedWords(): List<String> =
    Regex("""\S+\s*|\s+""").findAll(this).map { it.value }.toList()
