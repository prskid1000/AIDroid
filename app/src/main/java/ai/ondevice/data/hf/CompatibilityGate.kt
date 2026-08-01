package ai.ondevice.data.hf

import ai.ondevice.core.Fmt
import ai.ondevice.core.Verdict

/** What `flash_attn` is set to, which is three states and not two. */
enum class FlashAttention {
    AUTO,
    ON,
    OFF,
    ;

    companion object {
        /** From a parameter value, where absent means the runtime decides. */
        fun of(value: Boolean?): FlashAttention = when (value) {
            null -> AUTO
            true -> ON
            false -> OFF
        }
    }
}

/** The compatibility gate: SPEC §3.3, and the reason §1.2 ("honest refusal over silent failure") is enforceable rather than aspirational. */
object CompatibilityGate {

    const val DEFAULT_MICRO_BATCH = 512

    /** Whether a cache type is one ggml calls quantized, as `ggml_is_quantized` does. */
    fun isQuantizedCache(type: String): Boolean =
        type.lowercase() !in setOf("f32", "f16", "bf16")

    /** Bytes per element for each KV cache type. */
    fun cacheTypeBytes(type: String): Float = when (type.lowercase()) {
        "f32" -> 4f
        "f16", "bf16" -> 2f
        "q8_0" -> 1.0625f // 8 bits + a scale per 32-element block
        "q5_1" -> 0.75f
        "q5_0" -> 0.6875f
        "q4_1" -> 0.5625f
        "q4_0" -> 0.53125f
        else -> 2f
    }

    /** `2 × n_layer × n_ctx × n_embd_kv × bytes_per_elem(cache_type)`. */
    fun kvCacheBytes(
        layers: Int,
        contextTokens: Int,
        embeddingLengthKv: Int,
        cacheTypeK: String = "f16",
        cacheTypeV: String = cacheTypeK,
    ): Long {
        val perElement = (cacheTypeBytes(cacheTypeK) + cacheTypeBytes(cacheTypeV)) / 2f
        return (2.0 * layers * contextTokens * embeddingLengthKv * perElement).toLong()
    }

    /** Compute buffer: the graph's intermediates, which flash attention changes by more than any other single setting. */
    fun computeBufferBytes(
        nBatch: Int,
        embeddingLength: Int,
        contextTokens: Int = 0,
        heads: Int? = null,
        microBatch: Int = DEFAULT_MICRO_BATCH,
        flashAttention: Boolean = true,
    ): Long {
        val base = (nBatch.toLong() * embeddingLength * 4L * 6L)
            .coerceAtLeast(128L * 1024 * 1024)
        if (flashAttention || heads == null || contextTokens <= 0) return base
        val scores = microBatch.toLong() * contextTokens * heads * 4L
        return base + scores
    }

    /** The full resident estimate. */
    fun estimate(
        weightsBytes: Long,
        layers: Int?,
        contextTokens: Int,
        embeddingLengthKv: Int?,
        embeddingLength: Int?,
        nBatch: Int = 2048,
        cacheTypeK: String = "f16",
        cacheTypeV: String = "f16",
        heads: Int? = null,
        flashAttention: FlashAttention = FlashAttention.AUTO,
    ): FitEstimate {
        val kv = if (layers != null && embeddingLengthKv != null) {
            kvCacheBytes(layers, contextTokens, embeddingLengthKv, cacheTypeK, cacheTypeV)
        } else {
            (weightsBytes * 0.25).toLong()
        }
        val compute = computeBufferBytes(
            nBatch = nBatch,
            embeddingLength = embeddingLength ?: 4096,
            contextTokens = contextTokens,
            heads = heads,
            flashAttention = flashAttention != FlashAttention.OFF,
        )
        return FitEstimate(
            weightsBytes = weightsBytes,
            kvCacheBytes = kv,
            computeBufferBytes = compute,
            layers = layers,
            embeddingLengthKv = embeddingLengthKv,
            contextTokens = contextTokens,
            cacheTypeK = cacheTypeK,
            cacheTypeV = cacheTypeV,
            heads = heads,
            flashAttention = flashAttention,
            exact = layers != null && embeddingLengthKv != null &&
                (flashAttention != FlashAttention.OFF || heads != null),
        )
    }

    /** Turn an estimate into one of the five verdicts. */
    fun verdict(
        estimate: FitEstimate,
        availableRamBytes: Long,
        freeStorageBytes: Long,
        storageReserveBytes: Long,
        archSupported: Boolean,
        hasRuntimeForFormat: Boolean,
    ): Verdict = when {
        !hasRuntimeForFormat -> Verdict.NOT_RUNNABLE
        !archSupported -> Verdict.UNSUPPORTED_ARCH
        estimate.totalBytes > availableRamBytes -> Verdict.WONT_FIT
        estimate.weightsBytes + storageReserveBytes > freeStorageBytes -> Verdict.WONT_FIT
        availableRamBytes - estimate.totalBytes < TIGHT_HEADROOM_BYTES -> Verdict.TIGHT
        else -> Verdict.FAST
    }

    /** "fits, but headroom drops under 1 GB" is a warning, not a pass. */
    private const val TIGHT_HEADROOM_BYTES = 1_000_000_000L
}

/** A fit estimate with its terms intact, so the UI can show the working. */
data class FitEstimate(
    val weightsBytes: Long,
    val kvCacheBytes: Long,
    val computeBufferBytes: Long,
    val layers: Int?,
    val embeddingLengthKv: Int?,
    val contextTokens: Int,
    val cacheTypeK: String,
    val exact: Boolean,
    val cacheTypeV: String = cacheTypeK,
    val heads: Int? = null,
    val flashAttention: FlashAttention = FlashAttention.AUTO,
) {
    /** Whether llama.cpp will refuse this combination outright. */
    val quantizedVWithoutFlashAttention: Boolean
        get() = flashAttention == FlashAttention.OFF &&
            CompatibilityGate.isQuantizedCache(cacheTypeV)
    val totalBytes: Long get() = weightsBytes + kvCacheBytes + computeBufferBytes

    fun headroomBytes(availableRam: Long): Long = availableRam - totalBytes

    /** "≈ 5.20 GB at 8K context". */
    /** A KV cache is a property of autoregressive attention over a context. */
    val hasContext: Boolean get() = contextTokens > 0 && kvCacheBytes > 0

    fun summary(): String = if (hasContext) {
        "≈ ${Fmt.gb(totalBytes)} GB at ${Fmt.contextLabel(contextTokens)} context"
    } else {
        "≈ ${Fmt.gb(totalBytes)} GB resident"
    }

    /** "model 2.50 + KV 0.60 + compute 0.25" — the S1 one-liner. */
    fun shortWorking(): String = if (hasContext) {
        "model ${Fmt.gb(weightsBytes)}   +   KV ${Fmt.gb(kvCacheBytes)}   +   compute ${Fmt.gb(computeBufferBytes)}"
    } else {
        "weights ${Fmt.gb(weightsBytes)}   +   working set ${Fmt.gb(computeBufferBytes)}"
    }

    /** The three-line breakdown S2 prints, with the KV line showing the actual multiplication so the user can see where the number comes from. */
    fun longWorking(): List<String> = buildList {
        add("weights ${Fmt.gb(weightsBytes)}")
        if (layers != null && embeddingLengthKv != null) {
            val types = if (cacheTypeK == cacheTypeV) cacheTypeK else "$cacheTypeK/$cacheTypeV"
            add("KV cache 2 × $layers × ${contextTokens} × $embeddingLengthKv × $types = ${Fmt.gb(kvCacheBytes)}")
        } else {
            add("KV cache ${Fmt.gb(kvCacheBytes)} (estimated — architecture metadata unavailable)")
        }
        when {
            flashAttention != FlashAttention.OFF ->
                add("compute buffer ${Fmt.gb(computeBufferBytes)} (flash attention on)")
            heads != null ->
                add(
                    "compute buffer ${Fmt.gb(computeBufferBytes)} — includes attention scores " +
                        "${CompatibilityGate.DEFAULT_MICRO_BATCH} × $contextTokens × $heads × f32, which flash " +
                        "attention would not materialise",
                )
            else ->
                add(
                    "compute buffer ${Fmt.gb(computeBufferBytes)} (understated — flash attention " +
                        "is off and the head count is unknown, so the attention scores are " +
                        "not counted)",
                )
        }
        if (quantizedVWithoutFlashAttention) {
            add(
                "cache_type_v is $cacheTypeV with flash_attn off — llama.cpp refuses this " +
                    "combination and the model will not load",
            )
        }
    }

}
