package ai.ondevice.data.hf

import ai.ondevice.core.Fmt
import ai.ondevice.core.SpeedClass
import ai.ondevice.core.Verdict

/**
 * The compatibility gate: SPEC §3.3, and the reason §1.2 ("honest refusal over
 * silent failure") is enforceable rather than aspirational.
 *
 * Two rules from the spec shape this file:
 *  - It runs **before any download**, and no path may reach a native load
 *    without having passed it (Appendix A #5).
 *  - It **shows the arithmetic**. Not a bare yes/no — the model detail screen
 *    prints `weights + KV + compute` and recomputes live as the context slider
 *    moves, because a user who can see the sum can fix the problem themselves.
 */
object CompatibilityGate {

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

    /**
     * `2 × n_layer × n_ctx × n_embd_kv × bytes_per_elem(cache_type)`.
     *
     * The literal formula from SPEC §3.3, and the one the model-detail screen
     * prints back to the user. The leading 2 is K and V.
     */
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

    /**
     * Compute buffer, which scales with the logical batch size rather than the
     * context. Approximated from `n_batch × n_embd × bytes` with headroom for
     * the graph's intermediate tensors; the canvas shows this landing around
     * 0.20–0.25 GB for a 4B at default batch, which this reproduces.
     */
    fun computeBufferBytes(nBatch: Int, embeddingLength: Int): Long =
        (nBatch.toLong() * embeddingLength * 4L * 6L).coerceAtLeast(128L * 1024 * 1024)

    /**
     * The full resident estimate. Every term is returned separately so the UI
     * can print the sum rather than just the total.
     */
    fun estimate(
        weightsBytes: Long,
        layers: Int?,
        contextTokens: Int,
        embeddingLengthKv: Int?,
        embeddingLength: Int?,
        nBatch: Int = 2048,
        cacheTypeK: String = "f16",
        cacheTypeV: String = "f16",
    ): FitEstimate {
        val kv = if (layers != null && embeddingLengthKv != null) {
            kvCacheBytes(layers, contextTokens, embeddingLengthKv, cacheTypeK, cacheTypeV)
        } else {
            // Without the architecture's own numbers, fall back to a coarse
            // proportion of the weights and say so, rather than pretending to
            // an exactness we don't have.
            (weightsBytes * 0.25).toLong()
        }
        val compute = computeBufferBytes(nBatch, embeddingLength ?: 4096)
        return FitEstimate(
            weightsBytes = weightsBytes,
            kvCacheBytes = kv,
            computeBufferBytes = compute,
            layers = layers,
            embeddingLengthKv = embeddingLengthKv,
            contextTokens = contextTokens,
            cacheTypeK = cacheTypeK,
            exact = layers != null && embeddingLengthKv != null,
        )
    }

    /**
     * Turn an estimate into one of the six verdicts.
     *
     * `archSupported` is answered by the runtime registry, which is generated
     * from the pinned upstream source — never a hand-maintained allowlist
     * (SPEC §2.3, Appendix A #3).
     */
    fun verdict(
        estimate: FitEstimate,
        availableRamBytes: Long,
        freeStorageBytes: Long,
        storageReserveBytes: Long,
        archSupported: Boolean,
        hasRuntimeForFormat: Boolean,
        speedClass: SpeedClass,
    ): Verdict = when {
        !hasRuntimeForFormat -> Verdict.NOT_RUNNABLE
        !archSupported -> Verdict.UNSUPPORTED_ARCH
        estimate.totalBytes > availableRamBytes -> Verdict.WONT_FIT
        estimate.weightsBytes + storageReserveBytes > freeStorageBytes -> Verdict.WONT_FIT
        availableRamBytes - estimate.totalBytes < TIGHT_HEADROOM_BYTES -> Verdict.TIGHT
        speedClass == SpeedClass.CPU_PATH -> Verdict.WORKS_SLOWER
        else -> Verdict.FAST
    }

    /**
     * Q4_0 is the only quant with an Adreno OpenCL kernel on this hardware, so
     * everything else falls back to CPU. The canvas states this verbatim under
     * the variant list: "Q4_0 hits the Adreno OpenCL fast path on this device.
     * Other quants fall back to CPU."
     */
    fun speedClassFor(quant: String?): SpeedClass =
        if (quant?.uppercase()?.contains("Q4_0") == true) SpeedClass.OPENCL_FAST else SpeedClass.CPU_PATH

    /** "fits, but headroom drops under 1 GB" is a warning, not a pass. */
    private const val TIGHT_HEADROOM_BYTES = 1_000_000_000L
}

/**
 * A fit estimate with its terms intact, so the UI can show the working.
 *
 * [exact] records whether the KV term came from real architecture metadata or
 * from the coarse fallback — the screens say which, rather than presenting a
 * guess in the same voice as a calculation.
 */
data class FitEstimate(
    val weightsBytes: Long,
    val kvCacheBytes: Long,
    val computeBufferBytes: Long,
    val layers: Int?,
    val embeddingLengthKv: Int?,
    val contextTokens: Int,
    val cacheTypeK: String,
    val exact: Boolean,
) {
    val totalBytes: Long get() = weightsBytes + kvCacheBytes + computeBufferBytes

    fun headroomBytes(availableRam: Long): Long = availableRam - totalBytes

    /** "≈ 5.20 GB at 8K context". */
    fun summary(): String = "≈ ${Fmt.gb(totalBytes)} GB at ${Fmt.contextLabel(contextTokens)} context"

    /** "model 2.50 + KV 0.60 + compute 0.25" — the S1 one-liner. */
    fun shortWorking(): String =
        "model ${Fmt.gb(weightsBytes)}   +   KV ${Fmt.gb(kvCacheBytes)}   +   compute ${Fmt.gb(computeBufferBytes)}"

    /**
     * The three-line breakdown S2 prints, with the KV line showing the actual
     * multiplication so the user can see where the number comes from.
     */
    fun longWorking(): List<String> = buildList {
        add("weights ${Fmt.gb(weightsBytes)}")
        if (exact && layers != null && embeddingLengthKv != null) {
            add("KV cache 2 × $layers × ${contextTokens} × $embeddingLengthKv × $cacheTypeK = ${Fmt.gb(kvCacheBytes)}")
        } else {
            add("KV cache ${Fmt.gb(kvCacheBytes)} (estimated — architecture metadata unavailable)")
        }
        add("compute buffer ${Fmt.gb(computeBufferBytes)}")
    }

    /**
     * SPEC §3.3 also asks whether a Hexagon session can hold this at all — past
     * ~3.5 GB the model needs layer-splitting across HTP0..HTP3 or an OpenCL
     * fallback.
     */
    val exceedsHexagonSession: Boolean
        get() = totalBytes > DeviceCapabilities.HEXAGON_SESSION_CAP_BYTES
}
