package ai.ondevice.data.hf

/**
 * Whether this build's ONNX Runtime can open a graph, decided before it is
 * downloaded.
 *
 * The question is worth asking early because an ONNX graph and its weights are
 * separate files. `int4/audio_embeddings_encoder.onnx` is 2 363 bytes against a
 * sidecar of 87 MB, so the part that says which operators are used costs
 * nothing to fetch, and the part that costs 750 MB says nothing about whether
 * it will load.
 *
 * This reads the serialised proto as bytes rather than parsing it. Operator
 * types and attribute names are plain length-delimited strings in there, and
 * the only question being asked is whether a name appears at all — which needs
 * no schema and cannot be broken by one changing. A parser would be the right
 * tool for reading the graph; this is not reading the graph.
 */
object OnnxGraphProbe {

    /**
     * A graph file worth probing is one whose weights live elsewhere. Anything
     * larger is self-contained — Kokoro's 86 MB `model_q8f16.onnx` is the graph
     * *and* every weight — and downloading it to find out whether it loads is
     * the same as downloading it.
     */
    const val MAX_GRAPH_BYTES: Long = 8L * 1024 * 1024

    /**
     * Operators no released ONNX Runtime can run, and why.
     *
     * `GatherBlockQuantized` itself is fine and has been for several releases.
     * What is not is its `bits` attribute, which says the table is packed at
     * something other than the width the released kernels assume. It exists in
     * ORT's operator spec on main and in no shipped version — 1.22, the newest
     * Android build published, has `block_size`, `gather_axis` and
     * `quantize_axis` and nothing else. So the op alone is not the signal; the
     * op *together with* that attribute is.
     */
    private const val QUANTISED_GATHER = "GatherBlockQuantized"

    /**
     * Protobuf writes a string as a length byte then the bytes, so an attribute
     * literally named `bits` appears as 0x04 'b' 'i' 't' 's'. Matching that
     * rather than the bare word keeps a tensor called something like
     * `logits_bits` from reading as an attribute.
     */
    private val BITS_FIELD = byteArrayOf(0x04, 'b'.code.toByte(), 'i'.code.toByte(), 't'.code.toByte(), 's'.code.toByte())

    /** Null when nothing known-unsupported is present. */
    fun blockedReason(graph: ByteArray): String? {
        if (!contains(graph, QUANTISED_GATHER.toByteArray(Charsets.US_ASCII))) return null
        if (!contains(graph, BITS_FIELD)) return null
        return "packs its weights at a bit width no released ONNX Runtime reads yet " +
            "(GatherBlockQuantized with a `bits` attribute)"
    }

    private fun contains(haystack: ByteArray, needle: ByteArray): Boolean {
        if (needle.isEmpty() || haystack.size < needle.size) return false
        outer@ for (start in 0..haystack.size - needle.size) {
            for (i in needle.indices) {
                if (haystack[start + i] != needle[i]) continue@outer
            }
            return true
        }
        return false
    }
}
