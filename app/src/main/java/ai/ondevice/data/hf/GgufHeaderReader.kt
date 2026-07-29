package ai.ondevice.data.hf

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * A real GGUF header parser, used when Hugging Face has not parsed a repo's
 * metadata for us.
 *
 * SPEC §3.1 is explicit that this is a *maintained fallback, not a stub*: the
 * HF `gguf` block is undocumented and may change shape at any time, and if it
 * does the app must keep resolving models rather than degrading to guesswork.
 * Both paths feed the same [GgufMetadata], so the rest of the app cannot tell
 * which one produced it.
 *
 * Format: magic "GGUF", u32 version, u64 tensor_count, u64 metadata_kv_count,
 * then that many key/value pairs. Little-endian throughout.
 */
object GgufHeaderReader {

    private const val MAGIC = 0x46554747 // "GGUF" little-endian

    /** How much of the file to pull with a Range request. 1 MB per SPEC §3.1. */
    const val HEADER_BYTES = 1_048_576

    private const val T_UINT8 = 0
    private const val T_INT8 = 1
    private const val T_UINT16 = 2
    private const val T_INT16 = 3
    private const val T_UINT32 = 4
    private const val T_INT32 = 5
    private const val T_FLOAT32 = 6
    private const val T_BOOL = 7
    private const val T_STRING = 8
    private const val T_ARRAY = 9
    private const val T_UINT64 = 10
    private const val T_INT64 = 11
    private const val T_FLOAT64 = 12

    fun parse(bytes: ByteArray): Result<GgufMetadata> = runCatching {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(buf.remaining() >= 24) { "Not enough bytes for a GGUF header" }

        val magic = buf.int
        require(magic == MAGIC) { "Not a GGUF file (bad magic)" }
        val version = buf.int
        require(version in 1..3) { "Unsupported GGUF version $version" }

        val tensorCount = buf.long
        val kvCount = buf.long
        require(kvCount in 0..100_000) { "Implausible metadata count $kvCount" }

        val kv = LinkedHashMap<String, Any?>()
        for (i in 0 until kvCount) {
            if (buf.remaining() < 12) break // truncated slice: keep what we have
            val key = readString(buf) ?: break
            val type = buf.int
            val value = readValue(buf, type) ?: break
            kv[key] = value
        }
        GgufMetadata(version = version, tensorCount = tensorCount, kv = kv)
    }

    fun parse(stream: InputStream): Result<GgufMetadata> = runCatching {
        val buffer = ByteArray(HEADER_BYTES)
        var read = 0
        while (read < HEADER_BYTES) {
            val n = stream.read(buffer, read, HEADER_BYTES - read)
            if (n <= 0) break
            read += n
        }
        parse(buffer.copyOf(read)).getOrThrow()
    }

    private fun readString(buf: ByteBuffer): String? {
        if (buf.remaining() < 8) return null
        val len = buf.long
        if (len < 0 || len > buf.remaining()) return null
        val out = ByteArray(len.toInt())
        buf.get(out)
        return String(out, Charsets.UTF_8)
    }

    private fun readValue(buf: ByteBuffer, type: Int): Any? = when (type) {
        T_UINT8, T_INT8 -> if (buf.remaining() >= 1) buf.get().toInt() else null
        T_UINT16, T_INT16 -> if (buf.remaining() >= 2) buf.short.toInt() else null
        T_UINT32, T_INT32 -> if (buf.remaining() >= 4) buf.int else null
        T_FLOAT32 -> if (buf.remaining() >= 4) buf.float else null
        T_BOOL -> if (buf.remaining() >= 1) buf.get().toInt() != 0 else null
        T_STRING -> readString(buf)
        T_UINT64, T_INT64 -> if (buf.remaining() >= 8) buf.long else null
        T_FLOAT64 -> if (buf.remaining() >= 8) buf.double else null
        T_ARRAY -> readArray(buf)
        else -> null
    }

    private fun readArray(buf: ByteBuffer): List<Any?>? {
        if (buf.remaining() < 12) return null
        val elementType = buf.int
        val count = buf.long
        if (count < 0) return null
        // Token vocabularies run to hundreds of thousands of strings and are of
        // no use here; skipping them keeps a 1 MB slice sufficient.
        if (count > 65_536) return emptyList()
        val out = ArrayList<Any?>(count.toInt().coerceAtMost(1024))
        for (i in 0 until count) {
            out.add(readValue(buf, elementType) ?: return out)
        }
        return out
    }
}

/**
 * Normalised GGUF metadata. Keys are architecture-prefixed upstream
 * (`qwen3.block_count`, `llama.block_count`, …), so the accessors resolve the
 * architecture first and then look under it — which is why nothing here needs
 * a per-model branch.
 */
data class GgufMetadata(
    val version: Int,
    val tensorCount: Long,
    val kv: Map<String, Any?>,
) {
    val architecture: String? get() = kv["general.architecture"] as? String

    private fun archKey(suffix: String): Any? = architecture?.let { kv["$it.$suffix"] }

    val contextLength: Int? get() = (archKey("context_length") as? Number)?.toInt()
    val blockCount: Int? get() = (archKey("block_count") as? Number)?.toInt()
    val embeddingLength: Int? get() = (archKey("embedding_length") as? Number)?.toInt()
    val headCount: Int? get() = (archKey("attention.head_count") as? Number)?.toInt()
    val headCountKv: Int? get() = (archKey("attention.head_count_kv") as? Number)?.toInt()
    val ropeFreqBase: Float? get() = (archKey("rope.freq_base") as? Number)?.toFloat()

    val chatTemplate: String? get() = kv["tokenizer.chat_template"] as? String
    val bosTokenId: Int? get() = (kv["tokenizer.ggml.bos_token_id"] as? Number)?.toInt()
    val eosTokenId: Int? get() = (kv["tokenizer.ggml.eos_token_id"] as? Number)?.toInt()
    val quantVersion: Int? get() = (kv["general.quantization_version"] as? Number)?.toInt()
    val name: String? get() = kv["general.name"] as? String
    val paramCount: Long? get() = (kv["general.parameter_count"] as? Number)?.toLong()

    /**
     * `n_embd_kv` — the per-token KV width, which the cache estimate multiplies
     * by. With grouped-query attention this is `head_dim × n_head_kv`, not
     * `n_embd`; getting that wrong overstates the cache by the GQA ratio, which
     * for a modern 4B is a factor of four.
     */
    val embeddingLengthKv: Int?
        get() {
            val embd = embeddingLength ?: return null
            val heads = headCount ?: return embd
            val kvHeads = headCountKv ?: heads
            if (heads == 0) return embd
            return (embd / heads) * kvHeads
        }
}
