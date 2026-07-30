package ai.ondevice.data.hf

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject

/**
 * Tensor names, read from the front of a safetensors file.
 *
 * The format puts a little-endian `u64` length first and a JSON object of
 * tensor descriptors immediately after it, so the names are readable from a
 * range request without touching the weights.
 *
 * This exists because a filename is not enough to know what a `.safetensors`
 * file *is*, and the gap is not academic. `madebyollin/taesd` publishes three
 * usable-looking files: `diffusion_pytorch_model.safetensors`, whose tensors are
 * `decoder.layers.*` and `encoder.layers.*` and which is the one
 * stable-diffusion.cpp documents, and `taesd_encoder.safetensors` /
 * `taesd_decoder.safetensors`, whose tensors are bare `nn.Sequential` indices —
 * `0.weight`, `1.conv.0.bias` — matching nothing sd.cpp looks for. Judged by
 * name alone the app offered the two that cannot load and hid the one that can.
 *
 * sd.cpp decides by tensor name, so to agree with it the app has to read the
 * same thing.
 */
object SafetensorsHeaderReader {

    /** Enough for the header of any auxiliary; a full SD VAE's is ~50 KB. */
    const val HEADER_BYTES: Int = 256 * 1024

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    data class Header(val tensorNames: Set<String>) {
        fun hasPrefix(prefix: String): Boolean = tensorNames.any { it.startsWith(prefix) }
    }

    /**
     * Null rather than an exception for every "cannot tell" case — a truncated
     * read, a declared header longer than [HEADER_BYTES], a file that is not
     * safetensors at all. The caller must treat null as *unknown* and fall back
     * to the filename, never as *unloadable*: refusing a file because a range
     * request was short would make a network hiccup look like an incompatible
     * model.
     */
    fun parse(bytes: ByteArray): Header? {
        if (bytes.size < 8) return null
        var length = 0L
        for (i in 7 downTo 0) length = (length shl 8) or (bytes[i].toLong() and 0xFF)
        if (length <= 0 || length > Int.MAX_VALUE) return null
        val end = 8 + length.toInt()
        if (end > bytes.size) return null

        val text = String(bytes, 8, length.toInt(), Charsets.UTF_8)
        val names = runCatching {
            json.parseToJsonElement(text).jsonObject.keys.filterNot { it == "__metadata__" }
        }.getOrNull() ?: return null
        return Header(names.toSet()).takeIf { it.tensorNames.isNotEmpty() }
    }
}
