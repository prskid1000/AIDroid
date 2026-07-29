package ai.ondevice.data.hf

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement

/**
 * The Hugging Face API surface verified in SPEC §3.1.
 *
 * Note the caveat that comes with it: the `gguf` block is effectively
 * undocumented and may change shape without notice. Every field here is
 * therefore optional, and [GgufHeaderReader] exists as a maintained fallback
 * rather than a stub — SPEC §3.1 and risk table §15.
 */

@Serializable
data class HfModelInfo(
    @SerialName("_id") val internalId: String? = null,
    val id: String? = null,
    val modelId: String? = null,
    val author: String? = null,
    val sha: String? = null,
    val lastModified: String? = null,
    val gated: JsonElement? = null,
    val private: Boolean = false,
    val downloads: Long = 0,
    val likes: Long = 0,
    @SerialName("pipeline_tag") val pipelineTag: String? = null,
    val tags: List<String> = emptyList(),
    val siblings: List<HfSibling> = emptyList(),
    val gguf: HfGgufBlock? = null,
    val config: JsonElement? = null,
    val cardData: JsonElement? = null,
) {
    /**
     * `gated` comes back as `false`, `"auto"` or `"manual"` — a bare boolean
     * check would read `"auto"` as untruthy and let a gated repo through.
     */
    val isGated: Boolean
        get() {
            val p = gated as? kotlinx.serialization.json.JsonPrimitive ?: return false
            return if (p.isString) p.content != "false" else p.content.toBooleanStrictOrNull() == true
        }
}

@Serializable
data class HfSibling(
    val rfilename: String,
    val size: Long? = null,
)

/** The parsed GGUF metadata HF exposes for GGUF repos. */
@Serializable
data class HfGgufBlock(
    val architecture: String? = null,
    @SerialName("context_length") val contextLength: Int? = null,
    @SerialName("chat_template") val chatTemplate: String? = null,
    @SerialName("bos_token") val bosToken: String? = null,
    @SerialName("eos_token") val eosToken: String? = null,
    /** Parameter count, e.g. 7_615_616_512 → size class. */
    val total: Long? = null,
    @SerialName("totalFileSize") val totalFileSize: Long? = null,
    @SerialName("block_count") val blockCount: Int? = null,
    @SerialName("embedding_length") val embeddingLength: Int? = null,
    @SerialName("head_count") val headCount: Int? = null,
    @SerialName("head_count_kv") val headCountKv: Int? = null,
)

/** `POST /api/models/{id}/paths-info/{revision}` with `{"paths":[…],"expand":true}`. */
@Serializable
data class HfPathInfo(
    val path: String,
    val size: Long = 0,
    val type: String? = null,
    val oid: String? = null,
    val lfs: HfLfs? = null,
    val lastCommit: HfCommit? = null,
    val securityFileStatus: HfSecurityStatus? = null,
) {
    /** The sha256 used for post-download integrity verification. */
    val sha256: String? get() = lfs?.oid
}

@Serializable
data class HfLfs(
    /** This *is* the sha256, despite the name. */
    val oid: String? = null,
    val size: Long? = null,
    val pointerSize: Long? = null,
)

@Serializable
data class HfCommit(
    val id: String? = null,
    val title: String? = null,
    val date: String? = null,
)

/**
 * Malware-scan verdicts from JFrog / ProtectAI. SPEC §3.1 is specific that an
 * `unscanned` verdict warrants a *warning*, not a block — GGUF has had SSTI
 * vulnerabilities, so the user is told and allowed to proceed.
 */
@Serializable
data class HfSecurityStatus(
    val status: String? = null,
    val avScan: HfScanResult? = null,
    val pickleImportScan: HfScanResult? = null,
) {
    val isSafe: Boolean get() = status.equals("safe", ignoreCase = true)
    val isUnscanned: Boolean get() = status == null || status.equals("unscanned", ignoreCase = true)
}

@Serializable
data class HfScanResult(
    val status: String? = null,
    val message: String? = null,
    val pickleImports: List<JsonElement> = emptyList(),
)

@Serializable
data class HfSearchResult(
    val id: String,
    val author: String? = null,
    val downloads: Long = 0,
    val likes: Long = 0,
    val tags: List<String> = emptyList(),
    @SerialName("pipeline_tag") val pipelineTag: String? = null,
    val gated: JsonElement? = null,
)
