package ai.ondevice.data.hf

/**
 * Which ONNX Runtime execution providers this build does *not* have.
 *
 * Publishers ship a folder per provider — `cuda/` beside `int4/` — and on a
 * phone that is not a choice, it is a download that cannot load. The resolver
 * lists those variants and refuses them rather than dropping them, so the page
 * matches what Hugging Face shows.
 *
 * This used to be eleven names written down here. A name missing from the list
 * was not merely unlabelled, it was offered as an ordinary variant, so the list
 * had to be complete to be correct — and it was only ever as current as the last
 * time somebody thought about it.
 *
 * Now both halves come from the runtime: [OrtProvider] enumerates every provider
 * ONNX Runtime knows, `getAvailableProviders()` reports the ones this AAR
 * registered, and foreign is the difference. Upgrading the runtime updates the
 * answer.
 */
object OnnxProviders {

    /**
     * Directory-name tokens for providers this build cannot execute.
     *
     * Computed once. Empty if ONNX Runtime is not on the classpath at all, which
     * is the safe direction: no variant gets refused on the strength of a
     * question we could not ask.
     */
    val foreign: Set<String> by lazy {
        runCatching {
            val available = ai.onnxruntime.OrtEnvironment.getAvailableProviders()
                .map { token(it.name) }
                .toSet()
            ai.onnxruntime.OrtProvider.values()
                .map { token(it.name) }
                .toSet()
                .minus(available)
                .flatMap { listOf(it) + ALIASES.filterValues { alias -> alias == it }.keys }
                .toSet()
        }.getOrDefault(emptySet())
    }

    fun isForeign(directoryName: String): Boolean = directoryName.lowercase() in foreign

    /**
     * `CUDAExecutionProvider` → `cuda`, which is what publishers name the folder.
     *
     * Enum constant names are used rather than `getName()` because they are the
     * shorter form and already match the convention: `OPEN_VINO` → `openvino`,
     * `TENSOR_RT` → `tensorrt`, `CORE_ML` → `coreml`.
     */
    private fun token(enumName: String): String = enumName.replace("_", "").lowercase()

    /**
     * Folder spellings publishers use that are not the provider's own.
     *
     * Keyed by the folder name, valued by the token [token] produces — the
     * direction matters. `DIRECT_ML` becomes `directml`, so the entry that
     * catches a `dml/` folder is `dml -> directml`, not the reverse.
     *
     * Only abbreviations belong here. Anything ONNX Runtime spells out is
     * derived above rather than listed.
     */
    private val ALIASES = mapOf(
        "trt" to "tensorrt",
        "dml" to "directml",
    )
}
