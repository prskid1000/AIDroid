package ai.ondevice.data.hf

/** Which ONNX Runtime execution providers this build does *not* have. */
object OnnxProviders {

    /** Directory-name tokens for providers this build cannot execute. */
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

    /** `CUDAExecutionProvider` → `cuda`, which is what publishers name the folder. */
    private fun token(enumName: String): String = enumName.replace("_", "").lowercase()

    /** Folder spellings publishers use that are not the provider's own. */
    private val ALIASES = mapOf(
        "trt" to "tensorrt",
        "dml" to "directml",
    )
}
