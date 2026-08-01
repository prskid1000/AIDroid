package ai.ondevice.data.hf

/** Sorting the files found for one role into parts, alternatives and choices. */
object CompanionGrouping {

    /** The precision vocabulary, as one segment. */
    private val PRECISION = Regex(
        "(?i)^(IQ\\d[_A-Z0-9]*|Q\\d[_A-Z0-9]*|BF16|F16|F32|FP16|FP32|FP8[_A-Z0-9]*" +
            "|INT8|UINT8|INT4|BNB4|Q4F16|QUANTIZED" +
            // fp8 is written as two segments — `t5xxl_fp8_e4m3fn` — and the second one names the exponent layout, not a different encoder.
            "|E4M3FN|E5M2|SCALED)$",
    )

    private val SEPARATORS = Regex("[-_.]")

    /** Container formats, best first: a pickle is worth avoiding when there is a choice. */
    private val CONTAINERS = listOf("safetensors", "gguf", "onnx", "bin", "pth", "pt", "ckpt")

    /** The precision this file is written at, or null when it does not say. */
    fun precisionToken(filename: String): String? =
        segments(filename).firstOrNull { PRECISION.matches(it) }?.uppercase()

    /** What the file *is*, with precision and container removed. */
    fun identity(filename: String): String =
        segments(filename).filterNot { PRECISION.matches(it) }.joinToString("-").lowercase()

    private fun segments(filename: String): List<String> {
        val name = filename.substringAfterLast('/')
        val extension = name.substringAfterLast('.', "")
        val stem = if (extension.lowercase() in CONTAINERS) name.dropLast(extension.length + 1) else name
        return stem.split(SEPARATORS).filter { it.isNotEmpty() }
    }

    /** Group [companions] by role and decide what to download. */
    fun group(companions: List<CompanionFile>): List<CompanionGroup> =
        companions.groupBy { it.role }.map { (role, candidates) ->
            val identities = candidates.groupBy { identity(it.file.filename) }
            val kind = when {
                role.cardinality == Cardinality.ALL -> CompanionGroup.Kind.PARTS
                identities.size <= 1 -> CompanionGroup.Kind.ALTERNATIVES
                else -> CompanionGroup.Kind.CHOICES
            }
            CompanionGroup(
                role = role,
                candidates = candidates,
                selected = defaultSelection(role, candidates, kind),
                kind = kind,
            )
        }.sortedBy { it.role.ordinal }

    private fun defaultSelection(
        role: CompanionRole,
        candidates: List<CompanionFile>,
        kind: CompanionGroup.Kind,
    ): Set<String> = when (kind) {
        CompanionGroup.Kind.PARTS -> candidates.map { it.file.filename }.toSet()

        CompanionGroup.Kind.ALTERNATIVES -> setOfNotNull(preferred(candidates)?.file?.filename)

        // Which ControlNet, or which upscale factor, is a question about the picture someone wants rather than about this device.
        CompanionGroup.Kind.CHOICES ->
            if (role.required) setOfNotNull(preferred(candidates)?.file?.filename) else emptySet()
    }

    /** The one to take when they are interchangeable. */
    fun preferred(candidates: List<CompanionFile>): CompanionFile? =
        candidates.minWithOrNull(
            compareBy<CompanionFile> { if (precisionToken(it.file.filename) == "F16") 0 else 1 }
                .thenBy { if (precisionToken(it.file.filename) == "FP16") 0 else 1 }
                .thenBy { containerRank(it.file.filename) }
                .thenBy { it.file.sizeBytes }
                .thenBy { it.file.filename },
        )

    private fun containerRank(filename: String): Int =
        CONTAINERS.indexOf(filename.substringAfterLast('.', "").lowercase())
            .let { if (it < 0) CONTAINERS.size else it }
}
