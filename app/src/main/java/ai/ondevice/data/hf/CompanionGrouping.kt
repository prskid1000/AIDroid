package ai.ondevice.data.hf

/**
 * Sorting the files found for one role into parts, alternatives and choices.
 *
 * The whole problem is that these look identical from the outside. Kokoro ships
 * 55 voice packs and gemma-3-4b ships three vision projectors; both are "several
 * files matching one role", and the resolver used to queue all of both. One of
 * those is right.
 *
 * The signal that separates them is precision. `mmproj-BF16`, `mmproj-F16` and
 * `mmproj-F32` are one file written three ways, and saying so needs no knowledge
 * of gemma, of llama.cpp, or of what a projector is — only that `BF16` is a
 * precision and `heart` is not. Strip the precision and see what collapses.
 *
 * Checked against every repo the app suggests: gemma's three projectors collapse
 * to one, Kokoro's 55 voices stay 55, ControlNet's 29 nets stay 29 even though
 * every one of them is `_fp16`, and Real-ESRGAN's ×2/×4/×8 stay three because a
 * scale factor is not a precision.
 */
object CompanionGrouping {

    /**
     * The precision vocabulary, as one segment.
     *
     * Deliberately anchored to a whole segment rather than matched anywhere in
     * the name. `control_v11p_sd15_canny_fp16` contains `fp16`, and so does
     * every one of its 28 siblings; matching loosely would collapse the entire
     * ControlNet pack into a single "precision alternative" and pick one net at
     * random. It also keeps `BF16` from reading as `F16`.
     *
     * The GGUF half is the vocabulary already in [ModelResolver.extractQuantName];
     * the rest is the ONNX convention its own comment describes.
     */
    private val PRECISION = Regex(
        "(?i)^(IQ\\d[_A-Z0-9]*|Q\\d[_A-Z0-9]*|BF16|F16|F32|FP16|FP32|FP8[_A-Z0-9]*" +
            "|INT8|UINT8|INT4|BNB4|Q4F16|QUANTIZED" +
            // fp8 is written as two segments — `t5xxl_fp8_e4m3fn` — and the
            // second one names the exponent layout, not a different encoder.
            // Without these, Flux's three T5 encoders read as three separate
            // things to choose between instead of one at three precisions.
            "|E4M3FN|E5M2|SCALED)$",
    )

    private val SEPARATORS = Regex("[-_.]")

    /** Container formats, best first: a pickle is worth avoiding when there is a choice. */
    private val CONTAINERS = listOf("safetensors", "gguf", "onnx", "bin", "pth", "pt", "ckpt")

    /** The precision this file is written at, or null when it does not say. */
    fun precisionToken(filename: String): String? =
        segments(filename).firstOrNull { PRECISION.matches(it) }?.uppercase()

    /**
     * What the file *is*, with precision and container removed.
     *
     * The container goes too, because `sd-vae-ft-mse-840000-ema-pruned` ships as
     * both `.ckpt` and `.safetensors` — the same weights in two wrappers, which
     * is an alternative and not a choice.
     */
    fun identity(filename: String): String =
        segments(filename).filterNot { PRECISION.matches(it) }.joinToString("-").lowercase()

    private fun segments(filename: String): List<String> {
        val name = filename.substringAfterLast('/')
        val extension = name.substringAfterLast('.', "")
        val stem = if (extension.lowercase() in CONTAINERS) name.dropLast(extension.length + 1) else name
        return stem.split(SEPARATORS).filter { it.isNotEmpty() }
    }

    /**
     * Group [companions] by role and decide what to download.
     *
     * The default is a real default, not a lock: every group carries its whole
     * candidate list so the screen can offer the rest.
     */
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

        // Which ControlNet, or which upscale factor, is a question about the
        // picture someone wants rather than about this device. Guessing for them
        // means a 723 MB download they did not ask for, so an optional role
        // starts empty and says so. A required one has to have something.
        CompanionGroup.Kind.CHOICES ->
            if (role.required) setOfNotNull(preferred(candidates)?.file?.filename) else emptySet()
    }

    /**
     * The one to take when they are interchangeable.
     *
     * F16 first, deliberately: it is what llama.cpp and the ONNX exports treat
     * as the on-device default, it is half the size of F32 for no accuracy that
     * survives quantised weights anyway, and BF16 buys range this never needs.
     * Then a real container over a pickle. Then the smaller file.
     */
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
