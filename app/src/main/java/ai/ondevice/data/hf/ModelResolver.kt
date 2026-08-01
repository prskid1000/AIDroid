package ai.ondevice.data.hf

import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.Fmt
import ai.ondevice.core.Modality
import ai.ondevice.core.ModelFormat
import ai.ondevice.core.RefusalKind
import ai.ondevice.engine.RuntimeRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** SPEC §3.2, the resolution pipeline. */
class ModelResolver(
    private val api: HfApi,
    private val registry: RuntimeRegistry,
) {

    /** Which architecture strings mean "diffusion". */
    private val diffusionArchitectures: Set<String> by lazy {
        registry.architecturesFor(RuntimeRegistry.STABLE_DIFFUSION) + setOf("unet", "dit")
    }

    fun normalize(input: String): NormalizedInput? {
        val trimmed = input.trim()
        if (trimmed.isEmpty()) return null

        if (trimmed.startsWith("content://")) {
            return NormalizedInput.LocalFile(trimmed)
        }

        // A direct .gguf link that isn't on huggingface.co — any host is allowed.
        if (trimmed.startsWith("http") && trimmed.substringBefore('?').endsWith(".gguf", ignoreCase = true) &&
            !trimmed.contains("huggingface.co")
        ) {
            return NormalizedInput.DirectUrl(trimmed)
        }

        var path = trimmed
            .removePrefix("https://")
            .removePrefix("http://")
            .removePrefix("www.")
            .removePrefix("huggingface.co/")
            .removePrefix("hf.co/")
            .trim('/')

        // Strip the site's routing segments and remember the revision if the
        // URL pinned one.
        var revision = "main"
        var filename: String? = null
        for (marker in listOf("/tree/", "/blob/", "/resolve/")) {
            val idx = path.indexOf(marker)
            if (idx >= 0) {
                val tail = path.substring(idx + marker.length)
                path = path.substring(0, idx)
                val parts = tail.split('/')
                revision = parts.firstOrNull()?.takeIf { it.isNotBlank() } ?: "main"
                if (parts.size > 1) filename = parts.drop(1).joinToString("/")
                break
            }
        }
        path = path.substringBefore('?').substringBefore('#').trim('/')

        val segments = path.split('/').filter { it.isNotBlank() }
        return when (segments.size) {
            2 -> NormalizedInput.Repo(segments[0], segments[1], revision, filename)
            // A bare name with no owner isn't resolvable; the caller offers search.
            else -> null
        }
    }

    /** Steps 2–7 of §3.2. */
    suspend fun resolve(input: String, blockPickle: Boolean = true): Resolution = withContext(Dispatchers.IO) {
        val normalized = normalize(input)
            ?: return@withContext Resolution.Refused(
                kind = RefusalKind.NOT_FOUND,
                title = "Not a model reference",
                subject = input.take(80),
                detail = "Paste an owner/repo ID, a Hugging Face URL, or a direct .gguf link. " +
                    "You can also import a .gguf from storage.",
                remedies = listOf(Remedy("Search Hugging Face", RemedyAction.SearchRepo(input), primary = true)),
            )

        when (normalized) {
            is NormalizedInput.LocalFile, is NormalizedInput.DirectUrl ->
                return@withContext resolveDirect(normalized)
            is NormalizedInput.Repo -> Unit
        }

        val repoRef = normalized as NormalizedInput.Repo
        val repoId = "${repoRef.owner}/${repoRef.repo}"

        val info = api.modelInfo(repoId).getOrElse { error ->
            val hf = error as? HfException
            return@withContext when {
                hf?.isAuthFailure == true -> gatedRefusal(repoId)
                hf?.isNotFound == true -> Resolution.Refused(
                    kind = RefusalKind.NOT_FOUND,
                    title = "Not found, or private",
                    subject = repoId,
                    detail = "Hugging Face has no public repo at this ID. If it is private, " +
                        "add a token that can read it.",
                    remedies = listOf(
                        Remedy("Enter token", RemedyAction.EnterToken, primary = true),
                        Remedy("Open repo page", RemedyAction.OpenUrl("${HfApi.BASE}/$repoId")),
                    ),
                )
                else -> Resolution.Refused(
                    kind = RefusalKind.NOT_FOUND,
                    title = "Couldn't reach Hugging Face",
                    subject = repoId,
                    detail = error.message ?: "Network error.",
                    remedies = listOf(Remedy("Retry", RemedyAction.SearchRepo(repoId), primary = true)),
                )
            }
        }

        if (info.isGated) return@withContext gatedRefusal(repoId)

        val files = info.siblings.map { it.rfilename }

        // Step 3 — classify from file shape and architecture, never repo name.
        val ggufFiles = files.filter { it.endsWith(".gguf", ignoreCase = true) }
        val ggmlBins = files.filter { it.matches(Regex("""(?i).*ggml-.*\.bin""")) }
        val onnxFiles = files.filter { it.endsWith(".onnx", ignoreCase = true) }
        val safetensors = files.filter { it.endsWith(".safetensors", ignoreCase = true) }
        val pickles = files.filter { it.endsWith(".pkl") || it.endsWith(".pt") || it.endsWith(".pth") }

        // The pickle block exists because `torch.load` executes whatever the file tells it to.
        val pickleAuxiliaries = pickles.filter { AttachmentRole.classify(it, info.tags) != null }
        val unsafePickles = pickles - pickleAuxiliaries.toSet()

        if (blockPickle && unsafePickles.isNotEmpty()) {
            return@withContext Resolution.Refused(
                kind = RefusalKind.PICKLE_BLOCKED,
                title = "Pickle files present",
                subject = repoId,
                detail = "This repo ships ${unsafePickles.size} pickle-format file(s), which can " +
                    "execute arbitrary code on load. Blocked by default; you can override this in " +
                    "Settings under expert options.",
                remedies = listOf(Remedy("Open repo page", RemedyAction.OpenUrl("${HfApi.BASE}/$repoId"))),
            )
        }

        val auxiliaries = (
            safetensors +
                files.filter { it.endsWith(".ckpt", ignoreCase = true) } +
                pickleAuxiliaries
            // With the repo's tags, not without them. A ControlNet from the
            // diffusers world is called `diffusion_pytorch_model.safetensors`
            // and says nothing about itself; the only thing that identifies it
            // is the repo's `controlnet` tag, which classify already reads and
            // was never being given.
            ).filter { AttachmentRole.classify(it, info.tags) != null }

        if (ggufFiles.isEmpty() && ggmlBins.isEmpty() && onnxFiles.isEmpty() && auxiliaries.isEmpty()) {
            return@withContext if (safetensors.isNotEmpty() || files.any { it.endsWith(".bin") }) {
                pytorchOnlyRefusal(repoRef)
            } else {
                Resolution.Refused(
                    kind = RefusalKind.NO_RUNTIME,
                    title = "Nothing runnable in this repo",
                    subject = repoId,
                    detail = "No GGUF, GGML or ONNX artifacts were found, so none of the bundled " +
                        "runtimes can load it.",
                    remedies = listOf(Remedy("Open repo page", RemedyAction.OpenUrl("${HfApi.BASE}/$repoId"))),
                )
            }
        }

        val format = when {
            ggufFiles.isNotEmpty() -> ModelFormat.GGUF
            ggmlBins.isNotEmpty() -> ModelFormat.GGML_BIN
            onnxFiles.isNotEmpty() -> ModelFormat.ONNX
            else -> ModelFormat.SAFETENSORS
        }

        // Step 7 — pin *first*, then read everything at the pin.
        val pinnedRevision = info.sha?.takeIf { it.isNotBlank() } ?: repoRef.revision

        // Step 4/6 — enumerate quant variants, folding shard sets into one entry.
        val primaryFiles = when (format) {
            ModelFormat.GGUF -> ggufFiles.filterNot { isCompanionFilename(it) }
            ModelFormat.GGML_BIN -> ggmlBins
            ModelFormat.ONNX -> onnxFiles
            // For an auxiliary pack the "variants" are the individual auxiliaries — canny, depth, openpose — and picking one is the point, not a quality trade-off.
            else -> refineAuxiliaries(repoId, pinnedRevision, safetensors, auxiliaries, info.tags)
        }

        // A multi-graph ONNX model is grouped by directory, not by file — see
        // onnxGraphSets. Null for every other shape.
        val graphSets = if (format == ModelFormat.ONNX) onnxGraphSets(onnxFiles) else null

        // The sidecar weight files have to be priced too, or the variant reports the size of a graph stub.
        val graphFiles = graphSets?.let { (it.runnable + it.unrunnable).values.flatten() }
        val sidecars = (graphFiles ?: primaryFiles).flatMap { onnxSidecars(it, files) }
        val wanted = (
            (graphFiles ?: primaryFiles) +
                sidecars +
                files.filter { isCompanionFilename(it) }
            ).distinct()
        val sizeLookup = api.pathsInfo(repoId, wanted.take(128), pinnedRevision)
            .getOrDefault(emptyList())
            .associateBy { it.path }

        val enumerated = enumerateQuants(
            files = primaryFiles,
            sizes = sizeLookup,
            info = info,
            allFiles = files,
            preGrouped = graphSets?.runnable,
        ) + graphSets?.unrunnable
            ?.takeIf { it.isNotEmpty() }
            ?.let { foreign ->
                enumerateQuants(
                    files = primaryFiles,
                    sizes = sizeLookup,
                    info = info,
                    allFiles = files,
                    preGrouped = foreign,
                ).map {
                    // Shown, and refused.
                    it.copy(
                        blockedReason = "is built for the ${it.name} execution provider, " +
                            "which this build does not have",
                    )
                }
            }.orEmpty()

        // ESRGAN at any scale but four.
        //
        // The bundled upscaler builds one network shape, and the x2 and x8
        // Real-ESRGAN releases carry a different number of upsample blocks: the
        // file downloads, the tensor names line up, and the load fails. Shown
        // and refused is the honest answer — a x8 that installs and then never
        // runs is worse than one that says so before the 67 MB.
        val quants = enumerated.map { variant ->
            val name = variant.files.firstOrNull()?.filename.orEmpty().lowercase()
            val scale = Regex("""x(\d+)""").find(name)?.groupValues?.get(1)
            if (variant.blockedReason == null && name.contains("esrgan") && scale != null && scale != "4") {
                variant.copy(
                    blockedReason = "is the x$scale network, and this build's upscaler only has " +
                        "the upsample blocks for x4",
                )
            } else {
                variant
            }
        }

        if (quants.isEmpty()) {
            return@withContext Resolution.Refused(
                kind = RefusalKind.NO_RUNTIME,
                title = "No loadable weights",
                subject = repoId,
                detail = "The repo lists artifacts but none of them resolved to a downloadable file.",
                remedies = listOf(Remedy("Open repo page", RemedyAction.OpenUrl("${HfApi.BASE}/$repoId"))),
            )
        }

        // Step 3 continued — the architecture must be one the bundled runtime
        // knows. Worked out before the companions, because which of them the
        // model cannot run without is a property of its family: SDXL reads its
        // prompt through two CLIPs, FLUX.1 through CLIP-L and T5, FLUX.2
        // through a language model, and none of that is knowable from a role.
        val arch = info.gguf?.architecture ?: inferArchitectureFromTags(info)

        val companions = detectCompanions(
            files = files,
            sizes = sizeLookup,
            variantFiles = quants.flatMap { variant -> variant.files.map { it.filename } }.toSet(),
            architecture = arch,
        )
        val modality = classifyModality(info, format, files, companions)
        if (format == ModelFormat.GGUF && arch != null && !registry.supportsArchitecture(arch)) {
            return@withContext unsupportedArchRefusal(repoId, arch)
        }

        val security = sizeLookup.values.firstNotNullOfOrNull { it.securityFileStatus?.status }

        Resolution.Resolved(
            ResolvedModel(
                repoId = repoId,
                owner = repoRef.owner,
                repo = repoRef.repo,
                revision = pinnedRevision,
                displayName = repoRef.repo,
                architecture = arch,
                modality = modality,
                format = format,
                contextLength = info.gguf?.contextLength,
                chatTemplate = info.gguf?.chatTemplate,
                bosToken = info.gguf?.bosToken,
                eosToken = info.gguf?.eosToken,
                parameterCount = info.gguf?.total,
                layers = info.gguf?.blockCount,
                embeddingLength = info.gguf?.embeddingLength,
                embeddingLengthKv = deriveKvWidth(info.gguf),
                gated = info.isGated,
                quants = quants,
                companions = companions,
                metadataFromHeader = false,
                securityStatus = security,
                hasPickleFiles = pickles.isNotEmpty(),
            ),
        )
    }

    /** When HF hasn't parsed a repo's metadata, pull the first megabyte of the chosen file and read the GGUF header directly. */
    suspend fun enrichFromHeader(model: ResolvedModel, quant: QuantVariant): ResolvedModel {
        if (model.layers != null && model.contextLength != null && model.chatTemplate != null) return model
        val first = quant.files.firstOrNull() ?: return model
        val url = api.resolveUrl(model.repoId, first.filename, model.revision)
        val bytes = api.rangeGet(url, GgufHeaderReader.HEADER_BYTES).getOrNull() ?: return model
        val meta = GgufHeaderReader.parse(bytes).getOrNull() ?: return model
        val architecture = model.architecture ?: meta.architecture

        // Learning the architecture can change what the model *is*.
        val modality = if (model.modality == Modality.TEXT && architecture != null) {
            when (architecture.lowercase()) {
                in diffusionArchitectures -> Modality.DIFFUSION
                "whisper" -> Modality.SPEECH_TO_TEXT
                else -> model.modality
            }
        } else {
            model.modality
        }

        return model.copy(
            architecture = architecture,
            modality = modality,
            contextLength = model.contextLength ?: meta.contextLength,
            chatTemplate = model.chatTemplate ?: meta.chatTemplate,
            layers = model.layers ?: meta.blockCount,
            embeddingLength = model.embeddingLength ?: meta.embeddingLength,
            embeddingLengthKv = model.embeddingLengthKv ?: meta.embeddingLengthKv,
            parameterCount = model.parameterCount ?: meta.paramCount,
            metadataFromHeader = true,
        )
    }

    // — classification —

    /** Modality comes from architecture and file shape. */
    private fun classifyModality(
        info: HfModelInfo,
        format: ModelFormat,
        files: List<String>,
        companions: List<CompanionGroup>,
    ): Modality {
        val arch = info.gguf?.architecture?.lowercase()
        return when {
            format == ModelFormat.GGML_BIN || arch == "whisper" -> Modality.SPEECH_TO_TEXT
            format == ModelFormat.SAFETENSORS -> Modality.DIFFUSION
            files.any { it.contains("voices", true) && it.endsWith(".bin") } &&
                files.any { it.endsWith(".onnx") } -> Modality.TEXT_TO_SPEECH
            format == ModelFormat.ONNX && info.tags.any { it.contains("text-to-speech", true) } -> Modality.TEXT_TO_SPEECH
            arch != null && arch in diffusionArchitectures -> Modality.DIFFUSION
            files.any { it.equals("model_index.json", true) } -> Modality.DIFFUSION
            files.any { it.contains("unet", true) } && files.any { it.contains("vae", true) } -> Modality.DIFFUSION
            info.pipelineTag == "text-to-image" || info.pipelineTag == "image-to-image" -> Modality.DIFFUSION
            info.tags.any {
                it.equals("stable-diffusion", true) || it.equals("diffusers", true) ||
                    it.equals("text-to-image", true)
            } -> Modality.DIFFUSION
            companions.any { it.role == CompanionRole.VISION_PROJECTOR } -> Modality.VISION
            info.pipelineTag == "feature-extraction" || info.pipelineTag == "sentence-similarity" -> Modality.EMBEDDING
            format == ModelFormat.GGUF -> Modality.TEXT
            else -> Modality.UNKNOWN
        }
    }

    /** Step 5 — auto-pair companions so a multi-file model is never hand-assembled. */
    /** Companions are things a model needs *alongside* it — a vision projector, a TAESD decoder, Kokoro's voice packs. */
    private fun detectCompanions(
        files: List<String>,
        sizes: Map<String, HfPathInfo>,
        variantFiles: Set<String> = emptySet(),
        architecture: String? = null,
    ): List<CompanionGroup> = CompanionGrouping.group(
        files.mapNotNull { name ->
            if (name in variantFiles) return@mapNotNull null
            val role = companionRole(name) ?: return@mapNotNull null
            val info = sizes[name]
            CompanionFile(
                file = RemoteFile(
                    filename = name,
                    sizeBytes = info?.size ?: 0L,
                    sha256 = info?.sha256,
                    commitId = info?.lastCommit?.id,
                    securityStatus = info?.securityFileStatus?.status,
                ),
                role = role,
            )
        }.distinctBy { it.role to it.file.filename },
        architecture = architecture,
    )

    /** Correct the filename's verdict against the file's own tensor names. */
    private suspend fun refineAuxiliaries(
        repoId: String,
        revision: String,
        safetensors: List<String>,
        classified: List<String>,
        tags: List<String> = emptyList(),
    ): List<String> {
        val ambiguous = safetensors.filter {
            it !in classified || AttachmentRole.classify(it, tags) == AttachmentRole.TAESD
        }
        if (ambiguous.isEmpty() || safetensors.size > HEADER_PROBE_LIMIT) return classified

        val verdicts = ambiguous.associateWith { filename ->
            val url = api.resolveUrl(repoId, filename, revision)
            api.rangeGet(url, SafetensorsHeaderReader.HEADER_BYTES).getOrNull()
                ?.let(SafetensorsHeaderReader::parse)
        }

        fun isTaesd(filename: String) = verdicts[filename]?.hasPrefix(TAESD_TENSOR_PREFIX) == true
        fun unreadable(filename: String) = filename in verdicts && verdicts[filename] == null

        val kept = classified.filter { filename ->
            AttachmentRole.classify(filename, tags) != AttachmentRole.TAESD ||
                isTaesd(filename) || unreadable(filename)
        }
        val recovered = ambiguous.filter { it !in classified && isTaesd(it) }
        return (kept + recovered).distinct()
    }

    private fun companionRole(name: String): CompanionRole? {
        val n = name.lowercase()
        return when {
            n.contains("mmproj") -> CompanionRole.VISION_PROJECTOR
            n.contains("taesd") -> CompanionRole.TAESD
            n.contains("vae") -> CompanionRole.VAE
            n.contains("clip_g") || n.contains("clip-g") -> CompanionRole.CLIP_G
            n.contains("clip_l") || n.contains("clip-l") -> CompanionRole.CLIP_L
            n.contains("t5xxl") || n.contains("t5-xxl") -> CompanionRole.T5XXL
            n.contains("control") && n.endsWith(".safetensors") -> CompanionRole.CONTROLNET
            n.contains("esrgan") || n.contains("upscal") -> CompanionRole.UPSCALER
            n.contains("voices") && n.endsWith(".bin") -> CompanionRole.VOICES
            n.contains("silero") || n.contains("vad") -> CompanionRole.VAD
            else -> null
        }
    }

    private fun isCompanionFilename(name: String) = companionRole(name) != null

    /** Fold shard sets — `model-00001-of-00003.gguf` and its siblings become one variant with three files, downloaded as one atomic job. */
    private fun enumerateQuants(
        files: List<String>,
        sizes: Map<String, HfPathInfo>,
        info: HfModelInfo,
        /** Every file in the repo, so a graph can find its weight sidecar. */
        allFiles: List<String> = files,
        /** Pre-grouped variants, when the repo's shape is not one-file-per-choice. */
        preGrouped: Map<String, List<String>>? = null,
    ): List<QuantVariant> {
        val shardPattern = Regex("""(?i)^(.*)-\d{5}-of-\d{5}(\.gguf)$""")
        val grouped = LinkedHashMap<String, MutableList<String>>()
        if (preGrouped != null) {
            preGrouped.forEach { (label, members) -> grouped[label] = members.toMutableList() }
        } else {
            files.forEach { f ->
                val match = shardPattern.find(f)
                val key = if (match != null) match.groupValues[1] + match.groupValues[2] else f
                grouped.getOrPut(key) { mutableListOf() }.add(f)
            }
        }

        // Captured before the companions below are folded in.
        val shardCounts = grouped.mapValues { (_, members) -> members.size }

        // ONNX keeps anything over 2 GB — and in practice anything at all — in a sibling data file, so the graph alone measures a couple of kilobytes.
        grouped.forEach { (label, members) ->
            val sidecars = members.flatMap { onnxSidecars(it, allFiles) }
            members.addAll(sidecars.filterNot { it in members })
            if (members.any { it.endsWith(".onnx", ignoreCase = true) }) {
                val tokenisers = onnxTextCompanions(label, allFiles)
                members.addAll(tokenisers.filterNot { it in members })
            }
        }

        // A quant suffix only identifies a variant when the repo holds one model.
        val proposed = grouped.keys.associateWith {
            if (preGrouped != null) it else extractQuantName(it, info)
        }
        val ambiguous = proposed.values.groupingBy { it }.eachCount().filterValues { it > 1 }.keys

        return grouped.map { (key, members) ->
            val sorted = members.sorted()
            val quantName = proposed.getValue(key).let { name ->
                if (name in ambiguous) distinguishingName(key) else name
            }
            val remoteFiles = sorted.map { name ->
                val pi = sizes[name]
                RemoteFile(
                    filename = name,
                    sizeBytes = pi?.size ?: 0L,
                    sha256 = pi?.sha256,
                    commitId = pi?.lastCommit?.id,
                    securityStatus = pi?.securityFileStatus?.status,
                )
            }
            QuantVariant(
                name = quantName,
                files = remoteFiles,
                // The character note is about the *quantisation*, so it reads the suffix even when the label had to be widened to stay unambiguous.
                note = quantNote(
                    proposed.getValue(key),
                    shards = shardCounts[key] ?: sorted.size,
                    fileCount = sorted.size,
                    onnx = key.endsWith(".onnx", ignoreCase = true),
                    graphSet = preGrouped != null,
                ),
                cautionReason = quantCaution(
                    totalBytes = remoteFiles.sumOf { it.sizeBytes },
                    parameterCount = info.gguf?.total,
                ),
            )
        }.sortedBy { it.totalBytes }
    }

    /** The whole filename, minus the noise every file in the repo shares. */
    private fun distinguishingName(filename: String): String =
        filename.substringAfterLast('/')
            .removeSuffix(".gguf").removeSuffix(".bin").removeSuffix(".onnx")
            .removePrefix("ggml-")
            .removePrefix("model-")

    /** The text-side files an ONNX model needs that are not graphs. */
    private fun onnxTextCompanions(label: String, allFiles: List<String>): List<String> {
        val directory = label.substringBeforeLast('/', "").takeIf { it.isNotEmpty() } ?: label
        fun pick(name: String): String? =
            allFiles.firstOrNull { it == "$directory/$name" }
                ?: allFiles.firstOrNull { it == "$label/$name" }
                ?: allFiles.firstOrNull { it == name }
        return listOfNotNull(pick("tokenizer.json"), pick("tokenizer_config.json"))
    }

    /** An ONNX graph's external weight file, under any of the three spellings in use. */
    private fun onnxSidecars(graph: String, allFiles: List<String>): List<String> {
        if (!graph.endsWith(".onnx", ignoreCase = true)) return emptyList()
        val candidates = setOf("$graph.data", "${graph}_data", "$graph.onnx_data")
        return allFiles.filter { it in candidates }
    }

    /** Group a multi-graph ONNX model by directory instead of by file. */
    private fun onnxGraphSets(onnxFiles: List<String>): GraphSets? {
        if (onnxFiles.size < 2) return null
        val byBase = onnxFiles.groupBy { it.substringAfterLast('/') }
        val repeated = byBase.filterValues { paths ->
            paths.map { it.substringBeforeLast('/', "") }.distinct().size > 1
        }
        if (repeated.isEmpty()) return null

        val allDirectories = onnxFiles.groupBy { it.substringBeforeLast('/', "") }
        // Provider folders are held out of the grouping so a CUDA build cannot vote on which layout is primary, then added back at the end marked unrunnable.
        val foreign = allDirectories.filterKeys { OnnxProviders.isForeign(it.substringAfterLast('/')) }
        val byDirectory = allDirectories - foreign.keys
        if (byDirectory.isEmpty()) return null

        // Directories are only alternatives to each other when they hold the *same* graphs.
        val signature = byDirectory.mapValues { (_, paths) ->
            paths.map { it.substringAfterLast('/') }.toSortedSet()
        }
        val primary = signature[""] ?: signature.values
            .groupingBy { it }.eachCount().maxByOrNull { it.value }?.key
            ?: return null

        val alternatives = byDirectory.filterKeys { signature[it] == primary }
        if (alternatives.isEmpty()) return null
        val components = byDirectory.filterKeys { signature[it] != primary }

        // One directory per component family, deepest path first.
        val chosenComponents = components.entries
            .groupBy { (_, paths) -> paths.map { it.substringAfterLast('/') }.toSortedSet() }
            .mapNotNull { (_, dirs) -> dirs.maxByOrNull { it.key.count { c -> c == '/' } } }
            .flatMap { it.value }

        val runnable = alternatives.entries.associate { (directory, graphs) ->
            val label = directory.ifEmpty { "root" }
            label to (graphs + chosenComponents)
        }
        val unrunnable = foreign.entries.associate { (directory, graphs) ->
            directory.substringAfterLast('/') to (graphs + chosenComponents)
        }
        return GraphSets(runnable, unrunnable)
    }

    /** The variant directories a multi-graph ONNX repo offers, split by whether this build could load one. */
    private data class GraphSets(
        val runnable: Map<String, List<String>>,
        val unrunnable: Map<String, List<String>> = emptyMap(),
    )

    /** `Qwen2.5-7B-Instruct-Q4_K_M.gguf` → `Q4_K_M`. */
    private fun extractQuantName(filename: String, info: HfModelInfo): String {
        val base = filename.substringAfterLast('/').removeSuffix(".gguf").removeSuffix(".bin").removeSuffix(".onnx")
        if (filename.endsWith(".onnx", ignoreCase = true)) {
            val stem = base.removePrefix("model").trim('_', '-', '.')
            return if (stem.isBlank()) ORIGINAL_EXPORT else stem.uppercase()
        }
        val match = Regex("""(?i)(IQ\d[_A-Z0-9]*|Q\d[_A-Z0-9]*|BF16|F16|F32)$""").find(base)
        return match?.value?.uppercase() ?: base.substringAfterLast('-').ifBlank { base }
    }

    /** A warning that this variant will run and should probably not be chosen. */
    private fun quantCaution(totalBytes: Long, parameterCount: Long?): String? {
        if (parameterCount == null || parameterCount <= 0L || totalBytes <= 0L) return null
        val bitsPerWeight = totalBytes.toDouble() * 8.0 / parameterCount.toDouble()
        val rounded = String.format("%.1f", bitsPerWeight)
        return when {
            bitsPerWeight < 2.0 ->
                "About $rounded bits per weight. Below two, models answer confidently and wrongly " +
                    "rather than just less well — this one is for experimenting with, not using."
            bitsPerWeight < 2.5 && parameterCount < SMALL_MODEL_PARAMS ->
                "About $rounded bits per weight, on a model this size. A large model absorbs that; " +
                    "one under ${SMALL_MODEL_PARAMS / 1_000_000_000}B has no redundancy to spare."
            else -> null
        }
    }

    /** The one-line character note under each variant. */
    private fun quantNote(
        rawQuant: String,
        shards: Int,
        fileCount: Int = shards,
        onnx: Boolean = false,
        graphSet: Boolean = false,
    ): String = buildString {
        // A multi-graph ONNX variant is neither quantised by its label nor sharded.
        if (graphSet) {
            append("$fileCount files · one model in parts")
            return@buildString
        }
        val quant = rawQuant.uppercase()
        if (onnx) {
            // Half-precision *activations*, which is a different claim from half-precision weights and the one that decides whether this file can run here at all.
            val halfPrecisionMaths = Regex("(F16|FP16)$").containsMatchIn(quant)
            append(
                when {
                    quant == ORIGINAL_EXPORT.uppercase() -> "as published"
                    quant.startsWith("QUANTIZED") -> "8-bit, dynamically quantised"
                    halfPrecisionMaths && quant.contains("8") -> "8-bit weights, half-precision maths"
                    halfPrecisionMaths && quant.contains("4") -> "4-bit weights, half-precision maths"
                    quant.startsWith("UINT8") || quant.startsWith("INT8") || quant.startsWith("Q8") ->
                        "8-bit weights"
                    quant.startsWith("Q4") || quant.startsWith("INT4") || quant.startsWith("BNB4") ->
                        "4-bit weights, smallest"
                    quant.startsWith("FP16") || quant.startsWith("F16") -> "half precision"
                    quant.startsWith("BF16") -> "half precision, bfloat"
                    quant.startsWith("FP32") || quant.startsWith("F32") -> "full precision"
                    else -> "as published"
                },
            )
            if (halfPrecisionMaths || quant.startsWith("FP16") || quant.startsWith("F16")) {
                append(" · silent on arm64, pick another")
            }
            if (fileCount > 1) append(" · $fileCount files")
            return@buildString
        }
        append(
            when {
                quant.startsWith("F32") -> "unquantised"
                quant.startsWith("F16") || quant.startsWith("BF16") -> "near-lossless, slow here"
                quant.startsWith("Q8") -> "near-lossless, slow here"
                quant.startsWith("Q6") -> "high quality"
                quant.endsWith("_K_M") -> "balanced K-quant"
                quant.endsWith("_K_S") -> "compact K-quant"
                quant.endsWith("_K_XL") || quant.endsWith("_K_L") -> "generous K-quant"
                quant.startsWith("IQ1") || quant.startsWith("IQ2") -> "very small, real quality cost"
                quant.startsWith("IQ") -> "small, quality cost"
                quant.startsWith("Q2") || quant.startsWith("Q3") -> "small, quality cost"
                // Q4_0/Q4_1/Q5_0/Q5_1 — the pre-K quantisations.
                quant.matches(Regex("""Q[45]_[01]""")) -> "legacy quant"
                quant.endsWith("_1") || quant.endsWith("_0") -> "legacy quant"
                // No quant suffix at all: whisper's plain `ggml-base.bin` and
                // friends ship at full precision.
                !quant.matches(Regex("""(?i)(IQ|Q)\d.*""")) -> "full precision"
                else -> "unrecognised quant name"
            },
        )
        if (shards > 1) append(" · $shards shards")
    }

    private fun deriveKvWidth(block: HfGgufBlock?): Int? {
        val embd = block?.embeddingLength ?: return null
        val heads = block.headCount ?: return embd
        val kvHeads = block.headCountKv ?: heads
        if (heads == 0) return embd
        return (embd / heads) * kvHeads
    }

    private fun inferArchitectureFromTags(info: HfModelInfo): String? =
        info.tags.firstOrNull { it in registry.knownArchitectures }

    // — the refusals of §3.2, each with its own remedy —

    private fun gatedRefusal(repoId: String) = Resolution.Refused(
        kind = RefusalKind.GATED,
        title = "Gated repo",
        subject = repoId,
        detail = "Accept the licence on Hugging Face, then paste a token. The token is stored in " +
            "the Android Keystore and used for nothing else.",
        remedies = listOf(
            Remedy("Open repo page", RemedyAction.OpenUrl("${HfApi.BASE}/$repoId"), primary = true),
            Remedy("Enter token", RemedyAction.EnterToken),
        ),
    )

    private fun pytorchOnlyRefusal(ref: NormalizedInput.Repo) = Resolution.Refused(
        kind = RefusalKind.PYTORCH_ONLY,
        title = "PyTorch weights only",
        subject = "${ref.owner}/${ref.repo}",
        detail = "This repo ships safetensors. Converting to GGUF needs a desktop — the app won't " +
            "pretend otherwise.",
        remedies = buildList {
            add(Remedy("Search for ${ref.repo}-GGUF", RemedyAction.SearchRepo("${ref.repo}-GGUF"), primary = true))
            HfApi.GGUF_MIRRORS.forEach { add(Remedy(it, RemedyAction.OpenMirror(it, "${ref.repo}-GGUF"))) }
        },
    )

    private fun unsupportedArchRefusal(repoId: String, arch: String) = Resolution.Refused(
        kind = RefusalKind.UNKNOWN_ARCHITECTURE,
        title = "Unsupported architecture",
        subject = "arch $arch",
        detail = "llama.cpp ${registry.llamaBuildTag} — the build installed on this device — has " +
            "${registry.architectureCount} architectures and this isn't one of them. A newer " +
            "runtime may add it.",
        remedies = listOf(
            Remedy("Check for runtime update", RemedyAction.CheckRuntimeUpdate, primary = true),
            Remedy(
                "Upstream issues",
                RemedyAction.OpenUrl("https://github.com/ggml-org/llama.cpp/issues?q=$arch"),
            ),
        ),
    )

    private suspend fun resolveDirect(input: NormalizedInput): Resolution {
        val (name, url) = when (input) {
            is NormalizedInput.DirectUrl -> input.url.substringAfterLast('/').substringBefore('?') to input.url
            is NormalizedInput.LocalFile -> "Imported model" to input.uri
            else -> return Resolution.Refused(
                RefusalKind.NOT_FOUND,
                "Unsupported reference",
                "",
                input.toString(),
            )
        }
        // A direct file has no repo metadata, so the header parser is the only
        // source — which is exactly the case §3.1 keeps it maintained for.
        return Resolution.Resolved(
            ResolvedModel(
                repoId = url,
                owner = "",
                repo = name,
                revision = "direct",
                displayName = name.removeSuffix(".gguf"),
                architecture = null,
                modality = Modality.TEXT,
                format = ModelFormat.GGUF,
                contextLength = null,
                chatTemplate = null,
                bosToken = null,
                eosToken = null,
                parameterCount = null,
                layers = null,
                embeddingLength = null,
                embeddingLengthKv = null,
                gated = false,
                quants = listOf(
                    QuantVariant(
                        name = extractQuantName(name, HfModelInfo()),
                        files = listOf(RemoteFile(name, 0L)),
                        note = "direct file — metadata read from the header",
                    ),
                ),
                companions = emptyList(),
                metadataFromHeader = true,
                securityStatus = null,
                hasPickleFiles = false,
            ),
        )
    }

    private companion object {
        /** The label for a bare `model.onnx` when the filename says nothing about precision. */
        const val ORIGINAL_EXPORT = "original"

        /** Below this, a heavy quantisation has no redundancy to eat into. */
        const val SMALL_MODEL_PARAMS = 4_000_000_000L

        /** How many safetensors a repo may hold before header probing is skipped. */
        const val HEADER_PROBE_LIMIT = 8

        /** What sd.cpp's TinyDecoder block is named — `src/model/vae/tae.hpp`. */
        const val TAESD_TENSOR_PREFIX = "decoder.layers."
    }
}

sealed interface NormalizedInput {
    data class Repo(
        val owner: String,
        val repo: String,
        val revision: String = "main",
        val filename: String? = null,
    ) : NormalizedInput

    data class DirectUrl(val url: String) : NormalizedInput
    data class LocalFile(val uri: String) : NormalizedInput
}
