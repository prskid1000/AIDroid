package ai.ondevice.data.hf

import ai.ondevice.core.Fmt
import ai.ondevice.core.Modality
import ai.ondevice.core.ModelFormat
import ai.ondevice.core.RefusalKind
import ai.ondevice.core.SpeedClass
import ai.ondevice.engine.RuntimeRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * SPEC §3.2, the resolution pipeline.
 *
 * The whole point of this class is that it contains **no model-specific
 * knowledge**. It classifies from architecture and file shape, reads templates
 * and context lengths from metadata, and asks the runtime registry — which is
 * generated from pinned upstream source — whether an architecture is supported.
 * If a `when (modelName)` branch ever appears in here, §1.1 and §1.3 have both
 * been violated (Appendix A #2, #3).
 */
class ModelResolver(
    private val api: HfApi,
    private val registry: RuntimeRegistry,
) {

    /**
     * Accepted inputs, per §3.2: `owner/repo`, a huggingface.co URL in any of
     * its `/tree/` and `/blob/` variants, a direct `.gguf` URL on any host, and
     * a local `content://` SAF URI.
     */
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

    /**
     * Steps 2–7 of §3.2. Returns either a [ResolvedModel] or a [Resolution.Refused]
     * carrying a specific message and at least one actionable remedy.
     */
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

        // The pickle block exists because `torch.load` executes whatever the
        // file tells it to. This app never calls torch. The one loader that
        // reads these is sd.cpp's, and it does not execute: it walks the opcodes
        // and, on REDUCE, pushes a null unless the global is a known
        // tensor-rebuild — there is no interpreter behind it (see
        // src/model_io/pickle_io.cpp).
        //
        // That distinction matters here and nowhere else, because the ESRGAN
        // ecosystem is entirely .pth. Fourteen repos surveyed, not one shipping
        // safetensors — so a blanket block means no upscaler can ever be
        // installed. The exception is therefore as narrow as it can be: a pickle
        // is tolerated only when its own filename classifies it as a diffusion
        // auxiliary, which is also the only route by which it reaches that
        // non-executing reader. Anything else still refuses.
        val pickleAuxiliaries = pickles.filter { ai.ondevice.core.AttachmentRole.classify(it) != null }
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

        // Diffusion auxiliaries are the exception to the GGUF rule, and it is not
        // a small one: safetensors is the *native* format for a LoRA, a
        // ControlNet, a VAE, a TAESD decoder or an IP-Adapter, and sd.cpp loads
        // all of them directly. Refusing them for "needing a desktop to convert"
        // was advice for a conversion that must not happen — and because every
        // published auxiliary ships this way, it made the Image screen's
        // Attachments section impossible to fill by any route. That section
        // looked unimplemented for exactly this reason.
        val auxiliaries = (
            safetensors +
                files.filter { it.endsWith(".ckpt", ignoreCase = true) } +
                pickleAuxiliaries
            ).filter { ai.ondevice.core.AttachmentRole.classify(it) != null }

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

        // Step 4/6 — enumerate quant variants, folding shard sets into one entry.
        val primaryFiles = when (format) {
            ModelFormat.GGUF -> ggufFiles.filterNot { isCompanionFilename(it) }
            ModelFormat.GGML_BIN -> ggmlBins
            ModelFormat.ONNX -> onnxFiles
            // For an auxiliary pack the "variants" are the individual
            // auxiliaries — canny, depth, openpose — and picking one is the
            // point, not a quality trade-off.
            else -> auxiliaries
        }

        // Step 7 — pin *first*, then read everything at the pin.
        //
        // This ordering is the whole point. "main" is a moving target: a repo
        // that is re-uploaded keeps its filenames and changes their contents,
        // so sizes and hashes read at `main` describe a different file from the
        // one a URL pinned to an older commit will serve. That mismatch does
        // not fail loudly at resolve time — it fails after a 382 MB download,
        // as a checksum error blaming the file. Resolving the revision to a
        // concrete commit once, and using that same commit for paths-info and
        // for every download URL, is what makes the sha256 check meaningful.
        val pinnedRevision = info.sha?.takeIf { it.isNotBlank() } ?: repoRef.revision

        // A multi-graph ONNX model is grouped by directory, not by file — see
        // onnxGraphSets. Null for every other shape.
        val graphSets = if (format == ModelFormat.ONNX) onnxGraphSets(onnxFiles) else null

        // The sidecar weight files have to be priced too, or the variant reports
        // the size of a graph stub. They are asked for alongside the graphs
        // rather than discovered later, because paths-info is one round trip.
        val sidecars = (graphSets?.values?.flatten() ?: primaryFiles)
            .flatMap { onnxSidecars(it, files) }
        val wanted = (
            (graphSets?.values?.flatten() ?: primaryFiles) +
                sidecars +
                files.filter { isCompanionFilename(it) }
            ).distinct()
        val sizeLookup = api.pathsInfo(repoId, wanted.take(128), pinnedRevision)
            .getOrDefault(emptyList())
            .associateBy { it.path }

        val quants = enumerateQuants(
            files = primaryFiles,
            sizes = sizeLookup,
            info = info,
            allFiles = files,
            preGrouped = graphSets,
        )
        if (quants.isEmpty()) {
            return@withContext Resolution.Refused(
                kind = RefusalKind.NO_RUNTIME,
                title = "No loadable weights",
                subject = repoId,
                detail = "The repo lists artifacts but none of them resolved to a downloadable file.",
                remedies = listOf(Remedy("Open repo page", RemedyAction.OpenUrl("${HfApi.BASE}/$repoId"))),
            )
        }

        val companions = detectCompanions(
            files = files,
            sizes = sizeLookup,
            variantFiles = quants.flatMap { variant -> variant.files.map { it.filename } }.toSet(),
        )
        val modality = classifyModality(info, format, files, companions)

        // Step 3 continued — the architecture must be one the bundled runtime
        // knows. The registry's list is generated from upstream source, so this
        // question has a real answer rather than a maintained guess.
        val arch = info.gguf?.architecture ?: inferArchitectureFromTags(info)
        if (format == ModelFormat.GGUF && arch != null && !registry.supportsArchitecture(arch)) {
            return@withContext unsupportedArchRefusal(repoId, arch)
        }

        // Deliberately *not* a per-file `lastCommit.id`: that is the commit
        // which last touched one particular file, which for a repo uploaded in
        // several passes is older than the tree the hashes were read from.
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

    /**
     * When HF hasn't parsed a repo's metadata, pull the first megabyte of the
     * chosen file and read the GGUF header directly. Same output shape, so the
     * caller can't tell which path produced it.
     */
    suspend fun enrichFromHeader(model: ResolvedModel, quant: QuantVariant): ResolvedModel {
        if (model.layers != null && model.contextLength != null && model.chatTemplate != null) return model
        val first = quant.files.firstOrNull() ?: return model
        val url = api.resolveUrl(model.repoId, first.filename, model.revision)
        val bytes = api.rangeGet(url, GgufHeaderReader.HEADER_BYTES).getOrNull() ?: return model
        val meta = GgufHeaderReader.parse(bytes).getOrNull() ?: return model
        val architecture = model.architecture ?: meta.architecture

        // Learning the architecture can change what the model *is*. The first
        // classification ran before this file had been read at all, so a repo
        // whose HF metadata block is empty — which is exactly the case that
        // brings us here — was classified as text by default. SD-Turbo lands
        // that way: `sd2` in the header, nothing in the API. Re-deriving the
        // modality once the header is known is the difference between offering
        // a diffusion model in the chat picker and offering it on the Image
        // screen where it can actually run.
        val modality = if (model.modality == Modality.TEXT && architecture != null) {
            when (architecture.lowercase()) {
                in DIFFUSION_ARCHITECTURES -> Modality.DIFFUSION
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

    /**
     * Modality comes from architecture and file shape. A repo called
     * "whisper-something" that ships a text GGUF is a text model; a repo with
     * an `mmproj-*.gguf` beside its weights is a vision model whatever it's
     * called.
     */
    private fun classifyModality(
        info: HfModelInfo,
        format: ModelFormat,
        files: List<String>,
        companions: List<CompanionFile>,
    ): Modality {
        val arch = info.gguf?.architecture?.lowercase()
        return when {
            format == ModelFormat.GGML_BIN || arch == "whisper" -> Modality.SPEECH_TO_TEXT
            // A safetensors repo that got this far did so because its files
            // classify as diffusion auxiliaries, so it belongs to the diffusion
            // runtime — not to whatever its tags claim. h94/IP-Adapter is tagged
            // text-to-image, which would otherwise file an adapter as a base
            // model and offer it in the Image screen's model picker.
            format == ModelFormat.SAFETENSORS -> Modality.DIFFUSION
            files.any { it.contains("voices", true) && it.endsWith(".bin") } &&
                files.any { it.endsWith(".onnx") } -> Modality.TEXT_TO_SPEECH
            format == ModelFormat.ONNX && info.tags.any { it.contains("text-to-speech", true) } -> Modality.TEXT_TO_SPEECH
            arch != null && arch in DIFFUSION_ARCHITECTURES -> Modality.DIFFUSION
            files.any { it.equals("model_index.json", true) } -> Modality.DIFFUSION
            files.any { it.contains("unet", true) } && files.any { it.contains("vae", true) } -> Modality.DIFFUSION
            // A single-file SD/SDXL GGUF has none of the shapes above: no
            // `model_index.json`, no separate unet, and an architecture string
            // llama.cpp's enum has never heard of. What it does have is the
            // repo's declared pipeline, which is metadata rather than a name —
            // so this stays inside §1.3 while catching the case that would
            // otherwise install a diffusion model into the chat picker.
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
    /**
     * Companions are things a model needs *alongside* it — a vision projector, a
     * TAESD decoder, Kokoro's voice packs. They are never the alternatives the
     * user is choosing between.
     *
     * [variantFiles] is excluded for that reason, and it matters most for an
     * auxiliary pack: every file in the ControlNet v1.1 repo contains "control"
     * and ends in .safetensors, so all fifteen matched the companion rule and
     * were auto-paired behind whichever one was chosen — about 1.9 GB of rival
     * ControlNets attached to a 723 MB download. Kokoro is unaffected, since its
     * voice packs are .bin and its variants are .onnx, which is the case
     * companion detection exists for.
     */
    private fun detectCompanions(
        files: List<String>,
        sizes: Map<String, HfPathInfo>,
        variantFiles: Set<String> = emptySet(),
    ): List<CompanionFile> = files.mapNotNull { name ->
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
    }.distinctBy { it.role to it.file.filename }

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

    /**
     * Fold shard sets — `model-00001-of-00003.gguf` and its siblings become one
     * variant with three files, downloaded as one atomic job.
     */
    private fun enumerateQuants(
        files: List<String>,
        sizes: Map<String, HfPathInfo>,
        info: HfModelInfo,
        /** Every file in the repo, so a graph can find its weight sidecar. */
        allFiles: List<String> = files,
        /**
         * Pre-grouped variants, when the repo's shape is not one-file-per-choice.
         * Keyed by the label to show.
         */
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

        // ONNX keeps anything over 2 GB — and in practice anything at all — in a
        // sibling data file, so the graph alone measures a couple of kilobytes.
        // Reporting that as the download made a 411 MB model read "2 KB" and
        // "weights 0.00", and downloading it would have produced a graph with no
        // weights behind it.
        grouped.forEach { (_, members) ->
            val sidecars = members.flatMap { onnxSidecars(it, allFiles) }
            members.addAll(sidecars.filterNot { it in members })
        }

        // A quant suffix only identifies a variant when the repo holds one model.
        // whisper.cpp's repo holds tiny/base/small/medium/large *and* several
        // quantisations of each, so keying on the suffix alone produced six rows
        // all labelled "Q5_1" — a list in which you cannot tell base from small,
        // which is worse than no list. Where the suffix does not distinguish,
        // fall back to the part of the filename that does.
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
            val speed = CompatibilityGate.speedClassFor(quantName, registry.hasOpenClBackend)
            QuantVariant(
                name = quantName,
                files = remoteFiles,
                speedClass = speed,
                // The character note is about the *quantisation*, so it reads
                // the suffix even when the label had to be widened to stay
                // unambiguous.
                note = quantNote(proposed.getValue(key), speed, sorted.size),
            )
        }.sortedBy { it.totalBytes }
    }

    /**
     * The whole filename, minus the noise every file in the repo shares.
     *
     * `ggml-base.en-q5_1.bin` → `base.en-q5_1`. Used only when the quant suffix
     * fails to tell two variants apart, so the label stays short in the common
     * case and stays *correct* in the awkward one.
     */
    private fun distinguishingName(filename: String): String =
        filename.substringAfterLast('/')
            .removeSuffix(".gguf").removeSuffix(".bin").removeSuffix(".onnx")
            .removePrefix("ggml-")
            .removePrefix("model-")

    /** An ONNX graph's external weight file, under any of the three spellings in use. */
    private fun onnxSidecars(graph: String, allFiles: List<String>): List<String> {
        if (!graph.endsWith(".onnx", ignoreCase = true)) return emptyList()
        val candidates = setOf("$graph.data", "${graph}_data", "$graph.onnx_data")
        return allFiles.filter { it in candidates }
    }

    /**
     * Group a multi-graph ONNX model by directory instead of by file.
     *
     * Some ONNX "models" are a *set* of graphs that only work together —
     * OmniVoice is an embedding encoder, a Qwen3 backbone, a codebook head and a
     * vocoder — and the publisher ships the whole set once per precision, in
     * sibling folders. Listing the graphs as quant variants asks the user to pick
     * one of seventeen when they need four, and the three identical
     * `audio_embeddings_encoder` rows are indistinguishable anyway.
     *
     * The signal is structural rather than a filename: **a basename that appears
     * in more than one directory** means the directories are the alternatives and
     * the files within one are components. Kokoro, whose eight graphs are genuine
     * precision variants, keeps them in a single folder under distinct names — so
     * it does not trip this and still lists eight choices. Verified against both.
     *
     * Returns null when the repo is the ordinary one-file-per-choice shape.
     */
    private fun onnxGraphSets(onnxFiles: List<String>): Map<String, List<String>>? {
        if (onnxFiles.size < 2) return null
        val byBase = onnxFiles.groupBy { it.substringAfterLast('/') }
        val repeated = byBase.filterValues { paths ->
            paths.map { it.substringBeforeLast('/', "") }.distinct().size > 1
        }
        if (repeated.isEmpty()) return null

        val byDirectory = onnxFiles.groupBy { it.substringBeforeLast('/', "") }
        // Graphs missing from a given directory are supplied from wherever they
        // do exist, preferring the largest copy — for OmniVoice that pairs the
        // int4 backbone with the fp16 tokenizer graphs, which is the combination
        // that actually produces speech rather than noise.
        return byDirectory.entries.associate { (directory, graphs) ->
            val present = graphs.map { it.substringAfterLast('/') }.toSet()
            val missing = byBase.filterKeys { it !in present }.values.mapNotNull { paths ->
                paths.maxByOrNull { it.length }
            }
            val label = directory.ifEmpty { "root" }
            label to (graphs + missing)
        }
    }

    /** `Qwen2.5-7B-Instruct-Q4_K_M.gguf` → `Q4_K_M`. */
    private fun extractQuantName(filename: String, info: HfModelInfo): String {
        val base = filename.substringAfterLast('/').removeSuffix(".gguf").removeSuffix(".bin").removeSuffix(".onnx")
        val match = Regex("""(?i)(IQ\d[_A-Z0-9]*|Q\d[_A-Z0-9]*|BF16|F16|F32)$""").find(base)
        return match?.value?.uppercase() ?: base.substringAfterLast('-').ifBlank { base }
    }

    /**
     * The one-line character note under each variant. It must never repeat the
     * speed class shown on the right — the two columns answer different
     * questions ("what does this quant cost you?" vs "which backend runs it?").
     */
    private fun quantNote(rawQuant: String, speed: SpeedClass, shards: Int): String = buildString {
        val quant = rawQuant.uppercase()
        append(
            when {
                speed == SpeedClass.OPENCL_FAST -> "Adreno fast path"
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
                // Q4_0/Q4_1/Q5_0/Q5_1 — the pre-K quantisations. Still common
                // in whisper.cpp's own repo, so they need a real description
                // rather than falling through to "unrecognised".
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
                        speedClass = CompatibilityGate.speedClassFor(name, registry.hasOpenClBackend),
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
        /** Known to stable-diffusion.cpp; matched on architecture, not repo name. */
        val DIFFUSION_ARCHITECTURES = setOf("sd1", "sd2", "sdxl", "sd3", "flux", "unet", "dit")
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
