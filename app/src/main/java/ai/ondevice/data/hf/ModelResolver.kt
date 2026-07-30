package ai.ondevice.data.hf

import ai.ondevice.core.AttachmentRole
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
     * Which architecture strings mean "diffusion".
     *
     * Read from the registry rather than written here. §1.3 and Appendix A #3
     * say the allowlist must come from the source the runtime is built from, and
     * this file had its own copy of sd.cpp's SDVersion enum — five names kept by
     * hand, guaranteed to fall behind the day sd.cpp gains a sixth.
     *
     * The two additions are not architectures and are not claimed to be: `unet`
     * and `dit` are the tensor-prefix names a safetensors header exposes when
     * the repo never states a version at all, so they cannot come from the enum.
     */
    private val diffusionArchitectures: Set<String> by lazy {
        registry.architecturesFor(RuntimeRegistry.STABLE_DIFFUSION) + setOf("unet", "dit")
    }

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
        val pickleAuxiliaries = pickles.filter { AttachmentRole.classify(it) != null }
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
            ).filter { AttachmentRole.classify(it) != null }

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

        // Step 4/6 — enumerate quant variants, folding shard sets into one entry.
        val primaryFiles = when (format) {
            ModelFormat.GGUF -> ggufFiles.filterNot { isCompanionFilename(it) }
            ModelFormat.GGML_BIN -> ggmlBins
            ModelFormat.ONNX -> onnxFiles
            // For an auxiliary pack the "variants" are the individual
            // auxiliaries — canny, depth, openpose — and picking one is the
            // point, not a quality trade-off.
            else -> refineAuxiliaries(repoId, pinnedRevision, safetensors, auxiliaries)
        }

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
            arch != null && arch in diffusionArchitectures -> Modality.DIFFUSION
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

    /**
     * Correct the filename's verdict against the file's own tensor names.
     *
     * `madebyollin/taesd` is the case this exists for. It publishes three files
     * that look installable, and the naming is exactly backwards: the two called
     * `taesd_encoder`/`taesd_decoder` are standalone `nn.Sequential` dumps whose
     * tensors are `0.weight`, `1.conv.0.bias`, matching nothing sd.cpp looks
     * for — and each is half an autoencoder besides — while the one that
     * actually loads is `diffusion_pytorch_model.safetensors`, whose name says
     * nothing at all. So the app offered two files that cannot work, side by side
     * as if they were alternatives, and hid the one that can.
     *
     * sd.cpp resolves TAESD by looking for `decoder.layers.*` (see
     * `src/model/vae/tae.hpp`), so agreeing with it means reading the same names.
     *
     * Deliberately narrow. Only files that are unclassified or classified TAESD
     * are read, and only for repos with a handful of candidates — a probe per
     * file would otherwise add 29 round trips to resolving the ControlNet pack,
     * whose role never depended on tensor names in the first place. An
     * unreadable header leaves the filename's verdict standing: "cannot tell"
     * must not become "refused".
     */
    private suspend fun refineAuxiliaries(
        repoId: String,
        revision: String,
        safetensors: List<String>,
        classified: List<String>,
    ): List<String> {
        val ambiguous = safetensors.filter {
            it !in classified || AttachmentRole.classify(it) == AttachmentRole.TAESD
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
            AttachmentRole.classify(filename) != AttachmentRole.TAESD ||
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

        // Captured before the companions below are folded in. Only files the
        // shard pattern actually collapsed are shards; a graph plus its weight
        // sidecar plus a tokenizer is three files and one part. Counting after
        // the fold is what made every Kokoro variant claim "3 shards".
        val shardCounts = grouped.mapValues { (_, members) -> members.size }

        // ONNX keeps anything over 2 GB — and in practice anything at all — in a
        // sibling data file, so the graph alone measures a couple of kilobytes.
        // Reporting that as the download made a 411 MB model read "2 KB" and
        // "weights 0.00", and downloading it would have produced a graph with no
        // weights behind it.
        grouped.forEach { (label, members) ->
            val sidecars = members.flatMap { onnxSidecars(it, allFiles) }
            members.addAll(sidecars.filterNot { it in members })
            if (members.any { it.endsWith(".onnx", ignoreCase = true) }) {
                val tokenisers = onnxTextCompanions(label, allFiles)
                members.addAll(tokenisers.filterNot { it in members })
            }
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
                note = quantNote(
                    proposed.getValue(key),
                    speed,
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

    /**
     * The text-side files an ONNX model needs that are not graphs.
     *
     * A GGUF carries its vocabulary inside the file; an ONNX model does not, and
     * ships it beside the graphs as `tokenizer.json`. Collecting only `.onnx`
     * and its weight sidecars therefore installed OmniVoice complete in every
     * respect except the one that lets it read a sentence, and the failure
     * arrived as "onnx-community_OmniVoice-Onnx_int4 has no tokenizer.json"
     * *after* three quarters of a gigabyte had been fetched.
     *
     * Publishers keep a per-precision copy next to each variant — OmniVoice has
     * `int4/tokenizer.json` and `cuda/tokenizer.json` as well as one at the root
     * — so prefer the copy belonging to this variant and fall back to the root.
     * They are picked by name rather than by extension because a repo's `.json`
     * files also include configs no runtime opens.
     */
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
            // An execution-provider folder is not a choice on this device. A
            // CUDA export cannot run on a phone, and offering `cuda` beside
            // `int4` invites a gigabyte of download that will never load. These
            // are ONNX Runtime's own provider names, so the list describes our
            // runtime rather than any model.
            .filterKeys { it.substringAfterLast('/').lowercase() !in FOREIGN_PROVIDERS }
        if (byDirectory.isEmpty()) return null

        // Directories are only alternatives to each other when they hold the
        // *same* graphs. Grouping on "has any .onnx" made OmniVoice's four-graph
        // Higgs tokenizer — a component every variant needs — appear as two more
        // variants to choose between, so a five-entry list held two real choices,
        // two components and a CUDA build.
        val signature = byDirectory.mapValues { (_, paths) ->
            paths.map { it.substringAfterLast('/') }.toSortedSet()
        }
        // The publisher puts the model itself at the repo root, so the family
        // the root belongs to is the one being chosen between; everything else
        // is a part that gets added to whichever choice is made. With no root
        // graphs, the family with the most directories is the one with variants.
        val primary = signature[""] ?: signature.values
            .groupingBy { it }.eachCount().maxByOrNull { it.value }?.key
            ?: return null

        val alternatives = byDirectory.filterKeys { signature[it] == primary }
        if (alternatives.isEmpty()) return null
        val components = byDirectory.filterKeys { signature[it] != primary }

        // One directory per component family, deepest path first. Depth is how
        // publishers nest a precision under a component — OmniVoice's tokenizer
        // is `audio_tokenizer/` with `audio_tokenizer/fp16/` inside it — and the
        // nested one is both the smaller download and what its own manifest
        // says is used.
        val chosenComponents = components.entries
            .groupBy { (_, paths) -> paths.map { it.substringAfterLast('/') }.toSortedSet() }
            .mapNotNull { (_, dirs) -> dirs.maxByOrNull { it.key.count { c -> c == '/' } } }
            .flatMap { it.value }

        return alternatives.entries.associate { (directory, graphs) ->
            val label = directory.ifEmpty { "root" }
            label to (graphs + chosenComponents)
        }
    }

    /** `Qwen2.5-7B-Instruct-Q4_K_M.gguf` → `Q4_K_M`. */
    private fun extractQuantName(filename: String, info: HfModelInfo): String {
        val base = filename.substringAfterLast('/').removeSuffix(".gguf").removeSuffix(".bin").removeSuffix(".onnx")
        // ONNX exports carry their precision in a convention of their own —
        // `model_fp16`, `model_uint8`, `model_q4f16`, `model_quantized` — which
        // the GGUF suffix pattern below cannot read: `fp16` does not end in
        // `F16`, and `uint8` contains no `Q`. Both therefore fell through to the
        // whole filename and were then described as full precision, which is
        // wrong for every one of them. Reading the stem off `model` gives the
        // precision directly, and leaves a bare `model.onnx` blank so it can be
        // called what it is rather than guessed at.
        if (filename.endsWith(".onnx", ignoreCase = true)) {
            val stem = base.removePrefix("model").trim('_', '-', '.')
            return if (stem.isBlank()) ORIGINAL_EXPORT else stem.uppercase()
        }
        val match = Regex("""(?i)(IQ\d[_A-Z0-9]*|Q\d[_A-Z0-9]*|BF16|F16|F32)$""").find(base)
        return match?.value?.uppercase() ?: base.substringAfterLast('-').ifBlank { base }
    }

    /**
     * A warning that this variant will run and should probably not be chosen.
     *
     * Measured, not looked up. Bits per weight is the download size divided by
     * the parameter count the repo declares, so it needs no table of quant
     * names and cannot go stale when a new one appears — an unfamiliar
     * three-letter suffix is judged by the same arithmetic as a familiar one.
     *
     * The thresholds are where the published perplexity curves bend rather than
     * where they slope: under two bits per weight everything degrades sharply,
     * and under about two and a half a small model degrades much faster than a
     * large one, because a large one has redundancy to spend and a 1.7B does
     * not. Above that, "smaller is worse" is a trade-off the note already
     * describes and the user is entitled to make.
     *
     * Silent when the repo declares no parameter count, which is the honest
     * answer: the caution is arithmetic, and without both numbers there is none
     * to do.
     */
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

    /**
     * The one-line character note under each variant. It must never repeat the
     * speed class shown on the right — the two columns answer different
     * questions ("what does this quant cost you?" vs "which backend runs it?").
     */
    private fun quantNote(
        rawQuant: String,
        speed: SpeedClass,
        shards: Int,
        fileCount: Int = shards,
        onnx: Boolean = false,
        graphSet: Boolean = false,
    ): String = buildString {
        // A multi-graph ONNX variant is neither quantised by its label nor
        // sharded. Both of OmniVoice's read "full precision · 14 shards": the
        // label is a directory name that matches no quant pattern, and the file
        // count is four graphs plus their weight sidecars plus a tokenizer —
        // parts of one model, not slices of one file. Shards can be resumed
        // independently and parts cannot, so the word matters.
        if (graphSet) {
            append("$fileCount files · one model in parts")
            return@buildString
        }
        val quant = rawQuant.uppercase()
        // ONNX has its own vocabulary and the GGUF table gets it wrong three
        // ways on a single Kokoro screen: `model_fp16` and `model_uint8` both
        // read "full precision", and `model_q4` read "unrecognised quant name"
        // — a name the app itself had just extracted. None of these are GGUF
        // quantisations and none of them mean what the GGUF table says.
        if (onnx) {
            append(
                when {
                    quant == ORIGINAL_EXPORT.uppercase() -> "as published"
                    quant.startsWith("QUANTIZED") -> "8-bit, dynamically quantised"
                    quant.startsWith("UINT8F16") || quant.startsWith("INT8F16") ->
                        "8-bit weights, half-precision maths"
                    quant.startsWith("UINT8") || quant.startsWith("INT8") || quant.startsWith("Q8") ->
                        "8-bit weights"
                    quant.startsWith("Q4F16") || quant.startsWith("INT4F16") ->
                        "4-bit weights, half-precision maths"
                    quant.startsWith("Q4") || quant.startsWith("INT4") || quant.startsWith("BNB4") ->
                        "4-bit weights, smallest"
                    quant.startsWith("FP16") || quant.startsWith("F16") -> "half precision"
                    quant.startsWith("BF16") -> "half precision, bfloat"
                    quant.startsWith("FP32") || quant.startsWith("F32") -> "full precision"
                    else -> "as published"
                },
            )
            if (fileCount > 1) append(" · $fileCount files")
            return@buildString
        }
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
        /**
         * The label for a bare `model.onnx` when the filename says nothing about
         * precision. Naming it after what we know — that it is the export the
         * publisher put there first — beats guessing fp32 and being wrong on the
         * repos that export fp16 under that name.
         */
        const val ORIGINAL_EXPORT = "original"

        /** Below this, a heavy quantisation has no redundancy to eat into. */
        const val SMALL_MODEL_PARAMS = 4_000_000_000L

        /**
         * How many safetensors a repo may hold before header probing is skipped.
         * One request per file is fine for an autoencoder repo and absurd for a
         * fifteen-ControlNet pack, whose roles never needed probing anyway.
         */
        const val HEADER_PROBE_LIMIT = 8

        /**
         * ONNX Runtime execution providers this build does not have.
         *
         * Publishers ship a folder per provider, and a repo that offers `cuda/`
         * beside `int4/` is not offering a choice on a phone — it is offering a
         * download that cannot load. Named after providers rather than models,
         * so the list stays a statement about our runtime.
         */
        val FOREIGN_PROVIDERS = setOf(
            "cuda", "tensorrt", "trt", "dml", "directml", "openvino",
            "rocm", "migraphx", "cann", "webgpu", "coreml",
        )

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
