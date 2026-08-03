package ai.ondevice.ui.vm

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.Modality
import ai.ondevice.core.SparseParams
import ai.ondevice.core.Verdict
import ai.ondevice.data.ModelStorage
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.download.DownloadFile
import ai.ondevice.data.download.DownloadJob
import ai.ondevice.data.download.Downloader
import ai.ondevice.data.hf.CompatibilityGate
import ai.ondevice.data.hf.DeviceCapabilities
import ai.ondevice.data.hf.HfApi
import ai.ondevice.data.hf.ModelResolver
import ai.ondevice.data.hf.QuantVariant
import ai.ondevice.data.hf.RemedyAction
import ai.ondevice.data.hf.ResolvedModel
import ai.ondevice.data.hf.Resolution
import ai.ondevice.data.hf.VerdictResult
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.engine.RuntimeRegistry
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject

@HiltViewModel
class AddModelViewModel @Inject constructor(
    private val resolver: ModelResolver,
    private val api: HfApi,
    private val registry: RuntimeRegistry,
    private val capabilities: DeviceCapabilities,
    private val downloader: Downloader,
    private val storage: ModelStorage,
    private val prefs: AppPrefs,
    private val db: OnDeviceDatabase,
) : ViewModel() {

    private val _state = MutableStateFlow(AddModelState(totalRamBytes = capabilities.totalRamBytes))
    val state: StateFlow<AddModelState> = _state.asStateFlow()

    /**
     * The resolution before any role narrowing.
     *
     * Narrowing throws files away, so answering Role a second time has to start
     * from what the repo actually holds rather than from what the last answer
     * left behind.
     */
    private var unnarrowed: ResolvedModel? = null

    fun onQueryChange(value: String) {
        // Typed by hand, so any role the last starter card implied is stale.
        _state.value = _state.value.copy(query = value, intendedRole = null, roleWasSuggested = false)
    }

    /**
     * A starter card carries what it is for, and dropping that was why two
     * cards looked identical.
     *
     * `h94/IP-Adapter` holds the adapters *and* the CLIP-Vision encoders they
     * read through, so it appears twice in the list under two roles — and both
     * cards resolved to the same repo and offered the same twelve files, with
     * nothing to say which of them belonged to which card. The role travels
     * with the pick now: it seeds the Role answer, and the file list narrows to
     * what can fill it.
     */
    fun pickStarter(entry: ai.ondevice.core.StarterModel) {
        _state.value = _state.value.copy(
            query = entry.repoId,
            intendedRole = entry.role,
            roleWasSuggested = entry.role != null,
            selectedRole = entry.role,
            roleAnswered = entry.role != null,
            selectedModality = entry.modality,
        )
        resolve()
    }

    /** Resolve a repo, or search for one. */
    fun resolve() {
        val query = _state.value.query.trim()
        if (query.isEmpty()) return
        if (!query.contains('/')) {
            searchHub(query)
            return
        }
        _state.value = _state.value.copy(
            resolving = true,
            refusal = null,
            resolved = null,
            verdict = null,
            searchResults = emptyList(),
        )

        viewModelScope.launch {
            when (val outcome = resolver.resolve(query, blockPickle = prefs.blockPickle.first())) {
                is Resolution.Refused -> _state.value = _state.value.copy(resolving = false, refusal = outcome)
                is Resolution.Resolved -> {
                    // If HF hadn't parsed the metadata, read the GGUF header
                    // ourselves rather than guessing at the KV term.
                    val first = outcome.model.quants.firstOrNull()
                    val enriched = if (first != null) resolver.enrichFromHeader(outcome.model, first) else outcome.model
                    unnarrowed = enriched
                    // The card's role if it came from one, otherwise whatever
                    // Role says — a person who has answered "VAE" has stated
                    // the same intent the card would have.
                    val narrowed = narrowToRole(
                        enriched,
                        _state.value.intendedRole ?: _state.value.selectedRole,
                    )
                    // Every producer of a variant list passes through here, so
                    // this is where the names are made to stand apart — before
                    // one of them is picked as the default and stored.
                    val forRole = narrowed.model.withDistinctQuantNames()
                    val defaultQuant = pickDefaultQuant(forRole)
                    _state.value = _state.value.copy(
                        resolving = false,
                        resolved = forRole,
                        selectedQuant = defaultQuant?.name,
                        // Only claimed when it happened.
                        roleWasSuggested = narrowed.applied,
                        companionChoice = forRole.companions.associate { it.role to it.selected },
                        contextTokens = (forRole.contextLength ?: 8192).coerceAtMost(8192),
                    )
                    recomputeVerdict()
                }
            }
        }
    }

    /**
     * Offer the files that fill the role the card was for, and only those.
     *
     * A repo does not agree with us about what its main event is. Asked for
     * SD 3.5's VAE, the resolver calls the text encoders the primary files and
     * files the VAE away as a companion — so the list showed CLIP-L, CLIP-G and
     * two T5s, none of which is a VAE, under a heading claiming it had narrowed
     * to VAEs. Worse, taking the 168 MB VAE meant also taking a primary file,
     * the cheapest being 246 MB of an encoder nobody asked for.
     *
     * Where the role's files are companions, they are promoted to *be* the
     * choice and the companions are dropped: a component card then downloads
     * the component and nothing else. Where nothing in the repo fills the role,
     * the model is returned untouched and unlabelled — a repo whose filenames
     * do not announce a role is better shown whole than shown as nothing, and
     * better shown without a claim than with a false one.
     */
    private fun narrowToRole(model: ResolvedModel, role: AttachmentRole?): Narrowed {
        if (role == null) return Narrowed(model, applied = false)

        val fromQuants = model.quants.filter { variant ->
            variant.files.any { AttachmentRole.classify(it.filename) == role }
        }
        if (fromQuants.isNotEmpty()) {
            return Narrowed(model.copy(quants = fromQuants), applied = true)
        }

        // The two role enums do not know about each other; their names are the
        // only thing they share, and for these roles they agree.
        val promoted = model.companions
            .filter { group ->
                ai.ondevice.core.AttachmentRole.entries
                    .firstOrNull { it.name == group.role.name } == role
            }
            .flatMap { it.candidates }
            .map { it.file }
            // The group already knows what it holds — the resolver put it
            // there. Asking a second classifier to agree only invited them to
            // disagree, which they did: a diffusers VAE lives at
            // `vae/diffusion_pytorch_model.safetensors`, and one of the two
            // reads the directory while the other reads the filename.
            .distinctBy { it.filename }
            // The weights, not the config beside them.
            .filter { !it.filename.endsWith(".json", ignoreCase = true) }

        if (promoted.isEmpty()) return Narrowed(model, applied = false)

        return Narrowed(
            model.copy(
                quants = promoted
                    .map { file ->
                        QuantVariant(
                            name = file.filename.substringAfterLast('/').removeSuffix(".safetensors"),
                            files = listOf(file),
                            note = "${role.label} · nothing else is downloaded",
                        )
                    }
                    .sortedBy { it.totalBytes },
                // Already the thing being chosen; offering them twice would
                // download the file and then download it again.
                companions = emptyList(),
            ),
            applied = true,
        )
    }

    /** A narrowing, and whether it happened — the label may only claim the second. */
    private data class Narrowed(val model: ResolvedModel, val applied: Boolean)

    fun selectQuant(name: String) {
        _state.value = _state.value.copy(selectedQuant = name)
        recomputeVerdict()
    }

    /**
     * Choose which file fills a companion role — including none of them.
     *
     * A required role used to snap back to its selection when you tried to
     * clear it, on the reasoning that the model is unusable without one. That
     * reasoning aged out: a role is now filled from the library, by any
     * installed file that fits, so the copy sitting in this repo is one
     * candidate rather than the only one. Somebody who already has the SDXL
     * VAE has no use for a second, and refusing to let them say so meant
     * downloading it anyway.
     *
     * What it does not mean is that the model will run. Nothing chosen here
     * shows as a warning below, and the Image tab says the same before a run.
     */
    fun chooseCompanion(role: ai.ondevice.data.hf.CompanionRole, filename: String) {
        val group = _state.value.resolved?.companions?.firstOrNull { it.role == role } ?: return
        val current = _state.value.companionChoice[role] ?: group.selected
        val parts = group.kind == ai.ondevice.data.hf.CompanionGroup.Kind.PARTS
        apply(
            role,
            when {
                parts && filename in current -> current - filename
                parts -> current + filename
                filename in current -> emptySet()
                else -> setOf(filename)
            },
        )
    }

    /**
     * Take a role's files, or none of them.
     *
     * "All" means all only where the role takes several — voice packs. A role
     * that takes one file gets the preferred one, never every precision of it
     * at once, which is a bug this app has already had.
     */
    fun chooseAllCompanions(role: ai.ondevice.data.hf.CompanionRole, all: Boolean) {
        val group = _state.value.resolved?.companions?.firstOrNull { it.role == role } ?: return
        val everything = if (group.kind == ai.ondevice.data.hf.CompanionGroup.Kind.PARTS) {
            group.candidates.map { it.file.filename }.toSet()
        } else {
            setOfNotNull(
                ai.ondevice.data.hf.CompanionGrouping.preferred(group.candidates)?.file?.filename,
            )
        }
        apply(role, if (all) everything else emptySet())
    }

    /** "Just the weights" — every companion off in one tap, for a library that already has them. */
    fun skipAllCompanions() {
        val groups = _state.value.resolved?.companions ?: return
        _state.value = _state.value.copy(
            companionChoice = groups.associate { it.role to emptySet<String>() },
        )
        recomputeVerdict()
    }

    /** Back to what the resolver picked. */
    fun restoreCompanionDefaults() {
        val groups = _state.value.resolved?.companions ?: return
        _state.value = _state.value.copy(companionChoice = groups.associate { it.role to it.selected })
        recomputeVerdict()
    }

    private fun apply(role: ai.ondevice.data.hf.CompanionRole, selection: Set<String>) {
        _state.value = _state.value.copy(
            companionChoice = _state.value.companionChoice + (role to selection),
        )
        recomputeVerdict()
    }

    /** Recomputed live as the context slider moves — SPEC §3.3. */
    fun setContext(tokens: Int) {
        _state.value = _state.value.copy(contextTokens = tokens)
        recomputeVerdict()
    }

    private fun recomputeVerdict() {
        val resolved = _state.value.resolved ?: return
        val quant = resolved.quants.firstOrNull { it.name == _state.value.selectedQuant } ?: return

        viewModelScope.launch {
            val reserve = prefs.storageReserveMb.first() * 1_000_000L
            // Only autoregressive models have a KV cache.
            val autoregressive = resolved.modality == ai.ondevice.core.Modality.TEXT ||
                resolved.modality == ai.ondevice.core.Modality.VISION
            val estimate = CompatibilityGate.estimate(
                // Companions are resident too, and for a vision or diffusion model the companion can outweigh the model — a T5-XXL encoder is gigabytes.
                weightsBytes = quant.totalBytes + _state.value.companionBytes,
                layers = resolved.layers,
                contextTokens = if (autoregressive) _state.value.contextTokens else 0,
                embeddingLengthKv = resolved.embeddingLengthKv,
                embeddingLength = resolved.embeddingLength,
            )
            val verdict = CompatibilityGate.verdict(
                estimate = estimate,
                availableRamBytes = capabilities.availableRamBytes,
                freeStorageBytes = capabilities.freeStorageBytes,
                storageReserveBytes = reserve,
                archSupported = resolved.architecture?.let { registry.supportsArchitecture(it) } ?: true,
                hasRuntimeForFormat = registry.supportsFormat(resolved.format),
            )
            _state.value = _state.value.copy(
                verdict = VerdictResult(
                    verdict = verdict,
                    estimate = estimate,
                    availableRamBytes = capabilities.availableRamBytes,
                    freeStorageBytes = capabilities.freeStorageBytes,
                    architecture = resolved.architecture,
                ),
            )
        }
    }

    /** What to pre-select: the smallest that runs and is not a bad idea, not the smallest. */
    private fun pickDefaultQuant(model: ResolvedModel): QuantVariant? {
        val runnable = model.quants.filter { it.runnable }.ifEmpty { model.quants }
        val advisable = runnable.filter { it.cautionReason == null }.ifEmpty { runnable }
        return advisable.minByOrNull { it.totalBytes }
    }

    /** Queue the primary file, every shard, and every required companion as one atomic job (SPEC §3.4). */
    private fun searchHub(query: String) {
        _state.value = _state.value.copy(
            searching = true,
            refusal = null,
            resolved = null,
            verdict = null,
            searchResults = emptyList(),
        )
        viewModelScope.launch {
            val results = api.search(query).getOrElse { emptyList() }
            _state.value = _state.value.copy(searching = false, searchResults = results)
        }
    }

    /** Picking a result is the same as having typed its id: resolve it. */
    fun openSearchResult(repoId: String) {
        _state.value = _state.value.copy(query = repoId, searchResults = emptyList())
        resolve()
    }

    fun setModality(modality: Modality) {
        _state.value = _state.value.copy(selectedModality = modality)
    }

    /** Null is a real answer — "no role, this is a base model" — hence the flag. */
    fun setRole(role: AttachmentRole?) {
        _state.value = _state.value.copy(selectedRole = role, roleAnswered = true)

        // Answering Role is the clearest statement of what this download is
        // for, and it used to change nothing about what was offered: choosing
        // VAE still left a text encoder selected and added the VAE beside it,
        // so "just the VAE" downloaded 414 MB of which 246 MB was an encoder
        // nobody asked for.
        val base = unnarrowed ?: return
        val narrowed = narrowToRole(base, role)
        val defaultQuant = pickDefaultQuant(narrowed.model)
        _state.value = _state.value.copy(
            resolved = narrowed.model,
            selectedQuant = defaultQuant?.name,
            roleWasSuggested = narrowed.applied,
            companionChoice = narrowed.model.companions.associate { it.role to it.selected },
        )
        recomputeVerdict()
    }

    fun download() {
        val resolved = _state.value.resolved ?: return
        val quant = resolved.quants.firstOrNull { it.name == _state.value.selectedQuant } ?: return
        if (_state.value.verdict?.verdict?.runnable != true) return
        // The button is disabled for this, but a guard that only exists in the UI is one screen away from not existing.
        if (quant.blockedReason != null) return
        // Type and role are the user's to state.
        val modality = _state.value.selectedModality ?: return
        if (!_state.value.roleAnswered) return
        val role = _state.value.selectedRole

        viewModelScope.launch {
            val modelId = "${resolved.repoId}:${quant.name}"
            val wifiOnly = prefs.wifiOnly.first()
            val connections = prefs.parallelConnections.first()

            val primaries = quant.files.mapIndexed { index, file ->
                DownloadFile(
                    filename = file.filename,
                    url = api.resolveUrl(resolved.repoId, file.filename, resolved.revision),
                    destPath = storage.pathFor(modelId, file.filename),
                    sizeBytes = file.sizeBytes,
                    expectedSha256 = file.sha256,
                    shardIndex = if (quant.isSharded) index else null,
                    shardCount = if (quant.isSharded) quant.files.size else null,
                )
            }
            val companions = _state.value.chosenCompanions.map { companion ->
                DownloadFile(
                    filename = companion.file.filename,
                    url = api.resolveUrl(resolved.repoId, companion.file.filename, resolved.revision),
                    destPath = storage.pathFor(modelId, companion.file.filename),
                    sizeBytes = companion.file.sizeBytes,
                    expectedSha256 = companion.file.sha256,
                    companionRole = companion.role.name.lowercase(),
                )
            }

            // The library record is written up front so a resumed download after
            // a reboot still knows what it is installing.
            db.models().upsert(
                ModelEntity(
                    id = modelId,
                    hfRepo = resolved.repoId,
                    revision = resolved.revision,
                    localPath = primaries.first().destPath,
                    format = resolved.format,
                    architecture = resolved.architecture,
                    quant = quant.name,
                    sizeBytes = quant.totalBytes + companions.sumOf { it.sizeBytes },
                    sha256 = quant.files.firstOrNull()?.sha256,
                    modality = modality,
                    contextLength = resolved.contextLength,
                    parameterCount = resolved.parameterCount,
                    chatTemplate = resolved.chatTemplate,
                    bosToken = resolved.bosToken,
                    eosToken = resolved.eosToken,
                    companionPathsJson = SparseParams.of(
                        *companions.map { it.companionRole.orEmpty() to it.destPath }.toTypedArray(),
                    ).toJsonString(),
                    installedAt = System.currentTimeMillis(),
                    // Not installed yet — the downloader stamps this when the
                    // last file verifies, and until then no picker offers it.
                    completedAt = null,
                    lastUsedAt = null,
                    pinned = false,
                    favourite = false,
                    notes = null,
                    backendOverride = null,
                    paramOverridesJson = SparseParams.of(
                        "n_ctx" to _state.value.contextTokens,
                    ).toJsonString(),
                    defaultPresetId = null,
                    displayName = resolved.displayName,
                    attachmentRole = role,
                ),
            )

            downloader.enqueue(
                DownloadJob(
                    id = UUID.randomUUID().toString(),
                    modelId = modelId,
                    displayName = resolved.displayName,
                    hfRepo = resolved.repoId,
                    revision = resolved.revision,
                    files = primaries + companions,
                    state = ai.ondevice.core.DownloadState.QUEUED,
                    error = null,
                    connections = connections,
                    wifiOnly = wifiOnly,
                    attempts = 0,
                ),
            )
        }
    }

    fun applyRemedy(action: RemedyAction) {
        when (action) {
            is RemedyAction.SearchRepo -> {
                _state.value = _state.value.copy(query = action.query)
                resolve()
            }
            is RemedyAction.OpenMirror -> {
                _state.value = _state.value.copy(query = "${action.owner}/${action.repo}")
                resolve()
            }
            is RemedyAction.ShowSmallerQuants -> {
                _state.value = _state.value.copy(
                    selectedQuant = _state.value.resolved?.quants?.minByOrNull { it.totalBytes }?.name,
                )
                recomputeVerdict()
            }
            else -> _state.value = _state.value.copy(pendingAction = action)
        }
    }

    fun clearPendingAction() {
        _state.value = _state.value.copy(pendingAction = null)
    }

    fun importLocal() {
        _state.value = _state.value.copy(pendingAction = RemedyAction.OpenUrl("saf://import"))
    }
}

data class AddModelState(
    val query: String = "",
    val resolving: Boolean = false,
    val resolved: ResolvedModel? = null,
    val refusal: Resolution.Refused? = null,
    val selectedQuant: String? = null,
    val contextTokens: Int = 8192,
    val verdict: VerdictResult? = null,
    val totalRamBytes: Long = 0,
    val pendingAction: RemedyAction? = null,
    /** What this model *is*, and which add-on slot it fills — both chosen here rather than inferred, which is why they start unset. */
    val searching: Boolean = false,
    val searchResults: List<ai.ondevice.data.hf.HfSearchResult> = emptyList(),
    val selectedModality: Modality? = null,
    val selectedRole: AttachmentRole? = null,
    /** The role the starter card was listed under, when the pick came from one. */
    val intendedRole: AttachmentRole? = null,
    /** Whether the role above was suggested rather than answered by hand. */
    val roleWasSuggested: Boolean = false,
    /** True once [selectedRole] has been answered, including answered as "none". */
    val roleAnswered: Boolean = false,
    /** Which file fills each companion role, seeded from the resolver's defaults. */
    val companionChoice: Map<ai.ondevice.data.hf.CompanionRole, Set<String>> = emptyMap(),
) {
    val runnable: Boolean get() = verdict?.verdict?.runnable == true
    val isRefused: Boolean get() = refusal != null || verdict?.verdict?.runnable == false

    /** Type and role are required, so Download stays closed until both are set. */
    val classified: Boolean get() = selectedModality != null && roleAnswered

    /** The companion files this download will actually fetch. */
    val chosenCompanions: List<ai.ondevice.data.hf.CompanionFile>
        get() = resolved?.companions.orEmpty().flatMap { group ->
            val picked = companionChoice[group.role] ?: group.selected
            group.candidates.filter { it.file.filename in picked }
        }

    val companionBytes: Long get() = chosenCompanions.sumOf { it.file.sizeBytes }
}
