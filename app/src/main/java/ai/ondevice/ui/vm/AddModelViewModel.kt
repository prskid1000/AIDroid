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

/**
 * S1's state machine, and the place the "no path reaches a native load without
 * passing the gate" rule (Appendix A #5) is actually enforced: the download
 * button is only enabled once a [VerdictResult] says the model is runnable.
 */
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

    fun onQueryChange(value: String) {
        _state.value = _state.value.copy(query = value)
    }

    /**
     * Resolve a repo, or search for one.
     *
     * A Hugging Face id is `owner/name`, so text with no slash in it cannot be
     * one — and resolving it could only ever fail with "no such repo". Treating
     * that as a search costs the user nothing and turns a dead end into the
     * thing they were trying to do. Anything with a slash, a URL, or a direct
     * .gguf link still resolves exactly as before.
     */
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
                    val defaultQuant = pickDefaultQuant(enriched)
                    _state.value = _state.value.copy(
                        resolving = false,
                        resolved = enriched,
                        selectedQuant = defaultQuant?.name,
                        contextTokens = (enriched.contextLength ?: 8192).coerceAtMost(8192),
                    )
                    recomputeVerdict()
                }
            }
        }
    }

    fun selectQuant(name: String) {
        _state.value = _state.value.copy(selectedQuant = name)
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
            // Only autoregressive models have a KV cache. Passing a context to
            // the estimator for a diffusion or transcription model would invent
            // a number the user could act on, so the context is zeroed for them
            // and the summary drops the term rather than printing a fiction.
            val autoregressive = resolved.modality == ai.ondevice.core.Modality.TEXT ||
                resolved.modality == ai.ondevice.core.Modality.VISION
            val estimate = CompatibilityGate.estimate(
                weightsBytes = quant.totalBytes,
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
                speedClass = quant.speedClass,
            )
            _state.value = _state.value.copy(
                verdict = VerdictResult(
                    verdict = verdict,
                    estimate = estimate,
                    availableRamBytes = capabilities.availableRamBytes,
                    freeStorageBytes = capabilities.freeStorageBytes,
                ),
            )
        }
    }

    /** Prefer the Adreno fast path when one exists and fits. */
    private fun pickDefaultQuant(model: ResolvedModel): QuantVariant? =
        model.quants.firstOrNull { it.speedClass == ai.ondevice.core.SpeedClass.OPENCL_FAST }
            ?: model.quants.minByOrNull { it.totalBytes }

    /**
     * Queue the primary file, every shard, and every required companion as one
     * atomic job (SPEC §3.4).
     */
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
    }

    fun download() {
        val resolved = _state.value.resolved ?: return
        val quant = resolved.quants.firstOrNull { it.name == _state.value.selectedQuant } ?: return
        if (_state.value.verdict?.verdict?.runnable != true) return
        // Type and role are the user's to state. Without them there is nothing
        // to write, and falling back to the detected value here would quietly
        // reinstate the guessing this replaced.
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
            val companions = resolved.companions.filter { it.role.required || it.autoSelected }.map { companion ->
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
                    chatTemplate = resolved.chatTemplate,
                    bosToken = resolved.bosToken,
                    eosToken = resolved.eosToken,
                    companionPathsJson = SparseParams.of(
                        *companions.map { it.companionRole.orEmpty() to it.destPath }.toTypedArray(),
                    ).toJsonString(),
                    installedAt = System.currentTimeMillis(),
                    lastUsedAt = null,
                    pinned = false,
                    favourite = false,
                    notes = null,
                    backendOverride = null,
                    paramOverridesJson = SparseParams.of(
                        "n_ctx" to _state.value.contextTokens,
                    ).toJsonString(),
                    defaultPresetId = defaultPresetFor(modality),
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

    private fun defaultPresetFor(modality: Modality): String = when (modality) {
        Modality.DIFFUSION -> "image-quality"
        Modality.SPEECH_TO_TEXT -> "speech-accurate"
        else -> "text-balanced"
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
    /**
     * What this model *is*, and which add-on slot it fills — both chosen here
     * rather than inferred, which is why they start unset. The header parse
     * still supplies context length, architecture, chat template and the fit
     * estimate; it just no longer decides these two.
     */
    val searching: Boolean = false,
    val searchResults: List<ai.ondevice.data.hf.HfSearchResult> = emptyList(),
    val selectedModality: Modality? = null,
    val selectedRole: AttachmentRole? = null,
    /** True once [selectedRole] has been answered, including answered as "none". */
    val roleAnswered: Boolean = false,
) {
    val runnable: Boolean get() = verdict?.verdict?.runnable == true
    val isRefused: Boolean get() = refusal != null || verdict?.verdict == Verdict.WONT_FIT

    /** Type and role are required, so Download stays closed until both are set. */
    val classified: Boolean get() = selectedModality != null && roleAnswered
}
