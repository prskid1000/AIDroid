package ai.ondevice.ui.vm

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.ondevice.core.Modality
import ai.ondevice.core.SparseParams
import ai.ondevice.core.Tier
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.engine.EngineManager
import ai.ondevice.engine.ParamReport
import ai.ondevice.engine.RuntimeRegistry
import ai.ondevice.params.ParamRepository
import ai.ondevice.params.ParamSpec
import ai.ondevice.params.VisibleParams
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonPrimitive
import javax.inject.Inject

/** S8/S9 — the manifest renderer's state, and the escape hatch. */
@HiltViewModel
class ParamsViewModel @Inject constructor(
    private val repository: ParamRepository,
    private val registry: RuntimeRegistry,
    private val engines: EngineManager,
    private val db: OnDeviceDatabase,
    private val prefs: AppPrefs,
) : ViewModel() {

    private val _state = MutableStateFlow(ParamsState())
    val state: StateFlow<ParamsState> = _state.asStateFlow()

    /** The load in flight, so a newer one can cancel it. */
    private var loadJob: Job? = null

    init {
        startLoad(RuntimeRegistry.LLAMA)
    }

    private fun startLoad(runtimeId: String) {
        loadJob?.cancel()
        loadJob = viewModelScope.launch { load(runtimeId) }
    }

    /** The screen is not llama-only. */
    fun setRuntime(runtimeId: String) {
        if (_state.value.runtimeId == runtimeId && _state.value.allSpecs.isNotEmpty()) return
        startLoad(runtimeId)
    }

    private suspend fun load(runtimeId: String) {
        val manifest = repository.manifest()
        val model = modelFor(runtimeId)
        _state.value = _state.value.copy(
            manifestVersion = manifest.manifestVersion,
            bundledVersion = repository.bundledVersion(),
            runtimeId = runtimeId,
            buildTag = registry.buildTag(runtimeId),
            allSpecs = repository.specsFor(manifest, runtimeId),
            values = SparseParams.parse(model?.paramOverridesJson),
            modality = (model?.modality ?: modalityOf(runtimeId)).name.lowercase(),
            architecture = model?.architecture,
            modelId = model?.id,
            query = "",
        )
        _state.value = _state.value.copy(pathChoices = installedFiles())
        // Seed the display order from the stored value when there is one,
        // otherwise from the manifest default.
        val stored = _state.value.values.stringList("samplers")
        _state.value = _state.value.copy(samplerOrder = stored ?: defaultSamplerOrder())
        recompute()
    }

    /** Whose overrides this screen edits. */
    private suspend fun modelFor(runtimeId: String) = when (runtimeId) {
        RuntimeRegistry.LLAMA ->
            engines.state.value.loaded?.modelId?.let { db.models().get(it) }
                ?: db.models().observeInstalledByModality(Modality.TEXT).first().firstOrNull()
        else -> db.models().observeInstalledByModality(modalityOf(runtimeId)).first().firstOrNull()
    }

    /** Every installed file a `path` parameter could legitimately name. */
    private suspend fun installedFiles(): List<ai.ondevice.params.PathChoice> =
        db.models().getInstalled().mapNotNull { model ->
            val file = java.io.File(model.localPath)
            if (!file.isFile) return@mapNotNull null
            val role = model.attachmentRole
            ai.ondevice.params.PathChoice(
                label = if (role != null) file.name else model.displayName,
                detail = "${model.displayName} · ${file.name} · " +
                    ai.ondevice.core.Fmt.bytes(file.length()),
                path = model.localPath,
                role = role,
            )
        }

    private fun modalityOf(runtimeId: String) = when (runtimeId) {
        RuntimeRegistry.STABLE_DIFFUSION -> Modality.DIFFUSION
        RuntimeRegistry.WHISPER -> Modality.SPEECH_TO_TEXT
        // Both voice engines are text-to-speech models; they differ in which
        // parameter set describes them, not in what kind of model they are.
        RuntimeRegistry.KOKORO, RuntimeRegistry.OMNIVOICE -> Modality.TEXT_TO_SPEECH
        else -> Modality.TEXT
    }

    fun setTier(tier: Tier?) {
        _state.value = _state.value.copy(tier = tier)
        recompute()
    }

    fun setQuery(query: String) {
        _state.value = _state.value.copy(query = query)
        recompute()
    }

    /** Screen state, not a preference. */
    fun setShowAll(showAll: Boolean) {
        _state.value = _state.value.copy(showAll = showAll)
        recompute()
    }

    /** A value edit. */
    fun setValue(key: String, value: Any?) {
        val spec = _state.value.allSpecs.firstOrNull { it.key == key }
        val updated = if (value == null) {
            _state.value.values.without(key)
        } else {
            _state.value.values.overlaidWith(SparseParams.of(key to value))
        }
        _state.value = _state.value.copy(
            values = updated,
            pendingReloadKeys = if (spec?.requiresReload == true) {
                _state.value.pendingReloadKeys + key
            } else {
                _state.value.pendingReloadKeys
            },
        )
        recompute()
        if (spec?.requiresReload != true && _state.value.runtimeId == RuntimeRegistry.LLAMA) {
            viewModelScope.launch {
                val report = engines.llama?.applyParams(SparseParams.of(key to value))
                _state.value = _state.value.copy(lastReport = report)
            }
        }
        persist()
    }

    fun applyPendingReload() {
        val modelId = _state.value.modelId ?: return
        if (_state.value.runtimeId != RuntimeRegistry.LLAMA) return
        viewModelScope.launch {
            val model = db.models().get(modelId) ?: return@launch
            engines.load(model, _state.value.values)
            _state.value = _state.value.copy(pendingReloadKeys = emptySet())
        }
    }

    fun resetAll() {
        _state.value = _state.value.copy(values = SparseParams.EMPTY, pendingReloadKeys = emptySet())
        recompute()
        persist()
    }

    /** SPEC §16.6 — arbitrary JSON straight through to the runtime. */
    fun setRawJson(json: String) {
        _state.value = _state.value.copy(rawJson = json)
    }

    fun applyRawJson() {
        val parsed = SparseParams.parse(_state.value.rawJson)
        if (parsed.isEmpty) {
            _state.value = _state.value.copy(rawError = "Not valid JSON, or an empty object.")
            return
        }
        viewModelScope.launch {
            // Unknown keys are stored too — §11 says never drop them.
            _state.value = _state.value.copy(
                values = _state.value.values.overlaidWith(parsed),
                rawError = null,
            )
            val report = engines.llama?.applyParams(parsed)
            _state.value = _state.value.copy(lastReport = report)
            recompute()
            persist()
        }
    }

    /** S9 — the sampler chain. */
    fun setSamplerOrder(order: List<String>) {
        _state.value = _state.value.copy(samplerOrder = order)
        writeChain()
    }

    fun toggleSampler(name: String) {
        val disabled = _state.value.disabledSamplers
        _state.value = _state.value.copy(
            disabledSamplers = if (name in disabled) disabled - name else disabled + name,
        )
        writeChain()
    }

    private fun writeChain() {
        val active = _state.value.samplerOrder.filterNot { it in _state.value.disabledSamplers }
        setValue("samplers", active)
    }

    /** The full display order, including anything currently switched off. */
    fun samplerOrder(): List<String> = _state.value.samplerOrder.ifEmpty { defaultSamplerOrder() }

    private fun defaultSamplerOrder(): List<String> {
        val spec = _state.value.allSpecs.firstOrNull { it.key == "samplers" }
        return (spec?.default as? kotlinx.serialization.json.JsonArray)
            ?.mapNotNull { (it as? JsonPrimitive)?.content }
            ?: emptyList()
    }

    fun resetSamplerChain() {
        _state.value = _state.value.copy(
            samplerOrder = defaultSamplerOrder(),
            disabledSamplers = emptySet(),
        )
        setValue("samplers", null)
    }

    /** Mirostat replaces the chain entirely; the app greys it out and says so. */
    fun mirostatActive(): Boolean =
        (_state.value.values.string("mirostat") ?: "0") != "0"

    private fun recompute() {
        val s = _state.value
        _state.value = s.copy(
            visible = repository.visible(
                specs = s.allSpecs,
                values = s.values,
                tier = s.tier,
                showAll = s.showAll,
                loadedBuildTag = s.buildTag,
                modality = s.modality,
                architecture = s.architecture,
                query = s.query,
            ),
        )
    }

    /** Write the overrides to the model row. */
    private fun persist() {
        val modelId = _state.value.modelId
        if (modelId == null) {
            _state.value = _state.value.copy(
                unsavedReason = "No model is installed for this runtime, so there is nothing to " +
                    "save these against. Install one and they will stick.",
            )
            return
        }
        viewModelScope.launch {
            db.models().setParamOverrides(modelId, _state.value.values.toJsonString())
            if (_state.value.unsavedReason != null) {
                _state.value = _state.value.copy(unsavedReason = null)
            }
        }
    }
}

data class ParamsState(
    val manifestVersion: Int = 0,
    val bundledVersion: Int = 0,
    val runtimeId: String = RuntimeRegistry.LLAMA,
    val buildTag: String = "",
    val allSpecs: List<ParamSpec> = emptyList(),
    val visible: VisibleParams = VisibleParams(emptyList(), 0, 0, 0),
    val values: SparseParams = SparseParams.EMPTY,
    val tier: Tier? = Tier.BASIC,
    val query: String = "",
    val showAll: Boolean = false,
    val modality: String = "text",
    val architecture: String? = null,
    val modelId: String? = null,
    val pendingReloadKeys: Set<String> = emptySet(),
    /** Every sampler, in display order — including any switched off. */
    val samplerOrder: List<String> = emptyList(),
    val disabledSamplers: Set<String> = emptySet(),
    val rawJson: String = "{ \"some_new_upstream_flag\": 0.7 }",
    val rawError: String? = null,
    /** Non-null when edits cannot be stored, so the screen can stop pretending. */
    val unsavedReason: String? = null,
    /** Installed files that a `path` parameter can be pointed at. */
    val pathChoices: List<ai.ondevice.params.PathChoice> = emptyList(),
    val lastReport: ParamReport? = null,
) {
    val totalCount: Int get() = allSpecs.size
    val basicCount: Int get() = allSpecs.count { it.tier == Tier.BASIC }
    val advancedCount: Int get() = allSpecs.count { it.tier == Tier.ADVANCED }
    val expertCount: Int get() = allSpecs.count { it.tier == Tier.EXPERT }
    val needsReload: Boolean get() = pendingReloadKeys.isNotEmpty()
}
