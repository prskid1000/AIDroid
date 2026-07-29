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
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonPrimitive
import javax.inject.Inject

/**
 * S8/S9 — the manifest renderer's state, and the escape hatch.
 *
 * There is nothing parameter-specific in here either: it holds a manifest, a
 * sparse value map and a tier, and asks the repository what to show.
 */
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

    init {
        viewModelScope.launch {
            val manifest = repository.manifest()
            val model = engines.state.value.loaded?.modelId?.let { db.models().get(it) }
            val specs = manifest.paramsFor(RuntimeRegistry.LLAMA)
            _state.value = _state.value.copy(
                manifestVersion = manifest.manifestVersion,
                bundledVersion = repository.bundledVersion(),
                runtimeId = RuntimeRegistry.LLAMA,
                buildTag = registry.llamaBuildTag,
                allSpecs = specs,
                values = SparseParams.parse(model?.paramOverridesJson),
                modality = model?.modality?.name?.lowercase() ?: Modality.TEXT.name.lowercase(),
                architecture = model?.architecture,
                showAll = prefs.showAllParameters.first(),
                modelId = model?.id,
            )
            // Seed the display order from the stored value when there is one,
            // otherwise from the manifest default.
            val stored = _state.value.values.stringList("samplers")
            _state.value = _state.value.copy(samplerOrder = stored ?: defaultSamplerOrder())
            recompute()
        }
    }

    fun setTier(tier: Tier?) {
        _state.value = _state.value.copy(tier = tier)
        recompute()
    }

    fun setQuery(query: String) {
        _state.value = _state.value.copy(query = query)
        recompute()
    }

    fun setShowAll(showAll: Boolean) {
        _state.value = _state.value.copy(showAll = showAll)
        viewModelScope.launch { prefs.setShowAllParameters(showAll) }
        recompute()
    }

    /**
     * A value edit. Parameters that need a reload are *batched* — collected in
     * [ParamsState.pendingReloadKeys] and applied once, rather than thrashing
     * the model on every slider tick (SPEC §9).
     */
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
        if (spec?.requiresReload != true) {
            viewModelScope.launch {
                val report = engines.llama?.applyParams(SparseParams.of(key to value))
                _state.value = _state.value.copy(lastReport = report)
            }
        }
        persist()
    }

    fun applyPendingReload() {
        val modelId = _state.value.modelId ?: return
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

    /**
     * SPEC §16.6 — arbitrary JSON straight through to the runtime. Unknown keys
     * are reported, never fatal, so anything the loaded `.so` supports is
     * always reachable even if manifest generation missed it.
     */
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

    /**
     * S9 — the sampler chain.
     *
     * Two lists, deliberately. The *display* order keeps every sampler so a
     * disabled one stays on screen, greyed, with an "off" badge — the canvas is
     * explicit that tapping a row disables it rather than deleting it, and a
     * row that vanishes gives the user no way to bring it back. What goes over
     * the wire in `samplers` is that order minus the disabled entries, because
     * that is the chain the runtime should actually build.
     */
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

    private fun persist() {
        val modelId = _state.value.modelId ?: return
        viewModelScope.launch {
            db.models().setParamOverrides(modelId, _state.value.values.toJsonString())
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
    val lastReport: ParamReport? = null,
) {
    val totalCount: Int get() = allSpecs.size
    val basicCount: Int get() = allSpecs.count { it.tier == Tier.BASIC }
    val advancedCount: Int get() = allSpecs.count { it.tier == Tier.ADVANCED }
    val expertCount: Int get() = allSpecs.count { it.tier == Tier.EXPERT }
    val needsReload: Boolean get() = pendingReloadKeys.isNotEmpty()
}
