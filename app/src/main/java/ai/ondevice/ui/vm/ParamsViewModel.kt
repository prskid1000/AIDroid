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

    /**
     * The load in flight, so a newer one can cancel it.
     *
     * This screen is activity-scoped and its default load races the runtime the
     * caller actually asked for: the Voice screen's "Advanced" would open, ask
     * for kokoro, and then have llama.cpp's 74 parameters land on top of
     * Kokoro's eight because the default load started first and finished last.
     * The symptom was a screen titled "llama.cpp" full of samplers, reached
     * from a button that promised the phonemiser.
     */
    private var loadJob: Job? = null

    init {
        startLoad(RuntimeRegistry.LLAMA)
    }

    private fun startLoad(runtimeId: String) {
        loadJob?.cancel()
        loadJob = viewModelScope.launch { load(runtimeId) }
    }

    /**
     * The screen is not llama-only. S11's "Advanced" button opens this same
     * renderer against stable-diffusion.cpp, and the manifest already carries
     * every runtime's parameters — so the only thing that changes is which
     * `runtimes[…]` block is read and which model's overrides are edited.
     * Nothing about the renderer knows a runtime name.
     */
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
            allSpecs = manifest.paramsFor(runtimeId),
            values = SparseParams.parse(model?.paramOverridesJson),
            modality = (model?.modality ?: modalityOf(runtimeId)).name.lowercase(),
            architecture = model?.architecture,
            showAll = prefs.showAllParameters.first(),
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

    /**
     * Whose overrides this screen edits.
     *
     * The loaded model is preferred, because the build gate and the context
     * readout should describe what is actually in memory. But it cannot be the
     * *only* answer: nothing is loaded until the first message, so opening
     * Settings → Advanced parameters from a cold start left this null, and with
     * it null [persist] returned early and **every edit was silently
     * discarded** — the screen showed the new value, the database kept the old
     * one, and n_ctx set to 2048 came back as 8192 on the next load.
     *
     * So fall back to the model that *would* be loaded: the most recently used
     * one of the right modality, which is what the chat picks too.
     */
    private suspend fun modelFor(runtimeId: String) = when (runtimeId) {
        RuntimeRegistry.LLAMA ->
            engines.state.value.loaded?.modelId?.let { db.models().get(it) }
                ?: db.models().observeInstalledByModality(Modality.TEXT).first().firstOrNull()
        else -> db.models().observeInstalledByModality(modalityOf(runtimeId)).first().firstOrNull()
    }

    /**
     * Every installed file a `path` parameter could legitimately name.
     *
     * Taken from the library rather than by scanning the filesystem, so what is
     * offered is exactly what the app knows it downloaded and can vouch for.
     *
     * Labelled by filename, not by role. The role used to be the label, on the
     * reasoning that it is what the user chooses by — but each dropdown is
     * already filtered to one role, so "ControlNet" under the control_net field
     * only repeats the field's own name, and two installed ControlNets both
     * read "ControlNet" with nothing to tell them apart. The filename is the
     * part that differs: repo-derived display names collide too, since every
     * ControlNet in comfyanonymous/ControlNet-v1-1_fp16_safetensors shares one.
     * [PathChoice.detail] still carries the display name and size underneath.
     */
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
        // Only the text engine is warm; a diffusion or transcription parameter
        // is stored now and passed at the next run, so it must not be pushed
        // into the loaded llama context.
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

    /**
     * Write the overrides to the model row.
     *
     * If there is nothing to write them to, say so. This used to `return`
     * quietly, which turned the whole screen into a very convincing no-op —
     * sliders moved, values updated, nothing was saved. An edit that cannot be
     * stored is a refusal, and §1.2 says a refusal names itself.
     */
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
