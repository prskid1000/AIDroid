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

/** Laid out rather than minified: this text is read and edited by hand. */
private val PRETTY = kotlinx.serialization.json.Json {
    prettyPrint = true
    prettyPrintIndent = "  "
    isLenient = true
}

/** S8/S9 — the manifest renderer's state, and the escape hatch. */
@HiltViewModel
class ParamsViewModel @Inject constructor(
    private val repository: ParamRepository,
    private val registry: RuntimeRegistry,
    private val engines: EngineManager,
    // The four runtimes that are not llama.cpp, so this screen can drop each
    // one's loaded context. Each is a @Singleton, so these are the same
    // instances the tabs generate through.
    private val diffusion: ai.ondevice.engine.DiffusionEngine,
    private val transcriber: ai.ondevice.engine.Transcriber,
    private val kokoro: ai.ondevice.speech.KokoroEngine,
    private val omniVoice: ai.ondevice.speech.OmniVoiceEngine,
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

    /**
     * Whose overrides this screen edits.
     *
     * The base model, never an add-on. A VAE and a ControlNet are rows in the
     * same table with the same modality, so "the first diffusion model" could
     * be either — and writing the component choices onto a VAE's row meant the
     * Image screen, which reads them off the base model, never saw them.
     */
    private suspend fun modelFor(runtimeId: String) = when (runtimeId) {
        RuntimeRegistry.LLAMA ->
            engines.state.value.loaded?.modelId?.let { db.models().get(it) }
                ?: db.models().observeInstalledByModality(Modality.TEXT).first().firstOrNull()
        else -> db.models().observeInstalledByModality(modalityOf(runtimeId)).first()
            .firstOrNull { it.attachmentRole == null }
    }

    /** Every installed file a `path` parameter could legitimately name. */
    private suspend fun installedFiles(): List<ai.ondevice.params.PathChoice> {
        val installed = db.models().getInstalled().filter { java.io.File(it.localPath).isFile }
        // The filename was the label, and the diffusers layout gives every
        // component the same one — `vae/diffusion_pytorch_model.safetensors`
        // and `controlnet/diffusion_pytorch_model.safetensors` read as one row
        // twice. Each label is the shortest tail of its path that no other
        // installed file shares, so a folder appears only when it has to.
        val fileLabels = ai.ondevice.core.FileLabels.distinguish(installed.map { it.localPath })
        return installed.map { model ->
            val file = java.io.File(model.localPath)
            val role = model.attachmentRole
            ai.ondevice.params.PathChoice(
                // A name given by hand wins here as everywhere else; the
                // derived label — a filename, grown by a folder where two
                // files share one — is the fallback rather than the rule.
                label = model.customLabel?.takeIf { it.isNotBlank() }
                    ?: if (role != null) fileLabels.getValue(model.localPath) else model.displayName,
                detail = "${model.label} · ${fileLabels.getValue(model.localPath)} · " +
                    ai.ondevice.core.Fmt.bytes(file.length()),
                path = model.localPath,
                role = role,
            )
        }
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
        val updated = when {
            value != null -> _state.value.values.overlaidWith(SparseParams.of(key to value))
            // Clearing a file is an answer, and dropping the key threw it away.
            //
            // The Image tab fills an empty role slot when exactly one installed
            // file fits, which is what makes a bare diffusion model usable
            // without a scavenger hunt. It can only do that by looking for a
            // slot with nothing in it — so removing the key here put the slot
            // straight back into the state that invites filling, and the VAE
            // you had just cleared reappeared on the next refresh. An empty
            // string says "asked and answered: none".
            spec?.type == ai.ondevice.params.ParamType.PATH ->
                _state.value.values.overlaidWith(SparseParams.of(key to ""))
            else -> _state.value.values.without(key)
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

    /**
     * Make the batched edits take effect.
     *
     * This used to return at the first line for every runtime but llama.cpp,
     * so on the sd.cpp screen the card announcing "2 change(s) need a model
     * reload" offered a button that did nothing at all — and the settings it
     * named, `vae` and `threads`, are exactly the two that cannot be changed on
     * a live context.
     *
     * llama.cpp is the one runtime the app holds loaded between turns, so it is
     * rebuilt here and now. The other four load at the start of a run, from the
     * overrides already written to the model's row, so dropping the context is
     * the whole of the reload: the next run picks the new values up on its own.
     */
    fun applyPendingReload() {
        viewModelScope.launch {
            when (_state.value.runtimeId) {
                RuntimeRegistry.LLAMA -> {
                    val modelId = _state.value.modelId ?: return@launch
                    val model = db.models().get(modelId) ?: return@launch
                    engines.load(model, _state.value.values, force = true)
                }
                RuntimeRegistry.STABLE_DIFFUSION -> diffusion.unload()
                RuntimeRegistry.WHISPER -> transcriber.unload()
                RuntimeRegistry.KOKORO -> kokoro.unload()
                RuntimeRegistry.OMNIVOICE -> omniVoice.unload()
            }
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
        _state.value = _state.value.copy(rawJson = json, rawError = null)
    }

    /**
     * Re-indent what is in the box, if it parses.
     *
     * The box is where a parameter the manifest has never heard of gets typed,
     * which means it is where JSON gets typed by hand on a phone keyboard. A
     * button that lays it out is the difference between reading it back and
     * squinting at it — and it doubles as the check that it parses at all,
     * because it silently does nothing when it does not.
     */
    fun formatRawJson() {
        val parsed = runCatching {
            PRETTY.parseToJsonElement(_state.value.rawJson)
        }.getOrNull() ?: run {
            _state.value = _state.value.copy(rawError = "Not valid JSON, so there is nothing to lay out.")
            return
        }
        _state.value = _state.value.copy(
            rawJson = PRETTY.encodeToString(kotlinx.serialization.json.JsonElement.serializer(), parsed),
            rawError = null,
        )
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

    /** Null while the box is empty; otherwise whether what is in it parses. */
    val rawJsonParses: Boolean?
        get() = rawJson.takeIf { it.isNotBlank() }?.let {
            runCatching { PRETTY.parseToJsonElement(it) }.isSuccess
        }

    /**
     * Whether the reload happens on the button or at the start of the next run.
     *
     * llama.cpp is the only runtime held loaded between turns; the rest build
     * their context per run and read the stored overrides when they do.
     */
    val reloadsOnNextRun: Boolean get() = runtimeId != RuntimeRegistry.LLAMA
}
