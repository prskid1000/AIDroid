package ai.ondevice.engine

import android.content.ComponentCallbacks2
import android.content.Context
import android.content.res.Configuration
import ai.ondevice.core.BackendId
import ai.ondevice.core.SparseParams
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.hf.DeviceCapabilities
import ai.ondevice.data.prefs.AppPrefs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Owns residency: which model is loaded, on which backend, and when it must go.
 *
 * SPEC §3.5's residency policy in one place —
 *  - one model at a time by default;
 *  - an explicit "keep loaded" pin;
 *  - unload on `onTrimMemory` pressure;
 *  - warm-swap: the old model is unloaded *before* the new one loads, so the
 *    app never holds two at once.
 *
 * It also registers as a [ComponentCallbacks2] so §8.4's memory-pressure rule
 * is enforced by the system rather than by remembering to call something.
 */
class EngineManager(
    private val context: Context,
    private val registry: RuntimeRegistry,
    private val db: OnDeviceDatabase,
    private val prefs: AppPrefs,
    private val capabilities: DeviceCapabilities,
    private val scope: CoroutineScope,
) : ComponentCallbacks2 {

    private val loadMutex = Mutex()

    private val engines = mutableMapOf<String, InferenceEngine>()

    private val _state = MutableStateFlow(EngineState())
    val state: StateFlow<EngineState> = _state.asStateFlow()

    init {
        context.registerComponentCallbacks(this)
    }

    private fun engineFor(runtimeId: String): InferenceEngine? {
        val descriptor = registry.descriptor(runtimeId)?.takeIf { it.installed } ?: return null
        if (!registry.contractSatisfied(descriptor)) return null
        return engines.getOrPut(runtimeId) {
            when (runtimeId) {
                RuntimeRegistry.LLAMA -> LlamaEngine(descriptor)
                // The remaining runtimes are not text engines and are driven
                // through their own managers; nothing asks this for them.
                else -> LlamaEngine(descriptor)
            }
        }
    }

    val llama: InferenceEngine? get() = engineFor(RuntimeRegistry.LLAMA)

    /**
     * Load a model, honouring the per-model backend override and otherwise
     * taking the backend from [resolveBackend].
     */
    suspend fun load(model: ModelEntity, paramOverrides: SparseParams = SparseParams.EMPTY): Result<LoadedModel> =
        loadMutex.withLock {
            val engine = llama ?: return Result.failure(
                IllegalStateException("No llama.cpp runtime is installed."),
            )

            if (engine.loadedModelId == model.id) {
                return _state.value.loaded?.let { Result.success(it) }
                    ?: Result.failure(IllegalStateException("Inconsistent load state"))
            }

            _state.value = _state.value.copy(loading = true, error = null)

            // Warm-swap — never hold both.
            if (engine.isLoaded) engine.unload()

            val backend = resolveBackend(model)
            val params = SparseParams.parse(model.paramOverridesJson).overlaidWith(paramOverrides)

            val result = engine.load(
                LoadRequest(
                    modelId = model.id,
                    modelPath = model.localPath,
                    companionPaths = emptyMap(),
                    backend = backend,
                    params = params,
                    chatTemplate = model.chatTemplate,
                ),
            )

            result.onSuccess { loaded ->
                _state.value = EngineState(loaded = loaded, backend = backend, loading = false)
                db.models().touch(model.id, System.currentTimeMillis())
            }.onFailure { error ->
                _state.value = EngineState(loading = false, error = describeFailure(error))
            }
            result
        }

    suspend fun unload() = loadMutex.withLock {
        llama?.unload()
        _state.value = EngineState()
    }

    /**
     * Backend selection: an explicit per-model override wins, then the global
     * setting, then the first backend the installed runtime registered.
     *
     * Both preferences are clamped to what the runtime actually has. This used
     * to end in a bare `return BackendId.OPENCL`, and since the value is what
     * the Chat header reads, the badge said "OpenCL" on a build with no GPU
     * backend compiled in at all — runtimes.json honestly reports ["CPU"], and
     * whisper_jni sets `use_gpu = false` for the same reason. A cosmetic lie,
     * but an expensive one: it sent a plain-CPU slowness straight at the GPU.
     *
     * The middle step used to be a stored benchmark result. That measurement was
     * the right instinct — §8.2 says not to assume backend performance — but a
     * measurement over a one-element set is not a measurement. With ["CPU"]
     * registered, the benchmark ran CPU against itself, wrote one row, and the
     * selection then "chose" the only candidate. Falling back to the first
     * registered backend says the same thing without the ceremony, and the day a
     * GPU backend is compiled in is the day measuring is worth rebuilding.
     */
    private suspend fun resolveBackend(model: ModelEntity): BackendId {
        val available = registry.backendsFor(RuntimeRegistry.LLAMA)
        fun clamp(backend: BackendId) = backend.takeIf { it in available } ?: BackendId.CPU

        model.backendOverride?.let { return clamp(it) }
        val mode = prefs.backendMode.first()
        if (mode != AppPrefs.BACKEND_AUTO) {
            runCatching { BackendId.valueOf(mode) }.getOrNull()?.let { return clamp(it) }
        }
        return available.firstOrNull() ?: BackendId.CPU
    }

    /**
     * SPEC §8.4 — a native allocation failure surfaces as a real message with
     * numbers and a suggestion, never a bare crash.
     */
    private fun describeFailure(error: Throwable): EngineError {
        val message = error.message.orEmpty()
        return when {
            message.contains("alloc", true) || error is OutOfMemoryError -> EngineError(
                message = "Not enough memory to load this model. " +
                    "${ai.ondevice.core.Fmt.bytes(capabilities.availableRamBytes)} free of " +
                    "${ai.ondevice.core.Fmt.bytes(capabilities.totalRamBytes)}.",
                suggestion = "Lower the context size, pick a smaller quant, or set cache_type_k " +
                    "and cache_type_v to q8_0.",
            )
            message.contains("backend", true) -> EngineError(
                message = "The selected backend failed to initialise.",
                suggestion = "Switch to CPU in Settings → Backend.",
            )
            else -> EngineError(message = message.ifBlank { "Model load failed." }, suggestion = null)
        }
    }

    // — ComponentCallbacks2: §3.5 and §8.4 —

    override fun onTrimMemory(level: Int) {
        if (level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW) {
            scope.launch {
                val pinned = db.models().getPinned()
                val loadedId = llama?.loadedModelId
                // A pinned model survives screen-off but not genuine pressure at
                // the critical levels; below that the pin is honoured.
                if (pinned?.id != loadedId || level >= ComponentCallbacks2.TRIM_MEMORY_COMPLETE) {
                    unload()
                }
            }
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) = Unit

    @Deprecated("Required by ComponentCallbacks2")
    override fun onLowMemory() {
        onTrimMemory(ComponentCallbacks2.TRIM_MEMORY_COMPLETE)
    }
}

data class EngineState(
    val loaded: LoadedModel? = null,
    val backend: BackendId? = null,
    val loading: Boolean = false,
    val error: EngineError? = null,
    val tokensPerSecond: Float = 0f,
    val contextUsed: Int = 0,
)

data class EngineError(val message: String, val suggestion: String?)
