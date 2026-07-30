package ai.ondevice.ui.vm

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.ondevice.core.BackendId
import ai.ondevice.core.Modality
import ai.ondevice.core.SparseParams
import ai.ondevice.data.ModelStorage
import ai.ondevice.data.OrphanReport
import ai.ondevice.data.db.BenchmarkEntity
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.hf.CompatibilityGate
import ai.ondevice.data.hf.DeviceCapabilities
import ai.ondevice.data.hf.FitEstimate
import ai.ondevice.data.download.DownloadJob
import ai.ondevice.data.download.Downloader
import ai.ondevice.engine.Benchmarker
import ai.ondevice.engine.EngineManager
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

/** S3 — the installed library, storage breakdown and orphan sweep. */
@HiltViewModel
class ModelsViewModel @Inject constructor(
    private val db: OnDeviceDatabase,
    private val storage: ModelStorage,
    private val engines: EngineManager,
    private val capabilities: DeviceCapabilities,
) : ViewModel() {

    private val filter = MutableStateFlow("")
    private val orphans = MutableStateFlow<OrphanReport?>(null)

    val state: StateFlow<ModelsState> = combine(
        db.models().observeAll(),
        filter,
        orphans,
        engines.state,
        db.downloads().observeAll(),
    ) { models, query, orphanReport, engineState, jobs ->
        val filtered = if (query.isBlank()) {
            models
        } else {
            models.filter {
                it.displayName.contains(query, true) ||
                    it.architecture?.contains(query, true) == true ||
                    it.quant?.contains(query, true) == true
            }
        }
        ModelsState(
            groups = groupByModality(filtered),
            loadedModelId = engineState.loaded?.modelId,
            filter = query,
            usedBytes = models.sumOf { it.sizeBytes },
            totalBytes = capabilities.totalRamBytes.coerceAtLeast(1),
            freeStorageBytes = capabilities.freeStorageBytes,
            byModality = models.groupBy { it.modality }.mapValues { (_, v) -> v.sumOf { it.sizeBytes } },
            orphans = orphanReport,
            // A download that failed used to be invisible from here: the queue
            // was only reachable from the orphan card, which is not shown when
            // there are no orphans. A failed install with no way back to it is
            // the opposite of §1.2's "say what went wrong and offer the remedy".
            activeDownloads = jobs.count {
                it.state == ai.ondevice.core.DownloadState.RUNNING ||
                    it.state == ai.ondevice.core.DownloadState.QUEUED ||
                    it.state == ai.ondevice.core.DownloadState.VERIFYING
            },
            pausedDownloads = jobs.count { it.state == ai.ondevice.core.DownloadState.PAUSED },
            failedDownloads = jobs.count { it.state == ai.ondevice.core.DownloadState.FAILED },
            // The library row is written when a download starts, not when it
            // finishes, so a model still arriving looked installed and said
            // "never used". Every picker offered it, and selecting one meant
            // handing a runtime a half-written file. The pickers now filter on
            // the same condition; here the row stays visible — you want to see
            // that it is coming — but says so.
            pending = jobs
                .filter {
                    it.state == ai.ondevice.core.DownloadState.QUEUED ||
                        it.state == ai.ondevice.core.DownloadState.RUNNING ||
                        it.state == ai.ondevice.core.DownloadState.PAUSED ||
                        it.state == ai.ondevice.core.DownloadState.VERIFYING
                }
                .associate { job ->
                    job.modelId to PendingInstall(
                        fraction = if (job.bytesTotal > 0) {
                            (job.bytesDone.toFloat() / job.bytesTotal).coerceIn(0f, 1f)
                        } else {
                            0f
                        },
                        paused = job.state == ai.ondevice.core.DownloadState.PAUSED,
                    )
                },
        )
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), ModelsState())

    init {
        viewModelScope.launch { orphans.value = storage.findOrphans() }
    }

    fun onFilterChange(value: String) {
        filter.value = value
    }

    fun togglePin(model: ModelEntity) {
        viewModelScope.launch {
            if (!model.pinned) db.models().clearPins()
            db.models().setPinned(model.id, !model.pinned)
        }
    }

    fun load(model: ModelEntity) {
        viewModelScope.launch { engines.load(model) }
    }

    fun delete(model: ModelEntity) {
        viewModelScope.launch {
            storage.deleteModel(model)
            orphans.value = storage.findOrphans()
        }
    }

    fun sweepOrphans() {
        viewModelScope.launch {
            orphans.value?.strayFiles?.forEach { runCatching { it.delete() } }
            orphans.value?.recordsWithoutFiles?.forEach { db.models().deleteById(it.id) }
            orphans.value = storage.findOrphans()
        }
    }

    private fun groupByModality(models: List<ModelEntity>): List<ModelGroup> =
        Modality.entries
            .mapNotNull { modality ->
                val items = models.filter { it.modality == modality }
                if (items.isEmpty()) null else ModelGroup(modality, items)
            }
}

/** A model still downloading, and how far along. */
data class PendingInstall(val fraction: Float, val paused: Boolean) {
    val label: String
        get() = if (paused) {
            "paused — ${(fraction * 100).toInt()}% downloaded"
        } else {
            "downloading — ${(fraction * 100).toInt()}%"
        }
}

data class ModelsState(
    val groups: List<ModelGroup> = emptyList(),
    val loadedModelId: String? = null,
    val filter: String = "",
    val usedBytes: Long = 0,
    val totalBytes: Long = 1,
    val freeStorageBytes: Long = 0,
    val byModality: Map<Modality, Long> = emptyMap(),
    val orphans: OrphanReport? = null,
    val activeDownloads: Int = 0,
    val pausedDownloads: Int = 0,
    val failedDownloads: Int = 0,
    /** Models whose bytes have not all arrived yet, keyed by model id. */
    val pending: Map<String, PendingInstall> = emptyMap(),
) {
    val hasDownloadNews: Boolean get() = activeDownloads + pausedDownloads + failedDownloads > 0

    /** What the Downloads row says, in the order that matters most. */
    val downloadSummary: String
        get() = buildList {
            if (activeDownloads > 0) add("$activeDownloads in progress")
            if (pausedDownloads > 0) add("$pausedDownloads paused")
            if (failedDownloads > 0) add("$failedDownloads failed")
        }.joinToString(" · ").ifBlank { "Nothing queued" }
}

data class ModelGroup(val modality: Modality, val models: List<ModelEntity>)

/** S2 — model detail, with the verdict recomputing under the context slider. */
@HiltViewModel
class ModelDetailViewModel @Inject constructor(
    private val db: OnDeviceDatabase,
    private val engines: EngineManager,
    private val benchmarker: Benchmarker,
    private val storage: ModelStorage,
    private val capabilities: DeviceCapabilities,
    private val registry: ai.ondevice.engine.RuntimeRegistry,
) : ViewModel() {

    private val _state = MutableStateFlow(ModelDetailState())
    val state: StateFlow<ModelDetailState> = _state.asStateFlow()

    fun bind(modelId: String) {
        if (_state.value.model?.id == modelId) return
        viewModelScope.launch {
            val model = db.models().get(modelId) ?: return@launch
            val benchmarks = db.benchmarks().getFor(modelId)
            val overrides = SparseParams.parse(model.paramOverridesJson)
            _state.value = ModelDetailState(
                model = model,
                benchmarks = benchmarks,
                contextTokens = overrides.int("n_ctx") ?: model.contextLength ?: 8192,
                maxContext = model.contextLength ?: 262_144,
                loaded = engines.state.value.loaded?.modelId == modelId,
                totalRamBytes = capabilities.totalRamBytes,
                availableRamBytes = capabilities.availableRamBytes,
                companions = SparseParams.parse(model.companionPathsJson).values
                    .map { (role, path) -> role to path.toString().trim('"') },
                files = installedFilesOf(model),
            )
            recompute()
        }
    }

    /**
     * Everything actually on disk for this model, largest first.
     *
     * A model is a directory, and until now no screen said what was in it. That
     * is the difference between "OmniVoice is installed" and being able to see
     * that what landed was the audio tokeniser's four graphs and none of the
     * decoder — a distinction that otherwise only surfaces as a runtime error
     * naming a file nobody can check. Read from the filesystem rather than from
     * the download record, because the question being asked is what is there
     * now, not what was once meant to be.
     */
    private fun installedFilesOf(model: ModelEntity): List<InstalledFile> {
        val directory = runCatching { storage.modelDir(model.id) }.getOrNull() ?: return emptyList()
        val root = directory.absolutePath
        return directory.walkTopDown()
            .filter { it.isFile }
            .map {
                InstalledFile(
                    name = it.absolutePath.removePrefix(root).trimStart('/', '\\'),
                    sizeBytes = it.length(),
                    isPrimary = it.absolutePath == model.localPath,
                )
            }
            .sortedByDescending { it.sizeBytes }
            .toList()
    }

    /**
     * The live recompute S2 is built around: drag the slider, the KV term and
     * the verdict move with it, arithmetic visible.
     */
    fun setContext(tokens: Int) {
        _state.value = _state.value.copy(contextTokens = tokens)
        recompute()
    }

    private fun recompute() {
        val model = _state.value.model ?: return
        val estimate = CompatibilityGate.estimate(
            weightsBytes = model.sizeBytes,
            layers = engines.state.value.loaded?.layers ?: DEFAULT_LAYERS,
            contextTokens = _state.value.contextTokens,
            embeddingLengthKv = engines.state.value.loaded?.embeddingLengthKv ?: DEFAULT_KV_WIDTH,
            embeddingLength = engines.state.value.loaded?.embeddingLength,
            cacheTypeK = SparseParams.parse(model.paramOverridesJson).string("cache_type_k") ?: "f16",
            cacheTypeV = SparseParams.parse(model.paramOverridesJson).string("cache_type_v") ?: "f16",
        )
        val verdict = CompatibilityGate.verdict(
            estimate = estimate,
            availableRamBytes = capabilities.availableRamBytes,
            freeStorageBytes = capabilities.freeStorageBytes,
            storageReserveBytes = 1_000_000_000L,
            archSupported = true,
            hasRuntimeForFormat = true,
            speedClass = CompatibilityGate.speedClassFor(model.quant, registry.hasOpenClBackend),
        )
        _state.value = _state.value.copy(estimate = estimate, verdict = verdict)
    }

    fun commitContext() {
        val model = _state.value.model ?: return
        viewModelScope.launch {
            val updated = SparseParams.parse(model.paramOverridesJson)
                .with("n_ctx", _state.value.contextTokens)
            db.models().setParamOverrides(model.id, updated.toJsonString())
        }
    }

    fun runBenchmark() {
        val model = _state.value.model ?: return
        _state.value = _state.value.copy(benchmarking = true)
        viewModelScope.launch {
            val results = benchmarker.run(model, listOf(BackendId.OPENCL, BackendId.HEXAGON, BackendId.CPU)) { backend ->
                _state.value = _state.value.copy(benchmarkingBackend = backend)
            }
            _state.value = _state.value.copy(
                benchmarks = results,
                benchmarking = false,
                benchmarkingBackend = null,
            )
        }
    }

    fun togglePin() {
        val model = _state.value.model ?: return
        viewModelScope.launch {
            if (!model.pinned) db.models().clearPins()
            db.models().setPinned(model.id, !model.pinned)
            _state.value = _state.value.copy(model = db.models().get(model.id))
        }
    }

    fun delete(onDone: () -> Unit) {
        val model = _state.value.model ?: return
        viewModelScope.launch {
            storage.deleteModel(model)
            onDone()
        }
    }

    private companion object {
        /** Until a model is loaded the native side hasn't told us; the estimate says so. */
        const val DEFAULT_LAYERS = 36
        const val DEFAULT_KV_WIDTH = 1024
    }
}

data class ModelDetailState(
    val model: ModelEntity? = null,
    val benchmarks: List<BenchmarkEntity> = emptyList(),
    val contextTokens: Int = 8192,
    val maxContext: Int = 262_144,
    val estimate: FitEstimate? = null,
    val verdict: ai.ondevice.core.Verdict? = null,
    val loaded: Boolean = false,
    val benchmarking: Boolean = false,
    val benchmarkingBackend: BackendId? = null,
    val totalRamBytes: Long = 0,
    val availableRamBytes: Long = 0,
    val companions: List<Pair<String, String>> = emptyList(),
    val files: List<InstalledFile> = emptyList(),
) {
    val filesTotalBytes: Long get() = files.sumOf { it.sizeBytes }
}

/** One file inside a model's directory, as it exists on disk right now. */
data class InstalledFile(
    val name: String,
    val sizeBytes: Long,
    /** The file the library record points at — the one a runtime is handed. */
    val isPrimary: Boolean,
)

/** S4 — the download queue. */
@HiltViewModel
class DownloadsViewModel @Inject constructor(
    private val downloader: Downloader,
    private val db: OnDeviceDatabase,
) : ViewModel() {

    val jobs: StateFlow<List<DownloadJob>> = downloader.observeJobs()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    fun pause(id: String) = downloader.pause(id)
    fun resume(id: String) = downloader.start(id)
    fun cancel(id: String) = downloader.cancel(id)
    fun retry(id: String) = downloader.start(id)

    /**
     * Forget finished transfers. Nothing is uninstalled and no file is touched:
     * these rows are a receipt, and "installed" is defined by the absence of an
     * *active* job rather than the presence of a completed one.
     */
    fun clearFinished() = viewModelScope.launch { db.downloads().clearFinished() }
}
