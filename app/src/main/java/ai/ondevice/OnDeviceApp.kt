package ai.ondevice

import android.app.Application
import ai.ondevice.core.Modality
import ai.ondevice.core.RuntimeState
import ai.ondevice.core.SparseParams
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.PersonaEntity
import ai.ondevice.data.db.PresetEntity
import ai.ondevice.data.db.RuntimeBundleEntity
import ai.ondevice.di.ApplicationScope
import ai.ondevice.engine.HexagonSkels
import ai.ondevice.engine.RuntimeRegistry
import dagger.hilt.android.HiltAndroidApp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * SPEC §13 — offline-first, no telemetry, no account, no crash reporting that
 * transmits content. There is deliberately nothing to initialise here beyond
 * seeding the local database.
 */
@HiltAndroidApp
class OnDeviceApp : Application() {

    @Inject lateinit var db: OnDeviceDatabase

    @Inject lateinit var registry: RuntimeRegistry

    @Inject @ApplicationScope lateinit var scope: CoroutineScope

    @Inject lateinit var downloader: ai.ondevice.data.download.Downloader

    @Inject lateinit var computeDeviceSwitch: ai.ondevice.engine.ComputeDeviceSwitch

    override fun onCreate() {
        super.onCreate()
        // Before anything can touch ggml. Its backend registry builds itself
        // once, on the first call from any of the three runtimes, and the NPU
        // opens its DSP session while registering — so the search path for the
        // DSP's own code has to already be set by then. Synchronous for the
        // same reason: a coroutine would race the first Chat screen.
        HexagonSkels.stage(this)
        // A context is built for one device and cannot move, so changing the
        // Compute device has to drop what is loaded rather than leave it
        // resident on the device the user just stopped using.
        computeDeviceSwitch.start(scope)
        scope.launch {
            seed()
            // A download interrupted by a crash, a force-stop or a reinstall
            // leaves a row saying RUNNING with nothing behind it. §3.4 promises
            // downloads survive the app being killed, so picking those up is
            // part of keeping that promise rather than a nicety.
            downloader.resumeInterrupted()
        }
    }

    /**
     * Built-in presets and personas per SPEC §10 — all editable and deletable,
     * nothing locked. Runtime rows mirror the CI-generated registry so the
     * Runtimes screen has something real to show before the first update check.
     */
    private suspend fun seed() {
        if (db.presets().count() == 0) {
            db.presets().insertAll(builtInPresets())
        }
        if (db.personas().count() == 0) {
            db.personas().insertAll(builtInPersonas())
        }
        if (db.runtimes().count() == 0) {
            db.runtimes().insertAll(
                registry.descriptors.map { descriptor ->
                    RuntimeBundleEntity(
                        engine = descriptor.id,
                        buildTag = descriptor.version.takeIf { descriptor.installed },
                        upstreamCommit = descriptor.upstreamCommit,
                        jniContract = descriptor.jniContract,
                        installedAt = if (descriptor.installed) System.currentTimeMillis() else null,
                        sizeBytes = descriptor.sizeBytes,
                        state = if (descriptor.installed) RuntimeState.INSTALLED else RuntimeState.NOT_INSTALLED,
                        previousBuildTag = null,
                        availableBuildTag = null,
                        availableSizeBytes = null,
                        availableNotes = null,
                        architectureCount = descriptor.architectures.size,
                        backendsJson = descriptor.backends.joinToString(",") { it.name },
                        initFailureCount = 0,
                        rolledBackFrom = null,
                    )
                },
            )
        }
    }

    /** Insertion order is the display order — see [ai.ondevice.data.db.PresetDao]. */
    private fun builtInPresets(): List<PresetEntity> = listOf(
        PresetEntity(
            id = "text-precise",
            modality = Modality.TEXT,
            name = "Precise",
            paramsJson = SparseParams.of(
                "temp" to 0.2f,
                "top_p" to 0.9f,
                "min_p" to 0.05f,
                "repeat_penalty" to 1.05f,
            ).toJsonString(),
            isBuiltIn = true,
        ),
        PresetEntity(
            id = "text-balanced",
            modality = Modality.TEXT,
            name = "Balanced",
            paramsJson = SparseParams.of(
                "temp" to 0.7f,
                "top_p" to 0.95f,
                "min_p" to 0.05f,
                "repeat_penalty" to 1.0f,
            ).toJsonString(),
            isBuiltIn = true,
        ),
        PresetEntity(
            id = "text-creative",
            modality = Modality.TEXT,
            name = "Creative",
            paramsJson = SparseParams.of(
                "temp" to 1.1f,
                "top_p" to 0.98f,
                "min_p" to 0.02f,
                "xtc_probability" to 0.5f,
            ).toJsonString(),
            isBuiltIn = true,
        ),
        // SDXL and SD1.5 want very different step and CFG defaults (SPEC §5.4),
        // which is why the diffusion presets exist at all.
        PresetEntity(
            id = "image-fast-draft",
            modality = Modality.DIFFUSION,
            name = "Fast draft",
            paramsJson = SparseParams.of(
                "steps" to 8,
                "cfg_scale" to 2.0f,
                "width" to 512,
                "height" to 512,
                "vae_tiling" to true,
            ).toJsonString(),
            isBuiltIn = true,
        ),
        PresetEntity(
            id = "image-quality",
            modality = Modality.DIFFUSION,
            name = "Quality",
            paramsJson = SparseParams.of(
                "steps" to 28,
                "cfg_scale" to 7.0f,
                "sampling_method" to "dpm++2m",
                "schedule" to "karras",
                "vae_tiling" to true,
            ).toJsonString(),
            isBuiltIn = true,
        ),
        PresetEntity(
            id = "speech-accurate",
            modality = Modality.SPEECH_TO_TEXT,
            name = "Accurate",
            paramsJson = SparseParams.of("beam_size" to 5, "audio_ctx" to 0).toJsonString(),
            isBuiltIn = true,
        ),
        PresetEntity(
            id = "speech-fast",
            modality = Modality.SPEECH_TO_TEXT,
            name = "Fast",
            paramsJson = SparseParams.of("beam_size" to -1, "audio_ctx" to 512).toJsonString(),
            isBuiltIn = true,
        ),
    )

    private fun builtInPersonas(): List<PersonaEntity> = listOf(
        PersonaEntity(
            id = "persona-terse",
            name = "Terse",
            avatarPath = null,
            systemPrompt = "Answer in as few words as the question allows. Show working for any " +
                "arithmetic. Never apologise.",
            defaultModelId = null,
            defaultPresetId = "text-balanced",
            defaultVoice = "af_heart",
            memoryNotes = null,
        ),
        PersonaEntity(
            id = "persona-duck",
            name = "Rubber duck",
            avatarPath = null,
            systemPrompt = "Ask one clarifying question at a time. Do not offer a solution until " +
                "the problem has been stated precisely.",
            defaultModelId = null,
            defaultPresetId = "text-precise",
            defaultVoice = "bm_george",
            memoryNotes = null,
        ),
    )
}
