package ai.ondevice.engine

import android.content.Context
import ai.ondevice.core.Capability
import ai.ondevice.core.ModelFormat
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** SPEC §2.3 — each engine registers a capability descriptor at boot, and the resolver queries this registry rather than knowing anything about models. */
class RuntimeRegistry(private val context: Context) {

    private val json = Json { ignoreUnknownKeys = true }

    private val manifest: RuntimeManifestFile by lazy {
        runCatching {
            context.assets.open(ASSET).bufferedReader().use { it.readText() }
                .let { json.decodeFromString(RuntimeManifestFile.serializer(), it) }
        }.getOrElse { RuntimeManifestFile() }
    }

    val descriptors: List<RuntimeDescriptor> by lazy {
        manifest.runtimes.map { r ->
            RuntimeDescriptor(
                id = r.id,
                version = r.buildTag,
                upstreamCommit = r.upstreamCommit,
                jniContract = r.jniContract,
                formats = r.formats.mapNotNull { f -> runCatching { ModelFormat.valueOf(f) }.getOrNull() }.toSet(),
                architectures = r.architectures.toSet(),
                languages = r.languages.toSet(),
                capabilities = r.capabilities.mapNotNull { c -> runCatching { Capability.valueOf(c) }.getOrNull() }.toSet(),
                backends = r.backends,
                installed = r.installed,
                sizeBytes = r.sizeBytes,
            )
        }
    }

    fun descriptor(id: String): RuntimeDescriptor? = descriptors.firstOrNull { it.id == id }

    /**
     * Every architecture the installed runtimes can load.
     *
     * Kokoro's entry declares [RuntimeDescriptor.languages] and no
     * architectures, and that separation is the point: its espeak language list
     * used to live in this field, so `en`, `es`, `fr`, `hi`, `id`, `it` and `pt`
     * were all "architectures". The resolver infers an architecture from a
     * repo's tags when the GGUF header does not carry one, and an ordinary HF
     * language tag then matched — which is how FLUX.2 Klein and Real-ESRGAN
     * both ended up recorded as architecture `en`.
     */
    val knownArchitectures: Set<String> by lazy {
        descriptors.filter { it.installed }.flatMap { it.architectures }.toSet()
    }

    /** Only *installed* runtimes count. */
    fun supportsArchitecture(arch: String): Boolean = arch.lowercase() in knownArchitectures.map { it.lowercase() }

    /** The architectures one runtime declares, lower-cased for matching. */
    fun architecturesFor(runtimeId: String): Set<String> =
        descriptor(runtimeId)?.architectures?.map { it.lowercase() }?.toSet() ?: emptySet()

    fun supportsFormat(format: ModelFormat): Boolean =
        descriptors.any { it.installed && format in it.formats }

    /**
     * What ggml registered in this process, per runtime, asked once.
     *
     * Nothing chooses on this any more — the app is CPU-only and there is
     * nothing to choose between. It is still read, and still shown on the
     * Runtimes screen, because it is the one answer that comes from the loaded
     * binary rather than from a file describing it: if a build ever registered
     * something other than "CPU", that screen would say so.
     */
    fun registeredBackends(runtimeId: String): List<String> = reported.getOrPut(runtimeId) {
        val info = runCatching {
            when (runtimeId) {
                LLAMA -> if (LlamaBridge.available) LlamaBridge.nativeSystemInfo() else null
                WHISPER -> if (WhisperBridge.available) WhisperBridge.nativeSystemInfo() else null
                STABLE_DIFFUSION -> if (SdBridge.available) SdBridge.nativeSystemInfo() else null
                else -> null
            }
        }.getOrNull() ?: return@getOrPut emptyList()

        runCatching {
            val names = json.decodeFromString(ReportedInfo.serializer(), info).backends
            android.util.Log.i("RuntimeRegistry", "$runtimeId registered ${names.joinToString()}")
            names.distinct()
        }.getOrElse { emptyList() }
    }

    private val reported = mutableMapOf<String, List<String>>()

    val llamaBuildTag: String get() = buildTag(LLAMA)

    /** The build tag the parameter screen gates `sinceBuild` against, per runtime. */
    fun buildTag(runtimeId: String): String = descriptor(runtimeId)?.version ?: "not installed"

    val architectureCount: Int get() = descriptor(LLAMA)?.architectures?.size ?: 0

    /** The JNI contract the Kotlin side hard-requires. */
    fun contractSatisfied(descriptor: RuntimeDescriptor): Boolean =
        descriptor.jniContract == REQUIRED_JNI_CONTRACT

    companion object {
        const val ASSET = "runtimes.json"
        const val LLAMA = "llama.cpp"
        const val WHISPER = "whisper.cpp"
        const val STABLE_DIFFUSION = "stable-diffusion.cpp"
        const val KOKORO = "kokoro"

        /** OmniVoice keeps its own parameter set: it shares no controls with Kokoro. */
        const val OMNIVOICE = "omnivoice"

        /** Bumped only when the Kotlin↔JNI signature changes — §16.7 makes that rare. */
        const val REQUIRED_JNI_CONTRACT = 4
    }
}

data class RuntimeDescriptor(
    val id: String,
    val version: String,
    val upstreamCommit: String?,
    val jniContract: Int,
    val formats: Set<ModelFormat>,
    val architectures: Set<String>,
    /**
     * What this runtime can *pronounce*, for the one where that is the gate.
     *
     * Kokoro publishes a single ONNX graph; what decides whether an install is
     * usable is whether the staged espeak data can turn text into the phonemes
     * that graph expects. That is a language list, not an architecture list, and
     * keeping it in its own field stops [RuntimeRegistry.knownArchitectures]
     * treating `en` as a model architecture.
     */
    val languages: Set<String> = emptySet(),
    val capabilities: Set<Capability>,
    /** What runtimes.json declares this runtime is built to run on, verbatim. */
    val backends: List<String>,
    val installed: Boolean,
    val sizeBytes: Long,
)

/** The slice of a bridge's `nativeSystemInfo()` this file reads. */
@Serializable
private data class ReportedInfo(
    val backends: List<String> = emptyList(),
)

@Serializable
private data class RuntimeManifestFile(
    val runtimes: List<RuntimeEntry> = emptyList(),
)

@Serializable
private data class RuntimeEntry(
    val id: String,
    val buildTag: String,
    val upstreamCommit: String? = null,
    val jniContract: Int = 3,
    val formats: List<String> = emptyList(),
    val architectures: List<String> = emptyList(),
    val languages: List<String> = emptyList(),
    val capabilities: List<String> = emptyList(),
    val backends: List<String> = emptyList(),
    val installed: Boolean = false,
    val sizeBytes: Long = 0,
)
