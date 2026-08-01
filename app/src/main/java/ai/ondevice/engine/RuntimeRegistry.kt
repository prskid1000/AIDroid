package ai.ondevice.engine

import android.content.Context
import ai.ondevice.core.BackendId
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
                capabilities = r.capabilities.mapNotNull { c -> runCatching { Capability.valueOf(c) }.getOrNull() }.toSet(),
                backends = r.backends.mapNotNull { b -> runCatching { BackendId.valueOf(b) }.getOrNull() },
                installed = r.installed,
                sizeBytes = r.sizeBytes,
            )
        }
    }

    fun descriptor(id: String): RuntimeDescriptor? = descriptors.firstOrNull { it.id == id }

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

    /** Whether the text runtime has a GPU it can actually reach. */
    val hasOpenClBackend: Boolean
        get() = BackendId.OPENCL in backendsFor(LLAMA)

    /** The backends [runtimeId] actually registered, which is not the same question as which one the user would like. */
    fun backendsFor(runtimeId: String): List<BackendId> {
        val declared = descriptor(runtimeId)?.takeIf { it.installed }?.backends ?: return emptyList()
        return registered(runtimeId).ifEmpty { declared }
    }

    /** What ggml registered in this process, per runtime, asked once. */
    private fun registered(runtimeId: String): List<BackendId> = reported.getOrPut(runtimeId) {
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
            names
                .mapNotNull { name -> BackendId.entries.firstOrNull { it.matches(name) } }
                .distinct()
        }.getOrElse { emptyList() }
    }

    private val reported = mutableMapOf<String, List<BackendId>>()

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
    val capabilities: Set<Capability>,
    val backends: List<BackendId>,
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
    val capabilities: List<String> = emptyList(),
    val backends: List<String> = emptyList(),
    val installed: Boolean = false,
    val sizeBytes: Long = 0,
)
