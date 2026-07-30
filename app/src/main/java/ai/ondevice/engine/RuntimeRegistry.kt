package ai.ondevice.engine

import android.content.Context
import ai.ondevice.core.BackendId
import ai.ondevice.core.Capability
import ai.ondevice.core.ModelFormat
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * SPEC §2.3 — each engine registers a capability descriptor at boot, and the
 * resolver queries this registry rather than knowing anything about models.
 *
 * The architecture list is **not hand-maintained**. §2.3 and Appendix A #3 both
 * insist it be generated from the pinned upstream source at build time, because
 * a hand-kept list rots and silently reintroduces model-locking. It therefore
 * ships as a CI-generated asset that the registry reads; regenerating it is a
 * step in the runtime-bump pipeline (§17.5), not an edit to this file.
 */
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

    /**
     * Only *installed* runtimes count. A model whose architecture lives in an
     * engine the user hasn't installed is a different message from one nothing
     * supports — §17.4 makes engines separately installable, so the distinction
     * is real.
     */
    fun supportsArchitecture(arch: String): Boolean = arch.lowercase() in knownArchitectures.map { it.lowercase() }

    /**
     * The architectures one runtime declares, lower-cased for matching.
     *
     * Unlike [knownArchitectures] this does not filter on `installed`, because
     * callers use it to decide what a model *is* rather than whether it can run
     * — an SDXL checkpoint is a diffusion model whether or not sd.cpp is here.
     *
     * It exists so classification can ask runtimes.json instead of keeping a
     * second copy of the same list in Kotlin. That file is generated from each
     * engine's own source (`tools/generate_runtimes.py` reads sd.cpp's SDVersion
     * enum), so a hand-written duplicate could only ever drift away from it.
     */
    fun architecturesFor(runtimeId: String): Set<String> =
        descriptor(runtimeId)?.architectures?.map { it.lowercase() }?.toSet() ?: emptySet()

    fun supportsFormat(format: ModelFormat): Boolean =
        descriptors.any { it.installed && format in it.formats }

    /**
     * Whether the installed text runtime actually has a GPU backend compiled
     * in. The quant list quotes a fast path off the back of this, so it has to
     * describe the binary rather than the hardware the binary is running on.
     */
    val hasOpenClBackend: Boolean
        get() = descriptor(LLAMA)?.let { it.installed && BackendId.OPENCL in it.backends } == true

    /**
     * The backends [runtimeId] actually registered, which is not the same
     * question as which one the user would like. Empty means the runtime is not
     * installed; callers treat CPU as the floor rather than inventing a device.
     */
    fun backendsFor(runtimeId: String): List<BackendId> =
        descriptor(runtimeId)?.takeIf { it.installed }?.backends ?: emptyList()

    val llamaBuildTag: String get() = buildTag(LLAMA)

    /** The build tag the parameter screen gates `sinceBuild` against, per runtime. */
    fun buildTag(runtimeId: String): String = descriptor(runtimeId)?.version ?: "not installed"

    val architectureCount: Int get() = descriptor(LLAMA)?.architectures?.size ?: 0

    /**
     * The JNI contract the Kotlin side hard-requires. A bundle declaring
     * anything else is refused with a clear message, never loaded optimistically
     * (SPEC §17.3).
     */
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
        const val REQUIRED_JNI_CONTRACT = 3
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
