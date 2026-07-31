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
 * The architecture list is not written here. §2.3 and Appendix A #3 both insist
 * it come from the pinned upstream source, because a hand-kept list rots and
 * silently reintroduces model-locking. It ships as `runtimes.json`, the output
 * of `tools/generate_runtimes.py`; bumping a runtime means running that script,
 * not editing this file.
 *
 * The comment here used to say "CI-generated asset" and cite a runtime-bump
 * pipeline. There is no CI — no workflow directory, and no Gradle task invokes
 * the script. Naming it as something a person runs is the version of the rule
 * that can actually be followed.
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
     * Whether the text runtime has a GPU it can actually reach.
     *
     * The quant list quotes a fast path off the back of this. It used to have
     * to describe the *binary*, because no build had a GPU backend and the only
     * honest question was what was compiled; now that one is compiled into
     * every arm64 build, the binary is no longer the interesting half — a
     * compiled backend on a phone with no driver behind libOpenCL.so is not a
     * fast path. [backendsFor] asks the device.
     */
    val hasOpenClBackend: Boolean
        get() = BackendId.OPENCL in backendsFor(LLAMA)

    /**
     * The backends [runtimeId] actually registered, which is not the same
     * question as which one the user would like. Empty means the runtime is not
     * installed; callers treat CPU as the floor rather than inventing a device.
     *
     * Asked of the loaded binary first, and only then of the manifest. The two
     * answer different questions: runtimes.json says what this *build compiles*,
     * which is a property of the APK, while ggml's registry says what it found
     * *on this phone* — and those diverge exactly where it matters. The OpenCL
     * backend is compiled into every arm64 build; whether a device has a driver
     * behind libOpenCL.so is not something the APK can know, and a manifest that
     * claimed OPENCL on a phone without one would send work at a device that
     * does not exist.
     */
    fun backendsFor(runtimeId: String): List<BackendId> {
        val declared = descriptor(runtimeId)?.takeIf { it.installed }?.backends ?: return emptyList()
        return registered(runtimeId).ifEmpty { declared }
    }

    /**
     * What ggml registered in this process, per runtime, asked once.
     *
     * Per runtime rather than once for all three, because they are three shared
     * objects with a static ggml each: they are built from one CMake invocation
     * and so far always agree, but "so far always agree" is not a thing to hard
     * code — whisper's answer comes from whisper's binary.
     *
     * A runtime with no way to answer, or one whose library is not loadable at
     * all, returns empty and the caller falls back to the manifest. That is the
     * honest shape: an unanswered question is not the answer "CPU".
     */
    private fun registered(runtimeId: String): List<BackendId> = reported.getOrPut(runtimeId) {
        val info = runCatching {
            when (runtimeId) {
                LLAMA -> if (LlamaBridge.available) LlamaBridge.nativeSystemInfo() else null
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

/**
 * The slice of a bridge's `nativeSystemInfo()` this file reads.
 *
 * Deliberately not the whole document: it also carries build numbers and a
 * device list, and parsing fields nobody reads is how a JSON contract acquires
 * requirements it never meant to have.
 */
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
