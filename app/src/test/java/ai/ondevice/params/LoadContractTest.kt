package ai.ondevice.params

import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.RoleFamily
import ai.ondevice.engine.DiffusionEngine
import ai.ondevice.engine.RuntimeRegistry
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * That everything the loader will act on can be reached from the app.
 *
 * `nativeLoad` used to take eleven positional strings, and the cost was not the
 * length of the call — it was that the fields it *did not* take were invisible.
 * `uncond_diffusion_model_path` unset meant Ideogram 4 could not run and
 * `high_noise_diffusion_model_path` unset meant Wan 2.2 could not; neither said
 * so, because the app had no way to name the file. The architecture looked
 * unsupported when it was merely unplumbed.
 *
 * These assert the three lists that now have to agree: the roles the engine
 * sends, the keys it sends them under, and the manifest rows that let a person
 * choose one.
 */
class LoadContractTest {

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    private val manifest: ParamManifest by lazy {
        json.decodeFromString(
            ParamManifest.serializer(),
            File("src/main/assets/params-manifest.json").readText(),
        )
    }

    private fun described(key: String) =
        manifest.spec(RuntimeRegistry.STABLE_DIFFUSION, key)

    /**
     * A companion denoiser is the model in more than one piece. If one is not
     * sent, the run is wrong rather than plainer — so this family in
     * particular must never sit outside the set the loader is told about.
     */
    @Test
    fun `every companion denoiser reaches the loader`() {
        val companions = AttachmentRole.entries.filter { it.family == RoleFamily.COMPANION_DENOISER }
        assertTrue("the family exists to hold these", companions.isNotEmpty())
        companions.forEach {
            assertTrue(
                "${it.label} is a companion denoiser the loader is never told about",
                it.paramKey in DiffusionEngine.LOAD_SETTING_KEYS ||
                    it in loadTimeRoles(),
            )
        }
    }

    /**
     * The engine keeps LOAD_TIME_ROLES private — it is the list it sends and
     * the list it reports as resident — so this reads it the way the manifest
     * does, by asking which roles have a described path row.
     */
    private fun loadTimeRoles(): List<AttachmentRole> =
        AttachmentRole.entries.filter { described(it.paramKey)?.type == ParamType.PATH }

    @Test
    fun `every load setting the engine sends is described`() {
        DiffusionEngine.LOAD_SETTING_KEYS.forEach { key ->
            val spec = described(key)
            assertTrue(
                "$key is sent to sd_ctx_params_t and the manifest does not describe it, " +
                    "so it renders as an untyped text box",
                spec != null,
            )
        }
    }

    /**
     * sd.cpp settles each of these while building the context. A row that does
     * not say so offers an edit that appears to work and is discarded.
     */
    @Test
    fun `every load setting asks for a reload`() {
        DiffusionEngine.LOAD_SETTING_KEYS.forEach { key ->
            assertEquals("$key is load-time and must say so", true, described(key)?.requiresReload)
        }
    }

    /**
     * The roles added for the missing paths, named individually.
     *
     * A list-to-list comparison would pass if both lists lost the same entry.
     * These are the four architectures that were unreachable, so they are
     * spelled out.
     */
    @Test
    fun `the paths that blocked whole architectures are described`() {
        listOf(
            AttachmentRole.UNCOND_DIFFUSION to "Ideogram 4",
            AttachmentRole.HIGH_NOISE_DIFFUSION to "Wan 2.2",
            AttachmentRole.MOTION_MODULE to "AnimateDiff",
            AttachmentRole.AUDIO_VAE to "LTX-AV audio",
        ).forEach { (role, blocked) ->
            val spec = described(role.paramKey)
            assertTrue("${role.paramKey} is undescribed, so $blocked stays unreachable", spec != null)
            assertEquals("${role.paramKey} names a file", ParamType.PATH, spec!!.type)
        }
    }

    /** Two roles pointing at one key would send one file under both names. */
    @Test
    fun `no two roles share a param key`() {
        val byKey = AttachmentRole.entries.groupBy { it.paramKey }.filterValues { it.size > 1 }
        assertEquals("roles sharing a key: $byKey", emptyMap<String, List<AttachmentRole>>(), byKey)
    }

    /**
     * Image and video share one runtime and one parameter set, and must not
     * offer each other's settings.
     *
     * The video ones were gated by a hand-typed list of every video
     * architecture, which is the kind of list that stops being true the week
     * one is added; the image ones were not gated at all, so a Wan checkpoint
     * was offered `ip_adapter_strength` and `batch_count` — neither of which
     * `sd_vid_gen_params_t` has a field for. Both now ask
     * `DiffusionFamily.isVideo`, which is derived from upstream's own branch.
     */
    @Test
    fun `the two outputs do not offer each other's settings`() {
        val specs = manifest.paramsFor(RuntimeRegistry.STABLE_DIFFUSION)
        fun outputOf(key: String) = specs.firstOrNull { it.key == key }?.appliesTo?.output

        // No field in the video struct at all.
        listOf("ip_adapter", "ip_adapter_strength", "clip_vision", "batch_count", "photo_maker", "pulid")
            .forEach {
                assertEquals("$it has no field in sd_vid_gen_params_t", listOf("image"), outputOf(it))
            }
        // No field in the image struct.
        listOf("video_frames", "fps", "moe_boundary", "high_noise_diffusion_model", "audio_vae")
            .forEach {
                assertEquals("$it has no field in sd_img_gen_params_t", listOf("video"), outputOf(it))
            }
        // Shared by both, and gated by neither.
        listOf("steps", "cfg_scale", "seed", "width", "sampling_method", "loras", "vae", "hires_fix")
            .forEach {
                assertEquals("$it reaches both structs and must not be gated", null, outputOf(it))
            }
    }

    @Test
    fun `the video architectures are the ones upstream claims`() {
        listOf("Wan 2.x", "Wan 2.2 I2V", "Hunyuan Video", "LingBot Video", "LTXAV", "SVD")
            .forEach {
                assertEquals("$it generates video", true, ai.ondevice.core.DiffusionFamily.isVideo(it))
            }
        listOf("SDXL", "SD1.x", "Flux", "Z-Image", "Krea2")
            .forEach {
                assertEquals("$it generates stills", false, ai.ondevice.core.DiffusionFamily.isVideo(it))
            }
        // A name nothing recognises gets every setting rather than half of them.
        assertEquals(null, ai.ondevice.core.DiffusionFamily.isVideo("something shipped next year"))
    }

    /**
     * The three roles that belong to the other runtimes.
     *
     * They appeared on the diffusion parameter screen under "not described
     * yet" — empty text boxes for `mmproj`, `vad_model` and `voices`, none of
     * which `sd_ctx_params_t` has any field for. A row you can type into that
     * nothing reads is worse than no row.
     */
    @Test
    fun `the other runtimes' roles are not diffusion auxiliaries`() {
        listOf(
            AttachmentRole.VISION_PROJECTOR to "llama.cpp",
            AttachmentRole.VAD to "whisper.cpp",
            AttachmentRole.VOICES to "kokoro",
        ).forEach { (role, owner) ->
            assertEquals(
                "${role.paramKey} belongs to $owner and sd.cpp has no field for it",
                false,
                role.isDiffusionAuxiliary,
            )
        }
    }

    /** Everything the diffusion loader is actually told about is a diffusion role. */
    @Test
    fun `every described diffusion path is a diffusion auxiliary`() {
        AttachmentRole.entries.filter { described(it.paramKey)?.type == ParamType.PATH }
            .forEach {
                assertEquals(
                    "${it.paramKey} has a diffusion row and is marked as not a diffusion role",
                    true,
                    it.isDiffusionAuxiliary,
                )
            }
    }

    /**
     * A strength for a component that is not attached does nothing at all.
     *
     * `ip_adapter_strength` said so and `control_strength` did not, so the
     * ControlNet slider was live, draggable and marked as modified with no
     * ControlNet installed — the same shape of silent no-op as a LoRA that
     * matches no tensor.
     */
    @Test
    fun `a strength is gated on the file it scales`() {
        listOf("control_strength" to "control_net", "ip_adapter_strength" to "ip_adapter")
            .forEach { (strength, file) ->
                val spec = described(strength)
                assertTrue("$strength is not described", spec != null)
                assertEquals(
                    "$strength scales $file and must be disabled until one is chosen",
                    file,
                    spec!!.dependsOn?.key,
                )
            }
    }
}
