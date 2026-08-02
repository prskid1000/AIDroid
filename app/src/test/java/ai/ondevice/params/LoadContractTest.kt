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
        //
        // The two ControlNet rows joined them late: `sd_vid_gen_params_t` has
        // no `control_strength`, and `generate_video` hands the sampler an
        // empty control image and a strength of zero — a clip's control map
        // goes to VACE, whose blocks are part of the checkpoint. Both rows were
        // ungated, so a Wan model offered a file picker for weights nothing
        // would read and a slider with no field to reach.
        listOf(
            "ip_adapter", "ip_adapter_strength", "clip_vision", "batch_count", "photo_maker",
            "pulid", "control_net", "control_strength",
        ).forEach {
            assertEquals("$it has no field in sd_vid_gen_params_t", listOf("image"), outputOf(it))
        }
        // No field in the image struct.
        listOf("video_frames", "fps", "moe_boundary", "high_noise_diffusion_model", "audio_vae")
            .forEach {
                assertEquals("$it has no field in sd_img_gen_params_t", listOf("video"), outputOf(it))
            }
        // Shared by both, and gated by neither.
        listOf(
            "steps", "cfg_scale", "seed", "width", "sampling_method", "loras", "vae", "hires_fix",
            "slg_scale", "skip_layers",
        ).forEach {
            assertEquals("$it reaches both structs and must not be gated", null, outputOf(it))
        }
    }

    /**
     * That skip-layer guidance has the one setting that turns it on.
     *
     * Upstream gates the whole feature on `(slg_scale != 0) && !layers.empty()`
     * and `sd_sample_params_init` leaves `layer_count` at 0. The app set the
     * scale, the start and the end and never the list, so three dials were
     * live, draggable, marked as modified, and read by a sampler that had
     * already decided SLG was off. Nothing said so, because nothing could —
     * the picture it produced was simply the picture without SLG.
     *
     * The video path did not even set the other three, which is why the pair
     * of assertions differ in what they are guarding against.
     */
    @Test
    fun `skip-layer guidance has a layer list`() {
        val spec = described("skip_layers")
        assertTrue("skip_layers is undescribed, so SLG cannot be switched on at all", spec != null)
        assertEquals("it names block indices", ParamType.INT_ARRAY, spec!!.type)
        assertEquals(
            "the scale is what SLG is gated on, so the list follows it",
            "slg_scale",
            spec.dependsOn?.key,
        )
        assertEquals("sd.cpp reads it per run, not per context", false, spec.requiresReload)

        val cpp = File("src/main/cpp/sd_jni.cpp").readText()
        assertTrue(
            "the layer list is never handed to sd_slg_params_t, so SLG stays off",
            cpp.contains("slg.layers"),
        )
        assertEquals(
            "both generate paths have to set SLG; the video one used to set none of it",
            2,
            Regex("""apply_slg\(\*e,""").findAll(cpp).count(),
        )
    }

    /**
     * That the three per-architecture defaults are left to the architecture.
     *
     * `sd_sample_params_init` writes INFINITY into `flow_shift`, `eta` and
     * `img_cfg`, and each is a sentinel the runtime resolves later — the flow
     * shift from the version it detected (5 for Wan, 7 for Hunyuan Video, 3
     * for most), eta from the sampler, img_cfg from the model. The app held
     * `flow_shift` at 0.0f and assigned it on every run, which replaced that
     * resolution with a literal zero. `time_snr_shift(0, t)` is `0*t/(1-t)`,
     * zero for every t, so the whole sigma schedule went to zero and
     * `noise_scaling` handed the sampler back its own latent. Every flow model
     * — Flux, SD3, Wan, Qwen-Image, LTX, Chroma — sampled a schedule with no
     * noise in it, and nothing anywhere said so.
     *
     * A negative value is the sentinel on this side of the boundary, so the
     * defaults have to be negative and the minimums have to allow it.
     */
    @Test
    fun `the per-model sampler defaults are left to the model`() {
        listOf("flow_shift", "eta", "img_cfg").forEach { key ->
            val spec = described(key)
            assertTrue("$key is undescribed", spec != null)
            assertEquals(
                "$key must default to the sentinel, not to a number that replaces the " +
                    "architecture's own",
                -1.0,
                (spec!!.default as? kotlinx.serialization.json.JsonPrimitive)?.content?.toDouble(),
            )
            assertTrue("$key cannot reach its sentinel: min is ${spec.min}", (spec.min ?: 0.0) <= -1.0)
        }

        val cpp = File("src/main/cpp/sd_jni.cpp").readText()
        assertTrue(
            "nothing converts the negative sentinel back to INFINITY, so the runtime reads " +
                "a literal -1 instead of resolving its own default",
            cpp.contains("or_model_default"),
        )
        assertTrue(
            "flow_shift is still assigned raw somewhere; it has to go through the sentinel",
            !cpp.contains("flow_shift                  = e->flow_shift") &&
                !cpp.contains("flow_shift           = e->flow_shift"),
        )
    }

    /**
     * That the identity adapters are given someone to keep.
     *
     * Both are load-time paths and both were being passed, so the weights went
     * resident. The generate-time half — `pm_params.id_images` and the two
     * strengths — was left as `sd_img_gen_params_init` wrote it: no images,
     * therefore no identity. Several hundred megabytes each, and a picture of
     * nobody in particular.
     */
    @Test
    fun `the identity adapters get an identity`() {
        // The two adapters take different inputs, and assuming otherwise is how
        // PuLID came to be "fixed" while still doing nothing: PhotoMaker reads
        // `pm_params.id_images`, actual photographs; PuLID reads a precomputed
        // embedding off disk and returns early from `before_diffusion` while it
        // is empty. So the weight is gated on the embedding, not on the model.
        listOf("style_strength" to "photo_maker", "id_weight" to "pulid_id_embedding").forEach { (dial, file) ->
            val spec = described(dial)
            assertTrue("$dial is undescribed", spec != null)
            assertEquals("$dial scales $file", file, spec!!.dependsOn?.key)
            assertEquals("$dial has no field in sd_vid_gen_params_t", listOf("image"), spec.appliesTo?.output)
        }
        val cpp = File("src/main/cpp/sd_jni.cpp").readText()
        assertTrue(
            "pm_params.id_images is never filled, so PhotoMaker loads and does nothing",
            cpp.contains("pm_params.id_images"),
        )
        assertTrue(
            "pulid_params.id_embedding_path is never filled, so PuLID loads and does nothing",
            cpp.contains("pulid_params.id_embedding_path"),
        )
        assertEquals(
            "the embedding names a file, not a picture",
            ParamType.PATH,
            described("pulid_id_embedding")?.type,
        )
    }

    /** The cache is the one setting here that buys minutes rather than quality. */
    @Test
    fun `the step cache is reachable`() {
        val mode = described("cache_mode")
        assertTrue("cache_mode is undescribed", mode != null)
        assertTrue("disabled has to be selectable to be returned to", "disabled" in mode!!.values)
        listOf("easycache", "ucache", "taylorseer").forEach {
            assertTrue("$it is one of upstream's modes and is not offered", it in mode.values)
        }
        assertEquals("the cache is per run, not per context", false, mode.requiresReload)
        assertEquals(
            "the options only mean anything once a mode is chosen",
            "cache_mode",
            described("cache_option")?.dependsOn?.key,
        )
    }

    /**
     * That a clip's hi-res stage is not offered as if it were a still's.
     *
     * They share the manifest rows and are different features. On an image,
     * hi-res is generate-small-then-denoise-larger and every upscaler mode
     * works. On a clip it is LTX-AV's latent spatial upsampler, a separate
     * file, and `generate_video` returns false for any other architecture, for
     * any upscaler but MODEL, and for a missing path. All three came back as
     * "The run produced no frames. This is usually memory" — after minutes of
     * sampling, about a setting the person had turned on deliberately.
     */
    @Test
    fun `the video hi-res stage answers for itself`() {
        val cpp = File("src/main/cpp/sd_jni.cpp").readText()
        assertTrue(
            "the video path still shares apply_hires, so a Wan run with hi-res on fails as OOM",
            cpp.contains("apply_video_hires"),
        )
        assertTrue(
            "nothing checks the architecture, and only LTX-AV has a latent upsampler",
            cpp.contains("LTX"),
        )
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
