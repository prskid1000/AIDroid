package ai.ondevice.params

import ai.ondevice.engine.DiffusionEngine
import ai.ondevice.engine.RuntimeRegistry
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * That the weight types the app offers are spellings the runtime will take.
 *
 * The name goes through JNI to `str_to_sd_type`, which compares it against
 * ggml's own registry, and ggml writes the k-quants with a capital K. The
 * manifest offered `q3_k` and `q2_k` for as long as the row has existed —
 * never noticed, because nothing reported the key and so the row was never
 * shown. Now that it is shown, a value nobody can select successfully is a
 * setting that appears to work and does not, which is the failure this
 * codebase keeps finding.
 *
 * Two spellings could not be reconciled by lowering both: `q4_0` and `q4_K`
 * differ in case and only in case, so the case is load-bearing.
 */
class WeightTypeTest {

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    private val manifest: ParamManifest by lazy {
        json.decodeFromString(
            ParamManifest.serializer(),
            File("src/main/assets/params-manifest.json").readText(),
        )
    }

    private val spec: ParamSpec
        get() = manifest.spec(RuntimeRegistry.STABLE_DIFFUSION, DiffusionEngine.WEIGHT_TYPE_KEY)
            .also { assertNotNull("the weight type row is missing from the manifest", it) }!!

    /**
     * ggml's names, from `type_traits` in ggml.c — the same table
     * `str_to_sd_type` walks. Only the ones a phone would ask for; anything
     * outside this set is either larger than the file or a type this build
     * cannot write.
     */
    private val ggmlNames = setOf(
        "f32", "f16", "bf16",
        "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
        "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
    )

    /** The value that means "leave every tensor as the file wrote it". */
    private val unchanged = "as-is"

    @Test
    fun `every offered weight type is spelled the way ggml spells it`() {
        val offered = spec.values.filterNot { it == unchanged }
        assertTrue("the row offers no types at all", offered.isNotEmpty())
        offered.forEach {
            assertTrue(
                "\"$it\" is not a ggml type name; str_to_sd_type would refuse it. " +
                    "The k-quants are written with a capital K.",
                it in ggmlNames,
            )
        }
    }

    /**
     * Loading at the file's own precision is what every load did before this
     * setting existed, and a default of `f16` would have quietly converted
     * every checkpoint on the device the moment the row became visible.
     */
    @Test
    fun `the default leaves the file alone`() {
        assertEquals(unchanged, (spec.default as? kotlinx.serialization.json.JsonPrimitive)?.content)
        assertTrue("the unchanged option has to be selectable to be returned to", unchanged in spec.values)
    }

    /** sd.cpp settles the weight type while building the context; there is no later. */
    @Test
    fun `changing the weight type asks for a reload`() {
        assertTrue(spec.requiresReload)
    }
}
