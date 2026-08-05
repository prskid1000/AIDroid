package ai.ondevice.params

import ai.ondevice.engine.RuntimeRegistry
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The tiling controls, and specifically that the useless one stays gated.
 *
 * `vae_tiling` on its own only decides *that* the VAE decodes in tiles, not
 * that it decodes in small ones. The runtime's default tile is 32 latent units,
 * and a latent unit is 16 output pixels on Wan 2.2 — a 512px tile, which splits
 * a 704x384 frame into two and leaves the peak roughly three quarters of where
 * it was. That is why the switch could be on and the process still be killed,
 * and why `vae_tile_size` exists.
 *
 * `vae_tiling_temporal` is the trap. It reads like the answer to a clip that
 * runs out of memory as frames go up, and upstream implements it for LTX-AV's
 * VAE alone: Wan's runner never overrides `set_temporal_tiling_enabled`, so on
 * Wan the flag is accepted and does nothing. An inert switch beside a real
 * memory error is worse than no switch, so it is gated to the architecture that
 * honours it and shows its reason everywhere else.
 */
class VaeTilingParamsTest {

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    private val manifest: ParamManifest by lazy {
        json.decodeFromString(
            ParamManifest.serializer(),
            File("src/main/assets/params-manifest.json").readText(),
        )
    }

    private fun spec(key: String): ParamSpec =
        manifest.spec(RuntimeRegistry.STABLE_DIFFUSION, key)
            ?: throw AssertionError("$key is not in the manifest")

    @Test
    fun `tile size defaults to the runtime rather than to a number of ours`() {
        val size = spec("vae_tile_size")
        assertEquals(ParamType.INT, size.type)
        assertEquals(
            "0 is the sentinel for the runtime's own tile; a non-zero default here " +
                "would silently overrule every architecture's idea of a sensible one",
            0,
            size.default?.jsonPrimitive?.intOrNull,
        )
        // Below the runtime's own floor of 4 the value is read as "use the
        // default", so a minimum under 0 would give two meanings to one slider.
        assertEquals(0.0, size.min)
        assertNotNull("a tile size needs an upper bound to stay a slider", size.max)
    }

    @Test
    fun `tile overlap cannot be set above the half the runtime clamps to`() {
        val overlap = spec("vae_tile_overlap")
        assertEquals(ParamType.FLOAT, overlap.type)
        assertEquals(0.0, overlap.min)
        assertEquals(
            "the runtime clamps target_overlap to 0.5; offering more would be a " +
                "control that stops responding halfway along",
            0.5,
            overlap.max,
        )
    }

    @Test
    fun `temporal tiling is gated to the one architecture that implements it`() {
        val temporal = spec("vae_tiling_temporal")
        val gate = temporal.appliesTo
            ?: throw AssertionError(
                "vae_tiling_temporal must stay gated — ungated it offers Wan a cure " +
                    "for an out-of-memory it cannot administer",
            )
        assertEquals(
            "only LTX-AV's VAE overrides set_temporal_tiling_enabled upstream",
            listOf("ltxav"),
            gate.arch,
        )
        assertEquals("it splits a clip along time, so it is video-only", listOf("video"), gate.output)
    }

    @Test
    fun `the tiling switch no longer promises a saving it cannot make alone`() {
        assertTrue(
            "vae_tiling's help should send the reader to the tile size, which is " +
                "what actually decides how much memory tiling saves",
            spec("vae_tiling").help.contains("tile size", ignoreCase = true),
        )
    }
}
