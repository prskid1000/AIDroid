package ai.ondevice.data.hf

import ai.ondevice.core.ModelFormat
import org.junit.Assert.assertEquals
import org.junit.Test

/** That no two variants answer to one name, whichever code path built them. */
class DistinctQuantNamesTest {

    private fun variant(name: String, path: String, bytes: Long = 1_000) = QuantVariant(
        name = name,
        files = listOf(RemoteFile(filename = path, sizeBytes = bytes)),
        note = "",
    )

    private fun model(vararg quants: QuantVariant) = ResolvedModel(
        repoId = "owner/repo",
        owner = "owner",
        repo = "repo",
        revision = "main",
        displayName = "repo",
        architecture = null,
        modality = ai.ondevice.core.Modality.DIFFUSION,
        format = ModelFormat.SAFETENSORS,
        contextLength = null,
        chatTemplate = null,
        bosToken = null,
        eosToken = null,
        parameterCount = null,
        layers = null,
        embeddingLength = null,
        embeddingLengthKv = null,
        gated = false,
        quants = quants.toList(),
        companions = emptyList(),
        metadataFromHeader = false,
        securityStatus = null,
        hasPickleFiles = false,
    )

    @Test
    fun `one filename in two folders becomes two names`() {
        val fixed = model(
            variant("pytorch_lora_weights", "ema/pytorch_lora_weights.safetensors"),
            variant("pytorch_lora_weights", "pytorch_lora_weights.safetensors"),
        ).withDistinctQuantNames()

        val names = fixed.quants.map { it.name }
        assertEquals(2, names.toSet().size)
        assertEquals(listOf("ema/pytorch_lora_weights", "pytorch_lora_weights"), names)
    }

    @Test
    fun `names that already stand alone are untouched`() {
        val before = model(
            variant("Q4_K_M", "model-Q4_K_M.gguf"),
            variant("Q8_0", "model-Q8_0.gguf"),
        )
        assertEquals(before, before.withDistinctQuantNames())
    }

    @Test
    fun `only the clashing names are rewritten`() {
        val fixed = model(
            variant("Q4_K_M", "model-Q4_K_M.gguf"),
            variant("weights", "a/weights.safetensors"),
            variant("weights", "b/weights.safetensors"),
        ).withDistinctQuantNames()

        assertEquals(listOf("Q4_K_M", "a/weights", "b/weights"), fixed.quants.map { it.name })
    }
}
