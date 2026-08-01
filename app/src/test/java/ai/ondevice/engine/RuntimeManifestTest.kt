package ai.ondevice.engine

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The shipped `runtimes.json`, read as a file.
 *
 * A unit test cannot open Android assets, and it does not need to: the asset is
 * a checked-in file and reading it here catches a bad regeneration before the
 * build that packages it.
 */
class RuntimeManifestTest {

    private val manifest: List<Map<String, Any?>> by lazy {
        val text = File("src/main/assets/runtimes.json").readText()
        Json.parseToJsonElement(text).jsonObject["runtimes"]!!.jsonArray.map { entry ->
            val obj = entry.jsonObject
            mapOf(
                "id" to obj["id"]!!.jsonPrimitive.content,
                "architectures" to obj["architectures"]?.jsonArray?.map { it.jsonPrimitive.content }.orEmpty(),
                "languages" to obj["languages"]?.jsonArray?.map { it.jsonPrimitive.content }.orEmpty(),
            )
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun architectures(entry: Map<String, Any?>) = entry["architectures"] as List<String>

    @Suppress("UNCHECKED_CAST")
    private fun languages(entry: Map<String, Any?>) = entry["languages"] as List<String>

    /**
     * The bug this file exists for.
     *
     * Kokoro's espeak language list used to be written into `architectures`.
     * The registry unions every runtime's architectures, and the resolver falls
     * back to a repo's Hugging Face tags when a GGUF header carries no
     * architecture of its own — so a repo tagged `en` resolved to *architecture*
     * `en`. FLUX.2 Klein and Real-ESRGAN both did.
     */
    @Test
    fun `no runtime declares a bare language code as an architecture`() {
        val languageCodes = setOf("en", "es", "fr", "hi", "it", "pt", "de", "ja", "ko", "zh", "ru", "ar")
        manifest.forEach { entry ->
            val offending = architectures(entry).filter { it.lowercase() in languageCodes }
            assertEquals(
                "${entry["id"]} lists ${offending.joinToString()} as architectures; " +
                    "language codes belong in `languages`",
                emptyList<String>(),
                offending,
            )
        }
    }

    @Test
    fun `kokoro declares languages and no architectures`() {
        val kokoro = manifest.first { it["id"] == RuntimeRegistry.KOKORO }
        assertEquals(emptyList<String>(), architectures(kokoro))
        assertTrue("kokoro should declare the languages espeak can pronounce", languages(kokoro).isNotEmpty())
    }

    /** The runtimes that load models still declare what they can load. */
    @Test
    fun `the model runtimes still declare architectures`() {
        listOf(RuntimeRegistry.LLAMA, RuntimeRegistry.WHISPER, RuntimeRegistry.STABLE_DIFFUSION).forEach { id ->
            val entry = manifest.first { it["id"] == id }
            assertTrue("$id declares no architectures", architectures(entry).isNotEmpty())
        }
    }
}
