package ai.ondevice.params

import ai.ondevice.core.DiffusionFamily
import org.junit.Assert.assertNotNull
import org.junit.Test

/**
 * That the encoder an architecture cannot run without can always be attached.
 *
 * The manifest's `appliesTo.arch` lists were typed by hand and stopped at
 * FLUX.2, so every architecture upstream added afterwards had its text-encoder
 * slot refused — the file it needs most was the one file it would not accept.
 * These assert the derived table names an encoder for each; that it reaches the
 * screen is [ParamRepository.readsEncoder]'s job, tested through `visible`.
 */
class EncoderGateTest {

    @Test
    fun `the language-model families name an llm encoder`() {
        listOf(
            "Qwen Image", "Qwen Image Layered", "Z-Image", "Anima", "Ovis Image",
            "Ernie Image", "Boogu Image", "Lens", "PiD", "Ideogram 4", "Krea2",
            "Mage Flow", "Longcat-Image", "SeFi-Image", "Flux.2 klein",
        ).forEach {
            val family = DiffusionFamily.forName(it)
            assertNotNull("$it should be recognised", family)
            org.junit.Assert.assertTrue(
                "$it reads its prompt with a language model",
                "llm" in family!!.encoders,
            )
        }
    }

    @Test
    fun `the T5 families name a t5xxl encoder`() {
        listOf("Chroma Radiance", "MiniT2I", "Wan 2.x").forEach {
            val family = DiffusionFamily.forName(it)
            assertNotNull("$it should be recognised", family)
            org.junit.Assert.assertTrue("$it reads its prompt with T5", "t5xxl" in family!!.encoders)
        }
    }
}
