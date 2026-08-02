package ai.ondevice.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * That a bundle cannot offer a part its architecture has no use for.
 *
 * The flat list recorded which base a component fitted in English, inside the
 * sentence shown to the user, where nothing could check it. These assert the
 * same claims against the tables that already know — [DiffusionFamily] for
 * encoders, [ComponentCheck] for the two roles sd.cpp only builds for a UNet.
 */
class StarterBundleTest {

    @Test
    fun `every bundle names an architecture the runtime can`() {
        StarterModels.BUNDLES.forEach {
            assertNotNull(
                "${it.label} names an architecture nothing recognises: ${it.architecture}",
                DiffusionFamily.forName(it.architecture),
            )
        }
    }

    @Test
    fun `an encoder is only offered where the architecture reads through it`() {
        val encoderRoles = setOf(
            AttachmentRole.CLIP_L, AttachmentRole.CLIP_G,
            AttachmentRole.T5XXL, AttachmentRole.LLM_ENCODER,
        )
        StarterModels.BUNDLES.forEach { bundle ->
            val family = DiffusionFamily.forName(bundle.architecture)!!
            bundle.parts.filter { it.role in encoderRoles }.forEach { part ->
                val key = part.role!!.paramKey
                assertTrue(
                    "${bundle.label} offers a ${part.role.label} it does not read through",
                    key in family.encoders || key in family.optionalEncoders,
                )
            }
        }
    }

    @Test
    fun `ControlNet and IP-Adapter are only offered on a UNet`() {
        val unetOnly = setOf(AttachmentRole.CONTROLNET, AttachmentRole.IP_ADAPTER)
        StarterModels.BUNDLES.forEach { bundle ->
            val offered = bundle.parts.mapNotNull { it.role }.toSet() intersect unetOnly
            if (offered.isEmpty()) return@forEach
            // ComponentCheck is the authority: ask it, and it must not object.
            val complaints = ComponentCheck.forDiffusion(
                available = bundle.parts.mapNotNull { part ->
                    part.role?.let {
                        ModelAttachment(part.repoId, it, part.repoId, part.repoId, enabled = true)
                    }
                },
                architecture = bundle.architecture,
                bareDenoiser = true,
            ).filter { it.state == MissingComponent.State.WONT_ATTACH }
            assertEquals(
                "${bundle.label} offers $offered, which sd.cpp cannot build for it",
                emptyList<MissingComponent>(),
                complaints,
            )
        }
    }

    @Test
    fun `an IP-Adapter always comes with the vision encoder it needs`() {
        StarterModels.BUNDLES.forEach { bundle ->
            val roles = bundle.parts.mapNotNull { it.role }.toSet()
            if (AttachmentRole.IP_ADAPTER !in roles) return@forEach
            assertTrue(
                "${bundle.label} offers an IP-Adapter with no CLIP-vision to read pictures with",
                AttachmentRole.CLIP_VISION in roles,
            )
        }
    }

    @Test
    fun `every bundle's base is one of the listed models`() {
        StarterModels.BUNDLES.forEach {
            assertTrue(
                "${it.label}'s base is not in ALL",
                it.base in StarterModels.ALL,
            )
            assertEquals(Modality.DIFFUSION, it.base.modality)
        }
    }
}
