package ai.ondevice.core

/** A part that is needed and is not going to be there when the run starts. */
data class MissingComponent(
    val what: String,
    val because: String,
    val state: State,
) {
    enum class State {
        /** Nothing installed can fill this slot. */
        NOT_INSTALLED,

        /** It is installed and is not attached, so it will not be passed. */
        INSTALLED_NOT_ATTACHED,

        /** It is attached and the model it is attached to cannot use it. */
        WONT_ATTACH,
    }

    val remedy: String get() = when (state) {
        State.NOT_INSTALLED -> "Add one from Models → Add"
        State.INSTALLED_NOT_ATTACHED -> "Switch it on under Attachments"
        State.WONT_ATTACH -> "Switch it off, or pick a base model it fits"
    }
}

/** Says what is missing before a run rather than after it. */
object ComponentCheck {

    /**
     * The two roles sd.cpp only knows how to build for a UNet.
     *
     * ControlNetBlock branches on SD1/SD2/SDXL/SVD and has no other shape, and
     * the IP-Adapter's injection map is a list of `input_blocks.N.1` names that
     * exist in a UNet and nowhere else. SD 3.5 and FLUX.2 are both DiTs: there
     * are no such blocks to inject into, so either file loads, costs its memory
     * and changes nothing about the picture.
     */
    private val UNET_ONLY = setOf(AttachmentRole.CONTROLNET, AttachmentRole.IP_ADAPTER)

    /** The architectures those two were written against, as sd.cpp spells them. */
    private val UNET_ARCHITECTURES = setOf("sd1", "sd2", "sdxl", "sdxl_refiner", "svd")

    /**
     * @param available every add-on the library holds for this model, ticked or not.
     * @param architecture the base model's architecture, or null when it is not known yet.
     */
    fun forDiffusion(
        available: List<ModelAttachment>,
        architecture: String? = null,
        installedRoles: Set<AttachmentRole> = emptySet(),
        /**
         * Whether the checkpoint is the denoiser alone.
         *
         * The loader's finding where there has been a load, and null before
         * one. This is a property of the *file*, not of the family: SDXL ships
         * both as a full checkpoint that carries its two CLIPs and its VAE, and
         * as a quantised denoiser that carries none of them. Saying "SDXL keeps
         * its decoder in the file" is true of the first and wrong about the
         * second, and the second is what anybody running SDXL on a phone has.
         */
        bareDenoiser: Boolean? = null,
    ): List<MissingComponent> {
        val enabled = available.filter { it.enabled }.map { it.role }.toSet()
        val installed = available.map { it.role }.toSet() + installedRoles

        // What this family reads its prompt with, and whether its decoder is a
        // separate file. Asked of the architecture, because the answer differs
        // per family and not per role: SDXL takes CLIP-L and CLIP-G, FLUX.1
        // CLIP-L and T5-XXL, FLUX.2 a language model, Chroma T5 alone.
        val family = DiffusionDefaults.forName(architecture)
        // A full checkpoint supplies its own everything, so nothing is missing.
        val selfContained = bareDenoiser == false
        val needed = if (selfContained) {
            emptyList()
        } else {
            buildList {
                family?.encoders?.forEach { key ->
                    AttachmentRole.entries.firstOrNull { it.paramKey == key }?.let(::add)
                }
                // Before a load, fall back to what the family usually ships.
                if (bareDenoiser == true || family?.vaeSeparate == true) add(AttachmentRole.VAE)
            }
        }
        val unfilled = needed.filter { it !in enabled }.map { role ->
            MissingComponent(
                what = "No ${role.label} for ${architecture ?: "this model"}",
                because = if (role == AttachmentRole.VAE) {
                    "a quantised release is the denoiser alone, so the decoder that turns the " +
                        "latent into pixels has to come from a separate file"
                } else {
                    "this architecture reads its prompt through it, and without one there is " +
                        "nothing to turn the words into conditioning"
                },
                state = when {
                    role in installed && role !in enabled ->
                        MissingComponent.State.INSTALLED_NOT_ATTACHED
                    else -> MissingComponent.State.NOT_INSTALLED
                },
            )
        }

        val mismatched = if (architecture.isNullOrBlank() ||
            architecture.lowercase() in UNET_ARCHITECTURES
        ) {
            emptyList()
        } else {
            (enabled intersect UNET_ONLY).map { role ->
                MissingComponent(
                    what = "${role.label} does nothing on $architecture",
                    because = "sd.cpp only builds one for a UNet, and this model is a diffusion " +
                        "transformer with no UNet blocks to attach to",
                    state = MissingComponent.State.WONT_ATTACH,
                )
            }
        }

        return unfilled + mismatched + enabled.mapNotNull { role ->
            val requirement = role.requires ?: return@mapNotNull null
            val needed = role.required ?: return@mapNotNull null
            if (needed in enabled) return@mapNotNull null
            MissingComponent(
                what = "${role.label} needs a ${needed.label}",
                because = requirement.because,
                state = if (needed in installed) {
                    MissingComponent.State.INSTALLED_NOT_ATTACHED
                } else {
                    MissingComponent.State.NOT_INSTALLED
                },
            )
        }.sortedBy { it.what }
    }

    /** Whether this text model can be shown a picture. */
    fun forChatImage(companionPaths: Map<String, String>): MissingComponent? =
        if (companionPaths.keys.any { it.contains("vision", true) || it.contains("mmproj", true) }) {
            null
        } else {
            MissingComponent(
                what = "This model cannot read images",
                because = "images reach a text model through a separate projector file, and this " +
                    "model was installed without one",
                state = MissingComponent.State.NOT_INSTALLED,
            )
        }

    /** What a voice model needs beside itself. */
    fun forSpeech(requiresVoicePacks: Boolean, voicePackCount: Int): MissingComponent? =
        if (!requiresVoicePacks || voicePackCount > 0) {
            null
        } else {
            MissingComponent(
                what = "No voice packs installed",
                because = "this engine keeps its speakers in separate files and has none to read",
                state = MissingComponent.State.NOT_INSTALLED,
            )
        }
}
