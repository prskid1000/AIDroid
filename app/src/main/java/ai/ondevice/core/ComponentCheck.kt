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
    ): List<MissingComponent> {
        val enabled = available.filter { it.enabled }.map { it.role }.toSet()
        val installed = available.map { it.role }.toSet()

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

        return mismatched + enabled.mapNotNull { role ->
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
