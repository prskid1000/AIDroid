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
    }

    val remedy: String get() = when (state) {
        State.NOT_INSTALLED -> "Add one from Models → Add"
        State.INSTALLED_NOT_ATTACHED -> "Switch it on under Attachments"
    }
}

/** Says what is missing before a run rather than after it. */
object ComponentCheck {

    /** @param available every add-on the library holds for this model, ticked or not. */
    fun forDiffusion(available: List<ModelAttachment>): List<MissingComponent> {
        val enabled = available.filter { it.enabled }.map { it.role }.toSet()
        val installed = available.map { it.role }.toSet()

        return enabled.mapNotNull { role ->
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
