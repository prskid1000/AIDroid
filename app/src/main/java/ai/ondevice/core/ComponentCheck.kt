package ai.ondevice.core

/**
 * A part that is needed and is not going to be there when the run starts.
 *
 * @property state the difference that matters to the user. "You do not have
 *   this" and "you have this and it is switched off" want opposite actions, and
 *   the app used to give the same answer to both — which was no answer, because
 *   the failure only surfaced once the runtime had loaded and refused.
 */
data class MissingComponent(
    val what: String,
    val because: String,
    val state: State,
) {
    enum class State {
        /** Nothing installed can fill this slot. */
        NOT_INSTALLED,

        /**
         * It is installed and is not attached, so it will not be passed.
         *
         * This is the one that reads as a bug rather than as a setting: the
         * Models screen shows the file present and the run behaves as though it
         * were absent, because `DiffusionEngine.load` resolves each path from
         * the *ticked* attachments and falls back to an empty string.
         */
        INSTALLED_NOT_ATTACHED,
    }

    val remedy: String get() = when (state) {
        State.NOT_INSTALLED -> "Add one from Models → Add"
        State.INSTALLED_NOT_ATTACHED -> "Switch it on under Attachments"
    }
}

/**
 * Says what is missing before a run rather than after it.
 *
 * Every rule here is derived from data the app already holds — the declared
 * dependencies on [AttachmentRole], and the difference between what is
 * installed and what is switched on. There is no list of model names and no
 * per-model special case; adding a dependency is a line in the enum.
 */
object ComponentCheck {

    /**
     * @param available every add-on the library holds for this model, ticked or
     *   not. [ModelAttachment.enabled] is what decides whether it is passed.
     */
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

    /**
     * Whether this text model can be shown a picture.
     *
     * Asked of the companion map rather than of the model's name or
     * architecture: a projector is a file that is either paired with the model
     * or is not. The chat screen used to find this out at send time, after the
     * user had picked an image and written a message.
     */
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

    /**
     * What a voice model needs beside itself.
     *
     * @param voicePackCount how many voice packs came with it. A synthesiser
     *   whose speaker embeddings are separate files has nothing to speak *as*
     *   without at least one, and reports a shape error deep in the graph rather
     *   than saying so.
     */
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
