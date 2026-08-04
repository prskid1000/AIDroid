package ai.ondevice.core.workflow

import ai.ondevice.core.Modality

/**
 * What a step *is*, derived from its stored type string.
 *
 * The catalogue is deliberately not the one ComfyUI uses. There a checkpoint
 * is loaded by one node, a prompt encoded by a second, a latent denoised by a
 * third and decoded by a fourth — which is right on a desktop, where the point
 * is reusing one loaded checkpoint across several branches. It is wrong here
 * for three reasons: the JNI has no such seams to expose, only whole runs; the
 * reuse it buys is impossible when the engine holds a load lock and runs one
 * thing at a time; and it would make the author wire up the encoders and
 * decoder by hand, which this app already chooses for them. So a model step is
 * one step, and the shape is closer to an automation graph than to a tensor
 * graph.
 */
sealed interface NodeKind {

    /** The stored discriminator. */
    val type: String

    /** What the palette and an unnamed step call it. */
    val title: String

    /** Which family it belongs to, for grouping in the palette. */
    val family: NodeFamily

    /** One line, said in the palette. */
    val blurb: String

    /** What it reads. Model steps derive this from the chosen model. */
    fun slots(context: NodeContext): List<SlotSpec> = emptyList()

    /** What it produces. */
    fun outputs(context: NodeContext): List<OutputSpec> = emptyList()

    // ── sources ──────────────────────────────────────────────────────────

    /**
     * A value from outside the graph.
     *
     * One node with several sources rather than one node per source: typing a
     * prompt and picking a text file are the same step with a different
     * origin, and splitting them would double the palette to no end.
     *
     * "Ask when run" is the setting that makes a workflow a tool rather than a
     * macro. Without it every reuse begins by editing the graph.
     */
    data object Input : NodeKind {
        override val type = "input"
        override val title = "Input"
        override val family = NodeFamily.SOURCE
        override val blurb = "Text, a picture, a recording or a file — typed here, picked, or asked for when the workflow runs."
        override fun outputs(context: NodeContext) =
            listOf(OutputSpec("value", context.portType(), "Value"))
    }

    /** Something this app has already made. */
    data object LibraryItem : NodeKind {
        override val type = "library"
        override val title = "From library"
        override val family = NodeFamily.SOURCE
        override val blurb = "A picture, clip, recording or transcript this app made earlier."
        override fun outputs(context: NodeContext) =
            listOf(OutputSpec("value", context.portType(), "Value"))
    }

    // ── models ───────────────────────────────────────────────────────────

    /**
     * Any model computation — one node type, six shapes.
     *
     * The slots are read off the chosen model's own row rather than typed
     * here, the same discipline that has the parameter manifest ask whether a
     * family makes video instead of listing the ones that do. Choose a
     * transcription model and this step takes audio and gives text; choose a
     * diffusion model and it takes a prompt and gives a picture. Nothing in
     * this file names a model.
     */
    data object Processor : NodeKind {
        override val type = "processor"
        override val title = "Model"
        override val family = NodeFamily.MODEL
        override val blurb = "Run one of the installed models. What it takes and gives follows from which model you choose."

        override fun slots(context: NodeContext): List<SlotSpec> = when (context.shape) {
            ProcessorShape.TEXT -> listOf(
                SlotSpec("prompt", PortType.TEXT, "Prompt"),
                SlotSpec("system", PortType.TEXT, "System prompt", required = false),
            )
            ProcessorShape.VISION -> listOf(
                SlotSpec("prompt", PortType.TEXT, "Prompt"),
                SlotSpec("image", PortType.IMAGE, "Picture to look at"),
                SlotSpec("system", PortType.TEXT, "System prompt", required = false),
            )
            ProcessorShape.IMAGE -> listOf(
                SlotSpec("prompt", PortType.TEXT, "Prompt"),
                SlotSpec("negative", PortType.TEXT, "Negative prompt", required = false),
                SlotSpec(
                    "init", PortType.IMAGE, "Start from", required = false,
                    help = "Denoises away from this picture rather than from noise.",
                ),
                SlotSpec("mask", PortType.IMAGE, "Mask", required = false),
                SlotSpec("control", PortType.IMAGE, "Control map", required = false),
            )
            ProcessorShape.VIDEO -> listOf(
                SlotSpec("prompt", PortType.TEXT, "Prompt"),
                SlotSpec("negative", PortType.TEXT, "Negative prompt", required = false),
                SlotSpec(
                    "first", PortType.IMAGE, "First frame", required = false,
                    help = "A distilled clip model is usually trained to animate a picture, and is much weaker without one.",
                ),
                SlotSpec("last", PortType.IMAGE, "Last frame", required = false),
            )
            ProcessorShape.TRANSCRIBE -> listOf(
                SlotSpec("audio", PortType.AUDIO, "Recording"),
            )
            ProcessorShape.SPEAK -> listOf(
                SlotSpec("text", PortType.TEXT, "What to say"),
            )
            ProcessorShape.UPSCALE -> listOf(
                SlotSpec("image", PortType.IMAGE, "Picture"),
            )
            ProcessorShape.NONE -> emptyList()
        }

        override fun outputs(context: NodeContext): List<OutputSpec> = when (context.shape) {
            ProcessorShape.TEXT, ProcessorShape.VISION ->
                listOf(OutputSpec("text", PortType.TEXT, "Answer"))
            ProcessorShape.IMAGE, ProcessorShape.UPSCALE ->
                listOf(OutputSpec("image", PortType.IMAGE, "Picture"))
            ProcessorShape.VIDEO ->
                listOf(OutputSpec("clip", PortType.CLIP, "Clip"))
            ProcessorShape.TRANSCRIBE -> listOf(
                OutputSpec("text", PortType.TEXT, "Transcript"),
                OutputSpec("subtitles", PortType.FILE, "Subtitles"),
            )
            ProcessorShape.SPEAK ->
                listOf(OutputSpec("audio", PortType.AUDIO, "Recording"))
            ProcessorShape.NONE -> emptyList()
        }
    }

    /** A tool the chat models can already call, as a step of its own. */
    data object Tool : NodeKind {
        override val type = "tool"
        override val title = "Tool"
        override val family = NodeFamily.MODEL
        override val blurb = "Search, fetch a page, read a file, or any connected server — the tools chat can call, called directly."
        override fun slots(context: NodeContext) =
            listOf(SlotSpec("arguments", PortType.TEXT, "Arguments", required = false))
        override fun outputs(context: NodeContext) =
            listOf(OutputSpec("text", PortType.TEXT, "Result"))
    }

    // ── media, no model ──────────────────────────────────────────────────

    /** One frame out of a clip — what makes video composable at all. */
    data object FrameExtract : NodeKind {
        override val type = "frame_extract"
        override val title = "Take a frame"
        override val family = NodeFamily.MEDIA
        override val blurb = "Pull one frame out of a clip, to upscale it or to start another clip from it."
        override fun slots(context: NodeContext) = listOf(SlotSpec("clip", PortType.CLIP, "Clip"))
        override fun outputs(context: NodeContext) =
            listOf(OutputSpec("image", PortType.IMAGE, "Frame"))
    }

    /** Stills into a clip. */
    data object Assemble : NodeKind {
        override val type = "assemble"
        override val title = "Assemble clip"
        override val family = NodeFamily.MEDIA
        override val blurb = "Turn a list of pictures into a clip at a chosen frame rate."
        override fun slots(context: NodeContext) = listOf(SlotSpec("images", PortType.LIST, "Pictures"))
        override fun outputs(context: NodeContext) = listOf(OutputSpec("clip", PortType.CLIP, "Clip"))
    }

    /**
     * Crop, scale and pad — the cheapest node here and among the most useful.
     *
     * A generative model is trained at particular sizes and shapes, and asking
     * for one far from them costs coherence before it costs anything else. A
     * step that fits a picture to the next model's shape is how a graph avoids
     * that, and it needs no model of its own.
     */
    data object Resize : NodeKind {
        override val type = "resize"
        override val title = "Resize"
        override val family = NodeFamily.MEDIA
        override val blurb = "Crop, scale or pad a picture to the size and shape the next model wants."
        override fun slots(context: NodeContext) = listOf(SlotSpec("image", PortType.IMAGE, "Picture"))
        override fun outputs(context: NodeContext) = listOf(OutputSpec("image", PortType.IMAGE, "Picture"))
    }

    /**
     * Long text into pieces that fit.
     *
     * An hour of transcript does not fit a context window, so without this
     * "summarise this recording" is not a workflow anyone can build.
     */
    data object TextSplit : NodeKind {
        override val type = "text_split"
        override val title = "Split text"
        override val family = NodeFamily.MEDIA
        override val blurb = "Break long text into pieces — by paragraph, by sentence, or to a length a model can hold."
        override fun slots(context: NodeContext) = listOf(SlotSpec("text", PortType.TEXT, "Text"))
        override fun outputs(context: NodeContext) = listOf(OutputSpec("pieces", PortType.LIST, "Pieces"))
    }

    /** Pieces back into one. */
    data object TextJoin : NodeKind {
        override val type = "text_join"
        override val title = "Join text"
        override val family = NodeFamily.MEDIA
        override val blurb = "Put a list of pieces back together with a separator."
        override fun slots(context: NodeContext) = listOf(SlotSpec("pieces", PortType.LIST, "Pieces"))
        override fun outputs(context: NodeContext) = listOf(OutputSpec("text", PortType.TEXT, "Text"))
    }

    // ── logic ────────────────────────────────────────────────────────────

    /**
     * The same step again, with a different seed each time.
     *
     * The best value for time on this hardware, and it is not close. A model
     * load is tens of seconds and gigabytes; a sample is a minute or two.
     * Four pictures from one load beats four loads by more than it beats
     * anything else in this list.
     */
    data object Batch : NodeKind {
        override val type = "batch"
        override val title = "Several times"
        override val family = NodeFamily.LOGIC
        override val blurb = "Run the steps below more than once with a different seed each time, without reloading the model."
        override fun slots(context: NodeContext) = emptyList<SlotSpec>()
    }

    /**
     * Stop and let a person choose.
     *
     * On a device where the next step is twenty minutes, choosing before
     * spending them is worth more than any amount of automation.
     */
    data object Pick : NodeKind {
        override val type = "pick"
        override val title = "Let me choose"
        override val family = NodeFamily.LOGIC
        override val blurb = "Pause, show what has been made so far, and carry on with the one you pick."
        override fun slots(context: NodeContext) = listOf(SlotSpec("options", PortType.LIST, "Options"))
        override fun outputs(context: NodeContext) =
            listOf(OutputSpec("chosen", PortType.FILE, "Chosen"))
    }

    /**
     * A bracket, not a container.
     *
     * A container would make the step list a tree, and a tree on a phone means
     * either indentation that eats the width or a second screen. A bracket
     * over a run of steps stays flat, and makes a loop that crosses another
     * loop's boundary impossible to express.
     */
    data object RepeatStart : NodeKind {
        override val type = "repeat_start"
        override val title = "Repeat"
        override val family = NodeFamily.LOGIC
        override val blurb = "Run the steps between here and the end of the repeat more than once."
    }

    /**
     * Where a Repeat, a Batch or an Only-if stops.
     *
     * It collects, the same way the end of a for-each does. Without that a
     * Batch was decorative: it ran the steps below four times, each pass
     * overwrote the last one's answer, and there was no way to name the four
     * results together — so the node that exists to make four pictures from one
     * model load could not hand four pictures to anything.
     */
    data object RepeatEnd : NodeKind {
        override val type = "repeat_end"
        override val title = "End repeat"
        override val family = NodeFamily.LOGIC
        override val blurb = "Where the repeat stops. Collects what each pass produced."
        override fun slots(context: NodeContext) =
            listOf(SlotSpec("collect", PortType.FILE, "Keep from each pass", required = false))
        override fun outputs(context: NodeContext) =
            listOf(OutputSpec("items", PortType.LIST, "Everything collected"))
    }

    /** Once per item. */
    data object ForEachStart : NodeKind {
        override val type = "for_each_start"
        override val title = "For each"
        override val family = NodeFamily.LOGIC
        override val blurb = "Run the steps below once for every item in a list."
        override fun slots(context: NodeContext) = listOf(SlotSpec("items", PortType.LIST, "Items"))
        override fun outputs(context: NodeContext) =
            listOf(OutputSpec("item", PortType.FILE, "This item"))
    }

    data object ForEachEnd : NodeKind {
        override val type = "for_each_end"
        override val title = "End for each"
        override val family = NodeFamily.LOGIC
        override val blurb = "Where the loop stops. Collects what each pass produced."
        override fun slots(context: NodeContext) =
            listOf(SlotSpec("collect", PortType.FILE, "Keep from each pass", required = false))
        override fun outputs(context: NodeContext) =
            listOf(OutputSpec("items", PortType.LIST, "Everything collected"))
    }

    /** Carry on, or skip to the end of the branch. */
    data object Branch : NodeKind {
        override val type = "branch"
        override val title = "Only if"
        override val family = NodeFamily.LOGIC
        override val blurb = "Skip the steps below unless a condition holds."
        override fun slots(context: NodeContext) =
            listOf(SlotSpec("value", PortType.TEXT, "Value to test", required = false))
    }

    // ── data ─────────────────────────────────────────────────────────────

    /**
     * Reshape text, without a model.
     *
     * A template with expressions rather than a language: it covers turning a
     * transcript into a prompt, which is what "transform" means in practice
     * here, and it is deterministic, cancellable and testable without a
     * device. A real embedded language is a different decision with a native
     * build behind it.
     */
    data object Script : NodeKind {
        override val type = "script"
        override val title = "Text template"
        override val family = NodeFamily.DATA
        override val blurb = "Build text out of earlier steps — {{1.text}} — with a handful of functions for trimming, joining and matching."
        override fun outputs(context: NodeContext) = listOf(OutputSpec("text", PortType.TEXT, "Text"))
    }

    /** Pull one piece out. */
    data object Extract : NodeKind {
        override val type = "extract"
        override val title = "Extract"
        override val family = NodeFamily.DATA
        override val blurb = "Take the part of some text that matches a pattern."
        override fun slots(context: NodeContext) = listOf(SlotSpec("text", PortType.TEXT, "Text"))
        override fun outputs(context: NodeContext) = listOf(OutputSpec("text", PortType.TEXT, "Match"))
    }

    // ── sinks ────────────────────────────────────────────────────────────

    /** Name a value as a keeper. */
    data object Output : NodeKind {
        override val type = "output"
        override val title = "Keep"
        override val family = NodeFamily.SINK
        override val blurb = "Save this into the library, where the rest of the app can find it."
        override fun slots(context: NodeContext) =
            listOf(SlotSpec("value", PortType.FILE, "What to keep"))
    }

    /** Say something about a step without changing it. */
    data object Note : NodeKind {
        override val type = "note"
        override val title = "Note"
        override val family = NodeFamily.SINK
        override val blurb = "A line of explanation for whoever opens this next."
    }

    /**
     * A type this build does not know.
     *
     * Kept rather than dropped, and refused rather than skipped: skipping
     * silently changes what the workflow means, and a graph that quietly does
     * less than it says is worse than one that stops and explains.
     */
    data class Unknown(val raw: String) : NodeKind {
        override val type = raw
        override val title = "Unrecognised step"
        override val family = NodeFamily.SINK
        override val blurb = "Saved by a newer version of this app. It is kept as it was and will not run."
    }

    companion object {
        val ALL: List<NodeKind> = listOf(
            Input, LibraryItem,
            Processor, Tool,
            FrameExtract, Assemble, Resize, TextSplit, TextJoin,
            Batch, Pick, RepeatStart, RepeatEnd, ForEachStart, ForEachEnd, Branch,
            Script, Extract,
            Output, Note,
        )

        fun of(type: String): NodeKind =
            ALL.firstOrNull { it.type == type } ?: Unknown(type)
    }
}

/** How the palette is grouped. */
enum class NodeFamily(val label: String) {
    SOURCE("Bring in"),
    MODEL("Run a model"),
    MEDIA("Change it"),
    LOGIC("Control"),
    DATA("Text"),
    SINK("Finish"),
}

/**
 * What a Processor turns out to be, once a model has been chosen.
 *
 * Read off the model's own row — its modality, and for diffusion whether the
 * family makes video. No list of model names anywhere.
 */
enum class ProcessorShape {
    NONE, TEXT, VISION, IMAGE, VIDEO, TRANSCRIBE, SPEAK, UPSCALE;

    companion object {
        fun of(modality: Modality?, makesVideo: Boolean, isUpscaler: Boolean): ProcessorShape = when {
            isUpscaler -> UPSCALE
            modality == Modality.TEXT -> TEXT
            modality == Modality.VISION -> VISION
            modality == Modality.SPEECH_TO_TEXT -> TRANSCRIBE
            modality == Modality.TEXT_TO_SPEECH -> SPEAK
            modality == Modality.DIFFUSION && makesVideo -> VIDEO
            modality == Modality.DIFFUSION -> IMAGE
            else -> NONE
        }
    }
}

/**
 * What the editor knows about a node when asking it for its shape.
 *
 * Slots depend on choices — which model, which kind of input — so they cannot
 * be constants on the kind. This is the little bit of context that makes them
 * derivable instead.
 */
data class NodeContext(
    val shape: ProcessorShape = ProcessorShape.NONE,
    /** For an Input or a Library step: what it is bringing in. */
    val declaredType: PortType = PortType.TEXT,
) {
    fun portType(): PortType = declaredType
}
