package ai.ondevice.core

/**
 * What a runtime is doing, for the one control that starts and stops it.
 *
 * Five screens answered this question five ways: an enum derived on the state
 * with a label and an enabled flag (Image), an inline `when` over three
 * booleans in the composable (Video), a single `busy` that swaps a Send icon
 * for a Stop one (Chat), two unrelated booleans and two buttons (Voice), and a
 * `running`/`cancelling` pair (Workflow). One question, five state machines,
 * and every one of them free to forget a state.
 *
 * It cost something real. An unload is not instant — freeing waits for the run
 * still inside the context to return — so while it works the run is neither
 * going nor stopped. Image and Video both went on showing live progress and an
 * enabled Cancel through it, and fixing that meant writing the same fix twice
 * in two different shapes, with three more shapes waiting on the other screens.
 *
 * So the phase lives here and the screens read it.
 *
 * Deliberately *not* the whole of what a screen's button says. "Add a diffusion
 * model", "Choose a source image" and "No diffusion runtime in this build" are
 * answers to a different question — what this screen still needs before it can
 * start — and mixing the two is what let a prerequisite and a running job share
 * one enum and one set of edge cases. This type only describes a runtime that
 * is already doing something; when it is [Idle] the screen decides.
 */
enum class RunPhase {
    /** Nothing running. The screen's own prerequisites decide what to offer. */
    Idle,

    /** Weights going in. Interruptible, and worth saying so. */
    Loading,

    /** Generating, transcribing, speaking — the work itself. */
    Running,

    /**
     * A cancel or an unload is in flight and there is nothing left to press.
     *
     * One phase for both, because the difference between "stopping" and
     * "freeing" is a word rather than a state: an unload cancels the run on its
     * way to the memory, so both are the same wait for the same native call to
     * return, and both must refuse a second press for the same reason. Two
     * phases would be two chances to handle only one of them.
     */
    Stopping,
    ;

    /** Whether a run is in progress in any sense — loading counts. */
    val busy: Boolean get() = this != Idle

    /**
     * Whether a progress readout belongs on screen.
     *
     * False while stopping: the numbers are still arriving from a run that has
     * been told to stop, and showing them says it is still going.
     */
    val showsProgress: Boolean get() = this == Loading || this == Running
}

/** A button's text and whether pressing it does anything. */
data class RunControl(val label: String, val enabled: Boolean)

/**
 * The control for a runtime that is doing something, or null when it is idle
 * and the screen should say what it wants instead.
 *
 * Null rather than a label of its own so a caller cannot accidentally render
 * "Cancel" over a screen with no model installed: there is no idle label here
 * to get wrong, because there is no idle label here at all.
 */
fun RunPhase.control(): RunControl? = when (this) {
    RunPhase.Idle -> null
    // A load is worth interrupting; upstream cannot stop mid-call, but the app
    // honours the intent by dropping the weights the moment it returns.
    RunPhase.Loading, RunPhase.Running -> RunControl("Cancel", enabled = true)
    RunPhase.Stopping -> RunControl("Stopping…", enabled = false)
}

/**
 * The phase, from the flags a screen already keeps.
 *
 * Ordered, and the order is the point. Stopping wins over running because a run
 * being torn down is not a run in progress, and that precedence inverted is
 * exactly the bug this type exists for. Loading wins over running because a
 * generate queued behind a load has not started.
 */
fun runPhaseOf(
    stopping: Boolean = false,
    loading: Boolean = false,
    running: Boolean = false,
): RunPhase = when {
    stopping -> RunPhase.Stopping
    loading -> RunPhase.Loading
    running -> RunPhase.Running
    else -> RunPhase.Idle
}
