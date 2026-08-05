package ai.ondevice.ui.components

import ai.ondevice.core.RunPhase
import ai.ondevice.engine.DiffusionPhase

/**
 * The sentence under a run, from the two phases that describe it.
 *
 * There are two, and conflating them is what went wrong. [RunPhase] is what the
 * *app* is doing — idle, loading, running, stopping — and [DiffusionPhase] is
 * what the *runtime* is doing inside a run — preparing, sampling, decoding. A
 * stopping run is still preparing, in the sense that the encode it was told to
 * abandon has not finished yet, and the clip screen printed exactly that: it
 * asked only the runtime, so a press of Cancel left "preparing" on screen as
 * though nothing had been pressed.
 *
 * The still screen had the answer already, spelled out in full. It lived inline
 * in one composable, which is why the clip screen did not have it. It lives
 * here now, and both read it.
 */
fun runStatusLine(
    run: RunPhase,
    stage: DiffusionPhase,
    step: Int,
    /** What to say when there is no run: "No preview yet", "No clip yet". */
    idle: String,
    /**
     * What sampling means on this screen, which is the one genuinely
     * screen-specific line: a still is waiting on its first preview, a clip is
     * not previewed at all. Parameterised rather than branched on the caller,
     * so the two cannot drift apart again the way they already had.
     */
    sampling: String,
): String = when {
    run == RunPhase.Loading -> "loading model…"
    run == RunPhase.Idle -> idle

    // Said out loud, because the press is honoured but not instantly and a
    // silent wait reads as a hang.
    //
    // Encoding the prompt is one ggml graph — FLUX.2 reads it through a 4B
    // language model, half a minute of it — and abandoning that graph hands
    // sd.cpp an empty result it asserts on rather than checks, which is an
    // abort and takes the process with it. So the press waits for the encode
    // and lands on the first step. Sampling and the decode stop inside the
    // current graph, in about a step.
    run == RunPhase.Stopping && stage == DiffusionPhase.PREPARING ->
        "stopping · the prompt encode can't be interrupted, so it finishes first"
    run == RunPhase.Stopping -> "stopping · leaving the current step"

    stage == DiffusionPhase.PREPARING -> "preparing · loading weights, no steps to count yet"
    stage == DiffusionPhase.DECODING -> "decoding the latent to pixels · almost done"
    step <= 0 -> "warming up…"
    else -> sampling
}
