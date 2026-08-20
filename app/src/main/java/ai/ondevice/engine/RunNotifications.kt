package ai.ondevice.engine

import ai.ondevice.core.Fmt
import ai.ondevice.proxy.ProxyActivity
import ai.ondevice.proxy.ProxyServer
import ai.ondevice.ui.vm.ChatState
import ai.ondevice.ui.vm.ImageState
import ai.ondevice.ui.vm.VideoState
import ai.ondevice.ui.vm.VoiceState

/**
 * Everything that could be happening at once, and what to say about it.
 *
 * Pulled out of [InferenceService] because the service was deciding this inline
 * and had grown a `when` that only knew about two of the five kinds of run. The
 * result was a notification that said "Model in memory" through a transcription
 * and "Serving the API" through a four-minute answer over HTTP — true
 * sentences, describing nothing that was actually going on.
 *
 * Pure, so it is testable without a device: given a snapshot, what should the
 * status bar say.
 */
data class RunSnapshot(
    val engine: EngineState = EngineState(),
    /** Runs in flight, counted across every engine. */
    val count: Int = 0,
    val clip: VideoState,
    val still: ImageState,
    val chat: ChatState,
    val voice: VoiceState,
    val served: ProxyServer.Status = ProxyServer.Status(),
    val remote: ProxyActivity? = null,
    /** A clip being made for somebody who is not holding a connection. */
    val videoJob: ai.ondevice.proxy.VideoJobs.Job? = null,
    /**
     * Which runtime holds weights, across all five.
     *
     * [engine] is llama's alone, so a resting line built from it announced
     * "no model loaded" while sd.cpp was holding five gigabytes.
     */
    val resident: String? = null,
)

/**
 * One line of status, in the shape a notification takes.
 *
 * [steps] of zero means an indeterminate bar. That distinction is load-bearing
 * on this hardware: a load and a VAE decode take minutes and report no step at
 * all, and a determinate bar frozen at zero for that long reads as a run that
 * has stalled rather than one that is working.
 */
data class RunLine(
    val title: String,
    val detail: String,
    val step: Int = 0,
    val steps: Int = 0,
) {
    val determinate: Boolean get() = steps > 0 && step > 0
}

object RunStatus {

    /**
     * What to show, in priority order.
     *
     * Only one thing can actually be sampling — the engines serialise — so the
     * first match is the whole story rather than an arbitrary pick among
     * several. The order is by how specific the answer is: a named run beats
     * "working", and "working" beats "a model is resident".
     */
    fun describe(s: RunSnapshot): RunLine {
        val clip = s.clip.takeIf { it.generating || it.loadingModel }
        val still = s.still.takeIf { it.generating || it.loadingModel }

        return when {
            clip != null -> RunLine(
                title = "Making a clip",
                detail = parts(
                    clip.model?.label,
                    if (clip.loadingModel) "loading weights" else clip.phase.label,
                    rate(clip.secondsPerStep),
                    elapsed(clip.elapsedMillis),
                ),
                step = clip.step,
                steps = clip.progressSteps,
            )

            still != null -> RunLine(
                title = "Making a picture",
                detail = parts(
                    still.model?.label,
                    if (still.loadingModel) "loading weights" else still.phase.label,
                    rate(still.secondsPerStep),
                    elapsed(still.elapsedMillis),
                ),
                step = still.step,
                steps = still.progressSteps,
            )

            // Before the remote branch, because a job outlives the request
            // that made it — by the time it is sampling there is no request.
            s.videoJob != null -> RunLine(
                title = "Making a clip",
                detail = parts(
                    s.videoJob.model.substringAfterLast('/'),
                    s.videoJob.phase.takeIf { it.isNotBlank() },
                    rate(s.videoJob.secondsPerStep),
                    s.videoJob.queuedBehind.takeIf { it > 0 }?.let { "$it queued" },
                ),
                step = s.videoJob.step,
                steps = s.videoJob.steps,
            )

            s.voice.speaking -> RunLine(
                title = "Speaking",
                detail = parts(s.voice.ttsModel?.label, elapsed(s.voice.elapsedMillis)),
            )

            s.voice.recording -> RunLine(
                title = "Recording",
                detail = parts(s.voice.sttModel?.label, elapsed(s.voice.elapsedMillis)),
            )

            transcribing(s.voice) -> RunLine(
                title = "Transcribing",
                detail = parts(
                    s.voice.sttModel?.label,
                    "${(s.voice.fileProgress * 100).toInt()}%",
                    elapsed(s.voice.elapsedMillis),
                ),
                step = (s.voice.fileProgress * 100).toInt(),
                steps = 100,
            )

            // A request from the network, named the way its own route named it.
            // The client is worth carrying: "Answering" on a phone nobody is
            // holding is otherwise indistinguishable from the app doing it.
            s.remote != null -> RunLine(
                title = s.remote.phase,
                detail = parts(
                    s.remote.client.takeIf { it.isNotBlank() }?.let { "via ${shorten(it)}" },
                    s.remote.model.takeIf { it.isNotBlank() },
                    tokens(s.remote.tokensPerSecond),
                    s.remote.rounds.takeIf { it > 1 }?.let { "round $it" },
                    (s.remote.inFlight - 1).takeIf { it > 0 }?.let { "$it queued" },
                ),
                step = s.remote.step,
                steps = s.remote.steps,
            )

            s.chat.generating || s.chat.loadingModel -> RunLine(
                title = "Answering",
                detail = parts(
                    s.chat.model?.label ?: s.engine.loaded?.modelId,
                    if (s.chat.loadingModel) "loading weights" else null,
                    tokens(s.chat.tokensPerSecond),
                    context(s.chat.contextUsed, s.engine.loaded?.contextLength ?: 0),
                ),
            )

            s.count > 0 -> RunLine("Working", parts(s.engine.loaded?.modelId))

            // Nothing running, but the port is open. Its own line because this
            // is the one resting state nobody started by tapping something, and
            // the address is what makes it recognisable rather than alarming.
            s.served.listening -> RunLine(
                title = "Serving the API",
                detail = parts(
                    s.served.url,
                    // Asked of the residency rather than of llama's own state,
                    // which said "no model loaded" while the diffusion engine
                    // held five gigabytes.
                    s.engine.loaded?.modelId?.let { "$it loaded" }
                        ?: s.resident?.let { "$it resident" }
                        ?: "no model loaded",
                ),
            )

            // Switched on and not serving. This is a state worth naming: the
            // service is alive precisely so it can start listening again, and
            // saying nothing here is how a proxy that has quietly given up
            // looks exactly like one that is working.
            s.served.enabled -> RunLine(
                title = "Proxy not listening",
                detail = parts(s.served.refusal?.substringBefore('.'), s.engine.loaded?.modelId),
            )

            s.engine.loaded != null -> RunLine(
                title = "Model in memory",
                detail = parts(s.engine.loaded.modelId),
            )

            // Reachable only for the instant between the service starting and
            // the first real state arriving. It used to read "Model in memory"
            // with nothing after it, which is a claim rather than a gap — and
            // it was read, reasonably, as the model being loaded.
            else -> RunLine("Idle", "")
        }
    }

    /**
     * Whether the service still has a reason to exist.
     *
     * The proxy term asks whether it is *switched on*, not whether it is
     * listening this instant. Those come apart during every rebind — `sync()`
     * closes the old socket before opening the new one — and reading the
     * instantaneous value meant a Tailscale reconnect stopped the service in
     * the gap. The process stayed up, still holding the port, but with nothing
     * keeping it foreground it was frozen: connections were accepted by the
     * kernel and then answered by nobody, so a client saw a timeout rather than
     * a refusal, which is the harder of the two to diagnose.
     */
    fun shouldStop(s: RunSnapshot): Boolean =
        s.engine.loaded == null && s.count == 0 && !s.served.enabled

    /**
     * The flag, not the fraction.
     *
     * This was written against `fileProgress` being strictly between its ends,
     * which read correctly and was wrong: the field defaulted to 0.74f — a
     * value off the design canvas — so a device that had never transcribed
     * anything reported a transcription in progress, and every other line in
     * this file was unreachable behind it.
     */
    private fun transcribing(voice: VoiceState): Boolean = voice.transcribing

    private fun parts(vararg values: String?): String =
        values.filterNotNull().filter { it.isNotBlank() }.joinToString(" · ")

    private fun tokens(rate: Float): String? =
        rate.takeIf { it > 0f }?.let { Fmt.tokensPerSecond(it) }

    private fun rate(secondsPerStep: Float): String? =
        secondsPerStep.takeIf { it > 0f }?.let { String.format("%.0f s/it", it) }

    private fun elapsed(millis: Long): String? = millis
        .takeIf { it > 1000L }
        ?.let { it / 1000 }
        ?.let { if (it >= 60) "${it / 60}m ${it % 60}s" else "${it}s" }

    private fun context(used: Int, limit: Int): String? =
        if (used > 0 && limit > 0) "$used/$limit ctx" else null

    /** A User-Agent is a paragraph; a notification line is not. */
    private fun shorten(client: String): String =
        client.substringBefore('/').take(24).ifBlank { client.take(24) }
}
