package ai.ondevice.engine

import ai.ondevice.proxy.ProxyActivity
import ai.ondevice.proxy.ProxyServer
import ai.ondevice.ui.vm.ChatState
import ai.ondevice.ui.vm.ImageState
import ai.ondevice.ui.vm.VideoState
import ai.ondevice.ui.vm.VoiceState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What the notification says, given what is happening.
 *
 * Every one of these was wrong before this test existed. The service decided
 * this inline with a `when` that knew about two of the five kinds of run, so a
 * transcription read "Model in memory", an answer over HTTP read "Serving the
 * API" for four minutes, and a conversation showed no token rate at all —
 * because the field it read is one nothing has ever written.
 *
 * None of it is visible on a device without waiting out a real run, which is
 * exactly the sort of thing CLAUDE.md means by logic worth a unit test.
 */
class RunStatusTest {

    private fun snapshot(
        engine: EngineState = EngineState(),
        count: Int = 0,
        clip: VideoState = VideoState(),
        still: ImageState = ImageState(),
        chat: ChatState = ChatState(),
        voice: VoiceState = VoiceState(),
        served: ProxyServer.Status = ProxyServer.Status(),
        remote: ProxyActivity? = null,
    ) = RunSnapshot(engine, count, clip, still, chat, voice, served, remote)

    @Test
    fun `an idle device with a model resident says so`() {
        val line = RunStatus.describe(
            snapshot(engine = EngineState(loaded = loaded("qwen"))),
        )
        assertEquals("Model in memory", line.title)
        assertTrue(line.detail.contains("qwen"))
    }

    /**
     * "Model in memory" with nothing after it is a claim, not a gap — and it
     * was read as one. The service can now be alive with nothing loaded,
     * because a listening proxy is reason enough, so this state is reachable
     * where before it was not.
     */
    @Test
    fun `nothing loaded and nothing running is idle, not a model in memory`() {
        assertEquals("Idle", RunStatus.describe(snapshot()).title)
    }

    @Test
    fun `a serving proxy says whether a model is actually loaded`() {
        val empty = RunStatus.describe(
            snapshot(served = ProxyServer.Status(enabled = true, listening = true, address = "100.64.0.1", port = 8080)),
        )
        assertTrue(empty.detail.contains("no model loaded"))

        val warm = RunStatus.describe(
            snapshot(
                engine = EngineState(loaded = loaded("qwen")),
                served = ProxyServer.Status(enabled = true, listening = true, address = "100.64.0.1", port = 8080),
            ),
        )
        assertTrue(warm.detail.contains("qwen loaded"))
    }

    /** A proxy that has given up must not look like one that is working. */
    @Test
    fun `a proxy switched on but not listening says so`() {
        val line = RunStatus.describe(
            snapshot(
                served = ProxyServer.Status(
                    enabled = true,
                    listening = false,
                    refusal = "Tailscale is not connected, so there is no address. Open the app.",
                ),
            ),
        )
        assertEquals("Proxy not listening", line.title)
        assertTrue(line.detail.contains("Tailscale is not connected"))
    }

    @Test
    fun `an idle server names the address rather than the model`() {
        val line = RunStatus.describe(
            snapshot(
                served = ProxyServer.Status(
                    listening = true, address = "100.64.0.1", port = 8080,
                ),
            ),
        )
        assertEquals("Serving the API", line.title)
        assertTrue(line.detail.contains("http://100.64.0.1:8080"))
    }

    /**
     * The failure this whole file exists for: a request from the network used
     * to fall through every branch and land on "Serving the API", so a
     * four-minute answer and an idle port read identically.
     */
    @Test
    fun `a remote request outranks the idle server line`() {
        val line = RunStatus.describe(
            snapshot(
                served = ProxyServer.Status(listening = true, address = "100.64.0.1", port = 8080),
                remote = ProxyActivity(
                    inFlight = 1,
                    phase = "Answering",
                    client = "claude-cli/2.0.1",
                    model = "qwen3.5-9b",
                    tokensPerSecond = 12.5f,
                ),
            ),
        )
        assertEquals("Answering", line.title)
        assertTrue(line.detail.contains("via claude-cli"))
        assertTrue(line.detail.contains("qwen3.5-9b"))
        assertTrue(line.detail.contains("12.5"))
    }

    @Test
    fun `queued requests are counted, and one alone is not`() {
        fun detail(inFlight: Int) = RunStatus.describe(
            snapshot(
                remote = ProxyActivity(
                    inFlight = inFlight, phase = "Answering", client = "c", model = "m",
                ),
            ),
        ).detail

        assertFalse(detail(1).contains("queued"))
        assertTrue(detail(3).contains("2 queued"))
    }

    /** A picture asked for over HTTP still counts steps, and they still show. */
    @Test
    fun `a remote picture carries its step count`() {
        val line = RunStatus.describe(
            snapshot(
                remote = ProxyActivity(
                    inFlight = 1, phase = "Making a picture", client = "curl",
                    model = "flux", step = 3, steps = 8,
                ),
            ),
        )
        assertEquals("Making a picture", line.title)
        assertTrue(line.determinate)
        assertEquals(3, line.step)
        assertEquals(8, line.steps)
    }

    @Test
    fun `an in-app answer reports its rate from the session, not the engine`() {
        val line = RunStatus.describe(
            snapshot(
                engine = EngineState(loaded = loaded("qwen")),
                chat = ChatState(generating = true, tokensPerSecond = 8.25f, contextUsed = 900),
            ),
        )
        assertEquals("Answering", line.title)
        assertTrue(line.detail.contains("8.2") || line.detail.contains("8.3"))
        assertTrue(line.detail.contains("900/4096 ctx"))
    }

    @Test
    fun `loading weights is said rather than shown as a stalled bar`() {
        val line = RunStatus.describe(snapshot(chat = ChatState(loadingModel = true)))
        assertEquals("Answering", line.title)
        assertTrue(line.detail.contains("loading weights"))
        assertFalse(line.determinate)
    }

    @Test
    fun `speaking, recording and transcribing each get their own line`() {
        assertEquals(
            "Speaking",
            RunStatus.describe(snapshot(voice = VoiceState(speaking = true))).title,
        )
        assertEquals(
            "Recording",
            RunStatus.describe(snapshot(voice = VoiceState(recording = true))).title,
        )
        val transcribing = RunStatus.describe(snapshot(voice = VoiceState(transcribing = true, fileProgress = 0.4f)))
        assertEquals("Transcribing", transcribing.title)
        assertTrue(transcribing.detail.contains("40%"))
        assertTrue(transcribing.determinate)
    }

    /**
     * The fraction alone cannot say. It defaulted to 0.74f — a number off the
     * design canvas — so reading it as "in progress when strictly between the
     * ends" reported a permanent transcription on a device that had never run
     * one, and hid every other line behind it.
     */
    @Test
    fun `a progress fraction without the flag is not a transcription`() {
        assertEquals(
            "Idle",
            RunStatus.describe(snapshot(voice = VoiceState(fileProgress = 0.74f))).title,
        )
        assertEquals(
            "Idle",
            RunStatus.describe(snapshot(voice = VoiceState(fileProgress = 1f))).title,
        )
    }

    @Test
    fun `a clip outranks everything, because it is the longest thing here`() {
        val line = RunStatus.describe(
            snapshot(
                clip = VideoState(generating = true, step = 12, progressSteps = 40, secondsPerStep = 9f),
                chat = ChatState(generating = true),
                remote = ProxyActivity(1, "Answering", "c", "m"),
            ),
        )
        assertEquals("Making a clip", line.title)
        assertTrue(line.detail.contains("9 s/it"))
        assertEquals(12, line.step)
    }

    // ── the stop condition ──────────────────────────────────────────────

    @Test
    fun `an idle service with nothing loaded stops`() {
        assertTrue(RunStatus.shouldStop(snapshot()))
    }

    /**
     * The one that killed the socket. An idle proxy has nothing loaded and
     * nothing running, so without the third term the service stopped itself
     * the moment the last generation finished — a server up for exactly as
     * long as the last request.
     */
    @Test
    fun `a listening proxy is a reason to stay alive`() {
        assertFalse(
            RunStatus.shouldStop(
                snapshot(served = ProxyServer.Status(enabled = true, listening = true)),
            ),
        )
    }

    /**
     * The one that froze the process. `sync()` closes the old socket before it
     * opens the new one, so a Tailscale reconnect passes through a moment of
     * enabled-but-not-listening — and stopping there left a background process
     * still holding the port, accepting connections and answering none.
     */
    @Test
    fun `a proxy mid-rebind is still a reason to stay alive`() {
        assertFalse(
            RunStatus.shouldStop(
                snapshot(served = ProxyServer.Status(enabled = true, listening = false)),
            ),
        )
    }

    @Test
    fun `a switched-off proxy is not`() {
        assertTrue(
            RunStatus.shouldStop(
                snapshot(served = ProxyServer.Status(enabled = false, listening = false)),
            ),
        )
    }

    @Test
    fun `a loaded model or a run in flight keeps it alive`() {
        assertFalse(RunStatus.shouldStop(snapshot(engine = EngineState(loaded = loaded("q")))))
        assertFalse(RunStatus.shouldStop(snapshot(count = 1)))
    }

    private fun loaded(id: String) = LoadedModel(
        modelId = id,
        contextLength = 4096,
        layers = 32,
        embeddingLength = 0,
        embeddingLengthKv = 0,
        chatTemplate = null,
        stopSequences = emptyList(),
        loadMillis = 0,
    )
}
