package ai.ondevice.data.db

import ai.ondevice.core.DownloadState
import ai.ondevice.core.Modality
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** What a tab says while a model is still arriving. */
class InstallingModelTest {

    private fun job(
        done: Long,
        total: Long,
        state: DownloadState = DownloadState.RUNNING,
    ) = InstallingModel(
        modelId = "m",
        displayName = "Qwen3.5 4B",
        modality = Modality.TEXT,
        attachmentRole = null,
        bytesDone = done,
        bytesTotal = total,
        state = state,
    )

    @Test
    fun `a job with no declared total does not divide by zero`() {
        assertEquals(0f, job(done = 1_000, total = 0).fraction, 0f)
    }

    @Test
    fun `a server that over-reports cannot push the bar past full`() {
        assertEquals(1f, job(done = 3_000, total = 2_000).fraction, 0f)
    }

    @Test
    fun `the label carries the percentage`() {
        assertEquals("Qwen3.5 4B · 62%", job(done = 620, total = 1_000).label)
    }

    @Test
    fun `a paused job says so rather than looking stalled`() {
        val paused = job(done = 500, total = 1_000, state = DownloadState.PAUSED)
        assertTrue(paused.paused)
        assertEquals("Qwen3.5 4B · paused at 50%", paused.label)
        assertFalse(job(done = 500, total = 1_000).paused)
    }
}
