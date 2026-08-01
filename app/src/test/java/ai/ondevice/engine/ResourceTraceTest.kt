package ai.ondevice.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The trace's arithmetic and its storage format. */
class ResourceTraceTest {

    private fun trace(
        cpu: List<Int>,
        rss: List<Int> = cpu.map { 1000 },
        avail: List<Int> = cpu.map { 4000 },
        intervalMillis: Int = SAMPLE_INTERVAL_MILLIS,
        baselineRssMb: Int = 800,
    ) = ResourceTrace(
        intervalMillis = intervalMillis,
        elapsedMillis = cpu.size.toLong() * intervalMillis,
        cpuPercent = cpu,
        rssMb = rss,
        availMb = avail,
        baselineRssMb = baselineRssMb,
        totalRamMb = 8192,
        cores = 8,
    )

    @Test
    fun `summary values describe the series`() {
        val subject = trace(cpu = listOf(10, 90, 50), rss = listOf(1000, 2400, 2000))
        assertEquals(90, subject.peakCpuPercent)
        assertEquals(50, subject.meanCpuPercent)
        assertEquals(2400, subject.peakRssMb)
        assertEquals(4000, subject.minAvailMb)
    }

    @Test
    fun `added memory subtracts the baseline and never goes negative`() {
        assertEquals(1600, trace(listOf(50), rss = listOf(2400), baselineRssMb = 800).deltaRssMb)
        assertEquals(0, trace(listOf(50), rss = listOf(600), baselineRssMb = 800).deltaRssMb)
    }

    @Test
    fun `a trace round-trips through storage`() {
        val subject = trace(cpu = listOf(12, 88, 44), rss = listOf(900, 2100, 2000))
        val restored = ResourceTrace.parse(subject.toJson())
        assertNotNull(restored)
        assertEquals(subject, restored)
    }

    /** A trace that cannot be read is no trace. */
    @Test
    fun `unreadable and empty traces parse to null`() {
        assertNull(ResourceTrace.parse(null))
        assertNull(ResourceTrace.parse(""))
        assertNull(ResourceTrace.parse("{not json"))
        // Well-formed but with no samples: there is nothing to draw, and a graph
        // of nothing reads as "this used no resources" rather than "no data".
        assertNull(ResourceTrace.parse(trace(emptyList()).toJson()))
    }

    /** Decimation is the part that could quietly lie. */
    @Test
    fun `decimation keeps peaks and loses only resolution`() {
        val halved = trace(
            cpu = listOf(20, 40, 60, 80),
            rss = listOf(1000, 4000, 1200, 1100),
            avail = listOf(4000, 900, 3800, 3900),
        ).halved()

        assertEquals(listOf(30, 70), halved.cpuPercent)
        assertEquals("the 4 GB spike must survive", listOf(4000, 1200), halved.rssMb)
        assertEquals("the moment of least free memory must survive", listOf(900, 3800), halved.availMb)
    }

    /** An odd-length series keeps its last sample rather than dropping it. */
    @Test
    fun `decimation carries an unpaired final sample`() {
        val halved = trace(cpu = listOf(20, 40, 90), rss = listOf(1, 1, 7), avail = listOf(9, 9, 2))
            .halved()
        assertEquals(listOf(30, 90), halved.cpuPercent)
        assertEquals(listOf(1, 7), halved.rssMb)
        assertEquals(listOf(9, 2), halved.availMb)
    }

    /** The window a graph covers must not shrink just because its resolution did. */
    @Test
    fun `a decimated trace still spans the whole run`() {
        val long = trace(cpu = List(180) { 50 })
        val halved = long.halved()
        assertEquals(long.elapsedMillis, halved.elapsedMillis)
        assertEquals(SAMPLE_INTERVAL_MILLIS * 2, halved.intervalMillis)
        assertEquals(90, halved.cpuPercent.size)
        assertTrue(halved.cpuPercent.size <= MAX_TRACE_POINTS)
        assertEquals(
            "halving describes the same span",
            long.cpuPercent.size.toLong() * long.intervalMillis,
            halved.cpuPercent.size.toLong() * halved.intervalMillis,
        )
    }
}
