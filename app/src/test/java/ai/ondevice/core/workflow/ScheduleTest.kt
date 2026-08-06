package ai.ondevice.core.workflow

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.time.ZoneId
import java.time.ZonedDateTime

/**
 * The calendar, which is the whole of scheduling that a device cannot help with.
 *
 * Alarms, permissions and foreground-service exemptions all need hardware to
 * exercise. Working out *when* seven o'clock next is needs a clock and an
 * argument — and it is where the mistakes hide, because they are all off by
 * exactly one day and only twice a year.
 */
class ScheduleTest {

    private val london = ZoneId.of("Europe/London")

    private fun at(text: String, zone: ZoneId = london): ZonedDateTime =
        ZonedDateTime.of(java.time.LocalDateTime.parse(text), zone)

    private fun daily(minute: Int) =
        Schedule(enabled = true, kind = Schedule.DAILY, atMinute = minute)

    // ── off ──────────────────────────────────────────────────────────────

    @Test
    fun `a disabled schedule never fires`() {
        assertNull(daily(7 * 60).copy(enabled = false).nextOccurrence(at("2026-08-06T06:00")))
    }

    // ── daily ────────────────────────────────────────────────────────────

    @Test
    fun `daily fires later today when the time is still ahead`() {
        val next = daily(7 * 60).nextOccurrence(at("2026-08-06T06:00"))
        assertEquals(at("2026-08-06T07:00"), next)
    }

    @Test
    fun `daily rolls to tomorrow once the time has passed`() {
        val next = daily(7 * 60).nextOccurrence(at("2026-08-06T09:00"))
        assertEquals(at("2026-08-07T07:00"), next)
    }

    /** Exactly on the minute is past, not present — otherwise it fires twice. */
    @Test
    fun `daily at exactly the appointed minute goes to tomorrow`() {
        val next = daily(7 * 60).nextOccurrence(at("2026-08-06T07:00"))
        assertEquals(at("2026-08-07T07:00"), next)
    }

    @Test
    fun `midnight is a time like any other`() {
        val next = daily(0).nextOccurrence(at("2026-08-06T23:30"))
        assertEquals(at("2026-08-07T00:00"), next)
    }

    // ── the reason a wall-clock time is stored ───────────────────────────

    /**
     * Spring forward. The clocks go from 01:00 to 02:00 on 29 March 2026, so
     * the day is 23 hours long — and an absolute instant stored once would come
     * back an hour early for the rest of the year.
     */
    @Test
    fun `seven in the morning stays seven across the spring change`() {
        val next = daily(7 * 60).nextOccurrence(at("2026-03-28T09:00"))
        assertEquals(7, next!!.hour)
        assertEquals(java.time.LocalDate.parse("2026-03-29"), next.toLocalDate())
    }

    /** And back again in October, where the day is 25 hours long. */
    @Test
    fun `seven in the morning stays seven across the autumn change`() {
        val next = daily(7 * 60).nextOccurrence(at("2026-10-24T09:00"))
        assertEquals(7, next!!.hour)
        assertEquals(java.time.LocalDate.parse("2026-10-25"), next.toLocalDate())
    }

    // ── weekly ───────────────────────────────────────────────────────────

    private fun weekly(days: Set<Int>, minute: Int = 7 * 60) =
        Schedule(enabled = true, kind = Schedule.WEEKLY, atMinute = minute, onDays = days)

    @Test
    fun `weekly finds the next chosen day`() {
        // 2026-08-06 is a Thursday; Monday is day 1.
        val next = weekly(setOf(1)).nextOccurrence(at("2026-08-06T09:00"))
        assertEquals(at("2026-08-10T07:00"), next)
    }

    /** Today counts, but only while the time is still ahead of now. */
    @Test
    fun `weekly on today fires today when the time has not passed`() {
        val next = weekly(setOf(4)).nextOccurrence(at("2026-08-06T06:00"))
        assertEquals(at("2026-08-06T07:00"), next)
    }

    @Test
    fun `weekly on today rolls a week once the time has passed`() {
        val next = weekly(setOf(4)).nextOccurrence(at("2026-08-06T09:00"))
        assertEquals(at("2026-08-13T07:00"), next)
    }

    @Test
    fun `weekly with no days chosen never fires`() {
        assertNull(weekly(emptySet()).nextOccurrence(at("2026-08-06T09:00")))
    }

    // ── once ─────────────────────────────────────────────────────────────

    private fun once(date: String?, minute: Int = 7 * 60) =
        Schedule(enabled = true, kind = Schedule.ONCE, atMinute = minute, onDate = date)

    @Test
    fun `once on a future date fires then`() {
        val next = once("2026-08-09").nextOccurrence(at("2026-08-06T09:00"))
        assertEquals(at("2026-08-09T07:00"), next)
    }

    /** A date that has gone is gone — a once-only run does not fire late. */
    @Test
    fun `once on a past date never fires`() {
        assertNull(once("2026-08-01").nextOccurrence(at("2026-08-06T09:00")))
    }

    /** With no date it means the next time that clock reads, which may be tomorrow. */
    @Test
    fun `once with no date rolls to tomorrow when the time has passed`() {
        val next = once(null).nextOccurrence(at("2026-08-06T09:00"))
        assertEquals(at("2026-08-07T07:00"), next)
    }

    // ── words ────────────────────────────────────────────────────────────

    @Test
    fun `an off schedule describes itself as nothing`() {
        assertEquals("", Schedule().describe(london))
    }

    @Test
    fun `a daily schedule says when`() {
        assertEquals("every day at 07:30", daily(7 * 60 + 30).describe(london))
    }
}
