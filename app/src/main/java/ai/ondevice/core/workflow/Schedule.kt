package ai.ondevice.core.workflow

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.time.DayOfWeek
import java.time.LocalDate
import java.time.LocalTime
import java.time.ZoneId
import java.time.ZonedDateTime

/**
 * When a workflow should start itself.
 *
 * **A wall-clock time, never an instant.** "Every morning at seven" has to stay
 * seven o'clock across a flight and across a daylight-saving change; an absolute
 * epoch stored once would drift by an hour twice a year and by a continent after
 * a journey. The cost is that a timezone change has to re-arm, which is cheap
 * and is the sort of thing that is only cheap if it was decided at the start.
 */
@Serializable
data class Schedule(
    val enabled: Boolean = false,
    /** [ONCE] · [DAILY] · [WEEKLY] */
    val kind: String = ONCE,
    /** Local minutes past midnight. */
    val atMinute: Int = 8 * 60,
    /** For [WEEKLY]: ISO day numbers, Monday = 1. */
    val onDays: Set<Int> = emptySet(),
    /** For [ONCE]: an ISO date, or null for the next occurrence of the time. */
    val onDate: String? = null,

    // — what makes an unattended run acceptable —
    val requireCharging: Boolean = true,
    val minBatteryPercent: Int = 40,

    // — written by the run, read by the screens —
    val lastFiredAt: Long? = null,
    val lastSkippedAt: Long? = null,
    val lastSkipReason: String? = null,
) {
    val isRecurring: Boolean get() = kind != ONCE

    /**
     * When this should next start, or null if it never should again.
     *
     * Pure, and the whole of the calendar logic — which is why it is here rather
     * than inside the thing holding an `AlarmManager`. Everything else about
     * scheduling needs a device to exercise; this needs a clock and an argument.
     */
    fun nextOccurrence(now: ZonedDateTime): ZonedDateTime? {
        if (!enabled) return null
        val time = LocalTime.of((atMinute / 60).coerceIn(0, 23), atMinute % 60)

        return when (kind) {
            ONCE -> {
                val date = onDate?.let { runCatching { LocalDate.parse(it) }.getOrNull() }
                val at = ZonedDateTime.of(date ?: now.toLocalDate(), time, now.zone)
                // A date in the past does not fire late; a bare time rolls to
                // tomorrow, which is what "once, at eight" means when it is nine.
                when {
                    at.isAfter(now) -> at
                    date != null -> null
                    else -> at.plusDays(1)
                }
            }

            DAILY -> {
                val today = ZonedDateTime.of(now.toLocalDate(), time, now.zone)
                if (today.isAfter(now)) today else today.plusDays(1)
            }

            WEEKLY -> {
                val days = onDays.filter { it in 1..7 }.toSet()
                if (days.isEmpty()) return null
                // Seven candidates is the whole search space, and stating it that
                // way avoids the off-by-one that "days until next" invites.
                (0..7).asSequence()
                    .map { ZonedDateTime.of(now.toLocalDate().plusDays(it.toLong()), time, now.zone) }
                    .firstOrNull { it.isAfter(now) && it.dayOfWeek.value in days }
            }

            else -> null
        }
    }

    /** What the list says under the workflow's name. */
    fun describe(zone: ZoneId = ZoneId.systemDefault()): String {
        if (!enabled) return ""
        val at = String.format("%02d:%02d", atMinute / 60, atMinute % 60)
        return when (kind) {
            ONCE -> onDate?.let { "once on $it at $at" } ?: "once at $at"
            DAILY -> "every day at $at"
            WEEKLY -> {
                val names = onDays.sorted()
                    .mapNotNull { runCatching { DayOfWeek.of(it) }.getOrNull() }
                    .joinToString(", ") { it.name.take(3).lowercase().replaceFirstChar(Char::uppercase) }
                if (names.isBlank()) "weekly, no days chosen" else "$names at $at"
            }
            else -> ""
        }
    }

    fun encode(): String = JSON.encodeToString(serializer(), this)

    companion object {
        const val ONCE = "once"
        const val DAILY = "daily"
        const val WEEKLY = "weekly"

        val JSON = Json {
            ignoreUnknownKeys = true
            isLenient = true
            encodeDefaults = true
        }

        fun decode(text: String?): Schedule {
            if (text.isNullOrBlank()) return Schedule()
            return runCatching { JSON.decodeFromString(serializer(), text) }
                .getOrElse { Schedule() }
        }
    }
}
