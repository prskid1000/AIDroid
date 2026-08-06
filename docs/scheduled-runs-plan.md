# Scheduled runs — design plan

**Status:** proposal, not implemented.

A workflow that starts itself: once at a chosen time, or again on a cadence.

Grounded in a read of `engine/workflow/WorkflowLauncher.kt`,
`engine/InferenceService.kt`, `data/download/BootSweepReceiver.kt`,
`engine/workflow/ResidencyPlanner.kt`, `data/hf/DeviceCapabilities.kt`,
`data/db/WorkflowEntities.kt`, `AndroidManifest.xml`, `app/build.gradle.kts`,
SPEC §8.3–8.4 and §17.2, and `docs/triggers-and-handoff-plan.md`.

---

## 0. What is already here, and what is missing

**Missing entirely.** No `AlarmManager`, no `WorkManager`, no `JobScheduler`
anywhere in app source, and no scheduling dependency. The one boot receiver
sweeps orphaned model downloads. Both prior plans deferred this deliberately —
`workflow-plan.md` §6 and `triggers-and-handoff-plan.md` §7.

**But the expensive half already exists**, because the trigger work needed the
same thing: a way to start a run with no editor, no screen, and nobody present.

```kotlin
launcher.launch(workflowId, payload = null)   // the whole call
```

`WorkflowLauncher` owns the wake lock, the run snapshot, the reporter, the
residency plan and the refusal when something is already running.
`InferenceService` is already a `specialUse` foreground service that keeps the
process alive across a run, and the share path proved the unattended shape end
to end. So this plan is not about running a graph. **It is about being allowed
to wake up, and about deciding whether waking up was a good idea.**

---

## 1. The platform constraint that picks the mechanism

A run needs a foreground service. Android 12 and later forbid starting one from
the background, with a short list of exemptions — and the list decides this
design:

> **Exact alarms aren't affected by foreground service launch restrictions.**

| mechanism | wakes at the right time | may start our FGS | permission |
|---|---|---|---|
| `setExactAndAllowWhileIdle` | yes, through Doze | **yes — exempt** | `SCHEDULE_EXACT_ALARM`, denied by default on 14+ |
| `setAndAllowWhileIdle` (inexact) | within a window | **no** | none |
| `WorkManager` periodic | 15-minute floor, batched | only via *its* service | none |

**WorkManager is the obvious answer and it is the wrong one here.** Its
`setForeground` runs the work inside WorkManager's own foreground service, which
since Android 14 must declare a service type — so `specialUse` and its
justification would have to be merged onto a service this app does not own,
beside the `InferenceService` that already does exactly this job. And Android 15
puts a shared six-hour ceiling on foreground services with `dataSync` and
`mediaProcessing` bounded hardest. Two foreground services for one run, to avoid
one permission prompt, is a bad trade.

**So: an exact alarm, a `BroadcastReceiver`, and `launcher.launch(id)`.** No new
service, no new dependency, and the receiver runs in an exemption window wide
enough to start the service we already have.

### 1.1 The permission, and the honest degradation

`SCHEDULE_EXACT_ALARM` is **denied by default** for apps targeting Android 13 or
above, and this app targets 35. `USE_EXACT_ALARM` — which is granted
automatically — is restricted by Play to alarm clocks and calendars, and this
app would not qualify. So it must be asked for, through
`ACTION_REQUEST_SCHEDULE_EXACT_ALARM`, and `canScheduleExactAlarms()` says
whether it was given.

**What happens when it is refused is the interesting part, and it should not be
a dead feature.** The answer already exists in this codebase — it is what a Send
step does when the app is in the background:

- **Granted** → the run starts by itself, at the time asked for.
- **Refused** → an inexact alarm still fires, and posts a notification: *"Your
  scheduled run is due. Tap to start it."* One tap, and the ordinary foreground
  path takes over.

That is the same shape as the parked hand-off, for the same reason: the platform
will not let this happen unattended, so say so and offer the tap rather than
failing silently. The editor must say which of the two a schedule will get,
**before** it is saved.

### 1.2 One more Play-policy fork, and the flavours already exist

Play's guidance is that `SCHEDULE_EXACT_ALARM` is for apps whose core function
needs exact timing. A workflow scheduler is arguable; it is not obviously
covered. This repo already splits on precisely this kind of question — SPEC
§17.2 has `sideload` self-updating runtimes and `play` degrading to a store link
because Play forbids the former.

**Scheduling maps onto that split with nothing new to invent:** `sideload` asks
for the exact-alarm permission; `play` ships the tap-to-start path only, and
says so. If the permission later proves acceptable for the store build, one
`buildConfigField` changes.

---

## 2. Data model

A schedule is genuinely new state — unlike the trigger work, where what a
workflow accepted was *derived* from its graph and nothing needed storing. So it
gets a column.

```
ALTER TABLE workflows ADD COLUMN scheduleJson TEXT NOT NULL DEFAULT '{}'
```

Room **v15**, additive only, the shape of `MIGRATION_9_10`.

```kotlin
@Serializable
data class Schedule(
    val enabled: Boolean = false,
    /** "once" · "daily" · "weekly" */
    val kind: String = "once",
    /** Local minutes past midnight. Wall-clock, not an instant — see below. */
    val atMinute: Int = 8 * 60,
    /** For "weekly": 1 = Monday, ISO order. */
    val onDays: Set<Int> = emptySet(),
    /** For "once": the day it should fire. */
    val onDate: String? = null,
    val requireCharging: Boolean = true,
    val minBatteryPercent: Int = 40,
    /** Written by the runner, read by the editor. */
    val lastFiredAt: Long? = null,
    val lastSkippedAt: Long? = null,
    val lastSkipReason: String? = null,
)
```

**Stored as a wall-clock time, not as an instant.** "Every morning at seven" must
stay seven o'clock across a flight and across a daylight-saving change; an
absolute epoch would drift by an hour twice a year and by a continent once. The
consequence is that `ACTION_TIMEZONE_CHANGED` and `ACTION_TIME_CHANGED` have to
re-arm — and both are on the same FGS exemption list, so that costs nothing.

**A "once" schedule disarms itself** by setting `enabled = false` when it fires,
in the same write that records `lastFiredAt`. A recurring one computes the next
occurrence and re-arms.

---

## 3. The guards, which are the actual design work

Waking up is easy. **Deciding that a forty-five-minute GPU run is a good idea at
three in the morning, unattended, is the part worth thinking about**, and it is
where this device differs from a laptop. A hot phone on a bedside table with
nobody watching is a worse outcome than a run that did not happen.

Checked in order, before anything loads:

| gate | source | why |
|---|---|---|
| nothing already running | `launcher.busy` | one engine holds the weights |
| charging, if required | `BatteryManager` | the default, and it should be |
| battery above the floor | `BatteryManager` | 40% by default |
| thermal headroom | SPEC §8.3 policy | a throttled run is slower *and* hotter |
| plan fits in RAM | `ResidencyPlanner.plan` | it already computes this |
| storage for the outputs | `ModelStorage` | a run that fills the disk is worse than one that waited |

**A skipped slot is recorded and said, never silently dropped.** `lastSkipReason`
is written and the workflow row shows it — *"Skipped at 07:00: battery was 22%
and this workflow is set to need 40%."* SPEC §1.2 applies as much to a run that
did not happen as to one that failed, and a schedule that quietly does nothing
is indistinguishable from a broken one.

**A missed slot is not made up by default.** If the phone was off at seven, the
run does not begin at half past nine when it comes back — it waits for tomorrow,
and says it missed one. Catch-up is a per-schedule opt-in, because the reasonable
answer differs: a nightly summary of yesterday is worth running late, and a
"good morning" briefing is not.

---

## 4. The pieces

```
core/workflow/Schedule.kt        the record above, plus nextOccurrence()
engine/workflow/Scheduler.kt     arm / cancel / rearmAll, and the guard checks
workflow/ScheduleReceiver.kt     the alarm lands here, calls launcher.launch
```

- **`Scheduler.arm(workflow)`** computes the next occurrence and calls
  `setExactAndAllowWhileIdle` with a `PendingIntent` keyed on the workflow id, so
  re-arming replaces rather than stacks.
- **`ScheduleReceiver`** runs the gates, then either `launcher.launch(id)` or
  records the skip; then re-arms for the next occurrence. `goAsync()` for the
  database read, as `BootSweepReceiver` already does.
- **Re-arming on boot** goes into the *existing* `BootSweepReceiver`, which
  already holds `RECEIVE_BOOT_COMPLETED` and already sweeps on start. Alarms do
  not survive a reboot; this is two lines beside something that already runs.
- **`nextOccurrence` is pure** — a wall-clock time, a day set and a "now", into
  an instant. It is the whole of the calendar logic and the only part testable
  without a device, so it is where the tests go: DST forward and back, a weekly
  schedule on the day it fires, a "once" already in the past, midnight.

---

## 5. UI

- **Editor, a *Schedule* card** beside Triggers: off · once · daily · weekly, a
  time, and the day chips for weekly. Under it, in plain words, which of the two
  worlds it will get — *"This will start on its own"* or *"This will notify you
  and start when you tap"*, decided by `canScheduleExactAlarms()` and the
  flavour, with the permission prompt one tap away when it can help.
- **Conditions** as two rows: *Only while charging* (on by default) and a battery
  floor. Not a general condition language — these two cover what the hardware
  actually cares about.
- **The workflow list** shows the next run under the name — *"next: tomorrow
  07:00"* — and the last skip when there is one. This is the only surface that
  answers "is this thing working", and without it a schedule is unfalsifiable.
- **No new screen and no new route.**

---

## 6. Phasing

**S1 — one workflow, one time, by itself.**
`Schedule` record, Room v15, `Scheduler`, `ScheduleReceiver`, boot re-arm, the
permission ask with the notification fallback, and the guards. Kind is `once`
only. This proves the whole mechanism, including the part that can refuse.

**S2 — cadence and honesty.**
`daily` and `weekly`, `nextOccurrence` with its DST tests, next-run and
last-skip on the list, the timezone and time-change re-arm.

**S3 — the rest.**
Catch-up opt-in, Wi-Fi-only as a condition, a per-run history of fired and
skipped slots, and a quick-settings tile if it turns out to be wanted.

**Not in scope:** a cron expression, sub-hourly cadence (a run is longer than the
interval would be), and chaining one workflow's schedule to another's completion
— that is a graph, and the graph already exists.

---

## 7. Decisions needed, and where this is uncertain

**For the author:**

1. **Ask for `SCHEDULE_EXACT_ALARM` in the `play` flavour, or ship tap-to-start
   there?** §1.2 proposes the flavour split. It is a policy judgement, not a
   technical one, and it is the only decision that changes what S1 builds.
2. **Is *only while charging* the default?** It makes the feature quieter and
   less surprising, and it also makes it useless to anyone who does not charge
   overnight on a schedule.
3. **Battery floor of 40%** — a guess, and it should be argued about once. A Wan
   clip is three quarters of an hour at full tilt.
4. **Does a skipped slot notify, or only get recorded?** Notifying every morning
   that nothing happened is its own problem.
5. **Once-only schedules after firing** — disarm and keep the time visible as
   history, or clear the row? Keeping it means the editor shows a schedule that
   will not fire again, which needs saying carefully.

**Uncertain:**

6. **`setExactAndAllowWhileIdle` is rate-limited in Doze** to roughly one firing
   per app per nine minutes. Irrelevant at these cadences, and worth writing down
   before somebody proposes a five-minute interval.
7. **OEM battery managers.** This device is a Motorola and its `moto_freezer`
   was observed freezing the app process during the trigger testing. Exact alarms
   are meant to survive that; whether every OEM honours it is a device-matrix
   question, and the notification fallback is the answer when they do not.
8. **The permission can be revoked at any time**, and the system sends
   `ACTION_SCHEDULE_EXACT_ALARM_PERMISSION_STATE_CHANGED` when it changes. A
   schedule armed exactly and then de-permissioned must fall back rather than go
   quiet.
9. **A run that outlives its next slot.** A forty-five-minute run starting at
   07:00 daily is fine; one that takes longer than its own interval is refused by
   `launcher.busy` and recorded as a skip, which is correct but will read as a
   bug the first time it happens.
10. **Android 15's six-hour foreground-service ceiling** is shared across an
    app's services. No run approaches it today, and a queue of scheduled runs
    one day might.
