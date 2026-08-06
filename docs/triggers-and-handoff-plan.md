# Triggers and handoff — design plan

**Status:** proposal, not implemented.

How a workflow gets started by something outside this app, and how what it
made gets handed to something outside this app.

Grounded in a read of `AndroidManifest.xml`, `MainActivity.kt`,
`ui/ExportUi.kt`, `data/AttachmentStore.kt`, `data/ExportStore.kt`,
`core/Export.kt`, `ui/vm/WorkflowViewModel.kt`, `ui/vm/WorkflowSession.kt`,
`engine/workflow/WorkflowRunner.kt`, `core/workflow/NodeKind.kt`,
`tools/*`, `ui/Navigation.kt`, and `docs/workflow-plan.md`.

---

## 0. The ask, and the word that has to be settled first

> *"Expose the workflow as activities … and trigger activities from workflow.
> Once translation is over it's automatically sent to OneNote or Gmail."*

Two directions, and the second one contains the whole difficulty:

1. **In** — another app hands this app something, and a workflow runs on it.
   Share sheet, the text-selection menu, a launcher shortcut, a deep link.
   This is straightforward, and part of it is already half-wired.
2. **Out** — a workflow finishes and the result goes somewhere else.

**"Automatically" is where this has to be honest.** On Android there are three
different things that all get called "sending it to Gmail", and they have
different costs:

| | what actually happens | needs |
|---|---|---|
| **Hand off** | Gmail's compose screen opens, filled in; a person taps Send | a tap, and the app in the foreground |
| **Deferred hand off** | a notification; tapping it opens the filled-in compose screen | a tap, whenever |
| **Delivered** | the mail is in the Sent folder; nobody touched anything | an account, a network call, an API |

**No intent on Android delivers a mail.** `ACTION_SEND` to
`com.google.android.gm` opens a composer — that is what it is for, and no flag
changes it. A third-party app cannot make Gmail send. Same for OneNote: the
share target creates a draft page in *its* UI.

So the plan is not one mechanism. It is: **intents for hand-off, and the
`Tool` node that already exists for delivery** — because delivery means an
HTTP API, this app already speaks MCP with OAuth (`tools/McpClient.kt`,
`tools/McpOAuth.kt`, callback scheme `ai.ondevice://oauth`), and a Gmail or
Microsoft Graph MCP server is a `Tool` step today with nothing new built.

**Building a third mechanism — an HTTP-with-credentials node — would be the
mistake here.** It re-implements the tool registry, the OAuth dance, the
consent surface and the settings screen, badly, for one case.

---

## 1. What already exists and does nothing

`AndroidManifest.xml:45-55` declares `MainActivity` as a share target for
`image/*` and `audio/*`. Nothing anywhere reads `EXTRA_STREAM`. Grep finds
`EXTRA_STREAM` in four files and every one of them is *writing* it.

So today: share a screenshot to this app, the app opens on whatever tab it was
last on, and the screenshot is dropped. That is a live instance of the thing
SPEC §1.2 forbids — a silent failure that looks like a working feature — and
it is the first thing to fix, before any of the rest.

---

## 2. In — a workflow as something another app can start

### 2.1 The payload, and the thing that will bite

An incoming `content://` URI arrives with `FLAG_GRANT_READ_URI_PERMISSION`.
**That grant dies with the activity that received it.** A workflow run is
minutes to the better part of an hour and outlives any screen by design — see
`WorkflowSession`'s own docstring. Read the URI lazily when the step needs it
and the run fails, in a place unrelated to the cause, with
`SecurityException: Permission Denial`.

**So the receiving activity copies in first and finishes second**, through
`AttachmentStore.copyIn(uri)` — which exists, is already used by chat and the
media screens, and puts a real file on disk. Text is simpler: `EXTRA_TEXT` is
a `CharSequence` already in the process.

```kotlin
data class TriggerPayload(
    val type: PortType,          // TEXT | IMAGE | AUDIO | FILE
    val text: String = "",
    val paths: List<String> = emptyList(),
    val fromPackage: String?,    // getReferrer(), for the consent line
    val readOnly: Boolean = true // PROCESS_TEXT: may the selection be replaced?
)
```

Mime → `PortType` is a pure function and gets a unit test. It is the only
piece of this whole feature that is testable without a device, so it is worth
having a seam for.

### 2.2 Where the payload lands: an Input node gains a source

`NodeKind.Input`'s own docstring already says the shape:

> *One node with several sources rather than one node per source: typing a
> prompt and picking a text file are the same step with a different origin.*

A shared payload is a third origin, not a fourth node type. `params` gains
`"from"`, one of `typed` (today's behaviour), `asked` (workflow-plan §2.1's
*ask when run*, still unbuilt), `shared`.

**What a workflow accepts is then derived, not declared** — the same
discipline that has `ProcessorShape` read the model's own row rather than
carry a list of names:

```kotlin
fun accepts(graph: WorkflowGraph): Set<PortType> =
    graph.nodes.filter { it.type == "input" && it.params.string("from") == "shared" }
        .map { PortType.valueOf(it.params.string("portType", "TEXT")) }
        .toSet()
```

A workflow with no `shared` Input never appears in a share sheet. Nothing to
keep in step, no second source of truth about what a graph takes.

More than one `shared` Input in a graph is an author error the editor names —
the payload is one value and there is no rule for which slot wins.

### 2.3 The four surfaces, in order of value

| surface | what it looks like | cost |
|---|---|---|
| **Share sheet** | this app in the list when anything is shared | one activity, filters already half there |
| **Direct share targets** | *specific workflows* as rows in the share sheet | `shortcuts.xml` + `ShortcutManagerCompat` |
| **Text-selection menu** | "Translate on device" in the floating toolbar over selected text, in any app | `ACTION_PROCESS_TEXT` |
| **Launcher shortcut** | long-press the app icon → a workflow | same shortcut machinery |
| **Deep link** | `ai.ondevice://workflow/<id>` for Tasker and friends | one more `<data>` element |

**`ACTION_PROCESS_TEXT` is the one that matches the user's example exactly.**
Select text anywhere, tap the workflow's name in the toolbar, and — when the
caller did *not* set `EXTRA_PROCESS_TEXT_READONLY` — return the result with
`setResult(RESULT_OK, Intent().putExtra(EXTRA_PROCESS_TEXT, translated))` and
**the selection is replaced in place**. No share sheet, no second app.

It has a hard constraint, and it is the interesting one: replace-in-place
requires the activity to still be alive when the answer arrives. A run holding
a 4 GB model for four minutes behind a dialog the user cannot leave is not a
feature.

**So the offer is conditional and the condition is structural, not a guess:**
replace-in-place is offered only when the graph is text-in-text-out, has no
diffusion or video step, and `ResidencyPlanner.plan` reports one load. Anything
else falls back to the ordinary route — run in the background, keep the result,
say where it went. The planner already computes exactly this and the run screen
already shows it.

**Direct share targets** are `ShortcutInfoCompat` with `setLongLived(true)` and
a category matched by a `<share-target>` in `res/xml/shortcuts.xml`, pushed
with `ShortcutManagerCompat.pushDynamicShortcut`. (`ChooserTargetService` is
deprecated since API 29 and does nothing on 30+; it is not an option.) The
system caps these — ask `getMaxShortcutCountPerActivity()`, typically 15 —
so the list is the most recently run workflows, republished on
`db.workflows().touch(...)`, which is already called at the end of every run.

### 2.4 One activity, not filters on `MainActivity`

```xml
<activity android:name=".workflow.TriggerActivity"
          android:exported="true"
          android:taskAffinity=""
          android:excludeFromRecents="true"
          android:theme="@style/Theme.OnDeviceAI.Transparent" />
```

Three reasons not to hang this off `MainActivity`:

1. `MainActivity` is `launchMode="singleTask"`, deliberately — commit
   `af8a01a` and its docstring are about a notification tap reusing the
   instance rather than rebuilding it and killing a generation. A share
   arriving into that same task while a run is in flight is the same hazard
   from a different direction.
2. `PROCESS_TEXT` must return a result to its caller, which means the
   receiving activity is `startActivityForResult`-shaped and must not be the
   app's main task.
3. `taskAffinity=""` keeps a share from leaving a second card in recents —
   the reasoning already written on `OAuthCallbackActivity`.

`TriggerActivity` is a trampoline: normalise the intent, copy the payload in,
pick the workflow, get consent, start the run, finish. Anything long-lived
belongs to `WorkflowSession` and `InferenceService`, which already outlive
every screen.

The manifest's existing `image/*` and `audio/*` filters move here and gain
`text/plain` and `*/*`, plus `SEND_MULTIPLE` (which maps to a `LIST` payload —
the port type already exists), `PROCESS_TEXT`, and `VIEW` on
`ai.ondevice://workflow`. Same scheme as the OAuth callback, different host.

### 2.5 Which workflow, and consent

Zero matching workflows → say so, in the `MissingComponent` voice, with the
remedy (*"no workflow takes a picture; add an Input marked 'from another app'"*).
One → run it. Several → a picker, most recent first.

**An exported activity that starts a run is an exported activity that spends
forty minutes of somebody's battery on request.** With a `Send` step in the
graph it is also an exfiltration path. So:

- **Off by default.** A workflow is reachable from outside only once its
  author turns on *"Let other apps start this"* in the editor.
- The trampoline shows a confirmation naming the caller (`getReferrer()`; note
  `getCallingPackage()` is null unless the caller used
  `startActivityForResult`), the workflow, what the plan will load, and
  **every `Send` step's target** — before anything runs.
- *"Don't ask again for this app"* is per calling package, stored in
  `AppPrefs`, revocable in Settings beside the tool providers.

### 2.6 One run at a time

`WorkflowRunner` holds one engine's worth of weights and `WorkflowSession` has
one `runJob`. A trigger arriving mid-run is refused with what is running and
how far along it is — not silently queued, because a translation that arrives
forty minutes late has already been done by hand.

---

## 3. Out — handing a result to another app

### 3.1 One new node kind

```kotlin
data object Send : NodeKind {
    override val type = "send"
    override val title = "Send it somewhere"
    override val family = NodeFamily.SINK
    override val blurb =
        "Hand this to another app — a mail, a note, the clipboard, or a folder."
    override fun slots(context: NodeContext) =
        listOf(SlotSpec("value", PortType.FILE, "What to send"),
               SlotSpec("subject", PortType.TEXT, "Subject or title", required = false))
}
```

Targets, as one node with several destinations — the `Input` pattern again:

| target | mechanism | already exists |
|---|---|---|
| a chooser | `ExportUi.shareExport` | yes, unchanged |
| a named app | the same, `setPackage(...)` | needs `<queries>`, see §3.3 |
| the clipboard | `ClipboardManager` | no, ~10 lines |
| a folder | `ExportStore` + SAF tree | yes; the tree uri must be picked once, ahead of the run |

Everything here goes through `Export` and the existing FileProvider staging.
`ExportUi.kt`'s comment — *"the two ways an artifact leaves this app, in one
place"* — stays true; this makes it three, in the same place.

The old-build story is already handled: a graph with a `send` step opened by a
build that predates it decodes to `NodeKind.Unknown` and refuses with a named
reason. Nothing to design.

### 3.2 The wall, and the way around it

**Android does not let a backgrounded app start an activity.** A workflow that
finishes while the user is in another app cannot pop a share sheet — the start
is dropped, silently, and Android 14 tightened this further with explicit
background-activity-launch opt-in on `PendingIntent`.

Which means a `Send` step has two paths and must have both:

- **App in the foreground** → launch the chooser now. This is the case that
  looks like what was asked for.
- **App in the background** → **stage the export, post a notification, and
  keep the handoff.** Tapping it delivers. The notification's `PendingIntent`
  must be `getActivity` straight to a small `HandoffActivity` — since Android
  12, a notification may not trampoline through a receiver or a service.

Unsent handoffs are listed on the run screen and survive the run, so a result
is never lost to a missed notification.

**Say this in the editor, at the point of choosing**, not in a release note: a
`Send` step's card reads *"If the app isn't open when this runs, it waits in a
notification."* The alternative — the run appearing to succeed while nothing
arrives — is exactly SPEC §1.2's silent failure.

### 3.3 Package visibility, which is easy to miss

Targeting a named app means `packageManager.resolveActivity(...)`, and since
Android 11 that returns null for anything not declared visible.
`QUERY_ALL_PACKAGES` is a restricted Play permission and unjustifiable here.
So the manifest gains:

```xml
<queries>
    <intent>
        <action android:name="android.intent.action.SEND" />
        <data android:mimeType="*/*" />
    </intent>
</queries>
```

Which makes every share-capable app visible and enumerable — so the target
picker is **the list of apps that can receive this type**, asked of the system,
with no package names in app source. §1.3, applied to a different registry.

### 3.4 What "sent to Gmail" gets you, precisely

`ACTION_SEND` with `setPackage("com.google.android.gm")`, `EXTRA_EMAIL`,
`EXTRA_SUBJECT`, `EXTRA_TEXT` and an attachment opens Gmail's composer with
everything filled. One tap sends it. That is the ceiling for intents.

OneNote (`com.microsoft.office.onenote`) is the same story: its share target
opens a page picker.

`ACTION_SENDTO` with a `mailto:` URI is the same ceiling with fewer options
and no attachment. Not worth a separate target.

### 3.5 Delivered, not handed off

For the case where nobody taps anything — the literal reading of *"once
translation is over it's automatically sent"* — the mechanism is
`NodeKind.Tool`, which is **already built and already runs in the graph**
(`WorkflowRunner.kt:554`). Point it at a Gmail or Microsoft Graph MCP server
connected in Settings → Tools, bind `arguments` to the translation step, done.

Two things this plan owes that path, and neither is a new node:

1. **A `Tool` step's arguments are JSON**, and building JSON by hand in a
   template is miserable. The `Script` node runs QuickJS
   (`QuickJsBridge.eval`, with `steps` in scope) — so *"build the request
   body"* is a one-line recipe, and it should ship as a starter template
   rather than being rediscovered.
2. **SPEC §13 says no account.** MCP OAuth already crossed that line
   deliberately and per-provider; this does not move the line, but the
   consent surface should say plainly that a workflow with a delivering Tool
   step sends content to a named service, and that is where the offline
   guarantee ends.

---

## 4. Data model

**One additive column and no new tables.**

```
ALTER TABLE workflows ADD COLUMN triggerJson TEXT NOT NULL DEFAULT '{}'
```

Room v15, the additive shape of `MIGRATION_9_10`.

`triggerJson` holds **consent, not capability**: `{ "fromOtherApps": false,
"inShareSheet": false, "shortcut": false, "trustedCallers": [] }`. What a
workflow *accepts* is derived from the graph (§2.2) — deriving it means it
cannot go stale; storing consent means it cannot be granted by an edit.

A run started by a trigger records the calling package in the existing
`WorkflowRunEntity.nodeStatesJson` bag, so the history can answer "what started
this".

---

## 5. Execution — the refactor both entry points need

`WorkflowViewModel.run()` (`ui/vm/WorkflowViewModel.kt:256`) is where a run
lives today, and its first two lines are

```kotlin
if (_state.value.running) return
val workflow = _state.value.editing ?: return
```

— a run can only start for the graph currently open in the editor, from a
`ViewModel` scoped to a screen that a trigger has not visited.

**Move the body to `@Singleton class WorkflowLauncher`** taking
`(workflowId, payload: TriggerPayload?)`, keeping the wake-lock bracket, the
recorder, the run row and the reporter exactly as they are. `WorkflowViewModel.run()`
becomes a call into it, and `TriggerActivity` becomes a second caller. Nothing
about the runner, the session or the reporter changes.

This is a prerequisite for §2 and worth doing on its own: the editor being the
only way to start a run is why there is no rerun-from-history either.

---

## 6. UI

- **Editor, workflow-level:** a *Triggers* card — *Let other apps start this*,
  *Show in the share sheet*, *Add to the app icon*, each off by default, each
  disabled with a reason when the graph has no `shared` Input.
- **Editor, Input card:** the source row gains *From another app*, beside
  *Typed here* and *Ask when it runs*.
- **Editor, Send card:** target picker (chooser · a named app · clipboard ·
  folder), and the background line from §3.2.
- **Trigger confirmation:** an `NBottomSheet` over the transparent trampoline —
  caller, workflow, the residency plan's cost line, every `Send` target, and
  *Don't ask again for this app*.
- **Run screen:** a *Waiting to be sent* block listing unsent handoffs. Reuses
  the existing card grammar; nothing new drawn.
- **Settings → Tools:** the per-package trust list, revocable, beside the tool
  providers it resembles.

No new navigation route: the trampoline is its own activity, and everything
else is on screens that exist.

---

## 7. Phasing

**T1 — the dead filters stop being dead.**
`TriggerActivity`, payload normalisation and copy-in, `from = "shared"` on
Input, the `WorkflowLauncher` refactor, workflow picker, consent sheet, the
`fromOtherApps` flag. Share a screenshot and a workflow runs on it. *The
share-sheet-to-nothing bug is fixed here, and everything else builds on the
launcher refactor.*

**T2 — results leave.**
The `Send` node: chooser and clipboard targets, foreground launch, notification
fallback, unsent-handoff list, `<queries>`, named-app targeting. The user's
example works end to end with one tap.

**T3 — the sharp edges.**
`ACTION_PROCESS_TEXT` with replace-in-place gated on the residency plan; direct
share targets and launcher shortcuts; folder target via SAF.

**T4 — delivered.**
`ai.ondevice://workflow/<id>` for automation apps; starter templates for
Gmail-via-MCP and Graph-via-MCP; the JSON-building `Script` recipe.

**Not in scope**, and consistent with workflow-plan §6's deferred list:
scheduled runs, `BOOT_COMPLETED` triggers, watching a folder, an assistant
role, a quick-settings tile.

---

## 8. Decisions needed, and where this is uncertain

**For the author:**

1. **Hand off or deliver** — is the target case a tap in Gmail's composer
   (intents, T2) or a mail actually sent (MCP, T4)? Both are in the plan; the
   order depends on the answer, and only the second needs an account.
2. **Replace-in-place** — worth the gating rule in §2.3, or is
   *run in the background and keep the result* enough? The rule is the only
   place in this plan where the UI depends on the residency planner.
3. **More than one `shared` Input** — editor error, or first-one-wins? Erroring
   is consistent with §1.2.
4. **Trust per calling package** — one blanket *"apps may start workflows"*
   switch, or a remembered list? The list is more code and much easier to
   reason about a year later.
5. **Does a triggered run reuse the trigger's payload on a rerun**, or ask
   again? A rerun with a stale screenshot is a confusing thing to debug.

**Uncertain:**

6. **`getReferrer()` is spoofable** by a caller that sets it. It is fine for a
   label and must not be the thing a trust decision is keyed on — that should
   key on the package the system reports for `startActivityForResult` callers,
   and fall back to asking every time when there is no such package.
7. **Direct-share slots are scarce** and the system's recency ranking is opaque;
   the most-recently-run heuristic is a guess that wants a look on a device.
8. **A `content://` from a cloud provider** (Drive, OneDrive) can be slow or
   fail on `copyIn`, and the trampoline is holding a transparent screen while
   it happens. Needs a spinner and a timeout with a real message.
9. **Notification permission is optional** (`MainActivity` asks once and does
   not insist). Without it the deferred-handoff notification never appears —
   so the run screen's unsent list is not a nicety, it is the fallback.
10. **`*/*` in the share filter** puts this app in every share sheet on the
    device, including for types no workflow accepts. Narrower filters mean a
    manifest that lists types, which cannot follow what the graphs actually
    take. Probably worth accepting `*/*` and refusing well.
11. **A `Send` step inside a loop** could fire thirty-two notifications. Needs
    a cap, or coalescing into one handoff of many files.
