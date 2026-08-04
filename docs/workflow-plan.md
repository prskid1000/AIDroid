# Workflow — design plan

**Status:** proposal, not implemented.

Grounded in a read of `Navigation.kt`, `ui/vm/*`, `engine/*`, `speech/*`,
`data/db/*`, `params/*`, `core/Attachments.kt`, `tools/*`, `ui/components/*`.

---

## 0. The note, and where it is ambiguous

Five component types: Input, Output, Processor, Loop, Script. Three readings
need settling first, because they change everything downstream.

1. **"Input → … OR another model's output"** conflates a value from outside the
   graph with an edge between nodes. If Input nodes were also required in front
   of every consuming node, every graph would double in length. **Read as an
   edge.**
2. **"Loop wrapper component"** could be a container that nests other nodes, or
   a `for-each` over a collection. Nesting on a phone means a tree.
   **Proposed: a bracket pair over a contiguous span** — see §2.4.
3. **"Script component that can transform or do something"** is the
   highest-variance item. There is no scripting surface in this repo to reuse
   (§2.5), and "transform" and "do something" are very different asks — the
   second is a native-build decision.

---

## 1. Data model

### 1.1 Where the graph lives

Two tables, and the graph itself as one JSON document.

```
workflows(
  id TEXT PK, name TEXT, notes TEXT,
  graphJson TEXT,              -- the whole graph
  createdAt INTEGER, updatedAt INTEGER, lastRunAt INTEGER NULL)

workflow_runs(
  id TEXT PK, workflowId TEXT,
  graphJson TEXT,              -- a snapshot, not a reference
  state TEXT,                  -- RUNNING | DONE | FAILED | CANCELLED
  startedAt INTEGER, finishedAt INTEGER NULL,
  error TEXT NULL, errorHint TEXT NULL,
  nodeStatesJson TEXT,
  index on workflowId, index on startedAt)
```

**One JSON blob, not normalised node/edge tables.** Every structured thing in
this schema is already a JSON column — `paramOverridesJson`,
`companionPathsJson`, `filesJson`, `segmentsJson`, `traceJson`. SPEC §11 gives
the reason for parameters: *"A key unknown to the current manifest is preserved
inert and re-activates if a later runtime supports it."* That argument is
stronger for a graph: normalising means a Room migration every time a node type
gains a field, and the schema is already at v13 with twelve migrations.

**The run snapshots the graph** so editing a workflow after a run does not
rewrite what that run did — the same reasoning that puts `generationParamsJson`
on every message and a `parameters` text chunk in every PNG.

### 1.2 The migration

v14, additive only: two `CREATE TABLE IF NOT EXISTS` plus indices, the shape of
`MIGRATION_9_10`. Nothing existing is touched.

### 1.3 Node schema — typed in Kotlin, untyped on the wire

```kotlin
@Serializable
data class NodeRecord(
    val id: String,
    /** "input", "output", "processor", "repeat_start", "repeat_end", "script". */
    val type: String,
    val label: String = "",
    /** Slot name → a producer reference "<nodeId>:<outputName>", or absent. */
    val slots: Map<String, String> = emptyMap(),
    val params: JsonObject = JsonObject(emptyMap()),
)
```

**Sealed classes on the wire would be a mistake.** kotlinx polymorphic decoding
throws on an unrecognised discriminator, so a graph saved by a build that knows
`for_each` would fail to open on one that does not — losing the whole workflow,
not one step. Instead a flat record with a `String` type, lenient decoding, and
a Kotlin-side sealed `NodeKind` derived from it with an `Unknown(raw)` arm —
the stance `Converters` already takes for every enum.

An `Unknown` node renders as a card naming the build that wrote it and refuses
to run, in the `MissingComponent` voice of *what, because, remedy*.

### 1.4 Node params reuse `SparseParams` and the manifest

A Processor's `params` is a `SparseParams`, resolved with the same two-layer
merge the Image tab already does:

```kotlin
val params = SparseParams.parse(model.paramOverridesJson)
    .overlaidWith(node.params.toSparseParams())
```

Which keys the editor offers comes from `ParamRepository.applicableKeys(...)`
and their starting values from `defaultsFor(...)`, both gated by `appliesTo`,
rendered by `params/ParamRenderer.kt`. **No new parameter declarations and no
model names** — the node knows a `modelId`, asks the row for its `modality` and
`architecture`, and the manifest answers the rest.

LoRAs and other add-ons come along as `List<ModelAttachment>`, stored per node
under each role's own `paramKey`, with `WeightedPaths` for stacking roles.

### 1.5 Run artifacts

`storage.root()/workflows/<runId>/<nodeId>.<ext>`, beside `galleryDir()` and
friends. Intermediates get no Room rows. Only what an Output node marks becomes
a `GeneratedImageEntity` / `GeneratedClipEntity` / `SynthesisEntity` /
`TranscriptEntity` and reaches the Library.

Resource traces need no new table: each model-running node records a
`PredictionRunEntity` through the existing
`db.predictionRuns().record(kind, artifactId, modelId, startedAt, trace, stats)`.

---

## 2. The five components

### 2.1 Input

Brings a value in from outside. No inputs; one output of the configured type.

- `TEXT` — an inline `NTextArea`, or a picked text file through
  `AttachmentStore.extractText`, which already refuses PDFs honestly. One node
  with two sources, not two node types.
- `IMAGE` / `AUDIO` / `FILE` — `AttachmentStore.copyIn(uri)`, picked with the
  same `PickVisualMedia` launcher the video screen uses, rendered with
  `PickedImageField`.

**One addition beyond the note, and the thing that makes this reusable rather
than a macro:** an Input can be marked *ask when run*. Without it, "summarise a
recording" is a workflow you edit before every use.

### 2.2 Output

Names a value as a keeper and persists it. One input, no outputs.

Maps onto the persistence half of each existing generate path. The *viewers*
already exist — `AsyncImage`, `NAudioPlayer`, the `VideoScreen` frame stepper,
plain `Text` — and belong to the run screen, which shows every node's result
whether marked or not.

### 2.3 Processor

One node type, five runtimes. The node asks the chosen model row what it is.

| model row | runtime | in → out |
|---|---|---|
| `TEXT` / `VISION` | llama.cpp | text [+ images] → text |
| `DIFFUSION`, not video | stable-diffusion.cpp | text [+ image] → image |
| `DIFFUSION`, video | stable-diffusion.cpp | text [+ frames] → clip |
| `SPEECH_TO_TEXT` | whisper.cpp | audio → text |
| `TEXT_TO_SPEECH` | kokoro / omnivoice | text → audio |
| *(no model)* + `UPSCALER` | stable-diffusion.cpp | image → image |

The slot list is derived from the modality, never typed by hand — the
discipline that makes `appliesTo.output` ask `DiffusionFamily.isVideo` rather
than carry a list of video architectures.

**Upscale deserves its own preset** because `DiffusionEngine.upscale`
unconditionally unloads the denoiser. A generate → upscale → generate graph
costs three loads, and the planner in §4 must say so out loud.

### 2.4 Loop — a bracket pair, not a container

`Repeat start` / `Repeat end` enclosing a contiguous span.

A container makes the step list a tree, and a tree on 360 dp means either
indentation that eats the content or a second navigation level. A bracket pair
renders as a left accent rule down the spanned cards, and makes a loop crossing
another loop's boundary structurally impossible.

Config on `Repeat start`: **a required maximum iteration count** with an
app-side ceiling; an optional feedback binding (slot X of the first node takes,
from iteration 2, the output of the last); an optional stop condition that may
only *shorten* the loop.

**A condition-only loop is not acceptable on this hardware.** One diffusion
iteration is minutes and a Wan clip is three quarters of an hour, against
~15.6 GB shared with Android. The count is mandatory; the condition is a
courtesy.

### 2.5 Script — the honest answer

**What exists today: nothing.** `ToolProvider` is name + JSON-Schema-as-String
+ `suspend fun call(...)` returning text. No JS engine, no expression
evaluator, no `ScriptEngine`, no WebView, no scripting dependency. The only
evaluator in the repo is `internal object Arithmetic` — ~110 lines of
recursive descent over `+ - * / % ^`, returning `Double`, with no variables,
strings or comparisons.

`Shell.kt` does run a real shell, but it is toybox only, W^X blocks executing
written files, and it is **cwd-confined rather than containment-checked** —
`Workspace.resolve` is applied to `FileTools` and not to shell commands.

| | what | verdict |
|---|---|---|
| **A** | Template + small expression language in Kotlin | **Ship this** |
| **B** | Route Script to the existing shell provider | **No** |
| **C** | QuickJS via JNI | Only on evidence |
| **D** | WebView `evaluateJavascript` | No |
| **E** | Rhino or similar | No |

**A**, concretely: interpolation over node outputs (`{{2.text}}`), plus
`trim / join / split / replace / match / slice / lower / upper / length`,
number formatting, comparisons returning a boolean, and the arithmetic that
already exists. Deterministic, cancellable, testable without a device.

**Why not B.** The shell exists as a *chat tool*, where a model asks for it one
call at a time behind a 30 s cap and a visible bubble. A graph node —
unattended, inside a loop, reaching outside `Workspace` — is a different risk
with a different consent story.

**C fits this codebase better than any Java-side engine.** Four upstream repos
are already pinned by commit in `native/VERSIONS` and built through
`app/src/main/cpp/CMakeLists.txt` with four JNI shims. QuickJS is ~200 KB of
dependency-free C with a real interrupt handler and a memory limit — the only
option giving a bounded sandbox *and* a real language. **But it is a
native-build decision and belongs in §7, not discovered in M4.**

---

## 3. Edge types

### 3.1 The lattice

| port | carried as |
|---|---|
| `TEXT` | `String` |
| `IMAGE` | a path to a PNG |
| `AUDIO` | a path to a WAV |
| `CLIP` | directory + frames + fps + optional WAV |
| `FILE` | a path + mime |

**Values on edges are paths, never pixels or samples.** Four nodes holding four
ARGB bitmaps is `4 × W × H × 4` bytes in a process already holding ten
gigabytes. The codebase made this call once already: `DiffusionClip` carries
paths, *"a five-second 480p clip is ~147 MB of raw RGB"*.

One subtyping rule: `IMAGE`, `AUDIO`, `CLIP` all satisfy `FILE`; nothing
satisfies them. One explicit opt-in coercion, `FILE → TEXT` via
`extractText`, shown as a visible adapter on the slot.

A transcript Processor has two named outputs — `text` and `subtitles` — rather
than a sixth port type.

### 3.2 Invalid connections prevented by construction

A slot's source picker lists only upstream nodes whose output type satisfies
it. An invalid edge is unrepresentable. Where nothing qualifies, the picker
says so in the `MissingComponent` voice.

**Cycles are also prevented by construction:** a slot may only reference a node
*earlier in the list*. Topological order is list order. No sort, no cycle
detector, no error state. This is the largest simplification the phone-shaped
UI buys.

### 3.3 The second gate, which is not a type

Before any node runs, `ComponentCheck` over every Processor — the check the
Image and Video tabs already do — so a missing T5, an unarmed VAE or a model at
9% is reported *before* the run rather than four minutes into a load.

---

## 4. Execution

### 4.1 The session

`@Singleton class WorkflowSession` with `state`, `scope`, `runJob`, and
`claimObservers()` — verbatim the shape of `ImageSession` / `VideoSession` /
`ChatSession` / `VoiceSession`.

**It must be injected into `InferenceService` and added to the `combine`**, or
a running workflow will not appear in the shade and the service will
`stopSelf()` mid-run.

### 4.2 The wake lock

Held once for the whole graph, in the bracket form:

```kotlin
runJob = session.scope.launch {
    InferenceService.holdingWakeLock(context) { runGraph(plan) }
}
```

Nesting is safe because `running` is a reference-counted counter.

### 4.3 Model loading — the constraint that shapes everything

**`DiffusionEngine.load` unloads first, inside a mutex held across the whole
call. `EngineManager.load` warm-swaps llama. But there is no cross-engine
coordinator** — llama and sd.cpp can both be resident, and nothing prevents it,
because until now no single flow ever crossed runtimes. A workflow is the first
thing that does, and that is where the OOM kill lives.

**A `ResidencyPlanner` computing a plan before the run, not a policy during it:**

1. **Enumerate** `(runtimeId, modelId, loadKey)` per node, where `loadKey`
   hashes the load-time settings only — already answered by
   `LOAD_TIME_ROLES` / `LOAD_SETTING_KEYS` for diffusion and
   `ParamSpec.requiresReload` for llama.
2. **Coalesce** adjacent nodes on the same triple into one load.
3. **Reorder only where provably safe, and only visibly** — show the step order
   and load count before the run: *"3 model loads, ~18 GB read from disk,
   longest single residency 10.2 GB."* A schedule the user cannot see is one
   they cannot debug when it costs 45 minutes.
4. **Unload explicitly when crossing runtimes**, with a reason —
   `DiffusionEngine.unload(because)` already takes one. **This line does not
   exist anywhere today and is the most important one in the runner.**
5. **Refuse impossible plans up front** against `availableRamBytes`, with the
   arithmetic, in the `describeFailure` voice.

### 4.4 Per-node reporting

`NodeRunState { WAITING, LOADING, RUNNING, DONE, FAILED, SKIPPED, CANCELLED }`,
plus per active node exactly the field names `ImageState` and `VideoState`
already use. **Mirroring those names is not cosmetic** —
`InferenceService.buildNotification` reads the tuple
`(what, model, phase, step, steps, rate)`, so matching them means the
notification needs one new `when` arm and nothing else.

### 4.5 Cancellation

Two-step, as everywhere. The runner must hold the live engine's `cancel()`,
because cancelling the Job does not reach a blocking JNI call. `cancelling` is
a real state: a cancel during a prompt encode can only land at the phase
boundary, and a seven-step workflow makes that more visible.

### 4.6 Failure and durability

- **A failed node stops the run.** On hardware where each step is minutes,
  carrying on past a failure to spend another twenty is worse than stopping.
- **Every output is written to disk as produced**, so a run killed by the OS
  can resume from the last completed node. Resume is phase 3, but the directory
  layout and incremental write must be right in phase 1.

---

## 5. UI — a vertical list of steps, not a canvas

1. **The screen.** ~360 dp inside an 18 dp gutter. A canvas needs pan, zoom,
   hit targets below `Touch.Min = 44.dp`, and edge routing.
2. **The design system does not contain one.** Ten component files, ~50
   composables, and the entire visual grammar is a vertical stack of `NCard`s.
   A canvas means a second design language beside Nocturne.
3. **A canvas would draw parallelism the runtime is forbidden to run.** One
   engine at a time. A canvas's expressive advantage is showing concurrent
   branches; here there are none. **The device constraint and the screen
   constraint point the same way** — and the list also buys acyclicity free.

### The editor

A column of `NCard`s, one per step: index badge, kicker for the kind, title,
grip for reorder, overflow menu. Input slots as `NRuledRow`s with an `NTag` for
the port type and a binding chip; tapping opens an `NBottomSheet` listing only
valid sources, with a remedy line where none qualify. Body collapsed to three
headline params, expanded to the manifest-rendered set. `NNudgeSlider` for
every integer, never a bare `NSlider`.

Reorder copies `SamplerChainScreen`'s idiom verbatim —
`detectDragGesturesAfterLongPress`, a `LaunchedEffect` that refuses to clobber
mid-drag, commit once on `onDragEnd`. **Caveat:** it depends on a fixed 52 dp
row quantum; step cards are variable height, so the target index must come from
accumulated measured heights.

### The run screen, which is not the editor

`GenerationProgress`, `ResidentCard`, `ResourceBlock(live = true)` — all reused
unmodified — plus a compact step rail, one dot per node tinted by state.

### Navigation

```kotlin
const val WORKFLOW      = "workflow"
const val WORKFLOW_EDIT = "workflow/edit/{workflowId}"
const val WORKFLOW_STEP = "workflow/edit/{workflowId}/step/{nodeId}"
const val WORKFLOW_RUN  = "workflow/run"
```

The bottom bar gains the sixth entry it has been holding. **`WORKFLOW_RUN` must
be scoped to `Routes.GRAPH`**, exactly as `Routes.VIDEO` is, for the reason
written there: a run outlives the entry that started it.

**Results do not get a new `LibrarySection`.** A workflow's outputs are images,
clips, audio and text, which already land in the existing four.

---

## 6. Phasing

### M1 — walking skeleton: text → image, across two runtimes

The smallest slice that is useful *and* discovers the real risk. A chat model
expands a prompt; a diffusion model renders it — what people do by hand today
by copy-pasting between two tabs.

Room v14; the flat `NodeRecord` with `Unknown` fallback; Input(text),
Processor(TEXT | DIFFUSION-image), Output; slot binding restricted to the
immediately preceding node; `WorkflowSession` wired into `InferenceService`;
one wake lock; **the `ResidencyPlanner` including the cross-runtime unload —
this is the point of M1.** SPEC §14 sequences by risk, and the unproven thing
is two multi-gigabyte runtimes in one process across one run.

### M2 — the rest of the modalities, and honesty before the run

STT, TTS, video; Upscale with its forced unload made visible; free slot binding
with the full type gate; `ComponentCheck` pre-flight; the residency plan preview.

### M3 — reusable workflows

Script (option A); `Repeat N` with feedback and a mandatory bound; Inputs
marked *ask when run*; resume-after-kill.

### M4 — the expensive half

`For each` over collections; per-node continue-on-error; tool-calling inside a
text Processor; QuickJS if the expression language proved insufficient; export
and import.

### Deferred indefinitely

Free-form canvas. Parallel execution. Branch-and-merge. MCP servers as nodes.
Scheduled runs. Sharing between devices.

---

## 7. Decisions needed, and where this is uncertain

**For the author:**

1. **What "Script" means** — templating plus expressions (M3, no dependency) or
   a real embedded language (QuickJS, a fifth `native/VERSIONS` entry and a new
   JNI shim). A native-build decision to make now, not in M4.
2. **Loop shape** — bracket pair, nested container, or for-each. If for-each,
   the collection port type moves to M2.
3. **"Input → … OR another model's output"** read as an edge. Confirm.
4. **Unknown node types** — refuse with a named reason, or skip? Refusing is
   consistent with SPEC §1.2.
5. **Does a workflow output belong in the Library as a first-class thing,** or
   only its artifacts? First-class means a column on four tables and a
   migration over the whole history.

**Uncertain:**

6. **Six bottom-bar tabs at 360 dp** is ~60 dp each — above `Touch.Min`, so it
   fits mathematically, but needs a look on the device.
7. **Drag-reorder with variable-height cards** has no precedent in this app.
   Fallback is an explicit move up/down in the overflow.
8. **A memory plan is a snapshot.** `availableRamBytes` moves during a run, and
   `onTrimMemory` can unload out from under a running graph.
9. **Cancel latency** exists today at one step; a workflow multiplies how often
   it is seen.
10. **`ModelStorage.findOrphans()` knows about `modelsDir` only.** Workflow
    intermediates are invisible to it.
11. **`params-manifest.json` has no workflow runtime.** Node settings that are
    not model parameters should not pretend to be — the precedent is
    `ToolSettings`, which builds `ParamSpec`s in code with namespaced keys.
