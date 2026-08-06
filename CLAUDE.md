# Working in this repository

Read [`SPEC.md`](SPEC.md) before changing behaviour; it is the contract, and its
section numbers are cited throughout the source. [`README.md`](README.md) says what
is built. This file says how to work here.

---

## The four rules that decide most arguments

1. **Honest refusal over silent failure (§1.2).** A thing that cannot work says so,
   name it, and say what would fix it. A step that quietly does less than it claims
   is the worst outcome available — worse than a crash, because nobody finds it.
2. **Parameters are data, never code (§1.5).** There are zero hardcoded parameter
   widgets. If adding an upstream parameter needs Kotlin UI, the design has been
   violated. The manifest describes; each engine reports what it will act on.
3. **Metadata over hardcoding (§1.3).** No `when (modelName)` anywhere. Ask the
   model's own row, the GGUF header, or the system — never a list of names.
4. **Runtime-locked, not model-locked (§1.1).** The boundary is which runtimes are
   compiled in, not a curated catalogue.

## Comments carry the reasoning, not the mechanics

The house style is unusual and deliberate: comments explain *why* a thing is the way
it is, and especially what went wrong that made it that way. A comment restating the
code is noise; a comment recording the failure the code prevents is the only place
that knowledge exists. Match the density and voice of the file you are editing.

## Shape of the thing

- **Sessions outlive screens.** `ChatSession`, `ImageSession`, `VideoSession`,
  `VoiceSession`, `WorkflowSession` are `@Singleton`s holding a run. A view model is
  a window onto one, never the owner — a run must survive the screen being left.
- **`InferenceService` keeps the process alive** and must know about every kind of
  run, or the system reclaims a process holding gigabytes mid-generation.
- **One engine at a time.** The diffusion engine holds a load lock; nothing may
  assume two runtimes can be resident. `WorkflowRunner.makeRoomFor` is the only
  cross-runtime unload in the app and the reason a graph can change runtime at all.
- **Long work does not run on the main thread.** `WorkflowSession.scope` is
  `Dispatchers.Main.immediate` because it holds state; anything CPU-bound must be
  launched on `Dispatchers.Default`. A QuickJS step on the main thread ANRs the app,
  and a blocked main thread also freezes the activity lifecycle callbacks — which is
  how a backgrounded app came to believe it was still on screen.
- **Values on edges are paths, never pixels or samples.** See `PortType`.

## Android surfaces, and what they actually allow

- **A workflow cannot be an activity.** The manifest is fixed at build time;
  workflows are database rows. One exported `TriggerActivity`, and a shortcut per
  workflow over it — that is what a share sheet draws as a row.
- **A shortcut's stored intent is not what the share sheet sends.** Tapping a Direct
  Share row delivers the *original* `ACTION_SEND` to the target class with
  `EXTRA_SHORTCUT_ID` added. The shortcut id must therefore be the workflow id. The
  stored intent is used only by the launcher long-press path.
- **`setActivity()` on a shortcut is load-bearing.** Unset, a shortcut is attributed
  to the launcher activity, which will not match a `<share-target targetClass=…>`.
- **An incoming `content://` grant dies with the receiving activity.** Copy the bytes
  in before finishing, or the run fails minutes later with a permission denial in a
  place unrelated to the cause.
- **`startActivity` does not throw when the platform blocks it.** A background
  activity launch logs `Background activity launch blocked!` and returns normally.
  Never infer success from the absence of an exception — decide from whether the app
  is foreground, and park anything uncertain.
- **Apps targeting Android 15+ do not grant BAL to their own `PendingIntent`s.** A
  notification meant to launch something must opt in with
  `setPendingIntentCreatorBackgroundActivityStartMode`.
- **Since Android 12 a notification may not trampoline** an activity start through a
  receiver or a service; use `PendingIntent.getActivity`.
- **Package visibility.** Anything resolving an intent against other apps needs the
  `<queries>` block. `QUERY_ALL_PACKAGES` is not available to us.
- **Exact alarms are exempt from the foreground-service background-start rule;
  inexact ones are not.** That single fact picks the scheduling mechanism: a run
  needs an FGS, so an inexact alarm wakes us with no way to do the thing we were
  woken for. `WorkManager` would run the graph inside *its* foreground service,
  which since Android 14 must declare a type — meaning `specialUse` merged onto a
  service we do not own, beside `InferenceService` which already does the job.
- **`SCHEDULE_EXACT_ALARM` is denied by default** from Android 13; `USE_EXACT_ALARM`
  is Play-restricted to alarm clocks and calendars. Refusal is a supported state,
  not a broken feature — fall back to a notification and a tap.
- **Alarms are lost on more than a reboot.** Force-stop drops them and a reinstall
  clears them, so re-arm at app start as well as on `BOOT_COMPLETED`,
  `TIME_SET` and `TIMEZONE_CHANGED`.
- **A host app decides which `PROCESS_TEXT` items its selection menu shows.**
  Resolving correctly is not the same as being offered: Keep's custom rich-text
  editor filters ours out, and that is the host's choice, not a manifest fault.

## Native

- arm64-v8a only. Four upstream repos pinned by commit in `native/VERSIONS`, built
  through `app/src/main/cpp/CMakeLists.txt`.
- **16 kB page alignment is mandatory.** Play refuses updates that use native code
  and target Android 15+ unless the `.so` files align to 16 kB. NDK r28 does it by
  default; r27 needs the linker flag that `app/build.gradle.kts` passes. Verify with
  `llvm-readelf -l <lib>.so` — every `LOAD` must read `0x4000`, not `0x1000`.
- `cppFlags` reaches `CMAKE_CXX_FLAGS` only. C files need `cFlags` separately, and
  forgetting that once left every ggml kernel compiled at `-O0`.
- **The vendored `CLAUDE.md` files under `native/` are upstream's.** Do not edit
  them; they belong to those projects and would conflict on the next bump.

## Building and testing

```
./gradlew :app:assembleSideloadDebug          # local build
./gradlew :app:installPlayDebug               # to a device
./gradlew :app:testPlayDebugUnitTest          # unit tests
```

Two flavours — `sideload` may self-update runtime bundles, `play` degrades that to a
store link because Play policy forbids downloading native code. Both are debug- and
release-buildable, so task names are `assemble<Flavour><BuildType>`; a bare
`compileDebugKotlin` is ambiguous and will fail.

**What is worth a unit test here** is the logic that is invisible on a device: type
lattices, matching rules, mime mapping, span arithmetic. The device work — share
sheets, shortcut ranking, insets — has to be checked on hardware, and `uiautomator
dump` giving node bounds is a far better instrument than a screenshot when something
is a few pixels wrong.

## Releases

Bump **both** `versionCode` and `versionName`; Android refuses an update whose
versionCode is not higher, and the only way past that is an uninstall that takes the
model files and the database with it.

Tag `vX.Y.Z`, release title `X.Y.Z`, and attach `aidroid-<version>-sideload.apk`.
Release notes are prose in the same voice as the code comments: what changed and
what was wrong before, not a list of commit subjects.
