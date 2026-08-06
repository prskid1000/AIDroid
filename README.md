# On-Device AI — Android

A runtime-locked local inference app for Android. Any model on Hugging Face whose
artifacts match a bundled runtime, and which fits the device, is usable by pasting
its model ID — there is no curated catalog and no app update needed for a model
released tomorrow.

Built to [`SPEC.md`](SPEC.md) and to the fifteen-screen design canvas in
[`Mobile AI model app design/`](Mobile%20AI%20model%20app%20design/), which carries
the Nocturne design system.

---

## What this build is

The app is a complete, running Kotlin/Compose program, and the native layer has
landed:

| Layer | State |
|---|---|
| Nocturne design system in Compose | Real — tokens transcribed from `styles.css` |
| All 15 screens | Real, wired to live data |
| Hugging Face resolver + compatibility gate (SPEC §3) | **Real** — hits the live HF API |
| GGUF header Range-parser fallback (§3.1) | Real |
| Downloader + foreground service (§3.4) | **Real** — resumable, sha256-verified |
| Parameter system (§16) | Real — engines report their own keys, generic renderer |
| Room / DataStore / Keystore / Hilt | Real |
| Residency, thermal and memory-pressure policy (§3.5, §8.3, §8.4) | Real |
| **Inference engines** (llama.cpp, whisper.cpp, sd.cpp, Kokoro, OmniVoice) | **Real** — JNI, built from source |

Building the native layer last was the point: SPEC §14 sequences by risk so the app
was already standing when the unproven parts landed. The swap cost no change above
the boundary, which is a string-keyed map from the first call (§16.7) — the stub it
replaced has since been deleted rather than left to rot.

## The three load-bearing decisions

**Parameters are data, never code (§1.5).** There are zero hardcoded parameter
widgets, and [`ParamRenderer.kt`](app/src/main/java/ai/ondevice/params/ParamRenderer.kt)
is one composable per *type*, so adding a parameter of an existing type requires no
Kotlin at all.

Which parameters exist is not our claim to make. Each engine reports the keys it will
act on — the native ones enumerate the same dispatch table `apply_params()` uses, over
JNI — and [`assets/params-manifest.json`](app/src/main/assets/params-manifest.json)
only *describes* them. llama.cpp reports 57 keys; the manifest describes 66, so nine
rows that moved nothing are no longer shown. Anything reported but undescribed appears
as a plain text field rather than disappearing.

**Honest refusal over silent failure (§1.2).** The compatibility gate runs before any
download and shows its arithmetic — `weights + KV + compute` — recomputed live as the
context slider moves. "Won't run" is an acceptable answer; a crash is not.

**Metadata over hardcoding (§1.3).** Chat templates, context lengths and architecture
come from GGUF metadata via the HF API, with a maintained header parser as fallback.
There is no `when (modelName)` anywhere in the resolver.

## Workflows reach the rest of the phone

A workflow can be started by another app and can hand its result back to one.

**In.** An Input step set to *from another app* is the whole switch. What a workflow
accepts is then derived from that step's port type through `PortType.satisfies`, so it
appears in exactly the share sheets that can feed it and in no others — there is no
second setting to keep in step, and a graph that stops taking pictures stops being
offered them. Text, a picture, a recording and any file all arrive; a shared `.txt`
counts as both prose and a file, because which one a graph wanted is the graph's
business and not the boundary's. Selected text in any app reaches the same place
through `ACTION_PROCESS_TEXT`, and `ai.ondevice://workflow/<id>` reaches it from an
automation app.

A workflow cannot itself be an activity — the manifest is fixed at build time and
workflows are rows made long afterwards — so there is one exported activity and a
shortcut per workflow over it, which is what the share sheet draws as its own row.

**Out.** A Send step hands a result to another app or to the clipboard, and the
editor says plainly what that is worth: no intent on Android sends a mail. Handing
text to Gmail opens its composer filled in and a person taps send. For delivery with
nobody present the mechanism is a Tool step against a connected server, which is an
API call and already exists.

The interesting half is that the platform refuses an activity start from an app that
is not on screen, which for a run measured in minutes is the ordinary case. A Send
step that finishes in the background parks its result in a notification *and* on the
run screen, because the notification permission is optional here and losing a result
to a permission somebody declined would be worse than the wait.

## Colour discipline

Nocturne is a mono, dark-only system, and the design canvas is explicit that verdicts
carry no red or green. Accent means *runnable*, an accent outline means *caveat*, and
neutral-800 with a slash mark means *no*. Weight comes from the mark, not the hue —
which is why `Verdict` carries a tone rather than a colour, and why the Material 3
`error` role is mapped onto the neutral ramp so no screen can reach for red.

## Build

```
./gradlew :app:assembleSideloadDebug
```

Two flavours from one source, per SPEC §17.2:

- **`sideload`** — the in-app updater installs signed runtime bundles through
  `PackageInstaller`.
- **`play`** — native code arrives as Play Feature Delivery modules; the updater
  degrades to a store link, because Play policy forbids downloading executable code.

Android's W^X enforcement rejects `System.load()` on writable storage, so a downloaded
`.so` is never `dlopen`'d in either flavour. That constraint is stated on the Runtimes
screen rather than hidden.

Requires JDK 17+, Android SDK 35. `arm64-v8a` only.

## Known gaps

- Native inference is stubbed; the four engines are not yet compiled in.
- The parameter count is 74, not the 63 shown in the design canvas — the canvas used
  sample data, whereas the manifest implements every parameter in SPEC §4. The count in
  the UI is read from the manifest, so it stays honest either way.
- Manifest OTA fetch and Ed25519 verification are modelled but not wired to a server.
- Tool-call and branch rendering in chat, multi-file diffusion assembly, and the
  first-run onboarding are the three items the design canvas itself lists as unresolved.
