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

The app is a complete, running Kotlin/Compose program. Everything above the native
boundary is real:

| Layer | State |
|---|---|
| Nocturne design system in Compose | Real — tokens transcribed from `styles.css` |
| All 15 screens | Real, wired to live data |
| Hugging Face resolver + compatibility gate (SPEC §3) | **Real** — hits the live HF API |
| GGUF header Range-parser fallback (§3.1) | Real |
| Downloader + foreground service (§3.4) | **Real** — resumable, sha256-verified |
| Parameter manifest system (§16) | Real — 74 llama.cpp params, generic renderer |
| Room / DataStore / Keystore / Hilt | Real |
| Residency, thermal and memory-pressure policy (§3.5, §8.3, §8.4) | Real |
| **Inference engines** (llama.cpp, whisper.cpp, sd.cpp, Kokoro) | **Stubbed** behind `InferenceEngine` |

The native layer is the next slice. It is deliberately last: SPEC §14 sequences the
build by risk so the app is already standing when the unproven parts land. Replacing
`FakeLlamaEngine` with a JNI implementation requires no change above it — the
boundary is a string-keyed map from the first call (§16.7).

## The three load-bearing decisions

**Parameters are data, never code (§1.5).** There are zero hardcoded parameter
widgets. Every parameter in SPEC §4–7 is a row in
[`assets/params-manifest.json`](app/src/main/assets/params-manifest.json), and
[`ParamRenderer.kt`](app/src/main/java/ai/ondevice/params/ParamRenderer.kt) is one
composable per *type*. Adding an upstream parameter of an existing type requires no
Kotlin at all. The Expert screen renders 74 llama.cpp parameters from that file with
tier, `dependsOn` and `sinceBuild` gating applied.

**Honest refusal over silent failure (§1.2).** The compatibility gate runs before any
download and shows its arithmetic — `weights + KV + compute` — recomputed live as the
context slider moves. "Won't run" is an acceptable answer; a crash is not.

**Metadata over hardcoding (§1.3).** Chat templates, context lengths and architecture
come from GGUF metadata via the HF API, with a maintained header parser as fallback.
There is no `when (modelName)` anywhere in the resolver.

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
