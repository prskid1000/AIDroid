# Proxy — design plan

**Status:** implemented. The contract is [`SPEC.md`](../SPEC.md) §18; this file
is the reasoning behind it and stays as the record of why the shape is what it
is.

Four things were decided during the build rather than here, and each is worth
reading against the section it changes:

- **The engine gate had to be re-entrant** (§6). A media tool is a
  `ToolProvider`, so a chat turn calling `generate_image` re-enters the lock it
  is already holding. A plain mutex there is a deadlock — the whole app stops
  with a request in flight. Re-entering makes room for the incoming runtime,
  which means the chat model is gone when the tool returns, which is why
  `ChatPipeline` loads before *every* round rather than once.
- **`tool_choice` is refused, not honoured** (§4). Both protocols can demand a
  tool; llama.cpp here cannot be constrained to one. Accepting the field and
  answering in prose is precisely the silent failure §1 is against.
- **The service's stop condition had to grow a third term** (§3.2), or the
  socket died with the last generation — a server up for exactly as long as the
  last request, which is indistinguishable from one that never worked.
- **`BOOT_COMPLETED` restarts it** when it was already on. Without that the
  server works until the phone reboots and then silently does not.

Two things named as open questions in §15 were settled: nothing is written to
the Library (the `keep` switch was removed rather than left storing an answer
nothing reads), and no conversation is persisted.

Grounded in a read of telecode's `proxy/` (`server.py`, `translate.py`,
`config.py`, `tool_search.py`, `managed_tools.py`, `request_log.py`,
`main._start_tailscale_funnels`, `tray/qt_sections.py`) and of this repo's
`engine/*`, `engine/workflow/WorkflowRunner.kt`, `speech/*`, `tools/*`,
`params/*`, `ui/components/*`, `ui/theme/*`, `ui/screens/SettingsScreen.kt`,
`ui/screens/ToolsScreen.kt`, `data/prefs/AppPrefs.kt`, `AndroidManifest.xml`.

---

## 0. What telecode's proxy is, and which half of it ports

Telecode's proxy is an aiohttp server sitting **in front of** `llama-server`.
It speaks two client protocols, translates both into one internal shape, runs
an intercept loop over that shape, forwards to llama.cpp over HTTP, and
translates the SSE stream back into whichever protocol the client used.

```
client ──Anthropic /v1/messages──┐                         ┌── llama-server
                                 ├─→ internal OpenAI shape ─┤   (HTTP, SSE)
client ──OpenAI /v1/chat/comp.───┘   + intercept loop       └──
```

**The single most important structural difference is that we are the upstream.**
There is no llama-server here and no HTTP hop; there is `LlamaEngine.generate()`
returning `Flow<GenerationEvent>`. So the internal OpenAI shape — the reason
`translate.py` is 1507 lines — mostly evaporates:

- **Inbound** still needs both request shapes decoded (Anthropic content blocks,
  `tool_use`/`tool_result` decomposition, OpenAI `content` parts). That work
  ports almost verbatim, into `List<EngineMessage>`.
- **Outbound does not.** Telecode translates OpenAI SSE → Anthropic SSE because
  that is what llama-server hands it. We emit *directly* from `GenerationEvent`,
  which already carries the distinctions Anthropic's wire format needs and
  OpenAI's does not — `ThinkingDelta` is a first-class event here, where
  telecode has to re-derive it by scanning a text stream for `<think>` tags
  (`ReasoningState`, `translate.py:911`). We have the structured thing; we
  should not round-trip it through a tag soup to get it back.

That is a large deletion, and it is the reason this is worth building rather
than porting.

### 0.1 Feature-by-feature verdict

| telecode feature | where | verdict here |
|---|---|---|
| Dual protocol surface (`/v1/messages`, `/v1/chat/completions`) | `server.py:1964` | **Port.** The whole point. |
| `ClientAdapter` per protocol | `server.py:175` | **Port**, but fed by `GenerationEvent`, not OpenAI chunks. |
| Anthropic ⇄ OpenAI request decomposition | `translate.py:213` | **Port.** Genuinely fiddly; the `tool_result`-carrying-images case especially. |
| Anthropic SSE assembly (`AnthropicStreamState`) | `translate.py:1029` | **Rewrite** against `GenerationEvent`. The block-index bookkeeping ports; the tag scanner does not. |
| `ReasoningState` `<think>` scanner | `translate.py:911` | **Drop.** `GenerationEvent.ThinkingDelta` already exists. |
| Model aliasing (`claude-opus-5 = qwen3.8-27b`) | `config.py:model_mapping` | **Port.** Load-bearing: it is what makes an unmodified Claude Code point at this phone. |
| Heartbeat / `event: ping` | `server.py:65` | **Port**, and keep configurable. A phone doing a four-minute prefill needs it more than a desktop does. |
| ToolSearch + BM25 over deferred tools | `tool_search.py` | **Port.** ~90 lines of Kotlin. Matters *more* here: a phone-sized context cannot hold forty tool schemas. |
| Auto-load tool schema on a blind call | `server.py:1257` | **Port.** |
| Hallucinated-tool guard + suggestions | `server.py:1279` | **Port.** |
| Managed tools (proxy runs the tool itself) | `managed_tools.py` | **Adapt.** This repo already has `ToolRegistry`/`ToolProvider`; that *is* the managed-tool registry. Wire it, do not rebuild it. |
| `strip_reminders` | `tool_registry.py` | **Port.** Pure context economy, worth more on a phone. |
| `sort_tools` (cache-stable prefix) | `config.py:sort_tools` | **Port.** llama.cpp's prefix cache behaves the same here. |
| `mid_system_messages` policy | `config.py:77` | **Port.** Same Qwen template constraint, same fix, same four modes. |
| Context-overflow policy | `server.py:690` | **Adapt.** We know `contextLength` from `LoadedModel` and can count exactly via `renderPrompt`; telecode has to guess and retry. |
| Date/location injection via `ip-api.com` | `server.py:131` | **Drop the network call.** Date from the device. Location only if the user typed one — an outbound geo-IP lookup contradicts SPEC §13. |
| Client profiles matched by request header | `config.py:client_profiles` | **Port.** This is how one server serves Claude Code and a browser client with different tool policies. |
| CORS origins | `server.py:1940` | **Port.** Needed for any browser client. |
| Debug request dumps | `request_log.py` | **Adapt** to an in-memory ring plus an optional file dump, off by default. |
| Session / task / agent / job / skill / routine REST APIs | `api_*.py` | **Drop.** Telecode's own orchestration surface, not a model-serving surface. |
| Tailscale Funnel subprocess | `main.py:164` | **Cannot port.** See §8 — there is no `tailscale` CLI on Android and Funnel is not offered there. |
| `/ui` control panel | `static/` | **Drop.** The Settings screen is the control panel. |

---

## 1. What this is for

One sentence: **a laptop, a browser, or Claude Code talks to the models on this
phone using the API it already speaks, over the tailnet, with no cloud in the
path.**

That framing settles several arguments before they start.

- It is a **server**, so it must be honest about being reachable (§8) and must
  refuse rather than half-serve (§6.3).
- It is **not a second app**. It runs the same engines, honours the same
  parameter overrides, and appears in the same notification. A picture made over
  HTTP is a picture this app made.
- It serves **every modality this app has**, not just chat. A phone that can
  transcribe, speak, draw and animate should expose all four, or the surface is
  arbitrary.

---

## 2. The endpoint surface

Two protocols, five modalities. The grid is deliberately explicit because the
gaps in it are the design.

| Capability | OpenAI surface | Anthropic surface | Engine |
|---|---|---|---|
| Chat / text | `POST /v1/chat/completions` | `POST /v1/messages` | `LlamaEngine` |
| Vision in | same (`image_url` parts) | same (`image` blocks) | `LlamaEngine` + mmproj |
| Tool calling | same | same | `ToolRegistry` (§5) |
| Token count | — | `POST /v1/messages/count_tokens` | `engine.renderPrompt` |
| Image generate | `POST /v1/images/generations` | server tool `generate_image` | `DiffusionEngine.generate` |
| Image edit / inpaint | `POST /v1/images/edits` | server tool | `DiffusionEngine` (init + mask) |
| Upscale | `POST /v1/images/upscales` *(non-standard)* | server tool | `DiffusionEngine.upscale` |
| Video generate | `POST /v1/videos` → job | server tool | `DiffusionEngine.generateVideo` |
| Speech (TTS) | `POST /v1/audio/speech` | server tool `speak` | `SpeechSynthesizer` |
| Transcribe (STT) | `POST /v1/audio/transcriptions` | server tool `transcribe` | `Transcriber` |
| Translate to English | `POST /v1/audio/translations` | server tool | `Transcriber`, `translate` param |
| Models | `GET /v1/models`, `/v1/models/{id}` | same, shape by header sniff | `ModelDao` |
| Health | `GET /health` | — | — |
| Embeddings | *(501 — see §2.3)* | — | — |

### 2.1 The Anthropic surface has no media endpoints, and that is not our bug

The Messages API is chat-only. There is no `/v1/images` in Anthropic's protocol,
and inventing one under that name would produce something no client speaks.

So media reaches the Anthropic surface **as server-side tools** — exactly
telecode's managed-tool mechanism, where the proxy injects a tool schema,
intercepts the `tool_use`, runs it itself, and feeds the result back without the
client ever seeing it. A model on `/v1/messages` calls `generate_image(prompt=…)`,
the proxy runs stable-diffusion.cpp, writes the PNG, and returns the path plus an
image block. The client got an image out of a chat endpoint, which is the only
shape that endpoint has.

The same tools are offered on the OpenAI chat surface, so "make me a picture and
then describe it" works identically on both. The dedicated `/v1/images/*` and
`/v1/audio/*` routes exist for clients that want the media API directly.

### 2.2 Long runs cannot hold a socket open, except when they can

A 512-square, 20-step diffusion on this hardware is minutes. A clip is tens of
minutes. Three shapes, chosen by how long the thing takes:

- **Chat** — SSE, always. Streaming end to end already.
- **Images** — SSE when `stream: true`, mirroring OpenAI's `partial_image`
  events, fed from `DiffusionEvent.Progress` (which carries `step`, `steps`,
  `secondsPerStep` — enough for a real progress bar rather than a spinner). Plain
  JSON when `stream` is absent. A client that waits gets one body; a client that
  wants to watch gets frames.
- **Video** — **a job, never a held connection.** `POST /v1/videos` returns
  `{id, status:"queued"}` immediately; `GET /v1/videos/{id}` polls;
  `GET /v1/videos/{id}/content` fetches the result; `GET /v1/videos/{id}/events`
  is SSE progress for anyone who wants it. This mirrors OpenAI's Videos API
  shape, and it is the only shape that survives a phone changing network halfway
  through a forty-minute render — which it will.

### 2.3 Embeddings are deferred, and the reason is stated

`Modality.EMBEDDING` exists in `Domain.kt` and `LoadedModel` carries
`embeddingLength`, but that is GGUF header metadata. There is no
`llama_get_embeddings` path through `LlamaBridge` — nothing in this app has ever
asked for a vector. `/v1/embeddings` therefore needs a **new JNI entry point and
a contract bump** (`RuntimeRegistry.REQUIRED_JNI_CONTRACT` is at 4), which is a
native change, not a proxy change. Out of scope for the first cut, and the route
returns a 501 saying exactly this — per SPEC §1.2, a thing that cannot work says
so and says what would fix it.

---

## 3. Transport

### 3.1 There is no HTTP server in this project

`grep -rn "ServerSocket\|ktor\|NanoHTTPD" app/src` returns nothing. OkHttp is a
client. Something has to be added.

| Option | For | Against |
|---|---|---|
| **Ktor 3 CIO server** | Coroutine-native, matches the codebase idiom exactly; first-class SSE plugin; multipart for `/v1/audio/transcriptions`; well-tested | ~2–3 MB of dex, a new dependency tree (kotlinx-io) |
| Hand-rolled on `ServerSocket` | No dependency; total control of flush semantics | ~500 lines of HTTP/1.1, chunked encoding, keep-alive and multipart — all of it a place for a bug nobody finds until one particular client behaves oddly |
| NanoHTTPD | Tiny | Thread-per-connection, blocking, no coroutine story, awkward SSE, effectively unmaintained |

**Recommendation: Ktor CIO.** The deciding argument is not size, it is that
telecode's proxy spends much of its complexity fighting SSE buffering and flush
timing — `_ensure_prepared`, the explicit `writer.drain()`, the `initial_frame`
emitted before any status block so clients stop buffering. Every one of those is
a place a hand-rolled server would be subtly wrong. Ktor has already had those
arguments.

It is a pure-JVM dependency, so it has no bearing on the 16 kB page-alignment
rule in CLAUDE.md.

### 3.2 Where it runs

**Inside `InferenceService`.** Not a second foreground service.

The manifest already declares `InferenceService` as `specialUse` with a subtype
string, and CLAUDE.md is explicit that it "must know about every kind of run, or
the system reclaims a process holding gigabytes mid-generation". A proxy request
*is* a kind of run. Declaring a second `specialUse` service to hold a socket
beside the service that already holds the process is two answers to one question.

**This forces a change to the service's stop condition.** Today:

```kotlin
if (progress.engine.loaded == null && progress.count == 0) stopSelf()
```

An idle proxy has nothing loaded and nothing running — so the service would stop
itself and take the listening socket with it, and the server would be up exactly
as long as the last generation lasted. The condition gains a third term: *or the
proxy is listening*. Its notification gains a resting state naming the address,
so "why is this app in my shade" has an answer on the screen rather than buried
in a settings page.

**Nothing HTTP touches the main thread.** `WorkflowSession.scope` is
`Dispatchers.Main.immediate` because it holds state; the same rule applies here.
The engine and every handler run on `Dispatchers.Default`/`IO`. A blocked main
thread does not just ANR — CLAUDE.md records that it also freezes the activity
lifecycle callbacks, which is how a backgrounded app came to believe it was
still on screen.

---

## 4. Core logic — the request pipeline

One pipeline, protocol-agnostic in the middle, mirroring
`_prepare_internal_body` but ending at an engine call rather than an HTTP
forward.

```
  bytes in
    │
 1. ├─ auth          bearer / x-api-key check (§9)
 2. ├─ profile       first client_profile whose header match hits (server.py:158)
 3. ├─ decode        Anthropic blocks | OpenAI parts  →  List<EngineMessage>
 4. ├─ resolve       body.model → alias map → ModelEntity, else 404 naming it
 5. ├─ system        profile instruction + date (+ location only if typed)
 6. ├─ tools         client tools + registry tools → core / deferred split
 7. ├─ strip         <system-reminder> and <total_tokens> lines, ours excepted
 8. ├─ sort          body.tools alphabetically, when the profile asks
 9. ├─ overflow      count via renderPrompt; drop by policy, or refuse
10. ├─ admit         residency queue (§6) — may 429/503 with Retry-After
11. ├─ run           ModelRunner → Flow<GenerationEvent>
12. ├─ intercept     ToolSearch / registry tool / auto-load / hallucination guard
13. └─ encode        GenerationEvent → Anthropic SSE | OpenAI SSE | JSON
```

Steps 1–9 are a `ProxyRequest` builder with no Android and no engine in it,
which is what makes them unit-testable — and CLAUDE.md is specific that the
logic worth a unit test here is "the logic that is invisible on a device: type
lattices, matching rules, mime mapping, span arithmetic". Protocol decoding is
squarely that.

### 4.1 The intercept loop

Telecode's loop (`_run_streaming`, `server.py:1111`) is the cleverest part of
that proxy and it ports whole, because it is protocol-independent. Per round:

1. Run the engine. Watch the **first** content signal.
2. If it is a `ToolCall` whose name is in the intercept set → capture it, write
   nothing to the client yet, and handle it:
   - `ToolSearch` → BM25 over the deferred schemas, append the matches to the
     live tool list, feed the schemas back as a tool result;
   - a registry tool (`web_search`, filesystem, MCP, `generate_image`) → run it
     through `ToolRegistry.call`, feed the result back;
   - a deferred tool called blind, with `auto_load_tools` on → inject its schema
     and tell the model to call again;
   - a name nobody knows → BM25 over everything, return "did you mean".
3. Emit a **status line** so the round is visible rather than a silent pause
   (`● ToolSearch("…") └ 3 schemas loaded`). Anthropic gets it as a synthetic
   text block; OpenAI as a content delta.
4. Append `[assistant tool_call, tool result]` and loop, to `max_roundtrips`.
5. Anything else → stream straight through.

Two details from telecode that are easy to lose and expensive to rediscover:

- **The initial frame must be emitted before any status block.** Anthropic
  clients buffer every event until they see `message_start`; OpenAI clients want
  the `role:"assistant"` opener first. Telecode emits both immediately at request
  start, before the first round. Without it the status lines arrive in a lump at
  the end, which is worse than not having them at all.
- **The intercept decision must be made from the accumulated tool name**, not
  the first fragment — the name arrives in pieces, and deciding early routes
  correctly-named tools into the hallucination branch.

### 4.2 Streaming out

`GenerationEvent` → wire, per protocol. The mapping is direct, which is the
payoff for not having an intermediate shape:

| `GenerationEvent` | Anthropic | OpenAI |
|---|---|---|
| `PromptProcessed` | seeds `message_start` usage | folded into the final `usage` |
| `ThinkingDelta` | `content_block_delta` / `thinking_delta` | `delta.reasoning_content` |
| `ThinkingDone` | closes the thinking block | — |
| `Token` | `content_block_delta` / `text_delta` | `delta.content` |
| `ToolCall` | `content_block_start` `tool_use` + `input_json_delta` | `delta.tool_calls[]` |
| `Stats` | accumulates into `message_delta.usage` | accumulates into `usage` |
| `Done(stopReason)` | `message_delta` + `message_stop` | `finish_reason` + `[DONE]` |
| `Failed(message, suggestion)` | `event: error` carrying **both** strings | `data: {error:…}` |

`Failed` carrying its `suggestion` through to the wire matters. The engine
already produces *"Not enough memory … lower the context size, pick a smaller
quant, or set cache_type_k and cache_type_v to q8_0"*. Throwing the second half
away at the protocol boundary and sending a bare 500 is precisely the silent
failure SPEC §1.2 forbids.

Block-index bookkeeping — `_next_index`, `_current_kind`, close-before-switch,
the OpenAI-tool-index → Anthropic-block-index map — ports directly from
`AnthropicStreamState`.

### 4.3 Non-streaming

The same loop, collected. Both protocols allow it and some clients still use it.
Not a second implementation — the encoder gains a "buffer and emit once" mode.

---

## 5. Tools

The app already has the registry telecode calls managed tools: `ToolProvider` /
`ToolRegistry`, with built-ins, filesystem, shell, web search and MCP servers,
each carrying `settings(): List<ParamSpec>`. The proxy does not build a second
one.

**Three tool populations meet at the proxy**, and keeping them straight is most
of the work:

1. **Client tools** — what the caller sent in `body.tools`. The client executes
   these; we pass the call back out.
2. **Registry tools** — what this app can run itself. Injected per the profile's
   `inject_managed` list, intercepted, executed here, never seen by the client.
3. **Deferred tools** — either of the above, held back behind `ToolSearch`.

The core/deferred split matters far more here than on a desktop. A phone model
at 4k or 8k context cannot afford forty tool schemas in its prefix: telecode's
`tool_search` is an optimisation there and closer to a requirement here.

**Name collisions resolve toward the client**, with one exception: a registry
tool named in `inject_managed` strips the client's same-named tool. That is
telecode's `strip_from_cc` behaviour and the reason it exists — the client's
`WebSearch` reaches a cloud API, ours reaches Brave without a key, and the point
of pointing a client at this phone is to get ours.

### 5.1 Media tools

New `ToolProvider` implementations — which means they appear in the Tools screen
with their own settings rows for free, and are available to the Chat tab as well
as to the proxy:

- `generate_image(prompt, negative, steps, size, seed)` → `DiffusionEngine`
- `edit_image(image, prompt, mask, strength)` → img2img / inpaint
- `generate_video(prompt, first, last)` → returns a job id, never blocks a turn
- `speak(text, voice, speed)` → `SpeechSynthesizer.synthesizeToFile`
- `transcribe(audio, translate)` → `Transcriber.transcribeFile`

Each returns a **file path plus a one-line summary** — the convention
`WorkflowRunner` already uses for values on edges (`PortType`: paths, never
pixels or samples). The HTTP layer turns a path into base64 or a URL at the
boundary; nothing inside the app moves media bytes around inside JSON.

---

## 6. Residency — the hardest constraint

CLAUDE.md: *"One engine at a time. The diffusion engine holds a load lock;
nothing may assume two runtimes can be resident. `WorkflowRunner.makeRoomFor` is
the only cross-runtime unload in the app and the reason a graph can change
runtime at all."*

A server invites exactly the thing that breaks: two clients, one asking for chat
and one for an image, arriving a second apart.

### 6.1 Generation logic lives in view models and cannot be called from a socket

This is the largest implementation finding. `ImageViewModel.generate()`,
`VoiceViewModel.speak()`, `VideoViewModel` — the orchestration for every
modality except text sits in `ui/vm/MediaViewModels.kt` (2571 lines) and
`ui/vm/VideoViewModel.kt` (1153 lines), reachable only from a Compose screen.

**But the headless layer already exists.** `WorkflowRunner` is a `@Singleton`
that drives `EngineManager`, `DiffusionEngine`, `Transcriber` and
`SpeechSynthesizer` directly — `runText`, `runDiffusion`, `runTranscribe`,
`runSpeak` — with `makeRoomFor` handling cross-runtime eviction and a
`RunReporter` callback for progress. That is precisely the shape an HTTP handler
needs.

**So: extract, do not duplicate.** Lift those four bodies out of
`WorkflowRunner` into a `ModelRunner` singleton taking plain request/result
types, and have both `WorkflowRunner` and the proxy call it. The alternative — a
second copy of "load the model, apply the params, collect the flow, write the
file" per modality — is four more places for the next diffusion parameter to be
forgotten.

This extraction is worth doing on its own merits, is the largest single piece of
work in the plan, and is what makes the proxy itself small.

### 6.2 Admission

One serialized queue in front of `ModelRunner`, because the engines are
serialized anyway and a queue makes that legible instead of a race.

```
proxy.queue_depth        4     beyond this → 429 with Retry-After
proxy.queue_timeout_sec  120   waited this long → 503, honestly
```

Concurrency stays at 1 and is **not** exposed as a knob, because raising it
cannot work: the load lock is real, and a number the engine ignores is a setting
that quietly does nothing. If per-runtime concurrency ever becomes real, the
knob arrives with it.

### 6.3 The model-swap question

A proxy request naming a different model than the Chat tab has loaded would,
done naively, evict a conversation's context mid-thought. Three honest policies,
as `proxy.model_policy`:

- **`queue`** *(default)* — wait for the interactive run to finish, then swap.
- **`refuse`** — 409, naming the currently loaded model in the body. Right for a
  phone in someone's hand.
- **`swap`** — evict immediately. Right for a phone on a desk being used as a
  server.

The screen says which is in force, in a sentence, because the difference between
them is only visible when it bites.

### 6.4 Battery and thermal

An always-listening server on a phone is a new failure mode for this app. Two
guards, both reusing what exists:

- `proxy.battery_floor` — below this percent, refuse *generation* (not the
  socket) with a 503 naming the battery level. `AppPrefs.batteryGuardPercent`
  already exists for downloads and this is the same idea.
- `proxy.charging_only` — off by default; on, the server accepts only while
  charging.

Both refuse loudly. A server that silently gets slower as the phone throttles is
the failure mode this repo's first principle is written against.

---

## 7. Configuration — data, not widgets

Telecode's Qt panel hardcodes one widget per setting; `qt_sections.py` is 5929
lines. This app already refuses that pattern.

**Every proxy setting is a `ParamSpec`, rendered by `ParamRow`.** The precedent
is `ToolSettings.kt`, which says it plainly: *"the numbers a tool wants tuned — a
result count, a timeout, a size cap — are the same shape as the numbers a sampler
wants tuned. A second little settings type would mean a second slider, a second
clamp and a second set of defaults that drift."*

The fit is exact. Look at what `ParamRow` already draws
(`params/ParamRenderer.kt:53`): the label, a modified dot, the current value in
mono accent, **the key on its own line in `Mono2Xs` muted**, the type-chosen
control, the help paragraph, and a "reset to …" affordance. That is, line for
line, the telecode proxy panel in the screenshots — `Tool Search (BM25)` / help
text / `proxy.tool_search` / a switch. The design already exists in this repo; it
has simply not been pointed at these keys yet.

So: a `ProxySpecs.kt` declaring the list, and `AppPrefs.proxyParams` holding a
**sparse** JSON blob keyed `proxy.*`, exactly like `toolParams` — sparse for the
same stated reason, that a default which moves in a later release moves for
everyone who never touched it.

```kotlin
object ProxySpecs {
    val ALL: List<ParamSpec> = listOf(
        ProxySettings.bool("enabled", false,
            label = "Serve the API",
            help = "Off until asked for. Nothing listens on any port while this is off.",
            tier = Tier.BASIC),
        ProxySettings.int("port", 8080, 1024, 65535,
            label = "Port", requiresReload = true,
            help = "Below 1024 is not available to an app."),
        ProxySettings.enum("bind", "tailnet", listOf("tailnet", "loopback", "all"),
            label = "Listen on", requiresReload = true,
            help = "Tailnet binds to this device's 100.x address only — reachable from " +
                "your other Tailscale machines and from nowhere else. All includes Wi-Fi."),
        ProxySettings.bool("protocol_anthropic", true, label = "Anthropic",
            help = "Serves /v1/messages."),
        ProxySettings.bool("protocol_openai", true, label = "OpenAI",
            help = "Serves /v1/chat/completions, /v1/images/*, /v1/audio/*."),
        ProxySettings.bool("tool_search", true, label = "Tool search",
            help = "Hold tool schemas back behind a search tool. A phone-sized " +
                "context cannot hold forty of them."),
        ProxySettings.enum("mid_system_messages", "demote",
            listOf("demote", "strip", "merge_top", "keep"),
            label = "Mid-conversation system messages", tier = Tier.EXPERT,
            help = "Qwen's template refuses a system message that is not first. " +
                "Demote keeps its position and re-roles it to user — the only " +
                "option that is both template-safe and cache-safe."),
        // … max_roundtrips, ping_interval, strip_reminders, sort_tools,
        //   auto_load_tools, model_policy, battery_floor, queue_depth, debug
    )
}
```

`requiresReload = true` on `port` and `bind` gets the `reload` tag drawn beside
the label for free — the same batched-restart affordance the model parameters
use.

Three things are **not** `ParamSpec`, because they are lists rather than values,
and live as JSON under the same DataStore key:

- **Model aliases** — `Map<String, String>`, e.g. `claude-opus-5 → qwen3-8b:Q4_K_M`
- **CORS origins** — `List<String>`
- **Client profiles** — `List<ProxyProfile>`: a name, a header match, and
  per-profile overrides of any boolean/enum key above

**No Room migration.** All of it is DataStore JSON, following the argument in
`workflow-plan.md` §1.1 — normalising means a migration every time a field is
added, and the schema is already at v15 with fourteen migrations. The request log
is an in-memory ring and survives nothing, deliberately.

---

## 8. Reachability — where the brief needs correcting

> *"tailscale mobile app should be sufficient to connect and provide tailscale
> url right"*

**Half right, and the wrong half matters.**

**What works.** The Tailscale Android app is a full tailnet node. With it
connected, this device holds a `100.64.0.0/10` address, and a server listening on
that address is reachable from every other machine on the tailnet at
`http://100.x.y.z:8080` — no port forwarding, no public exposure, WireGuard
encryption end to end. That really is all the user has to do, and it is the right
answer for this app.

**What does not work.** `tailscale funnel` — the thing that produces
`https://levy.tailabb811.ts.net` in the telecode screenshot — **is not available
on Android.** Funnel is a CLI feature, the Android app ships no CLI, and it is an
open feature request (tailscale/tailscale#17071). Telecode gets its funnel URL by
spawning `tailscale funnel --bg --https 443 1235` as a subprocess
(`main.py:208`); there is no subprocess to spawn here.

Three consequences to design for rather than discover:

1. **No public HTTPS URL, and no TLS at all.** The proxy serves plain HTTP inside
   the WireGuard tunnel. That is not a downgrade — tailnet traffic is already
   encrypted — but any client demanding `https://` will not work, and the screen
   must print `http://` rather than pretend.
2. **The MagicDNS name is not directly discoverable.** Without the CLI there is
   no `tailscale status --json` to read `Self.DNSName` from. Two fallbacks, in
   order: a reverse lookup on the tailnet address
   (`InetAddress.getByName("100.x.y.z").canonicalHostName`), which returns the
   MagicDNS name when MagicDNS is on; failing that, the raw IP, which always
   works. Never a guessed name.
3. **Detection is ours to do.** Enumerate `NetworkInterface`s and find an IPv4 in
   `100.64.0.0/10`. Present → the status card is live with a URL to copy. Absent →
   it says *"Tailscale is not connected — open the Tailscale app"*, with the
   reason, rather than showing a dead address.

### 8.1 Binding defaults to the tailnet interface, not `0.0.0.0`

Telecode binds `0.0.0.0` because a desktop behind a router is a different threat
model. Here, `0.0.0.0` also exposes the server to whatever café Wi-Fi the phone
is on.

`res/xml/network_security_config.xml` already states this repo's position, about
outbound MCP traffic: *"A LAN address is deliberately not included: an MCP
request carries whatever the model decided to send, and putting that on a shared
network in the clear is exactly the leak this app exists to avoid."* The same
sentence applies inbound, and harder — an open port on hotel Wi-Fi is a
stranger's prompt running on your phone.

So `bind` defaults to `tailnet`, and choosing `all` shows the reason rather than a
shrug.

---

## 9. Access control

**A bearer token, generated on first enable, required by default.**

Not optional-by-default. A generation server with no auth, listening on an
interface other machines can reach, is the failure this app's whole posture is
against — and unlike a missing feature, nobody notices it until someone else
does.

- Generated with `SecureRandom`, stored in `TokenStore` — the Keystore-backed
  store already holding the HF token (`data/secure/`).
- Accepted as `Authorization: Bearer …` **and** as `x-api-key`, because the
  Anthropic and OpenAI clients each send a different one and a server speaking
  both protocols must accept both.
- Shown once in full with a copy button, masked thereafter, with a Regenerate
  that invalidates the old one — the shape of the HF-token block already on the
  Settings screen.
- `proxy.require_auth` may be turned off, and the row's help text says that doing
  so lets anything on the tailnet generate on this phone.

CORS is separate and defaults to empty: no browser origin is allowed until one is
typed. Telecode's `https://pivot.claude.ai` entry is the model for what that list
is for.

---

## 10. The screen

### 10.1 Getting there

`SettingsScreen`'s `RootToolbar` already carries two trailing `NIconButton`s at
`size = 34.dp, iconSize = 15.dp` — Tools, then Models. A third joins them:

```kotlin
NIconButton(NIcons.Endpoint, "Proxy — serve the API",
            onClick = onOpenProxy, size = 34.dp, iconSize = 15.dp)
NIconButton(NIcons.Tools,  "Tools and MCP servers", …)
NIconButton(NIcons.Models, "Models", …)
```

Proxy goes **leftmost** of the three, so the group reads outward-facing to
inward-facing: what other machines reach, what the model reaches, what this
device holds.

Route `Routes.PROXY = "settings/proxy"`, pushed from the Settings tab exactly as
`Routes.TOOLS = "settings/tools"` is. Not a bottom-bar destination: the bar is at
six, and Settings is where the things that describe the device live.

### 10.2 A new icon

Nothing in `NIcons` means "a server other machines connect to". The nearest are
`Tools` (a plug — taken, and it means MCP), `Runtime` (a chip — means silicon)
and `Wifi` (status-bar chrome at a 14×10 viewport, not an interface icon).

So one new glyph, transcribed in the same idiom as the rest of the file: 24×24
viewport, 1.7 stroke, round cap and join, built from the existing `rrect` helper.

```kotlin
/** Endpoint: this device, with two arcs leaving it. What other machines reach. */
val Endpoint: ImageVector by lazy {
    icon(1.7f, stroked = listOf(
        rrect(4f, 5f, 9f, 14f, 2f),     // the device, seen edge-on
        "M15.5 9.5a4 4 0 0 1 0 5",      // near arc
        "M18.5 7a7.5 7.5 0 0 1 0 10",   // far arc
    ))
}
```

Two arcs rather than three, because at 15 dp a third collapses into the second —
the same reason `Wifi` uses two.

### 10.3 The screen itself

`PhoneScaffold` + `PushToolbar`, `contentPadding = PaddingValues(start = 18.dp,
end = 18.dp, top = 4.dp, bottom = 18.dp)`, a `verticalScroll` `Column` — the
`ToolsScreen` shape exactly.

```
PushToolbar("Proxy", subtitle = "http://100.94.12.7:8080 · 2 clients",
            subtitleMono = true)          ← an address is a technical value

┌ NCard ──────────────────────────────────────────────────┐
│  ● Listening              [Anthropic] [OpenAI]          │  NDot + NTag ×2
│  http://100.94.12.7:8080                      [copy] ⧉  │  MonoValue + NIcons.Copy
│  Reachable from your Tailscale machines. Not from the   │  NCardBody
│  public internet — Funnel is not available on Android.  │
└─────────────────────────────────────────────────────────┘
   NHelp: "Point Claude Code here with ANTHROPIC_BASE_URL."

ACCESS          ← SectionKicker           token, masked, [Copy] [Regenerate]
NETWORK         ParamRow × { enabled, bind, port }
PROTOCOLS       ParamRow × { protocol_anthropic, protocol_openai }
SURFACES        ParamRow × { serve_images, serve_audio, serve_video }
BEHAVIOR        ParamRow × { tool_search, auto_load_tools, strip_reminders,
                             sort_tools, mid_system_messages, model_policy }
LIMITS          ParamRow × { max_roundtrips, ping_interval, queue_depth,
                             battery_floor, charging_only }
MODEL MAPPING   alias rows: NInput = NInput  [×]    +  NButton("Add")
CORS            NInput rows                  [×]    +  NButton("Add")
CLIENTS         NCard per profile, expandable — the ToolsScreen ProviderRow shape
DIAGNOSTICS     ParamRow(debug) + NButton("Recent requests →")
```

Every one of those rows is `ParamRow(spec, values, viewModel::set)`. No `when` on
a key name anywhere in the screen — the same discipline SPEC §16.4 imposes on
model parameters, and the reason `ParamControl` is a table on the *type*.

Spacing follows the files it sits beside: `SectionKicker(…, Modifier.padding(top
= 20.dp, bottom = 8.dp))`, `NCard(gap = 9.dp)`, `NHelp(…, Modifier.padding(top =
6.dp))`. Type is `NocturneType.Row` for row labels, `MonoValue` for values,
`Mono2Xs` for the key line, `Help` for footnotes — all of which `ParamRow`
already applies.

### 10.4 The request log

`Routes.PROXY_LOG = "settings/proxy/log"`, pushed from Diagnostics. Telecode's
`request_log.py` viewer at phone scale: an `NTable` of the last N requests — time
(`MonoTimestamp`), client (from the User-Agent, or the matched profile name),
model, rounds, duration, status. Tapping one opens the intercept trace: what was
searched, what was auto-loaded, what was blocked. That trace is the only place
the answer to "why was that slow" exists.

In-memory, capped, cleared on start — like telecode's, which clears its disk
dumps on startup for the same reason. `proxy.debug` additionally writes full
request/response JSON under the app's files dir; off by default, and the row says
it writes prompts to disk.

---

## 11. Build order

Sequenced so the structural piece lands first and the rest is additive — the
logic SPEC §14 uses.

1. **`ModelRunner` extraction (§6.1).** Lift `runText` / `runDiffusion` /
   `runTranscribe` / `runSpeak` out of `WorkflowRunner`. No proxy yet; workflows
   must keep passing. **This is load-bearing and the step most likely to be
   skipped under pressure.** Every handler written before it exists is a handler
   that duplicates residency logic and has to be deleted.
2. **`ProxySpecs` + storage + the screen**, wired to nothing. Toggles that
   persist. Cheap, and it settles the design questions on a real display.
3. **Ktor inside `InferenceService`** — `GET /health` and `GET /v1/models` only,
   plus the stop-condition change and the resting notification. Proves the socket
   survives the screen going off, which is the thing most likely to be wrong.
4. **Tailnet detection and the status card.** Proves reachability from a laptop
   before any generation exists — answers "is Tailscale enough" with a fact.
5. **Auth.** Before the first generation endpoint, not after.
6. **`/v1/chat/completions`**, streaming, no tools. The OpenAI surface first,
   because it is the smaller translation and `curl` speaks it.
7. **`/v1/messages`**, streaming — the Anthropic block decoder and the SSE
   assembler, verified against a real Claude Code.
8. **Tool passthrough**, then the intercept loop: ToolSearch → BM25 → auto-load →
   hallucination guard.
9. **`/v1/audio/speech` and `/v1/audio/transcriptions`.** Small, and they prove
   the runner extraction generalises past text.
10. **`/v1/images/*`**, with SSE progress.
11. **`/v1/videos`** as a job, with the poll and content routes.
12. **Media tools on the Anthropic surface** (§5.1) — what makes modality
    reachable from `/v1/messages` at all.
13. **Client profiles, CORS, request log, per-profile overrides.**

Steps 6 and 7 are each verifiable from a laptop with `curl` and with a real
client, which is why they come before anything media-shaped.

---

## 12. Testing

Per CLAUDE.md, the split is between what is invisible on a device and what is not.

**Unit tests** (`:app:testPlayDebugUnitTest`) — the decoders and the loop, none
of which need Android:

- Anthropic blocks → `EngineMessage`: text, images (base64 and url), `tool_use`,
  `tool_result` carrying images, the interleaved-thinking case.
- OpenAI parts → `EngineMessage`, and `tool_calls` round-tripping.
- `GenerationEvent` → SSE for both protocols: block indices,
  close-before-switch, a tool call interrupting text, `Failed` carrying its
  suggestion.
- BM25 ranking and the `select:Name` exact-match path.
- Profile header matching, alias resolution, the mid-system-message policy, the
  core/deferred split, `strip_reminders` span arithmetic.

**On hardware**, because nothing else will find it:

- The socket surviving screen-off, doze, and an app swipe from recents.
- A model swap arriving mid-generation, under each of the three `model_policy`
  values.
- Tailnet detection across Tailscale connect / disconnect / network change.
- A forty-minute video job surviving Wi-Fi → cellular.
- SSE flush behaviour against a real Claude Code, a real `openai` SDK client, and
  a browser `fetch`. This is where telecode's hard-won `initial_frame` and
  `drain()` lessons either replicate or do not.

---

## 13. Deliberately out of scope

- **Funnel / public HTTPS.** Not available on Android (§8). If it ever is, it is
  a status-card change and nothing else.
- **`/v1/embeddings`.** Needs a JNI contract bump (§2.3). Returns 501 saying so.
- **Telecode's session / task / agent / job / skill / routine APIs.** That is an
  orchestration surface; this app's orchestration surface is Workflow, and
  exposing *that* over HTTP is a separate proposal with a separate threat model.
- **Concurrency above 1.** The load lock is real, and a knob that lies is worse
  than no knob (§6.2).
- **Serving another device's models.** This proxy serves what is on this phone.
- **The `/ui` control panel.** The Settings screen is the control panel.

---

## 14. Risks

| Risk | Reading | Mitigation |
|---|---|---|
| The `ModelRunner` extraction destabilises Workflow | High cost, high likelihood if rushed | Step 1 alone, with the workflow path green before anything else starts |
| Android kills the process despite the FGS | Medium | `specialUse` already declared; add a battery-optimisation exemption prompt, and say plainly on the screen that the OS may still stop it |
| SSE buffering differs per client | High likelihood, low cost each | Port telecode's `initial_frame` + explicit-flush discipline from the start rather than rediscovering it |
| An open port on café Wi-Fi | Low likelihood, very high cost | `bind` defaults to `tailnet`; auth on by default; `all` warns |
| Thermal throttling makes served generation crawl | Certain, eventually | Refuse loudly at the battery floor; surface tok/s in the log rather than letting it degrade silently |
| Ktor's dex and method count | Low | Measure at step 3; the hand-rolled fallback is real but is a last resort |
| A phone model too small to be worth serving | Medium | Not ours to fix — but the alias map and `/v1/models` make it visible which model a client actually got |

---

## 15. Open questions

1. **Does anything write into the Library?** A picture generated over HTTP — does
   it appear in the Library tab? Argued yes, for the reason in §1: it is the same
   app. But it means a remote client can fill the gallery, so a
   `proxy.keep_outputs` toggle may be wanted.
2. **Whose parameter overrides apply?** A request naming a model gets that
   model's stored `paramOverridesJson`, and the request body overrides on top —
   the `SparseParams.overlaidWith` shape already used everywhere. Worth
   confirming that is the wanted precedence.
3. **Does a proxy conversation persist?** Proposed no: the protocols are
   stateless and the client holds the history. Nothing is written to
   `conversations`.
4. **Profile matching by header only?** Telecode matches one header substring. A
   per-profile token would be stronger, because a token is what actually
   identifies a client — a header is a hint.
