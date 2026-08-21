# On-Device AI — Android App Specification

**Target device (reference hardware):** Snapdragon 8 Gen 5 (non-Elite), Adreno 829 GPU, Hexagon NPU.
**Platform:** Android only. No iOS, no cross-platform.
**Status:** Design specification for implementation. Not yet built.

---

## 0. How to read this document

This spec is written to be handed to an implementing agent. It is organized as:

- **§1–3** — principles, architecture, and the model pipeline (what makes the app work)
- **§4–8** — exhaustive parameter reference per runtime (the bulk of the document)
- **§9–13** — UX surfacing, persistence, screens, hardware config, risks

**On parameter defaults:** every default listed in §4–8 must be verified against the *pinned upstream build* at implementation time. llama.cpp in particular renames, deprecates, and re-defaults sampler parameters between releases. Treat the defaults here as "what to look for," not as ground truth. Where a parameter is version-sensitive it is marked ⚠.

---

## 1. Core principles

### 1.1 Runtime-locked, not model-locked

The app does **not** ship a curated model catalog as its boundary. The boundary is which inference runtimes are compiled in. Any model on Hugging Face whose artifacts match a bundled runtime, and which fits the device, is usable by pasting its model ID.

The practical consequence: a model released tomorrow works without an app update, provided its architecture is already known to the bundled runtime. When it *doesn't* work, the app says which runtime version lacks support — a fact about the build, not a curation decision.

### 1.2 Honest refusal over silent failure

The app must never OOM-crash on a model it could have predicted wouldn't fit. Every model gets a compatibility verdict *before* download, with the reasoning shown. "Won't run" is an acceptable answer; a crash is not.

### 1.3 Metadata over hardcoding

Chat templates, context lengths, special tokens, and architecture come from GGUF metadata (via the HF API or a direct header read). Nothing model-specific is hardcoded in app source. This is the mechanism that makes §1.1 true rather than aspirational.

### 1.4 Everything tweakable, sensibly defaulted

Every parameter the underlying runtime exposes is reachable in the UI. They are tiered (Basic / Advanced / Expert, see §9) so the default experience stays clean, but nothing is hidden permanently and nothing is decided for the user irreversibly.

### 1.5 Parameters are data, never code

**There are zero hardcoded parameter widgets in the app.** Every parameter in §4–7 is a row in a machine-generated manifest (§16). The UI is a generic renderer over that manifest. A new sampler added upstream appears in the app because the manifest regenerated — not because anyone wrote a slider.

The test: if implementing a new upstream parameter requires touching Kotlin UI source, the design has been violated.

### 1.6 Runtimes are versioned, replaceable artifacts

The bundled engines are not baked into the app's identity. Each is a signed, versioned bundle built by CI from a pinned upstream commit, exposing a versioned JNI contract (§17). The app can check for, download, and install newer runtime bundles without the user visiting a store.

**Hard constraint discovered during design:** Android's W^X / dynamic-code-loading enforcement rejects `System.load()` on any file in writable app storage (`UnsatisfiedLinkError: Attempt to load writable file`), hard-enforced on Android 14+ at higher target SDKs. Google Play policy separately prohibits downloading native executable code from non-Play sources. **True in-process `.so` hot-swap is therefore impossible.** §17 specifies the mechanisms that do work.

---

## 2. Architecture

### 2.1 Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Kotlin | Direct JNI to ggml; no bridge on the token stream |
| UI | Jetpack Compose + Material 3 Expressive | Dynamic color, motion system, predictive back |
| Native build | CMake + Android NDK, **single shared ggml** | llama.cpp / whisper.cpp / sd.cpp all sit on ggml |
| Async | Coroutines + `Flow` | Token streaming maps to a cold Flow; structured cancellation |
| Inference host | Foreground Service | Survives backgrounding; owns memory-pressure negotiation |
| Persistence | Room (chats, models, presets) + DataStore (prefs) | |
| Images | Coil 3 | Compose-native bitmap lifecycle |
| DI | Hilt | |
| Audio | `AudioTrack` (TTS out) / `AudioRecord` (STT in) | Raw PCM; Media3 is overkill |
| Secrets | Android Keystore | HF token |

### 2.2 Native module layout

```
native/
  ggml/                 # ONE copy, shared submodule
  llama.cpp/            # submodule, links shared ggml
  whisper.cpp/          # submodule, links shared ggml
  stable-diffusion.cpp/ # submodule, links shared ggml
  kokoro/               # ONNX Runtime + espeak-ng (or sherpa-onnx)
  jni/
    llama_jni.cpp
    whisper_jni.cpp
    sd_jni.cpp
    tts_jni.cpp
  CMakeLists.txt        # top-level; one ggml target, four consumers
```

**Critical:** do not vendor three copies of ggml. One ggml target, one backend configuration (OpenCL / Hexagon / CPU), one set of NDK flags, one memory-pressure policy. This is the highest-leverage structural decision in the project.

**ABI:** `arm64-v8a` only. Do not ship `armeabi-v7a` — no device that can run these models is 32-bit, and it doubles APK size.

### 2.3 Runtime registry

Each engine registers a capability descriptor at boot. The resolver (§3) queries this registry and contains **zero** model-specific knowledge.

```kotlin
data class RuntimeDescriptor(
    val id: String,                       // "llama.cpp"
    val version: String,                  // "b6xxx"
    val formats: Set<ModelFormat>,        // [GGUF]
    val architectures: Set<String>,       // ["llama","qwen2","qwen3","gemma3",...]
    val capabilities: Set<Capability>,    // [TEXT, VISION, TOOLS, EMBEDDING]
    val backends: List<BackendId>,        // [OPENCL, HEXAGON, CPU]
    val quantPreferences: QuantPolicy,    // fast path = Q4_0 on Adreno OpenCL
)
```

The architecture list should be **generated from the pinned upstream source at build time** (parse `llama.cpp`'s arch enum), not hand-maintained. Hand-maintained lists rot and silently reintroduce model-locking.

Settings must surface each runtime's version and architecture count.

---

## 3. Model resolution, download, and management

### 3.1 Verified Hugging Face API surface

These endpoints were tested and confirmed working. They provide everything needed for resolution **without downloading weights**.

**`GET https://huggingface.co/api/models/{owner}/{repo}`**

For GGUF repos, returns a parsed `gguf` block:

| Field | Example | Use |
|---|---|---|
| `gguf.architecture` | `"qwen2"` | Match against runtime registry |
| `gguf.context_length` | `32768` | Max `n_ctx`; clamp UI slider |
| `gguf.chat_template` | full Jinja string | **Prompt formatting — do not hardcode** |
| `gguf.bos_token` / `eos_token` | `"<\|endoftext\|>"` / `"<\|im_end\|>"` | Tokenizer config, stop conditions |
| `gguf.total` | `7615616512` | Parameter count → size class |
| `gguf.totalFileSize` | `15237853600` | Repo total across all quants |
| `siblings[].rfilename` | `Qwen2.5-7B-Instruct-Q4_0.gguf` | Quant variant enumeration |
| `gated` | `false` | Whether HF token + license acceptance needed |
| `pipeline_tag`, `tags`, `config` | | Modality hints, secondary signal only |

**`POST https://huggingface.co/api/models/{id}/paths-info/{revision}`**
Body: `{"paths": ["file.gguf"], "expand": true}`

| Field | Use |
|---|---|
| `size` | Exact bytes — storage check |
| `lfs.oid` | **sha256** — post-download integrity verification |
| `lastCommit.id` | Revision pinning |
| `securityFileStatus` | Malware scan verdicts (JFrog / ProtectAI) |

Surface `securityFileStatus` before install. GGUF has had SSTI vulnerabilities; an `unscanned` verdict warrants a warning, not a block.

**Fallback:** for repos where HF has not parsed metadata, issue an HTTP **Range request** for the first ~1 MB and parse the GGUF header directly. The HF `gguf` block is effectively undocumented and may change shape — keep this parser as a maintained fallback, not a stub.

### 3.2 Resolution pipeline

**Accepted inputs:**
- `owner/repo`
- `https://huggingface.co/owner/repo` (and `/tree/`, `/blob/` variants — normalize)
- Direct `.gguf` URL (any host)
- Local file via SAF (`content://` URI)

**Steps:**

1. Normalize input → `{owner, repo, revision}`
2. `GET /api/models/{id}` → metadata block
3. **Classify modality** from architecture + file shape (not repo name):
   - text LLM, vision LLM (has `mmproj-*.gguf`), embedding model
   - diffusion (sd.cpp-known arch, or `model_index.json` / UNet+VAE+text-encoder shape)
   - whisper (`ggml-*.bin` / whisper arch)
   - Kokoro (ONNX + voices file)
4. **Enumerate quant variants** from siblings — present all, annotated with size and speed class
5. **Detect companions** and auto-pair:
   - `mmproj-*.gguf` → vision projector, required for image input
   - VAE / CLIP-L / CLIP-G / T5-XXL → diffusion text encoders and decoder
   - `voices-*.bin` → Kokoro style vectors
   - Never make the user manually assemble a multi-file model
6. **Detect sharding** — `model-00001-of-00003.gguf` → one logical model, download all parts
7. `paths-info` → exact sizes, sha256, commit SHA
8. Compute compatibility verdict (§3.3)

**Explicit failure cases** (each gets its own message and remedy):

| Case | Message | Remedy offered |
|---|---|---|
| safetensors/`.bin` only | "Ships PyTorch weights only — conversion requires a desktop." | One-tap search for `{repo}-GGUF`, and known mirrors (bartowski, mradermacher, unsloth) |
| Unknown architecture | "llama.cpp b6xxx doesn't support `arch_x` yet." | Link to upstream issue search; note it may work after an app update |
| Gated repo, no token | "This repo requires accepting its license on HF." | Deep-link to the repo page; token entry |
| Repo not found / private | "Not found, or private." | Token entry |
| Pickle-format files present | Security warning | Block by default, expert override |

### 3.3 Compatibility gate

Computed before any download. Rendered as a badge in search results and on the model detail sheet.

| Verdict | Condition |
|---|---|
| ✅ **Fast** | arch supported, Q4_0 (Adreno OpenCL fast path), fits GPU/NPU budget |
| ⚠️ **Works, slower** | arch supported, non-Q4_0 quant → CPU path |
| ⚠️ **Tight** | fits RAM but <1 GB headroom at chosen context |
| ❌ **Won't fit** | estimated runtime RAM > device RAM, or size > free storage |
| ❌ **Unsupported architecture** | not in runtime registry — *name the runtime and version* |
| ❌ **Not runnable** | format has no bundled runtime |

**Fit estimate must account for:**
- model file size on disk
- KV cache: `2 × n_layer × n_ctx × n_embd_kv × bytes_per_elem(cache_type)` — recompute live as the user moves the context slider
- compute buffer (~`n_batch`-proportional)
- **Hexagon session cap ≈ 3.5 GB** — beyond this, multi-device layer split is required
- free storage, with a configurable reserve

Show the arithmetic: *"≈5.2 GB at 8K context (model 4.4 + KV 0.6 + compute 0.2). You have 11 GB."* Not a bare yes/no.

### 3.4 Downloader

| Feature | Detail |
|---|---|
| Transport | OkHttp, HTTP Range for resume |
| Host | Foreground Service with progress notification |
| Concurrency | Configurable parallel connections per file (default 4); queue across models |
| Resume | Across app kill and reboot; persist byte offsets in Room |
| Integrity | sha256 vs `lfs.oid`; reject and offer re-download on mismatch |
| Revision | Pin to `lastCommit.id`; detect upstream change, offer update, never silently swap |
| Auth | HF token in Keystore; per-request bearer |
| Network policy | Wi-Fi-only toggle; metered warning; pause on network loss |
| Storage | Internal or SD card via SAF; per-model location |
| Sharded | Treat N parts as one job; atomic completion |
| Companions | Auto-queue mmproj / VAE / encoders alongside the primary file |
| Retry | Exponential backoff, max attempts configurable |
| Cleanup | Remove partial files on cancel; orphan sweep on boot |

### 3.5 Library and residency

**Per-model record:** id, revision, arch, quant, file size, install date, last used, measured tok/s per backend, user notes, favourite flag.

**Residency policy:**
- One model loaded at a time by default
- Explicit "keep loaded" pin (survives screen-off, not process death)
- Unload on `onTrimMemory` / `ComponentCallbacks2` pressure signals
- Warm-swap: when switching models, unload old before loading new (never hold both)
- Optional preload-on-launch for the pinned model

**Also:**
- Import local GGUF via SAF — a first-class path, equal prominence to HF download
- Disk usage view grouped by modality; bulk delete
- Orphan cleanup (files with no Room record, records with no file)
- Export/import model list as JSON (device migration)

---

## 4. llama.cpp — text and vision parameters

⚠ = version-sensitive; verify against pinned build.

### 4.1 Model load parameters

These require a model reload to change. Group them under a "Model" section with a "reload required" indicator.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `n_ctx` | int | from GGUF `context_length` | Context window. Clamp to model max. Drives KV cache size |
| `n_batch` | int | 2048 ⚠ | Logical batch size for prompt processing |
| `n_ubatch` | int | 512 ⚠ | Physical batch size; lower = less memory |
| `n_gpu_layers` | int | auto | Layers offloaded. `0` = CPU only, `99` = all. Auto-compute from free VRAM estimate |
| `n_threads` | int | perf-core count | Generation threads |
| `n_threads_batch` | int | = `n_threads` | Prompt-processing threads |
| `use_mmap` | bool | true | Memory-map weights. Disable if storage is slow |
| `use_mlock` | bool | false | Lock in RAM — risky on mobile, expert only |
| `flash_attn` | bool/auto ⚠ | auto | Flash attention. Large memory saving |
| `cache_type_k` | enum | `f16` | KV cache K quantization: `f32`,`f16`,`q8_0`,`q5_1`,`q5_0`,`q4_1`,`q4_0` |
| `cache_type_v` | enum | `f16` | Same set. `q4_0` V-cache often needs flash_attn |
| `no_kv_offload` | bool | false | Keep KV cache on CPU |
| `defrag_thold` | float | -1.0 ⚠ | KV cache defragmentation threshold |
| `n_parallel` | int | 1 | Parallel sequences (slots) |
| `rope_freq_base` | float | from GGUF | RoPE base frequency |
| `rope_freq_scale` | float | from GGUF | RoPE frequency scale |
| `rope_scaling_type` | enum | `none` | `none`, `linear`, `yarn`, `longrope` ⚠ |
| `yarn_ext_factor` | float | -1.0 | YaRN extrapolation mix |
| `yarn_attn_factor` | float | 1.0 | YaRN attention scale |
| `yarn_beta_fast` | float | 32.0 | YaRN low correction dim |
| `yarn_beta_slow` | float | 1.0 | YaRN high correction dim |
| `yarn_orig_ctx` | int | 0 | Original context for YaRN |
| `split_mode` | enum | `layer` | `none`, `layer`, `row` — relevant for multi-Hexagon-session split |
| `tensor_split` | float[] | — | Proportional split across devices |
| `main_gpu` | int | 0 | Primary device index |
| `check_tensors` | bool | false | Validate tensor data on load |
| `lora` | path[] | — | LoRA adapter(s) |
| `lora_scale` | float[] | 1.0 | Per-adapter scale |
| `mmproj` | path | auto-paired | Vision projector |
| `mmproj_use_gpu` | bool | true ⚠ | Offload projector |
| `pooling_type` | enum | model default | Embedding models: `none`,`mean`,`cls`,`last`,`rank` |

### 4.2 Sampling parameters

Live-editable, no reload. This is the section power users will live in.

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| `temp` | float | 0.8 ⚠ | 0.0–2.0 | 0 = greedy |
| `top_k` | int | 40 ⚠ | 0–200 | 0 = disabled |
| `top_p` | float | 0.95 ⚠ | 0.0–1.0 | 1.0 = disabled |
| `min_p` | float | 0.05 ⚠ | 0.0–1.0 | 0 = disabled. Often preferred over top_p |
| `typical_p` | float | 1.0 | 0.0–1.0 | Locally typical sampling; 1.0 = disabled |
| `top_n_sigma` | float | -1.0 ⚠ | — | Sigma-based truncation; -1 = disabled |
| `min_keep` | int | 0 | — | Min tokens truncation samplers must keep |
| `repeat_penalty` | float | 1.0 ⚠ | 0.0–2.0 | 1.0 = disabled |
| `repeat_last_n` | int | 64 | -1..n_ctx | Tokens considered for repeat penalty; -1 = full ctx |
| `presence_penalty` | float | 0.0 | 0.0–2.0 | |
| `frequency_penalty` | float | 0.0 | 0.0–2.0 | |
| `penalize_nl` | bool | false ⚠ | | Apply penalties to newline |
| `dry_multiplier` | float | 0.0 | 0.0–5.0 | DRY repetition penalty; 0 = disabled |
| `dry_base` | float | 1.75 | 1.0–4.0 | |
| `dry_allowed_length` | int | 2 | | Sequence length before DRY engages |
| `dry_penalty_last_n` | int | -1 | | -1 = full context |
| `dry_sequence_breakers` | string[] | `["\n",":","\"","*"]` | | |
| `xtc_probability` | float | 0.0 | 0.0–1.0 | XTC (exclude top choices); 0 = disabled |
| `xtc_threshold` | float | 0.1 | 0.0–1.0 | |
| `mirostat` | enum | 0 | 0/1/2 | Overrides top_k/top_p/temp when active |
| `mirostat_tau` | float | 5.0 | | Target entropy |
| `mirostat_eta` | float | 0.1 | | Learning rate |
| `dynatemp_range` | float | 0.0 | | Dynamic temperature range; 0 = static |
| `dynatemp_exponent` | float | 1.0 | | |
| `samplers` | ordered list | runtime default ⚠ | | **Sampler chain order** — drag-to-reorder UI |
| `logit_bias` | map<token,float> | — | | Per-token bias; needs a token picker |
| `grammar` | GBNF string | — | | Constrained generation |
| `json_schema` | JSON | — | | Compiled to GBNF |
| `seed` | int | -1 | | -1 = random. Expose "reuse last seed" |
| `ignore_eos` | bool | false | | |
| `n_probs` | int | 0 | | Return top-N logprobs — powers a token-probability inspector |

**Sampler chain ordering** deserves first-class UI. The order materially changes output and most apps hide it. A drag-reorderable list with enable/disable per sampler is the right surface.

### 4.3 Generation control

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `n_predict` | int | -1 | Max tokens; -1 = until EOS/context full |
| `stop` | string[] | from template | Stop sequences; user-extendable |
| `n_keep` | int | 0 | Tokens kept when context shifts |
| `cache_prompt` | bool | true | Reuse KV across turns — major speed win |
| `context_shift` | bool | true ⚠ | Auto-shift when full vs. hard stop |

### 4.4 Chat and template

| Parameter | Notes |
|---|---|
| `chat_template` | From GGUF metadata by default. **Editable override** with live preview of the rendered prompt |
| `system_prompt` | Per-conversation and per-model default |
| `add_bos` / `parse_special` | Expert toggles |
| Reasoning format ⚠ | Models with thinking blocks (`<think>`): parse and render collapsed, configurable tag pair |
| Tool calling | Tool definitions injected per template; render tool calls/results as distinct message types |

**Prompt inspector** (expert): show the exact final string sent to the tokenizer, plus token count and token boundaries. This is the single most useful debugging affordance for a local LLM app and almost nothing ships it.

### 4.5 Vision (image → text)

- Requires `mmproj-*.gguf` paired with the base model — resolver auto-detects
- Image input sources: camera, gallery, file, screenshot share-target
- Preprocessing: resize policy, aspect handling — expose the target resolution the projector expects
- Multi-image where the architecture supports it
- ⚠ Newer builds expose `image_min_tokens` / `image_max_tokens` — expose if present
- Show per-image token cost before sending (images consume context fast)

---

## 5. stable-diffusion.cpp — image parameters

Covers text→image, image+text→image (img2img), inpainting, and ControlNet.

### 5.1 Model and component loading

| Parameter | Notes |
|---|---|
| `model` | Full checkpoint (SD1.x / SD2.x / SDXL) |
| `diffusion_model` | Standalone UNet/DiT (Flux, SD3) — used *instead of* `model` |
| `vae` | External VAE. Auto-pair from repo when present |
| `clip_l`, `clip_g`, `t5xxl` | Text encoders, required for SD3/Flux |
| `embd_dir` | Textual inversion embeddings directory |
| `lora_model_dir` | LoRA directory; prompt syntax `<lora:name:weight>` |
| `control_net` | ControlNet model path |
| `upscale_model` | ESRGAN-family upscaler |
| `type` | Weight type / on-the-fly quantization: `f32`,`f16`,`q8_0`,`q5_1`,`q5_0`,`q4_1`,`q4_0`,`q3_k`,`q2_k` ⚠ |
| `vae_tiling` | **Critical on mobile** — tiled VAE decode, large memory reduction |
| `vae_on_cpu` | Offload VAE to CPU |
| `clip_on_cpu` | Offload text encoder to CPU |
| `control_net_cpu` | Offload ControlNet to CPU |
| `offload_to_cpu` ⚠ | General weight offload between stages |
| `diffusion_fa` | Flash attention in the diffusion model |
| `threads` | CPU thread count |
| `rng` | RNG source: `std_default`, `cuda` — affects seed reproducibility |

### 5.2 Generation parameters (all modes)

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | Supports attention weighting `(word:1.2)` and LoRA tags |
| `negative_prompt` | string | — | Ignored by Flux-family (distilled) |
| `width` / `height` | int | 512×512 ⚠ | Step 64. Warn above device-safe limits |
| `steps` | int | 20 ⚠ | |
| `cfg_scale` | float | 7.0 ⚠ | Classifier-free guidance |
| `guidance` | float | 3.5 ⚠ | Distilled guidance (Flux) — separate from cfg_scale |
| `sampling_method` | enum | `euler_a` ⚠ | `euler`, `euler_a`, `heun`, `dpm2`, `dpm++2s_a`, `dpm++2m`, `dpm++2mv2`, `ipndm`, `ipndm_v`, `lcm`, `ddim_trailing`, `tcd` |
| `schedule` | enum | `discrete` | `discrete`, `karras`, `exponential`, `ays`, `gits` |
| `seed` | int | -1 | -1 = random. Show the used seed; one-tap reuse |
| `batch_count` | int | 1 | Sequential generations |
| `clip_skip` | int | -1 | -1 = arch default (1 for SD1.x, 2 for SD2.x/SDXL) |
| `flow_shift` ⚠ | float | — | SD3/Flux timestep shift |
| `slg_scale` ⚠ | float | 0.0 | Skip-layer guidance scale |
| `skip_layers` ⚠ | int[] | — | Layers to skip for SLG |
| `skip_layer_start` / `skip_layer_end` ⚠ | float | 0.01 / 0.2 | SLG active window |

### 5.3 Mode-specific

**text → image (txt2img)** — the parameters above, no extras.

**image + text → image (img2img)**

| Parameter | Default | Notes |
|---|---|---|
| `init_img` | — | Source image |
| `strength` | 0.75 | Denoise strength. 0 = unchanged, 1 = ignore source. **The key dial** |
| `style_ratio` ⚠ | — | Style-transfer blend where supported |

**inpainting**

| Parameter | Notes |
|---|---|
| `mask` | Mask image. Needs an in-app **brush mask editor**: brush size, hardness, erase, invert, clear, undo |
| `strength` | As above, applied within the mask |

**ControlNet**

| Parameter | Notes |
|---|---|
| `control_image` | Guidance image |
| `control_strength` | 0.0–1.0 |
| `canny` | Built-in Canny preprocessor. Expose thresholds if available |

**Upscaling**

| Parameter | Notes |
|---|---|
| `upscale_model` | ESRGAN-family |
| `upscale_repeats` | Iteration count |

### 5.4 Image generation UX

- **Live preview** during sampling — show intermediate latents, not just a spinner
- **Cancel mid-generation** — must actually free memory, not just detach the callback
- Progress: step N/M, elapsed, ETA, current backend
- **Gallery** with full generation parameters embedded in PNG metadata (EXIF/tEXt) so any image is reproducible
- "Reuse parameters" from any gallery image → repopulates the whole form
- Parameter presets per model (SDXL and SD1.5 want very different step/CFG defaults)
- Memory guardrails: warn when `width × height × batch` exceeds a measured-safe envelope; suggest `vae_tiling`

---

## 6. whisper.cpp — speech → text parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `model` | path | — | `ggml-*.bin`, tiny → large-v3 |
| `language` | enum | `auto` | ISO code or auto-detect |
| `detect_language` | bool | false | Detect only, no transcribe |
| `translate` | bool | false | Translate to English |
| `threads` | int | perf cores | |
| `processors` | int | 1 | Parallel chunk processors |
| `offset_t` | ms | 0 | Start time offset |
| `offset_n` | int | 0 | Start segment index |
| `duration` | ms | 0 | 0 = full |
| `max_context` | int | -1 | Max text tokens carried between chunks; 0 improves long-audio stability |
| `max_len` | int | 0 | Max chars per segment |
| `split_on_word` | bool | false | Split on word rather than token |
| `best_of` | int | 5 ⚠ | Greedy candidates |
| `beam_size` | int | 5 ⚠ | Beam search width; -1 = greedy |
| `audio_ctx` | int | 0 | Audio context size — **lower = much faster, less accurate** |
| `word_thold` | float | 0.01 | Word timestamp probability threshold |
| `entropy_thold` | float | 2.40 | Decoder fallback trigger |
| `logprob_thold` | float | -1.00 | Decoder fallback trigger |
| `no_speech_thold` | float | 0.60 | Silence detection |
| `temperature` | float | 0.0 | |
| `temperature_inc` | float | 0.2 | Fallback temperature increment |
| `no_fallback` | bool | false | Disable temperature fallback |
| `prompt` | string | — | Initial prompt — biases vocabulary/spelling |
| `diarize` | bool | false | Stereo speaker diarization |
| `tinydiarize` | bool | false | Requires a tdrz model |
| `flash_attn` | bool | false ⚠ | |
| `no_gpu` | bool | false | Force CPU |
| `suppress_blank` / `suppress_nst` ⚠ | bool | | Suppress blank / non-speech tokens |

**VAD (voice activity detection)**

| Parameter | Default | Notes |
|---|---|---|
| `vad` | false | Enable VAD preprocessing |
| `vad_model` | — | Silero VAD ggml model |
| `vad_threshold` | 0.5 | Speech probability threshold |
| `vad_min_speech_duration_ms` | 250 | |
| `vad_min_silence_duration_ms` | 100 | |
| `vad_max_speech_duration_s` | FLT_MAX | Force split for long speech |
| `vad_speech_pad_ms` | 30 | Padding around detected speech |
| `vad_samples_overlap` | 0.1 | Overlap between VAD chunks (s) |

**Streaming mode** (live mic)

| Parameter | Default | Notes |
|---|---|---|
| `step_ms` | 3000 | Audio step per inference |
| `length_ms` | 10000 | Audio window length |
| `keep_ms` | 200 | Audio carried between steps |
| `capture_device` | default | Mic selection |

**STT UX:** live partial transcript with confidence shading, segment timestamps, per-segment edit, export as TXT/SRT/VTT/JSON, file-based transcription with a progress bar and background service, share-target for audio files.

---

## 7. Kokoro — text → speech parameters

⚠ Kokoro on Android is the least-proven component. Evaluate `sherpa-onnx` (bundles Kokoro + G2P + Android bindings) against raw ONNX Runtime + espeak-ng before committing.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `voice` | enum | `af_heart` | 50+ voices. Naming: `{lang}{gender}_{name}` — `af_`/`am_` US, `bf_`/`bm_` UK, plus ja/zh/es/fr/hi/it/pt-br |
| `speed` | float | 1.0 | 0.5–2.0 |
| `lang_code` | enum | from voice | Drives phonemizer language |
| `voice_blend` | (voice, voice, ratio) | — | Style-vector interpolation between two voices |
| `sample_rate` | int | 24000 | Fixed by model |
| `phonemizer` | enum | espeak-ng | G2P backend |
| `split_pattern` | regex | sentence | Long-text chunking — Kokoro has a short effective context |
| `trim_silence` | bool | true | Trim leading/trailing silence per chunk |
| `pitch` / `volume` ⚠ | float | 1.0 | Only if the pipeline supports it; otherwise post-process |

**TTS UX:**
- Voice picker with **preview sample per voice** (pre-generated or on-demand)
- Streaming synthesis → playback: start audio on the first chunk, don't wait for full synthesis
- Read-aloud toggle on assistant messages, with sentence-level highlight-follow
- Background playback with media notification and lock-screen controls
- Export to WAV/M4A
- Per-persona default voice

---

## 8. Cross-cutting: backend and hardware configuration

### 8.1 Backend selection

Available backends detected at boot: **OpenCL** (Adreno), **Hexagon HTP** (NPU), **CPU**.

For Adreno, the OpenCL backend is Qualcomm-authored and upstream-recommended; Vulkan on Adreno is documented as slower without driver fixes. Default to OpenCL, keep Vulkan out of v1.

| Setting | Notes |
|---|---|
| Backend mode | Auto (benchmark-driven) / OpenCL / Hexagon / CPU |
| Per-model override | Persisted per model |
| `n_gpu_layers` auto-calc | Estimated from free memory; user-overridable |
| Hexagon session count | For models > 3.5 GB, layer-split across HTP0..HTP3 |
| Thread count | Default to performance-core count, not total cores |

### 8.2 On-device benchmarking

Do not assume backend performance on this hardware — measure it.

- On first load of a model, offer a **micro-benchmark**: fixed prompt, ~64 tokens, run per available backend
- Record prompt-processing tok/s and generation tok/s per backend, per model
- Auto-select the winner; show the numbers
- Re-runnable from Settings; results visible in the model detail sheet

This converts an unproven assumption (Adreno 829 + OpenCL, Hexagon eligibility on 8 Gen 5) into a measurement, and gives the user a real answer for their device.

### 8.3 Thermal and power

- Read `PowerManager.getCurrentThermalStatus()`; surface the state during generation
- Configurable policy on `THERMAL_STATUS_SEVERE`: continue / reduce threads / downshift backend / pause
- Battery-level guard: warn before starting a long diffusion run below a threshold
- Live tok/s and elapsed time in the generation UI
- Wake-lock only while generating, released on completion

### 8.4 Memory pressure

- Register `ComponentCallbacks2`; unload non-pinned models on `TRIM_MEMORY_RUNNING_LOW` and above
- Catch native allocation failure and surface a real message with the numbers, plus a suggestion (lower context, smaller quant, enable `vae_tiling`)
- Never let an OOM present as a bare crash

---

## 9. Parameter surfacing — the three tiers

Every parameter in §4–7 is reachable. Tiering controls default visibility only; a global "show all parameters" switch collapses the tiers.

| Tier | Contents | Surface |
|---|---|---|
| **Basic** | temp, max tokens, system prompt · steps, CFG, size, seed · voice, speed · language | Inline in the generation screen |
| **Advanced** | top_k, top_p, min_p, repeat penalty, context size, n_gpu_layers · sampler, schedule, strength, clip_skip · VAD, beam size | Expandable panel |
| **Expert** | Full sampler chain + ordering, mirostat, DRY, XTC, dynatemp, YaRN/RoPE, KV cache types, logit_bias, grammar/JSON schema · SLG, offload flags, quantization type · all whisper thresholds | Separate screen, per-model |

**Rules:**
- Every parameter has an inline explanation (one sentence, plain language) and its valid range
- Every parameter has a visible "reset to default" affordance; show modified-from-default state
- Parameters requiring a model reload are marked and batched — apply once, not per-edit
- Numeric inputs accept both slider and typed entry

**Implementation note:** tier, group, label, help, range, and widget type are all read from the parameter manifest (§16) — this screen is a generic renderer, not a hand-built form. §4–7 document *what the parameters mean*; the manifest is what the app actually reads.

---

## 10. Presets and profiles

- **Named parameter presets** per modality, user-creatable — e.g. "Deterministic", "Creative", "Balanced" for text; "Fast draft" / "Quality" for diffusion
- **Per-model default preset**, applied automatically on load
- **Import/export presets as JSON** — shareable
- Built-in starting presets, all editable and deletable (nothing locked)
- **Personas** (text): name, avatar, system prompt, default model, default preset, default TTS voice, persistent memory notes

---

## 11. Data model

```
Model(id, hfRepo, revision, localPath, format, architecture, quant,
      sizeBytes, sha256, modality, contextLength, chatTemplate,
      companionPaths[], installedAt, lastUsedAt, pinned, notes)

BenchmarkResult(modelId, backend, promptTokPerSec, genTokPerSec, measuredAt)

Preset(id, modality, name, paramsJson, isBuiltIn)

Conversation(id, title, modelId, personaId, systemPrompt, presetId,
             createdAt, updatedAt)

Message(id, conversationId, role, content, images[], toolCalls,
        tokenCount, generationParamsJson, createdAt)

GeneratedImage(id, path, prompt, negativePrompt, paramsJson, modelId,
               seed, createdAt)

Transcript(id, sourcePath, segmentsJson, modelId, paramsJson, createdAt)

DownloadJob(id, modelId, files[], bytesDone, bytesTotal, state, error)

RuntimeBundle(engine, buildTag, upstreamCommit, jniContract, installedAt,
              sizeBytes, state, previousBuildTag)

ParamManifest(version, source /* bundled|ota */, fetchedAt, signatureOk, json)
```

Store the **full parameter set** alongside every generated artifact (message, image, transcript). Reproducibility is a feature, and it costs almost nothing at write time.

**Preset forward-compatibility:** presets and per-model overrides are stored as sparse key→value JSON, *not* as typed columns. A key unknown to the current manifest is preserved inert and re-activates if a later runtime supports it. Never drop unknown keys on read — that silently destroys a user's settings across an engine downgrade or rollback.

---

## 12. Screens

| Screen | Contents |
|---|---|
| **Chat** | Message list, streaming output, image attach, tool-call rendering, thinking blocks collapsed, read-aloud, regenerate, edit-and-resend, branch, token counter, live tok/s |
| **Chat settings** | Model picker, persona, system prompt, preset, sampling tiers |
| **Image** | Prompt + negative, mode tabs (txt2img / img2img / inpaint), parameter panel, live preview, cancel, result actions |
| **Mask editor** | Brush, size, hardness, erase, invert, clear, undo/redo |
| **Gallery** | Grid, detail view with full parameters, reuse-parameters, share, delete |
| **Transcribe** | Live mic mode + file mode, partial transcript, segment list, export |
| **Models** | Installed library, storage usage, per-model settings + benchmarks, import local file |
| **Add model** | **Paste HF ID/URL field (primary)**, HF search with compatibility badges, quant variant picker with size/speed, download queue |
| **Model detail** | Metadata, compatibility verdict with reasoning, quant selector, companions, benchmark results, parameter overrides |
| **Settings** | Backend config, thermal policy, storage location, HF token, runtime versions + arch counts, network policy, theme, export/import |

---

## 13. Non-functional requirements

- **Offline-first.** After download, zero network required. Network is used only for the HF API and downloads
- **No telemetry.** No analytics, no crash reporting that transmits content
- **No account.** No login of any kind. HF token is optional and only for gated repos
- All model files user-accessible; nothing in a private opaque store
- Generation must be cancellable at every stage, and cancellation must free native memory
- App must survive: model load failure, OOM, corrupt GGUF, backend init failure, storage full — each with a specific message

---

## 14. Build order

Sequenced by risk — get the proven path working first, discover the unproven one last, with the app already standing.

1. **Skeleton** — Compose shell, Room, DI, foreground service, navigation
2. **Shared-ggml CMake** — one ggml target, NDK config, arm64-v8a, JNI scaffolding
3. **llama.cpp + OpenCL** — best-supported path on Adreno; derisks the whole native layout
4. **Manifest codegen + generic JNI map (§16.2, §16.7)** — do this *before* any parameter UI exists. Building forms first and retrofitting the manifest is the expensive mistake; the generic string-keyed JNI boundary must exist from the first native call
5. **Manifest-driven renderer (§16.4)** — one composable per type; validates the whole approach against llama.cpp's ~60 parameters
6. **Resolver + downloader** — HF API, compatibility gate, download service. Test against a wide spread of real repos
7. **Chat** — streaming, templates from metadata, prompt inspector
8. **whisper.cpp** — same ggml build, near-free once §2 is right. Second proof that manifest codegen generalizes
9. **Runtime bundle packaging + in-app updater (§17)** — needs ≥2 engines to be meaningful
10. **Kokoro TTS** — evaluate sherpa-onnx first
11. **stable-diffusion.cpp** — highest chance of Adreno trouble; needs a CPU fallback path from the start
12. **CI upstream tracking + contract tests (§17.5, §17.6)**
13. **Hexagon NPU** — optional v2. Requires Qualcomm-account-gated Hexagon SDK 6.x; will complicate CI
14. **Benchmarking, thermal policy, presets, gallery, polish**

**Steps 4–5 are load-bearing and easy to skip under delivery pressure.** Every parameter form written before the manifest exists is a form that must be deleted later.

---

## 15. Known risks

| Risk | Impact | Mitigation |
|---|---|---|
| sd.cpp on Adreno 829 immature | Diffusion slow or broken | CPU fallback from day one; benchmark early; don't block release on GPU diffusion |
| Adreno 829 Vulkan drivers immature | — | Use OpenCL, not Vulkan. (Corroborated by this GPU needing a custom Turnip driver for other Vulkan workloads) |
| Hexagon SDK is account-gated | CI complexity | Make NPU support v2; keep the build green without it |
| Kokoro G2P (espeak-ng/misaki) fiddly | TTS slips | Evaluate sherpa-onnx first; it packages G2P |
| HF `gguf` API block is undocumented | Resolver breaks silently | Maintain the GGUF header Range-parser as a real fallback; contract-test both paths |
| Arch allowlist rots | Silent model-locking | Generate from upstream source at build time |
| 3.5 GB Hexagon session cap | Large models fail on NPU | Layer-split across sessions, or fall back to OpenCL |
| Parameter defaults drift upstream | Wrong defaults shipped | Pin runtime versions; contract-test defaults against the pinned build (§17.6) |
| **Upstream refactors its arg parser** | Manifest codegen breaks | Codegen is a CI gate, not a runtime dependency — a parse failure blocks the version bump and opens an issue; the shipped app keeps working on the last good manifest |
| **W^X blocks `.so` hot-swap** | Naive runtime updating impossible | **Verified constraint.** Package-manager delivery only (§17.2). Do not attempt `System.load()` from app data dir |
| **Play policy bans native code download** | Play build can't self-update engines | Two channels (§17.2); Play build uses Feature Delivery and degrades the updater to a store link |
| Bad upstream commit ships | App fails to init an engine | Keep previous bundle; auto-rollback after two consecutive init failures (§17.8) |
| Engine update invalidates saved presets | Users lose tuning | Sparse JSON storage; unknown keys preserved inert, never dropped (§11) |
| OTA manifest signing key compromised | Malicious manifest | Manifest is inert data — worst case is bad ranges/labels, not code execution. Ed25519 pinned key; reject on failure |
| APK-install updater abused as a vector | Malware sideload | Verify APK signature against the pinned signing cert *before* invoking `PackageInstaller`; never install an unverified package |

---

## 16. Parameter manifest system — automatic support for new parameters

This section is what makes §1.5 real. It replaces hand-written parameter UI entirely.

### 16.1 The manifest

A single JSON document describes every parameter of every runtime. The app ships one in `assets/` as a baseline and can fetch newer ones OTA (§16.5).

```jsonc
{
  "manifestVersion": 7,
  "generatedAt": "2026-07-29T00:00:00Z",
  "runtimes": {
    "llama.cpp": {
      "sourceCommit": "a1b2c3d",
      "buildTag": "b6xxx",
      "jniContract": 3,
      "params": [
        {
          "key": "top_n_sigma",
          "group": "sampling",
          "type": "float",
          "default": -1.0,
          "min": -1.0,
          "max": 10.0,
          "step": 0.1,
          "tier": "expert",
          "label": "Top-N sigma",
          "help": "Sigma-based logit truncation. -1 disables.",
          "requiresReload": false,
          "sinceBuild": "b4356",
          "untilBuild": null,
          "dependsOn": null,
          "appliesTo": { "modality": ["text"], "arch": null },
          "cliFlag": "--top-nsigma"
        },
        {
          "key": "cache_type_k",
          "group": "model",
          "type": "enum",
          "values": ["f32","f16","q8_0","q5_1","q5_0","q4_1","q4_0"],
          "default": "f16",
          "tier": "expert",
          "requiresReload": true,
          "sinceBuild": "b2000"
        },
        {
          "key": "yarn_ext_factor",
          "group": "model",
          "type": "float",
          "default": -1.0,
          "tier": "expert",
          "requiresReload": true,
          "dependsOn": { "key": "rope_scaling_type", "equals": "yarn" }
        }
      ]
    }
  }
}
```

**Field semantics:**

| Field | Purpose |
|---|---|
| `key` | Canonical name; the wire key passed through JNI |
| `type` | `int`, `float`, `bool`, `enum`, `string`, `string[]`, `int[]`, `map`, `path`, `text` (multiline) |
| `default`, `min`, `max`, `step`, `values` | Drive widget selection and validation |
| `tier` | `basic` / `advanced` / `expert` → §9 surfacing |
| `group` | Section heading (`model`, `sampling`, `generation`, `vision`, `vad`, …) |
| `label`, `help` | Extracted from upstream help text; overridable by a curated overlay (§16.3) |
| `requiresReload` | Batches into a single reload rather than applying per-edit |
| `sinceBuild` / `untilBuild` | **Version gating** — hidden unless the loaded runtime is in range |
| `dependsOn` | Conditional visibility (e.g. YaRN params only when `rope_scaling_type = yarn`) |
| `appliesTo` | Restrict by modality and/or model architecture |
| `cliFlag` | Provenance; also drives the raw-args escape hatch |

### 16.2 Generation from upstream source

The manifest is **generated, never hand-written**. A CI job parses the pinned upstream sources:

| Runtime | Source of truth | Extract |
|---|---|---|
| llama.cpp | `common/arg.cpp` (arg registration table), `common/common.h` (`common_params` struct), `include/llama.h` (`llama_model_params`, `llama_context_params`, sampler chain API) | flag, key, type, default, help text |
| whisper.cpp | `examples/main/main.cpp` arg parser, `include/whisper.h` (`whisper_full_params`) | same |
| stable-diffusion.cpp | `examples/cli/main.cpp` arg parser, `stable-diffusion.h` | same |
| Kokoro / ORT | Model card + pipeline config; smallest surface, may need a curated overlay | voices, lang codes |

Defaults are cross-checked against the struct initializers, not only the help text — help strings drift from actual defaults.

**Pipeline:**
```
upstream repo tag → parse args/structs → raw manifest
                 → merge curated overlay (§16.3)
                 → validate against JSON Schema
                 → diff vs previous manifest
                 → emit report: N added, M removed, K changed defaults
                 → sign + publish
```

The diff report is the point. When llama.cpp adds a sampler, CI says so, and it lands in the app's Expert tier with upstream's own help text — no UI work.

### 16.3 Curated overlay

Generated help text is terse and sometimes cryptic. A small hand-maintained overlay file keyed by `runtime.paramKey` may override `label`, `help`, `tier`, `group`, `min`/`max`/`step`, and add `dependsOn`.

**The overlay may never add a parameter, only annotate one.** If a param is absent from the generated manifest, it does not exist in this build — that invariant is what keeps the app honest about what the native lib supports. CI fails if the overlay references an unknown key (catches upstream renames immediately).

### 16.4 Rendering

```kotlin
interface ParamRenderer { fun render(spec: ParamSpec, value: Any?, onChange: (Any?) -> Unit) }
```

One composable per `type`. Selection is table-driven off `spec.type`:

| Type | Widget |
|---|---|
| `float` / `int` with min+max | Slider + typed entry, `step`-snapped |
| `float` / `int` unbounded | Typed entry with validation |
| `bool` | Switch |
| `enum` | Segmented control (≤4) or dropdown |
| `string` | Single-line field |
| `text` | Multiline field |
| `string[]` | Chip editor (stop sequences, DRY breakers) |
| `int[]` | Comma-separated with validation |
| `map` | Key-value editor (logit_bias — with a token picker) |
| `path` | SAF file picker filtered by extension |
| ordered list | Drag-reorder (sampler chain) |

Adding a *new type* is the only case that requires app code. Adding a new *parameter* of an existing type requires none.

Every rendered control shows: current value, default marker when modified, inline `help`, and a reset affordance.

### 16.5 Manifest delivery and OTA update

The manifest is **data, not executable code** — it is unaffected by both the W^X restriction and Play's code-download policy. It can be updated freely at any time.

- Baseline manifest ships in `assets/params-manifest.json`
- App checks a well-known URL for a newer `manifestVersion` on a configurable schedule (default: weekly, Wi-Fi only, user-disableable)
- Downloaded manifests are **Ed25519-signed**; public key pinned in the APK; signature failure = reject and keep the current one
- App uses `max(bundled, downloaded)` by version
- Version gating (§16.1 `sinceBuild`/`untilBuild`) filters against the **actually loaded** runtime build tag, queried from the native lib at init

**Be precise about what this buys.** An OTA manifest can correct metadata, retier, improve help text, and *reveal* parameters the bundled `.so` already supports but the shipped manifest omitted. It cannot add capability the native lib lacks — that needs a runtime update (§17). The `sinceBuild` gate makes this automatic and invisible: a param for a newer build simply stays hidden until the runtime catches up, then appears on its own.

### 16.6 Raw parameter passthrough — the escape hatch

Regardless of the manifest, an Expert-tier **raw parameters** editor accepts arbitrary JSON passed straight to the runtime:

```json
{ "some_new_upstream_flag": 0.7 }
```

Unknown keys are forwarded to the native layer and ignored there if unrecognized (never fatal). This guarantees that a parameter existing in the loaded `.so` is *always* reachable, even if manifest generation missed it. Warn that these are unvalidated; log what the runtime accepted or rejected.

### 16.7 JNI wire format

The JNI boundary passes a **generic string-keyed map**, not a fixed struct:

```kotlin
external fun llamaSetParams(ctxHandle: Long, paramsJson: String): String  // returns applied/rejected report
```

Native side maps keys onto `common_params` fields and the sampler chain via a generated dispatch table (same codegen run as the manifest). **The JNI signature does not change when a parameter is added** — this is what allows §16 to work without touching Kotlin or the JNI header. It also means the native side must report unknown keys rather than crash.

---

## 17. Runtime update system

### 17.1 Constraints (verified, non-negotiable)

| Constraint | Consequence |
|---|---|
| Android W^X / dynamic-code-loading enforcement rejects `System.load()` on writable-storage files (hard-enforced Android 14+, higher targetSdk) | **No downloaded-`.so` dlopen.** Native code must arrive read-only via the package manager |
| Play policy: no downloading executable code, incl. native code, from non-Play sources | Play-channel builds cannot self-update native libs at all |
| llama.cpp's C API changes between releases | Native bundle and JNI shim must ship and version **together** |

The design consequence: **the updatable unit is not a `.so` file — it is a signed package containing `libggml + lib<engine> + lib<engine>_jni` built from one upstream commit**, delivered through a package-manager path.

### 17.2 Two distribution channels

Build both from the same source with a build flavor.

**Channel A — sideload / F-Droid / GitHub Releases (recommended primary):**

- In-app updater polls a signed `releases.json`
- Shows: upstream version bumps per engine, changelog, new parameter count from the manifest diff, download size
- Downloads a **split APK** or full APK, verifies signature, installs via `PackageInstaller` (`REQUEST_INSTALL_PACKAGES`)
- Installation is user-confirmed; app data and models are preserved
- This is "on the go" in the way that matters: one tap in-app, no store, tracking upstream within days

**Channel B — Google Play:**

- Native code delivered as **Play Feature Delivery** dynamic modules (`SplitInstallManager`) — Play-sanctioned, installed read-only into the app's lib dir, W^X-compliant
- On-demand per engine, which also solves APK size (see §17.4)
- Updates still require a Play release; the in-app updater degrades to "update available in Play"

The updater component is channel-aware and must be feature-flagged, not `#ifdef`-scattered.

### 17.3 Runtime bundle structure and JNI contract

```
runtime-llama-b6xxx-arm64.bundle
  ├── libggml.so, libggml-cpu.so, libggml-opencl.so
  ├── libllama.so
  ├── libllama_jni.so          # built against the same commit
  └── bundle.json              # { engine, upstreamCommit, buildTag,
                               #   jniContract: 3, abi, sha256, archs[] }
```

- The APK contains a **thin, stable loader** plus a **baseline bundle** for every engine, so the app always works offline with no updates applied
- `jniContract` is an integer the Kotlin side hard-requires. A bundle declaring an unsupported contract is **refused with a clear message**, never loaded optimistically
- Contract bumps happen only when the Kotlin↔JNI signature changes — which §16.7's generic map makes rare
- Each engine updates independently: llama.cpp can move while sd.cpp stays pinned

### 17.4 Engine modularity

Engines are separately installable. Diffusion weights and code dwarf the rest, and many users never generate images.

- First run installs text (llama.cpp) only
- Vision, STT, TTS, diffusion install on demand, each with its size shown
- Uninstalling an engine removes its bundle and offers to remove its models
- Settings shows per-engine: installed version, upstream latest, update button, size on disk

### 17.5 CI: upstream tracking

```
weekly (or on upstream release):
  for engine in [llama.cpp, whisper.cpp, sd.cpp]:
    fetch latest upstream tag
    if newer than pinned:
      build arm64-v8a bundle (shared ggml, OpenCL enabled)
      regenerate param manifest (§16.2)
      diff manifest → "3 params added, 1 default changed"
      regenerate architecture allowlist from source
      run contract tests (§17.6)
      if green: bump pin, sign bundle + manifest, publish release
      if red:   open an issue with the failing contract, do not publish
```

This closes the loop: **new upstream parameters reach users' Expert tier without a human writing UI code**, and a new architecture becomes loadable without touching the resolver.

### 17.6 Contract tests (gate every runtime bump)

1. **Manifest ↔ native agreement** — every manifest key is accepted by the built `.so`; no manifest key is rejected as unknown
2. **Defaults match** — each manifest default equals the value the native struct initializes to
3. **Removed-parameter detection** — a key present in the previous manifest but gone upstream must be marked `untilBuild`, never silently dropped (user presets reference it)
4. **JNI contract stability** — signature hash unchanged, or `jniContract` bumped
5. **Architecture allowlist** — regenerated list is a superset of the previous, or the removal is explicit
6. **Smoke inference** — load a tiny GGUF, generate 16 tokens, on every backend
7. **Preset migration** — presets saved under the old manifest still load; unknown keys are preserved-but-inert, never dropped

### 17.7 User-visible surface

Settings → Runtimes:

```
llama.cpp    b6xxx  ·  41 architectures  ·  OpenCL, CPU     [ Update → b6yyy ]
whisper.cpp  v1.7.x ·  installed                            [ Up to date ]
sd.cpp       —      ·  not installed  (412 MB)              [ Install ]
kokoro       v0.19  ·  installed                            [ Up to date ]

Parameter manifest  v7  (bundled v6)          Last checked: 2h ago
Auto-check for updates  [ Wi-Fi only ▾ ]
```

Update notes must state what actually changed — engine version, new parameters, new architectures — not a generic "improvements and bug fixes."

### 17.8 Rollback

Keep the previous bundle for each engine. If a runtime fails to initialize twice consecutively, automatically revert to the prior bundle and surface what happened. A bad upstream commit must not brick the app.

---

## 18. Proxy — serving the device's models over HTTP

A laptop, a browser or a coding client talks to this device using the API it
already speaks, over the tailnet, with no cloud in the path. Off by default.

Implemented in `proxy/`, hosted by `InferenceService`, and designed in
[`docs/proxy-plan.md`](docs/proxy-plan.md).

### 18.1 Surface

Two protocols, five modalities. The gaps are part of the contract.

| Capability | OpenAI | Anthropic | Engine |
|---|---|---|---|
| Chat, vision, tools | `POST /v1/chat/completions` | `POST /v1/messages` | llama.cpp |
| Token count | — | `POST /v1/messages/count_tokens` | llama.cpp |
| Image generate / edit | `POST /v1/images/generations`, `/edits` | server tool | sd.cpp |
| Upscale | `POST /v1/images/upscales` *(extension)* | server tool | sd.cpp |
| Video | `POST /v1/videos` → job, poll, content, cancel | server tool | sd.cpp |
| Speech | `POST /v1/audio/speech` | server tool | Kokoro / OmniVoice |
| Transcribe / translate | `POST /v1/audio/transcriptions`, `/translations` | server tool | whisper.cpp |
| Models, health | `GET /v1/models`, `/v1/models/{id}`, `/health` | same, by header sniff | — |
| Certificate | `GET /certificate` — the PEM, unauthenticated | same | — |
| Embeddings | `501`, naming the reason | — | — |

**The Anthropic Messages API is chat-only.** There is no `/v1/images` in that
protocol and inventing one would produce something no client speaks, so media
reaches it as server-side tools the proxy intercepts and runs itself.

**Embeddings return 501, not 404.** There is no `llama_get_embeddings` path
through the JNI boundary; adding one is a native change and a runtime contract
bump. §1.2 requires saying which.

**Video is a job, never a held connection.** A clip is tens of minutes on this
hardware — longer than any client's idle timeout and longer than a phone stays
on one access point.

### 18.2 Reachability

- Binds to this device's `100.64.0.0/10` Tailscale address by default, and to
  nothing at all when the tailnet is down. Falling back to `0.0.0.0` because a
  VPN was off would put a generation server on whatever Wi-Fi the phone is on.
- **Tailscale Funnel is not available on Android.** Funnel is a CLI feature and
  the Android app ships no CLI, so there is no public HTTPS address and no
  setting that would produce one. The screen says so rather than implying a
  missing toggle.
- **TLS is off by default and available.** Tailnet traffic is already encrypted
  end to end, so plain HTTP over the tailnet is not the hole it looks like — but
  a client that will only speak `https://` is a real client, and refusing it was
  refusing the whole feature. `proxy.tls` serves TLS with a certificate this
  device signs for itself: there is no authority that will issue one for an
  address only a tailnet can reach, and Tailscale's own `tailscale cert` needs
  the CLI the Android app does not ship. The certificate carries a real SAN list
  (bind address, every local address, the MagicDNS name where one resolves,
  loopback) and `CA:TRUE`, so a client can be *given* it — `curl --cacert`,
  `NODE_EXTRA_CA_CERTS` — rather than only told to skip the check. The screen
  shows its SHA-256 so the reader can tell they were handed the right one.
  Ktor's CIO engine cannot terminate TLS, so `TlsFront` does, in front of a
  plaintext server bound to loopback on a port nobody is told.
- **Nobody copies a certificate by hand.** The Proxy screen sends it through the
  Android share sheet — Tailscale's app exports a Taildrop target, so it lands in
  the other machine's Downloads — and the server serves it at `GET /certificate`,
  unauthenticated, because it is a public key and requiring the token to fetch
  the thing you need before you can connect is a lock with its key inside the
  box. Fetching it with verification off is not the leap it looks like on a
  tailnet: WireGuard has already authenticated the machine that answered, which
  is the same guarantee `tailscale cert` leans on.
- **`tailscale cert` is not available to us, for the same reason Funnel is not.**
  It needs tailscaled's LocalAPI or the CLI; the Android app exports exactly
  three components — `MainActivity`, `ShareActivity` (Taildrop) and `IPNReceiver`
  (connect/disconnect) — and none of them will issue a certificate.
- The MagicDNS name is found by reverse lookup where MagicDNS is on, and is
  never guessed. The raw address always works.

### 18.3 Access

- A bearer token, generated on first enable and **required by default**,
  accepted as `Authorization: Bearer` or `x-api-key` — the two protocols send
  different headers and a server speaking both must accept both. Compared in
  constant time. Stored in the Keystore beside the Hugging Face token.
- CORS origins default to empty: no browser page may call the server until one
  is typed in.
- Refuses generation below a battery floor, and optionally unless charging. A
  server that quietly gets slower as the phone throttles is the failure §1.2
  exists to forbid.

### 18.4 Residency

One run at a time, across every engine, arbitrated by `engine/ModelRunner.kt` —
which also owns the only cross-runtime unload in the app. Concurrency is **not**
a setting: the diffusion engine holds a load lock, so a number the engines
ignore would be a knob that does nothing.

The gate is **re-entrant**. A media tool is a `ToolProvider` like any other, so
a chat turn calling `generate_image` re-enters it; a plain mutex there is a
deadlock. Re-entering makes room for the incoming runtime, which is why
`ChatPipeline` re-loads the text model before every round rather than once.

`proxy.model_policy` decides what a request for another model does while one is
running: `queue` (default), `refuse` with the resident model named, or `swap`.

### 18.5 Configuration

Every setting is a `ParamSpec` in `proxy/ProxySpecs.kt`, rendered by `ParamRow`.
§1.5 and §9 apply unchanged: **no proxy setting may have a hand-written
widget.** The three list-shaped things — model aliases, CORS origins, client
profiles — are stored as JSON in the same DataStore key. No Room migration.

Stored sparsely, so a default that moves in a later release moves for everyone
who never touched that row.

**A setting that nothing reads must not exist.** `tool_choice` is refused rather
than accepted and ignored, because the runtime cannot constrain the model to a
tool and pretending otherwise returns a well-formed reply that quietly did not
do what was asked.

---

## Appendix A — implementation notes for the coding agent

1. **Do not vendor ggml more than once.** Three copies is the single most likely structural mistake.
2. **Do not hardcode chat templates.** Read `gguf.chat_template`. If you find yourself writing a `when (modelName)` branch for prompt formatting, the design has been violated.
3. **Do not hardcode an architecture allowlist.** Generate it from the pinned llama.cpp source at build time.
4. **Verify every default in §4–7** against the pinned upstream build before wiring the UI. They drift.
5. **Compatibility verdict before download, always.** No path may reach a native load without having passed the gate.
6. **Store full parameters with every artifact.** Cheap at write time, impossible to reconstruct later.
7. **Cancellation must free native memory**, not just detach a callback. Verify with a heap profile.
8. **The paste-an-HF-ID field is the primary affordance** on the Add Model screen. Curated lists are convenience shortcuts and must never be the only path.
9. **Never write a parameter widget.** If you are adding a `Slider` for a specific named parameter, stop — add it to the manifest instead. The only legitimate reason to touch renderer code is introducing a new *type* (§16.4).
10. **The JNI parameter boundary is a string-keyed map, from the very first native call** (§16.7). A typed struct across JNI means every upstream parameter addition becomes a code change, which defeats §1.5. Native side reports unknown keys; it never crashes on them.
11. **Never `System.load()` from app data directory.** W^X enforcement rejects it and it is a policy violation. Native code arrives only via the package manager (§17.2).
12. **Never drop unknown keys** when reading presets, per-model overrides, or stored generation params. Preserve inert.
13. **Manifest codegen runs in CI, not on device.** The app only ever consumes a signed manifest; it never parses upstream C++.
14. **Verify signatures before installing anything executable.** APK signature against the pinned cert; Ed25519 on manifests and bundles. This is the one place where a shortcut becomes a remote code execution vector.
15. **Nothing generates outside `ModelRunner`.** Orchestration in a view model is unreachable from a socket, an alarm or a share sheet, and copying it per surface is how a parameter comes to be applied in three places and forgotten in the fourth (§18.4).
16. **The proxy is a server, so it must refuse well.** Every route answers with the engine's own suggestion beside its message, and every switched-off surface says it is switched off rather than 404ing like a typo.
