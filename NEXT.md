# Handoff

Written at commit `0627edd` + uncommitted work. 128 unit tests green, native build clean,
both APKs built, `app-sideload-debug.apk` installed on the device.
**Nothing here has generated a picture on hardware.**

## Do not

- **Never run a Gradle install task against the device** — `connectedAndroidTest`,
  `installSideloadDebug`, anything that installs. AGP resolves a signature or
  version conflict by uninstalling, and an uninstall of this app deletes
  `/storage/emulated/0/Android/data/ai.ondevice/`, which is where every model
  lives. That is how ~24 GB was lost on 2026-08-02. Use `adb install -r` only.
- Never write a starter card for a repo you have not fetched from
  `https://huggingface.co/api/models/<id>`. A card that 404s is worse than no card.

## Done since `0627edd`, uncommitted

### `wtype` through the loader

`nativeLoad` takes a 12th argument, a ggml type name; `DiffusionEngine.load` reads it
from the `type` key, which the manifest has described since it was written and which
nothing reported, so the row had never once been shown. Default `as-is` keeps the file's
own precision — the old manifest default was `f16`, which would have quietly converted
every checkpoint on the device the moment the row appeared.

- the manifest offered `q3_k` and `q2_k`; ggml spells them `q3_K` and `q2_K` and matches
  case-sensitively, so both values would have failed. `WeightTypeTest` now asserts every
  offered value against ggml's spelling.
- `parse_wtype` refuses a type this build cannot *write*. That gate is not decoration:
  `ggml_quantize_chunk` falls through to `GGML_ABORT` for q8_1, q8_K and the integer
  types, which ends the process minutes into a load with nothing in the crash buffer.
- **unmeasured**: conversion costs load time and a working buffer. Measure before
  trusting the claim that 6 GB of fp16 becomes ~2 GB at q4_0 without also becoming
  unusable.

### All twelve tokenizer vocabularies are back

`sd_vocab.cpp` is deleted and the CMake source-swap with it; upstream's `vocab.cpp`
compiles as shipped.

**The size table in the last handoff was wrong.** It measured a debug APK — two ABIs, no
minification. Measured today:

| | with stubs | all twelve |
|---|---|---|
| `assemblePlayRelease` | 53.6 MB | **96.5 MB** |
| `assembleSideloadDebug` | 94.8 MB | 137.7 MB |

Against a 150 MB Play ceiling, so the ABI question the old table raised does not arise —
release has been arm64-only since the 63 MB accident was fixed, and there was never a
reason to trade a vocabulary for headroom that was never tight.

What the stubs cost is the part worth keeping in mind: `load_umt5_tokenizer_json` and the
Gemma pair returned an empty string, sd.cpp built a tokenizer with no merges from it, and
a model whose weights loaded correctly encoded its prompt to nonsense. Wan, LTX-AV, Lens,
PiD and Ernie Image were not unsupported — they were supported and silently wrong.

**Task #79 was not closed.** The task list in this session is empty, so there was nothing
named #79 to find. It points the opposite way to this commit and someone with access to
that list should decline it.

### Four more starter bundles

`StarterModels.BUNDLES` covers 8 architectures, and every remaining one has been surveyed
— see "Remaining starter bundles" below. Every repo, filename and byte count was fetched
from the HF API today.

- **ERNIE Image turbo** — `unsloth/ERNIE-Image-Turbo-GGUF` Q2_K, 3.18 GB, plus a
  Ministral 3.3B at 2.15 GB and a 168 MB decoder from Baidu's own repo. ~5.5 GB all in,
  which makes it the most mobile-sized of the modern transformers, and it is the best
  here at putting readable text inside a picture.
- **Krea 2 turbo** — `vantagewithai/Krea-2-Turbo-GGUF` Q2_K, 4.89 GB, plus Qwen3-VL-**4B**
  at UD-IQ2_M 1.53 GB and Qwen-Image's decoder at 254 MB. ~6.7 GB. The 4B is not a guess:
  Comfy-Org's repackage ships `qwen3vl_4b`, so that is the size it was trained against.

- **SD 1.5** — `gpustack/stable-diffusion-v1-5-GGUF` Q4_0, 1.75 GB, self-contained. With
  the LCM LoRA, a canny ControlNet, an IP-Adapter and its vision encoder, and
  `sd-vae-ft-mse`. The smallest base in the app and the only one that takes every add-on.
- **FLUX.1 schnell** — `city96/FLUX.1-schnell-gguf` Q2_K, 4.01 GB, plus CLIP-L, T5 at
  Q3_K_S and a VAE: ~6.5 GB resident before it will load. The device has 15.6 GB, so it
  fits there; it will not fit on a modest phone.

Three things deliberately left out, with reasons rather than as oversights:

- **No SD 1.5 style LoRA.** Hugging Face does not have one worth linking — searching by
  downloads and by likes returns person-likeness LoRAs in the tens of downloads. Every
  anime LoRA with traction on HF is SDXL or FLUX. The SD 1.5 ones live on Civitai, which
  this app cannot resolve. The LCM LoRA is offered instead, and it is the more useful one.
- **No Chroma.** `silveroxides/Chroma-GGUF` publishes BF16, Q4_0 and Q8_0 and nothing
  smaller; Q4_0 is 5.43 GB and it needs a 2.10 GB T5 beside it. There is no small option
  to offer, so there is no card.
- **No AnimateDiff motion module.** `guoyww/animatediff` exists and `mm_sd_v15_v2.ckpt` is
  the right file, but `nativeLoad` has no `motion_module_path` and there is no
  `nativeGenerateVideo`. A card for it today is a download that cannot be used. Add it in
  the same commit as the video work below.

### `nativeLoad` now passes the whole of `sd_ctx_params_t`

It took eleven positional strings and grew by one whenever a field turned out to matter.
The cost was not the length of the call — it was that the fields it *did not* take were
invisible. Two architectures were unrunnable because nothing in the app could name a file
they needed, which reads as "unsupported" and was "unplumbed".

Now two JSON objects: components keyed by `AttachmentRole.paramKey`, settings keyed by the
struct's own field names. Same shape `LlamaBridge.nativeLoad` always had. A field added
upstream costs one line on each side.

What that unblocked, none of which hardware could have fixed:

| now passed | was blocking |
|---|---|
| `uncond_diffusion_model_path` | **Ideogram 4** — hard block |
| `high_noise_diffusion_model_path` | **Wan 2.2 I2V / TI2V** — hard block |
| `motion_module_path` | AnimateDiff |
| `audio_vae_path` | LTX-AV's audio track |
| `photo_maker_path`, `pulid_weights_path` | identity-preserving generation |
| `llm_vision_path` | an edit model being shown the picture |
| `embeddings` / `embedding_count` | textual inversions as an array |
| `max_vram`, `stream_layers`, `eager_load`, `auto_fit` | **a model larger than memory** |
| `tensor_type_rules` | per-tensor precision — the decoder can stay f16 while the denoiser goes to q4 |
| `diffusion_flash_attn`, `*_conv_direct`, `force_sdxl_vae_conv_scale` | memory/speed trades |
| `vae_format`, `prediction`, `rng_type`, `sampler_rng_type`, `lora_apply_mode` | correctness knobs |

Seven new roles carry the new paths, in a new `RoleFamily.COMPANION_DENOISER` — deliberately
outside `ADOPTABLE_FAMILIES`, because a second denoiser is authorship and must never be
filled in on the user's behalf. `FileRoles` learned to spot them: `uncond`, `high_noise`,
and AnimateDiff's `mm_sd_*` / `*_mm` naming.

`rng_type` is the one worth knowing about — it decides *what a seed means*. `cuda` is what
reproduces a seed shared from a desktop UI, and it is upstream's default.

`taesd_path` is still deliberately not passed; the reasoning in `sd_jni.cpp` stands.

`LoadContractTest` now fails the build if a load setting is sent and undescribed, if a
described load setting forgets `requiresReload`, if two roles collide on a key, or if any
companion denoiser stops reaching the loader.

### Four bugs found by running it

- **The step readout counted tensors.** `134/1234`, then `1680/1680`, then decoding with a
  stale number. Weights load lazily unless `eager_load` is set, so the tensor loader reports
  its own progress *through the sampler's callback, during sampling* — which the phase check
  cannot filter. They are now told apart by the total: upstream's sampler declares `(0, steps)`
  once, before the first denoise touches a weight, so the first total seen in the phase is
  the sampler's and anything with a different one is dropped. That also fixes what the old
  exact-match check got wrong — an ancestral sampler or img2img has a real total that is not
  the one asked for, and this takes whatever it turns out to be.
- **The "In memory" card vanished for the models most likely to be run.** It listed
  *components*, and the checkpoint was not one, so a self-contained model with nothing
  attached produced an empty list and the card never drew. SDXL turbo is exactly that case:
  3.94 GB resident and nothing on screen. It now leads with the checkpoint and its size,
  totals the weights, and draws an "Unloaded" state carrying the reason — running the
  upscaler drops the denoiser, which was the least explained minutes in the app.
- **Three roles from other runtimes were offered on the image screen.** `mmproj`,
  `vad_model` and `voices` belong to llama, whisper and Kokoro; `AttachmentRole.entries`
  was mapped wholesale into the diffusion key set, so they rendered as empty
  "not described yet" boxes. `isDiffusionAuxiliary` used to return `true` for everything —
  a property that answered no question — and now answers it.
- **`enable_thinking` was an empty text box.** Four keys were reported by a runtime and
  described by nothing: `enable_thinking` and `chat_template_kwargs` (chat),
  `single_segment` and `step_ms` (voice). `DescribedParamsTest` now parses the C++ dispatch
  tables out of source and fails the build on any undescribed key, for all five runtimes.

### A bug the new bundle found

`ComponentCheck` decided "is this a UNet?" against a private list of exact strings —
`sd1`, `sd2`, `sdxl`, `sdxl_refiner`, `svd` — compared with `==`. sd.cpp prints `SD1.x`,
so SD 1.5 was told a ControlNet "does nothing" on it, in a warning worded with complete
confidence, for the one architecture with more ControlNets written for it than every
other one here put together. It now asks `DiffusionFamily.unet`, which is the same table
that already answers the encoder question. `ComponentCheckTest` covers both directions.

## Work, in order

### 1. Device verification — the only thing blocking everything else

`app-sideload-debug.apk` is installed (Motorola Signature, 15.6 GB RAM, arm64-v8a) and
the app launches clean. **The device has no models on it** — the uninstall took all of
them — so nothing further can be checked without a download.

Download **SDXL turbo Q4_0 alone** (3.94 GB, self-contained, one file). That single load
exercises four fixes from `0627edd` and its four predecessors, none of which have run:

- `models.selfContained` is written for the first time
- **nothing should be auto-adopted** — no CLIP-L, CLIP-G, T5 or LLM encoder. If any
  appears attached without being chosen, commit `422e02e` did not take
- the "In memory" card should name only the checkpoint, not the five it used to
- the phase readout should read preparing → sampling → decoding, in that order

Then three measurements, each same prompt and same seed:

- flash attention, `params.flash_attn` in `sd_jni.cpp:675`, true vs false. Currently
  **on and unverified**.
- `type` = `as-is` vs `q4_0` on a model that is *already* q4_0 — should be a no-op
  conversion and is the cheapest way to find out what the conversion path costs.
- `type` on an fp16 safetensors, which is the case the whole setting exists for.

### 2. Remaining starter bundles

**The survey is complete.** `model_version_to_str` in `stable-diffusion.cpp:76` names 46
architectures; every one is accounted for below. "~30 image architectures" was the right
order of magnitude — it is 38 once the 7 video entries and ESRGAN come out.

`StarterBundleTest` fails the build on an invalid pairing, so adding a bundle is safe.
Every repo, filename and byte count below was fetched from the HF API on 2026-08-02.

**Covered — 8 bundles**

SD 1.x · SDXL · SD3.x · Flux · Flux.2 klein · Z-Image · Ernie Image · Krea2

**Variants of a covered family — 16.** They take their parent's parts exactly; only the
base repo differs, so each is a one-line addition whenever a mobile-sized release of that
particular checkpoint turns up. SD 1.x Inpaint, SD 1.x Tiny UNet, Instruct-Pix2Pix,
SD 2.x, SD 2.x Inpaint, SD 2.x Tiny UNet, SDXS (512-DS), SDXS (09), SDXL Inpaint,
SDXL Instruct-Pix2Pix, SDXL (Vega), SDXL (SSD1B), Flux Fill, Flux Control, Flex.2,
Qwen Image Layered.

**Uncovered, with the reason — 14.** None of these is an oversight:

| | why not | what would change it |
|---|---|---|
| **Ideogram 4** | **no longer blocked** — `uncond_diffusion_model_path` is passed as of this commit. `leejet/ideogram-4-GGUF` is Q4_0 5.64 GB plus `ideogram4_uncond-Q4_0.gguf` at the same size, and its README pairs them with Qwen3-VL-8B and `flux2_ae`. ~14.5 GB resident. | write the bundle. Best-documented pairing of the survey; the only question left is whether 14.5 GB runs, which `max_vram` + `stream_layers` now also exist to answer. |
| **Ovis Image** | base and encoder verified — `leejet/Ovis-Image-7B-GGUF` Q4_0 4.20 GB, `Comfy-Org/Ovis-Image` `split_files/text_encoders/ovis_2.5.safetensors` 5.14 GB. **No separate decoder is published anywhere.** | find out whether the Q4_0 carries its own. The loader says so on first load; do not guess from the file size. If it does, this is a bundle today — and the 5.14 GB encoder has no GGUF, so it is also the best test of the weight-type work. |
| **Longcat-Image** | base verified (`stduhpf/LongCat-Image-gguf` Q4_K_M 3.74 GB). Encoder and decoder unresolved — Comfy-Org ships only `split_files/diffusion_models`. | identify its LLM. It falls through the `LLMEmbedder` chain to the `QWEN2_5_VL` default rather than being named, so read `conditioner.hpp:2425` before trusting that. |
| **SeFi-Image** | base verified (`Sam2x/SeFi-Image-5B-turbo-GGUF` Q4_1 3.18 GB, 270 downloads). Encoder is Qwen3-VL; decoder unresolved — `SeFi-Image/SeFi-Image-SemVAE` holds a DINOv2 and nothing else. | a decoder. Promising size otherwise. |
| **Mage Flow** | `gguf-org/mageflow-gguf` has only `mageflow-edit-turbo-nvfp4.gguf` 2.37 GB — the *edit* variant, in NVFP4. Its VAE is in the same repo at 346 MB. | confirm sd.cpp reads NVFP4 GGUF weights off disk. `SD_TYPE_NVFP4` exists; whether the loader takes it as a source type is untested. |
| **Boogu Image** | smallest published quant is Q4_0 **6.78 GB**, and it needs a Qwen3-VL beside it. Over budget with nothing smaller offered. | a Q2/Q3 release. |
| **Lens** | conditions through **GPT-OSS 20B** (`conditioner.hpp:1801`). The encoder alone is larger than any bundle here. | nothing realistic. |
| **Anima** | no release findable on HF under that name — searching returns only Qwen3 text-encoder packs that mention it. | a base checkpoint existing. |
| **PiD** | same: nothing on HF. A shame, because it reads through **Gemma2 2B**, the second-smallest encoder of any architecture the runtime supports. | a base checkpoint existing. |
| **MiniT2I** | `MiniT2I/MiniT2I` and friends are JAX research checkpoints. No GGUF, no safetensors release. | a usable release. |
| **HiDream O1** | `city96`'s HiDream GGUFs are **I1, not O1** — a different architecture the runtime does not list. `Comfy-Org/HiDream-O1-Image` exists and was not sized. | survey that one repo. The only genuinely unfinished row here. |
| **Chroma Radiance** | `silveroxides/Chroma-GGUF` publishes BF16, Q4_0 and Q8_0 only; Q4_0 is 5.43 GB and it wants a 2.10 GB T5. | a quant below Q4_0. |
| **Qwen Image** | 10.1 GB minimum, verified earlier. | nothing. |
| **Flux.2** (dev) | 32B, and its Mistral Small 3.2 encoder is 24B. | nothing. |

Two things the survey turned up that are worth keeping:

- **`leejet` publishes GGUFs for this exact runtime** — Z-Image, both Klein sizes, Ovis,
  Ideogram 4, FLUX.1. Check that account first for any new architecture; the conversions
  are made against the loader we ship.
- **Encoder size is the binding constraint, not denoiser size.** Ernie is in the app
  because Ministral 3.3B is 2.15 GB; Lens is out because GPT-OSS 20B is not. When
  surveying a new architecture, read its branch of `LLMEmbedder`'s arch chain
  (`conditioner.hpp:1795`) before pricing the denoiser.

### 3. Video — read this before designing a screen

The screen is the last problem. In order:

1. The load gate rejects every video architecture — `sd_ctx_supports_image_generation` is
   literally `!supports_video_generation`. This must become mode-aware first, and it is
   now the *only* thing in the load path standing in video's way: the vocabularies are
   back, and `high_noise_diffusion_model_path`, `motion_module_path` and `audio_vae_path`
   are all passed as of this commit.
2. Wan and LTX-AV can tokenise now. What they cannot do is fit: Gemma 3 12B for LTX-AV.
   `max_vram` + `stream_layers` are the first honest attempt at that and are now reachable.
3. **SVD is listed as supported and is not.** No branch in
   `prepare_video_generation_latents`, no image conditioning, no `motion_bucket` /
   `cond_aug` / `fps_id` anywhere in the source. It would run and produce garbage.
   Remove it from the video list rather than repeat a claim upstream does not honour.
4. Hunyuan and LingBot clear the tokenizer bar but need 4–7 B text encoders on top
   of the denoiser, CPU-only, no flash attention, ~56k attention tokens per layer
   against ~4k for a 512² still.
5. Memory: a 5 s 480p clip is ~147 MB of raw RGB held three times over. Frames must
   go to disk with a manifest returned — not a `byte[]`.

**AnimateDiff is the one path that works today.** SD 1.5 base, CLIP tokenizer, passes the
load gate untouched, frames capped at 32, ~3.5 MB for 8 frames at 384². Needs
`motion_module_path` on `nativeLoad` and a `nativeGenerateVideo`. The SD 1.5 bundle above
is now in the app, so the base and its LoRAs are one tap away. It also builds the
multi-frame state model that Qwen-Image-Layered needs.

## Known open, smaller

- **IP-Adapter on a non-UNet loads and does nothing.** Measured: it loaded fine on
  SD 3.5. `ComponentCheck` warns beforehand, but that is the app's prediction, not
  the runtime's confirmation — and it was wrong about SD1.x until today, which is
  the argument for reading a count off the runtime instead. Look for a count sd.cpp
  reports when matching injection sites, the way `note_lora` reads the LoRA tally.
- **Sampling parameters are offered on every architecture** with nothing saying
  `cfg_scale` is inert on a distilled model or `flow_shift` inert on SDXL. Same
  class of silent no-op. Would need the family table to say which knobs each family
  honours, derived from upstream's sampler dispatch — not typed by hand.
- **The downloader may misfile an LLM text encoder as a chat model.** Unchecked.
  `FileRoles` classifies by folder and filename token; a Qwen3 GGUF encoder looks
  exactly like a chat model.
- **FLUX.2 rows of the component matrix were never captured** — logcat rotated.
