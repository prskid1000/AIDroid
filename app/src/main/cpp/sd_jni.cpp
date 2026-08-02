// stable-diffusion.cpp behind the string-keyed contract (SPEC §16.7).

#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <android/log.h>

#include "stable-diffusion.h"
#include "nlohmann/json.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO_STDLIB_HEADERS
#include "stb_image_write.h"

#include "jni_util.h"


/** The fallback thread count: every core but one. */
static int od_default_threads() {
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
}

using json = nlohmann::ordered_json;

/** JSON on its way out to Kotlin, with invalid UTF-8 replaced rather than thrown at. */
static std::string dump_json(const json & value) {
    return value.dump(-1, ' ', false, json::error_handler_t::replace);
}

#define SLOGI(...) __android_log_print(ANDROID_LOG_INFO, "ondevice.sd", __VA_ARGS__)
#define SLOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ondevice.sd", __VA_ARGS__)

namespace {

struct od_sd {
    sd_ctx_t * ctx = nullptr;
    std::mutex mutex;

    // Strings the params struct points at; they must outlive the call.
    std::string prompt;
    std::string negative_prompt;
    std::string sampling_method = "euler_a";
    std::string schedule        = "discrete";

    int   steps       = 20;
    float cfg_scale   = 7.0f;
    float guidance    = 3.5f;

    /**
     * Three settings whose real default is "whatever this architecture uses",
     * and which upstream spells INFINITY. Negative here means that sentinel.
     *
     * `flow_shift` was the expensive one. `sd_sample_params_init` sets it to
     * INFINITY, `set_flow_shift` reads that as "use `default_flow_shift`", and
     * the loader had already picked one per architecture — 5 for Wan, 7 for
     * Hunyuan, 3.16 for Boogu, 3 for the rest of the flow models. This struct
     * defaulted it to 0 and assigned it unconditionally, so every run
     * overwrote the architecture's own shift with zero. On a flow denoiser
     * that is not a small change: `time_snr_shift(0, t)` is `0*t/(1-t)`, which
     * is zero for every t, so every sigma in the schedule collapsed to zero
     * and `noise_scaling` returned the latent untouched. Flux, SD3, Wan,
     * Qwen-Image, LTX, Chroma — every modern architecture is a flow model, and
     * on all of them the sampler was handed a schedule with no noise in it.
     *
     * `eta` and `img_cfg` are the same sentinel and were simply never sent;
     * they are sent now, and a negative value still means "let it resolve".
     */
    float flow_shift  = -1.0f;
    float img_cfg     = -1.0f;
    float eta         = -1.0f;
    /** Timestep shifting, upstream's own default of 0 meaning "off". */
    int   shifted_timestep = 0;
    float slg_scale   = 0.0f;
    float skip_layer_start = 0.01f;
    float skip_layer_end   = 0.2f;
    /**
     * Which blocks skip-layer guidance skips — and without which it is off.
     *
     * `sd_sample_params_init` leaves `slg.layer_count` at 0, and upstream gates
     * the whole feature on `(slg_scale != 0) && !skip_layers.empty()`. So a
     * scale set to anything, a start, an end — three dials, all live, all
     * marked as modified, none of them reaching a sampler that had already
     * decided SLG was off. The list is the switch, and nothing here set it.
     *
     * `{7, 8, 9}` is upstream's own default, from examples/common/common.h.
     */
    std::vector<int> skip_layers = { 7, 8, 9 };
    int   width       = 512;
    int   height      = 512;
    int   clip_skip   = -1;
    int   batch_count = 1;
    float strength    = 0.75f;
    float control_strength = 0.9f;
    float ip_adapter_strength = 1.0f;
    int64_t seed      = -1;
    bool  vae_tiling  = false;

    // Video. Upstream's own defaults: a second of 16 fps, which is the shortest
    // clip worth looking at and the longest most phones will finish.
    int   video_frames = 16;
    int   fps          = 16;
    /** VACE's hold on the control frames, and inert without them. */
    float vace_strength = 1.0f;
    /** Where Wan 2.2 hands over from its high-noise expert to its low-noise one. */
    float moe_boundary  = 0.875f;
    /** Steps for that high-noise expert; 0 means "the same as the other". */
    int   high_noise_steps = 0;

    /**
     * Step caching, which is the one setting here that buys minutes.
     *
     * EasyCache and the block-level caches reuse a step's residual when the
     * model is changing little, and on a phone that is the difference between
     * a run you wait for and one you abandon. Exposed the way upstream's own
     * CLI exposes it — a mode and a `key=value` list — rather than as
     * twenty-two rows, most of which belong to one mode and are meaningless
     * under the others.
     */
    std::string cache_mode;    // empty is upstream's SD_CACHE_DISABLED
    std::string cache_option;

    /**
     * The two identity adapters, which could be attached and never used.
     *
     * `photo_maker_path` and `pulid_weights_path` are load-time and were being
     * passed, so the weights went into memory. What neither had was the
     * generate-time half — `pm_params` and `pulid_params` were left as
     * `sd_img_gen_params_init` wrote them, which means no identity image and
     * therefore no identity. Several hundred megabytes resident, and a picture
     * of nobody in particular.
     */
    float pm_style_strength = 20.0f;
    float pulid_id_weight   = 1.0f;

    // The hi-res stage, which both stills and clips have.
    //
    // Generate small, enlarge, then denoise again at the larger size — which is
    // a different thing from running an upscaler over a finished picture. On a
    // clip it is the only *coherent* way to enlarge: the whole sequence's
    // latent is scaled and re-sampled together, so frames stay consistent with
    // one another. Running ESRGAN over each frame separately does not.
    bool  hires_enabled  = false;
    std::string hires_upscaler = "latent";
    float hires_scale    = 1.5f;
    int   hires_steps    = 0;      // 0 follows the main step count
    float hires_denoise  = 0.5f;
    int   hires_tile     = 0;

    // Published by the callbacks, read by a polling Kotlin coroutine.
    std::atomic<int>   step{0};
    std::atomic<int>   total_steps{0};
    std::atomic<float> seconds_per_step{0.0f};
    std::atomic<bool>  generating{false};
    std::atomic<int>   phase{0};
    std::atomic<bool>  sampling_started{false};
    /** The step total the sampler declared for this pass; 0 before it has. */
    std::atomic<int>   sampler_steps{0};

    // What the loaded checkpoint can make, as the runtime answered at load.
    // Exclusive for everything but an SD 1.x with a motion module attached,
    // which can do both.
    std::atomic<bool>  supports_image{false};
    std::atomic<bool>  supports_video{false};
    /** Set by Cancel, read by the graph callback before every ggml node. */
    std::atomic<bool>  cancelled{false};
    /**
     * A Cancel pressed before sampling began, waiting for it to begin.
     *
     * Upstream's own cancel abandons whatever graph is running, and the
     * conditioner asserts on the empty result it then gets —
     * `GGML_ASSERT(!hidden_states.empty())` in `LLMEmbedder::encode_prompt`,
     * which is `ggml_abort` and takes the process with it. So the flag is not
     * handed over until sampling has started, which is where the minutes are
     * and where an abandoned graph is handled rather than asserted on.
     */
    std::atomic<bool>  cancel_deferred{false};

    std::mutex           preview_mutex;
    std::vector<uint8_t> preview_rgb;
    int                  preview_width  = 0;
    int                  preview_height = 0;
    std::atomic<int>     preview_serial{0};

    ~od_sd() {
        if (ctx) {
            free_sd_ctx(ctx);
        }
    }
};

std::atomic<od_sd *> g_current{nullptr};

// The last error sd.cpp logged.
std::mutex  g_last_error_mutex;
std::string g_last_error;

// The architecture upstream decided this file is, caught from its own log.
//
// sd.cpp works the version out by matching tensor names and announces it —
// "Version: FLUX.2 Klein" — but keeps it on a private struct with no getter.
// Reading the line it already prints beats the app guessing from a filename,
// and it is the loader's own answer rather than a second opinion.
std::mutex  g_version_mutex;
std::string g_version;

/**
 * Whether the last load went through the bare-denoiser door.
 *
 * A full checkpoint carries its denoiser, its text encoders and its VAE in one
 * file and needs nothing supplied. Every quantised release is the denoiser
 * alone. Which one a file is cannot be told from its architecture — SDXL ships
 * both ways — and the loader is the only thing that finds out, by trying the
 * first and falling back to the second.
 */
std::atomic<bool> g_bare_diffusion{false};

/**
 * What the loader said it was doing, most recently.
 *
 * A load of a full diffusion stack is minutes: the checkpoint, then a text
 * encoder or three, then the decoder, each mapped and dequantised in turn. From
 * outside it is one opaque JNI call, so the screen could only say "loading" and
 * hope. sd.cpp narrates every one of those steps to its log callback already —
 * this keeps the last line it said, so the app can report the runtime's own
 * words rather than a guess about which stage it might be at.
 */
std::mutex  g_stage_mutex;
std::string g_stage;

constexpr int PHASE_PREPARING = 0;
constexpr int PHASE_SAMPLING  = 1;
constexpr int PHASE_DECODING  = 2;

/**
 * How much of each LoRA actually landed.
 *
 * A LoRA trained against another architecture is not refused — sd.cpp matches
 * its tensors by name, finds none, says so and carries on. The run takes the
 * same time and produces a picture the LoRA had no part in, which is
 * indistinguishable from it having worked unless you know what it should have
 * looked like.
 *
 * Upstream reports the tally itself, in one of two forms depending on whether
 * anything matched, and the *worse* case is the quieter one: when nothing is
 * even recognised as a LoRA tensor, applied and compatible are both zero, they
 * compare equal, and it reports at INFO. So both spellings are caught here,
 * because a silent nothing is exactly the case worth reporting.
 */
std::mutex  g_lora_mutex;
std::string g_lora_report;

void note_lora(const std::string & line) {
    const auto marker = line.find(" LoRA tensors have been applied");
    if (marker == std::string::npos) return;
    const auto open = line.find('(');
    const auto slash = line.find('/', open);
    const auto close = line.find(')', slash);
    if (open == std::string::npos || slash == std::string::npos || close == std::string::npos) return;

    const auto applied = line.substr(open + 1, slash - open - 1);
    const auto total   = line.substr(slash + 1, close - slash - 1);
    std::string file;
    const auto path = line.find("lora_file_path = ");
    if (path != std::string::npos) {
        file = line.substr(path + 17);
        const auto slash_at = file.find_last_of('/');
        if (slash_at != std::string::npos) file = file.substr(slash_at + 1);
    }

    const auto trim = [](std::string s) {
        while (!s.empty() && s.front() == ' ') s.erase(s.begin());
        while (!s.empty() && s.back() == ' ') s.pop_back();
        return s;
    };

    json entry;
    entry["file"]    = trim(file);
    entry["applied"] = std::strtol(trim(applied).c_str(), nullptr, 10);
    entry["total"]   = std::strtol(trim(total).c_str(), nullptr, 10);

    std::lock_guard<std::mutex> lock(g_lora_mutex);
    json all = g_lora_report.empty() ? json::array() : json::parse(g_lora_report, nullptr, false);
    if (!all.is_array()) all = json::array();
    // Upstream calls its own reporter twice per LoRA — once to size the buffer
    // and once to build the graph — and says so in a comment. One entry each.
    for (const auto & seen : all) {
        if (seen.value("file", std::string()) == entry["file"]) return;
    }
    all.push_back(entry);
    g_lora_report = dump_json(all);
}

/**
 * Which part of a run is happening, taken from what the runtime says it is
 * doing rather than inferred from the shape of a callback.
 *
 * The inference this replaces compared the callback's step total against the
 * step count we asked for, and called anything that did not match "decoding"
 * once sampling had been seen at least once. Two things went wrong with that.
 * A sampler whose real step count differs from the requested one — every
 * ancestral sampler with a scheduler that adds a final step, and img2img, whose
 * steps are scaled by denoising strength — never matched, so sampling was never
 * recognised. And the VAE tiling callback and the sampler callback are the same
 * callback, so the readout announced the last phase it had guessed at rather
 * than the one in progress. The screen said "decoding" and then counted steps,
 * which is the order those two happen in reversed.
 *
 * sd.cpp narrates all of it — `get_learned_condition completed`, `sampling
 * using euler_a method`, `decoding 1 latents` — so the phase is read off the
 * narration and the callback is left to do the one thing it is good for, which
 * is counting steps.
 */
void note_phase(const std::string & line) {
    od_sd * e = g_current.load();
    if (e == nullptr) return;
    const auto says = [&line](const char * needle) {
        return line.find(needle) != std::string::npos;
    };

    // Order matters: "sampling completed" has to be read as the end of
    // sampling, not as the start of it, so the completions are tested first.
    if (says("sampling completed") || says("decoding") || says("decode_first_stage") ||
        says("decoded, taking") || says("upscal")) {
        e->phase.store(PHASE_DECODING);
    } else if (says("sampling using") || says("generating image:") ||
               says("generating latent")) {
        e->phase.store(PHASE_SAMPLING);
        e->sampling_started.store(true);
        // A Cancel pressed during the prompt encode has been waiting for this.
        if (e->cancel_deferred.exchange(false) && e->ctx != nullptr) {
            SLOGI("cancel deferred through the prompt encode; applying it now");
            sd_cancel_generation(e->ctx, SD_CANCEL_ALL);
        }
        // Each image of a batch declares its own total, so the latch that tells
        // the sampler from the tensor loader is dropped here rather than once
        // per run. See progress_cb.
        e->sampler_steps.store(0);
    } else if (says("get_learned_condition") || says("apply lora") ||
               says("apply_loras") || says("encode_first_stage") ||
               says("computing condition")) {
        e->phase.store(PHASE_PREPARING);
    }
}

/**
 * Which components the loader said it took, as role → path.
 *
 * The screen used to list what the *app sent*, which is a different claim and
 * one it could not check. A component the loader declined, or never reached, or
 * found already inside the checkpoint, appeared as resident all the same — the
 * card said "in memory" about a file that was not.
 *
 * sd.cpp narrates each one — `loading vae from '…'`, `loading llm from '…'` —
 * and says so again with ` failed` on the end when it does not take. So this is
 * the loader's own account, in the same way `note_version` and `note_lora`
 * already are.
 */
std::mutex g_loaded_mutex;
std::vector<std::pair<std::string, std::string>> g_loaded;

/** The runtime's word for a component, mapped to the app's role name. */
const std::pair<const char *, const char *> COMPONENT_WORDS[] = {
    // Longest first: "llm vision" must not be read as "llm".
    { "llm vision",                     "LLM_VISION" },
    { "LTX audio VAE",                  "AUDIO_VAE" },
    { "motion module (AnimateDiff)",    "MOTION_MODULE" },
    { "high noise diffusion model",     "HIGH_NOISE_DIFFUSION" },
    { "unconditional diffusion model",  "UNCOND_DIFFUSION" },
    { "embeddings connectors",          "EMBEDDING" },
    { "PuLID weights",                  "PULID" },
    { "clip_vision",                    "CLIP_VISION" },
    { "clip_l",                         "CLIP_L" },
    { "clip_g",                         "CLIP_G" },
    { "t5xxl",                          "T5XXL" },
    { "llm",                            "LLM_ENCODER" },
    { "vae",                            "VAE" },
};

void note_component(const std::string & line) {
    // `loading <what> from '<path>'`, with an optional ` failed` after it.
    const auto loading = line.find("loading ");
    if (loading == std::string::npos) return;
    const auto from = line.find(" from '", loading);
    if (from == std::string::npos) return;
    const auto close = line.rfind('\'');
    if (close == std::string::npos || close <= from + 7) return;

    const auto what = line.substr(loading + 8, from - (loading + 8));
    const auto path = line.substr(from + 7, close - (from + 7));
    const bool failed = line.find("failed", close) != std::string::npos;

    const char * role = nullptr;
    for (const auto & [word, name] : COMPONENT_WORDS) {
        if (what.find(word) != std::string::npos) { role = name; break; }
    }
    if (role == nullptr) return;

    std::lock_guard<std::mutex> lock(g_loaded_mutex);
    auto at = std::find_if(g_loaded.begin(), g_loaded.end(),
                           [&](const auto & e) { return e.first == role; });
    if (failed) {
        // Announced and then declined. Taking it back out is the whole point:
        // the announcement alone is what the app used to believe.
        if (at != g_loaded.end()) g_loaded.erase(at);
        return;
    }
    if (at != g_loaded.end()) at->second = path;
    else g_loaded.emplace_back(role, path);
}

void note_stage(const char * text) {
    if (text == nullptr) return;
    std::string line(text);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
    }
    if (line.empty()) return;
    // Upstream prefixes every line with `file.hpp:72 - `; the prefix is for a
    // developer reading logcat, not for someone waiting on a picture.
    const auto dash = line.find(" - ");
    if (dash != std::string::npos && line.find(':') < dash) {
        line = line.substr(dash + 3);
    }
    if (line.empty()) return;
    {
        std::lock_guard<std::mutex> lock(g_stage_mutex);
        g_stage = line;
    }
    note_phase(line);
    note_lora(line);
    note_component(line);
}

void note_version(const char * text) {
    if (text == nullptr) return;
    const std::string line(text);
    const auto at = line.find("Version: ");
    if (at == std::string::npos) return;
    auto value = line.substr(at + 9);
    while (!value.empty() && (value.back() == 0x0a || value.back() == 0x0d || value.back() == ' ')) {
        value.pop_back();
    }
    if (value.empty()) return;
    std::lock_guard<std::mutex> lock(g_version_mutex);
    g_version = value;
}

void set_last_error(const char * text) {
    if (text == nullptr) return;
    std::string line(text);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
    }
    // Upstream prefixes every line with `file.hpp:72 - `.
    const auto dash = line.find(" - ");
    if (dash != std::string::npos && line.find(':') < dash) {
        line = line.substr(dash + 3);
    }
    if (line.empty()) return;
    std::lock_guard<std::mutex> lock(g_last_error_mutex);
    // Keep the *first* error of a load: it names the cause, while the ones that
    // follow are its consequences ("clip prepare graph weights failed").
    if (g_last_error.empty()) {
        g_last_error = line;
    }
}

std::string take_last_error() {
    std::lock_guard<std::mutex> lock(g_last_error_mutex);
    return g_last_error;
}

od_sd * as_sd(jlong handle) {
    return reinterpret_cast<od_sd *>(handle);
}

bool as_bool(const json & v, bool fallback) {
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number())  return v.get<double>() != 0.0;
    if (v.is_string()) {
        const auto s = v.get<std::string>();
        return s == "true" || s == "1" || s == "on";
    }
    return fallback;
}

int32_t as_int(const json & v, int32_t fallback) {
    if (v.is_number_integer()) return v.get<int32_t>();
    if (v.is_number())         return (int32_t) std::llround(v.get<double>());
    if (v.is_string()) { try { return std::stoi(v.get<std::string>()); } catch (...) { return fallback; } }
    return fallback;
}

int64_t as_long(const json & v, int64_t fallback) {
    if (v.is_number_integer()) return v.get<int64_t>();
    if (v.is_number())         return (int64_t) std::llround(v.get<double>());
    if (v.is_string()) { try { return std::stoll(v.get<std::string>()); } catch (...) { return fallback; } }
    return fallback;
}

float as_float(const json & v, float fallback) {
    if (v.is_number()) return v.get<float>();
    if (v.is_string()) { try { return std::stof(v.get<std::string>()); } catch (...) { return fallback; } }
    return fallback;
}

std::string as_string(const json & v) {
    return v.is_string() ? v.get<std::string>() : dump_json(v);
}

/** A list of block indices, however it was written: `[7,8,9]`, `"7, 8, 9"`, `7`. */
std::vector<int> as_int_list(const json & v, const std::vector<int> & fallback) {
    std::vector<int> out;
    if (v.is_array()) {
        for (const auto & item : v) {
            if (item.is_number()) {
                out.push_back((int) std::lround(item.get<double>()));
            } else if (item.is_string()) {
                try { out.push_back(std::stoi(item.get<std::string>())); } catch (...) {}
            }
        }
        // An emptied chip row is a person turning the feature off, not a parse
        // failure, so it is kept rather than replaced with the fallback.
        return out;
    }
    if (v.is_number()) {
        out.push_back((int) std::lround(v.get<double>()));
        return out;
    }
    if (v.is_string()) {
        const std::string text = v.get<std::string>();
        std::string       token;
        for (size_t i = 0; i <= text.size(); ++i) {
            const char c = i < text.size() ? text[i] : ',';
            if (c == ',' || c == ' ' || c == '\t') {
                if (!token.empty()) {
                    try { out.push_back(std::stoi(token)); } catch (...) {}
                    token.clear();
                }
            } else {
                token.push_back(c);
            }
        }
        return out;
    }
    return fallback;
}

struct row { void (*apply)(od_sd &, const json &); };

const std::map<std::string, row> & table() {
    static const std::map<std::string, row> t = {
        { "prompt",           { [](od_sd & e, const json & v) { e.prompt = as_string(v); } } },
        { "negative_prompt",  { [](od_sd & e, const json & v) { e.negative_prompt = as_string(v); } } },
        { "steps",            { [](od_sd & e, const json & v) { e.steps = std::max(1, as_int(v, e.steps)); } } },
        { "cfg_scale",        { [](od_sd & e, const json & v) { e.cfg_scale = as_float(v, e.cfg_scale); } } },
        { "guidance",         { [](od_sd & e, const json & v) { e.guidance = as_float(v, e.guidance); } } },
        { "flow_shift",       { [](od_sd & e, const json & v) { e.flow_shift = as_float(v, e.flow_shift); } } },
        { "img_cfg",          { [](od_sd & e, const json & v) { e.img_cfg = as_float(v, e.img_cfg); } } },
        { "eta",              { [](od_sd & e, const json & v) { e.eta = as_float(v, e.eta); } } },
        { "shifted_timestep", { [](od_sd & e, const json & v) { e.shifted_timestep = std::max(0, as_int(v, e.shifted_timestep)); } } },
        { "cache_mode",       { [](od_sd & e, const json & v) { e.cache_mode = as_string(v); } } },
        { "cache_option",     { [](od_sd & e, const json & v) { e.cache_option = as_string(v); } } },
        { "style_strength",   { [](od_sd & e, const json & v) { e.pm_style_strength = as_float(v, e.pm_style_strength); } } },
        { "id_weight",        { [](od_sd & e, const json & v) { e.pulid_id_weight = as_float(v, e.pulid_id_weight); } } },
        { "slg_scale",        { [](od_sd & e, const json & v) { e.slg_scale = as_float(v, e.slg_scale); } } },
        { "skip_layer_start", { [](od_sd & e, const json & v) { e.skip_layer_start = as_float(v, e.skip_layer_start); } } },
        { "skip_layer_end",   { [](od_sd & e, const json & v) { e.skip_layer_end = as_float(v, e.skip_layer_end); } } },
        // A chip row hands over strings, the raw-parameter box hands over an
        // array, and a person typing into either may write "7, 8, 9". All three
        // mean the same list.
        { "skip_layers",      { [](od_sd & e, const json & v) { e.skip_layers = as_int_list(v, e.skip_layers); } } },
        { "width",            { [](od_sd & e, const json & v) { e.width = as_int(v, e.width); } } },
        { "height",           { [](od_sd & e, const json & v) { e.height = as_int(v, e.height); } } },
        { "seed",             { [](od_sd & e, const json & v) { e.seed = as_long(v, e.seed); } } },
        { "clip_skip",        { [](od_sd & e, const json & v) { e.clip_skip = as_int(v, e.clip_skip); } } },
        { "batch_count",      { [](od_sd & e, const json & v) { e.batch_count = std::max(1, as_int(v, e.batch_count)); } } },
        { "strength",         { [](od_sd & e, const json & v) { e.strength = as_float(v, e.strength); } } },
        { "control_strength", { [](od_sd & e, const json & v) { e.control_strength = as_float(v, e.control_strength); } } },
        { "ip_adapter_strength", { [](od_sd & e, const json & v) { e.ip_adapter_strength = as_float(v, e.ip_adapter_strength); } } },
        { "vae_tiling",       { [](od_sd & e, const json & v) { e.vae_tiling = as_bool(v, false); } } },
        // Video. Inert on an image model, which is why they are in the same
        // table rather than a second one: the runtime reports what it will act
        // on, and `appliesTo` in the manifest decides what is worth showing.
        { "video_frames",     { [](od_sd & e, const json & v) { e.video_frames = std::max(1, as_int(v, e.video_frames)); } } },
        { "fps",              { [](od_sd & e, const json & v) { e.fps = std::max(1, as_int(v, e.fps)); } } },
        { "vace_strength",    { [](od_sd & e, const json & v) { e.vace_strength = as_float(v, e.vace_strength); } } },
        { "moe_boundary",     { [](od_sd & e, const json & v) { e.moe_boundary = as_float(v, e.moe_boundary); } } },
        { "high_noise_steps", { [](od_sd & e, const json & v) { e.high_noise_steps = std::max(0, as_int(v, e.high_noise_steps)); } } },
        // The hi-res stage. Both stills and clips have one, and on a clip it is
        // the only coherent way to enlarge — the whole sequence's latent is
        // scaled and re-denoised together.
        { "hires_fix",        { [](od_sd & e, const json & v) { e.hires_enabled = as_bool(v, false); } } },
        { "hires_upscaler",   { [](od_sd & e, const json & v) { e.hires_upscaler = as_string(v); } } },
        { "hires_scale",      { [](od_sd & e, const json & v) { e.hires_scale = as_float(v, e.hires_scale); } } },
        { "hires_steps",      { [](od_sd & e, const json & v) { e.hires_steps = std::max(0, as_int(v, e.hires_steps)); } } },
        { "hires_denoise",    { [](od_sd & e, const json & v) { e.hires_denoise = as_float(v, e.hires_denoise); } } },
        { "sampling_method",  { [](od_sd & e, const json & v) { e.sampling_method = as_string(v); } } },
        { "schedule",         { [](od_sd & e, const json & v) { e.schedule = as_string(v); } } },
    };
    return t;
}

// This build's defaults, read back off a default-constructed od_sd rather than asserted a second time in the manifest.
const std::map<std::string, json (*)(const od_sd &)> & default_table() {
    static const std::map<std::string, json (*)(const od_sd &)> t = {
        { "prompt",           [](const od_sd & e) { return json(e.prompt); } },
        { "negative_prompt",  [](const od_sd & e) { return json(e.negative_prompt); } },
        { "steps",            [](const od_sd & e) { return json(e.steps); } },
        { "cfg_scale",        [](const od_sd & e) { return json(e.cfg_scale); } },
        { "guidance",         [](const od_sd & e) { return json(e.guidance); } },
        { "flow_shift",       [](const od_sd & e) { return json(e.flow_shift); } },
        { "img_cfg",          [](const od_sd & e) { return json(e.img_cfg); } },
        { "eta",              [](const od_sd & e) { return json(e.eta); } },
        { "shifted_timestep", [](const od_sd & e) { return json(e.shifted_timestep); } },
        { "cache_mode",       [](const od_sd & e) { return json(e.cache_mode.empty() ? "disabled" : e.cache_mode); } },
        { "cache_option",     [](const od_sd & e) { return json(e.cache_option); } },
        { "style_strength",   [](const od_sd & e) { return json(e.pm_style_strength); } },
        { "id_weight",        [](const od_sd & e) { return json(e.pulid_id_weight); } },
        { "slg_scale",        [](const od_sd & e) { return json(e.slg_scale); } },
        { "skip_layer_start", [](const od_sd & e) { return json(e.skip_layer_start); } },
        { "skip_layer_end",   [](const od_sd & e) { return json(e.skip_layer_end); } },
        { "skip_layers",      [](const od_sd & e) { return json(e.skip_layers); } },
        { "width",            [](const od_sd & e) { return json(e.width); } },
        { "height",           [](const od_sd & e) { return json(e.height); } },
        { "seed",             [](const od_sd & e) { return json(e.seed); } },
        { "clip_skip",        [](const od_sd & e) { return json(e.clip_skip); } },
        { "batch_count",      [](const od_sd & e) { return json(e.batch_count); } },
        { "strength",         [](const od_sd & e) { return json(e.strength); } },
        { "control_strength", [](const od_sd & e) { return json(e.control_strength); } },
        { "ip_adapter_strength", [](const od_sd & e) { return json(e.ip_adapter_strength); } },
        { "vae_tiling",       [](const od_sd & e) { return json(e.vae_tiling); } },
        { "video_frames",     [](const od_sd & e) { return json(e.video_frames); } },
        { "fps",              [](const od_sd & e) { return json(e.fps); } },
        { "vace_strength",    [](const od_sd & e) { return json(e.vace_strength); } },
        { "moe_boundary",     [](const od_sd & e) { return json(e.moe_boundary); } },
        { "high_noise_steps", [](const od_sd & e) { return json(e.high_noise_steps); } },
        { "hires_fix",        [](const od_sd & e) { return json(e.hires_enabled); } },
        { "hires_upscaler",   [](const od_sd & e) { return json(e.hires_upscaler); } },
        { "hires_scale",      [](const od_sd & e) { return json(e.hires_scale); } },
        { "hires_steps",      [](const od_sd & e) { return json(e.hires_steps); } },
        { "hires_denoise",    [](const od_sd & e) { return json(e.hires_denoise); } },
        { "sampling_method",  [](const od_sd & e) { return json(e.sampling_method); } },
        { "schedule",         [](const od_sd & e) { return json(e.schedule); } },
    };
    return t;
}

void progress_cb(int step, int steps, float time, void *) {
    od_sd * e = g_current.load();
    if (e == nullptr) return;

    // The phase is the runtime's own word for it, set in note_phase. This
    // callback serves both the sampler and the VAE tiler, and only the
    // sampler's numbers are steps of the thing the progress bar measures — the
    // tiler's are tiles, and reporting them made a finished picture restart its
    // count from 1 of 12.
    if (e->phase.load() != PHASE_SAMPLING || steps <= 0) return;

    // A third caller, which the phase cannot filter because it runs *inside*
    // sampling: the tensor loader.
    //
    // Weights load lazily unless `eager_load` is set, so the first time a layer
    // is needed the loader reads it in and reports its own progress — through
    // this same callback, in tensors. The readout counted to 134 of 1234 and
    // then 1680 of 1680, neither of which is a step, and the real step number
    // was overwritten between every one of them.
    //
    // They are told apart by the total rather than by the count. Upstream's
    // sampler declares its own total once, as `(0, steps)`, before the first
    // denoise touches a weight — so the first total seen in this phase is the
    // sampler's, and anything reporting a different one is not the sampler.
    // Latching it also handles the case the old exact-match check got wrong:
    // an ancestral sampler or an img2img run has a real total that is not the
    // one that was asked for, and this takes whatever it turns out to be.
    int expected = e->sampler_steps.load();
    if (expected == 0) {
        e->sampler_steps.store(steps);
        expected = steps;
    }
    if (steps != expected) return;

    e->step.store(step);
    e->total_steps.store(steps);
    if (time > 0.0f) {
        e->seconds_per_step.store(time);
    }
}

void preview_cb(int step, int frame_count, sd_image_t * frames, bool, void *) {
    od_sd * e = g_current.load();
    if (e == nullptr || frames == nullptr || frame_count <= 0) return;
    const sd_image_t & frame = frames[0];
    if (frame.data == nullptr || frame.channel < 3) return;

    std::lock_guard<std::mutex> lock(e->preview_mutex);
    const size_t pixels = (size_t) frame.width * frame.height;
    e->preview_rgb.resize(pixels * 3);
    for (size_t i = 0; i < pixels; ++i) {
        e->preview_rgb[i * 3 + 0] = frame.data[i * frame.channel + 0];
        e->preview_rgb[i * 3 + 1] = frame.data[i * frame.channel + 1];
        e->preview_rgb[i * 3 + 2] = frame.data[i * frame.channel + 2];
    }
    e->preview_width  = (int) frame.width;
    e->preview_height = (int) frame.height;
    e->preview_serial.fetch_add(1);
    (void) step;
}

/** Copy a Java byte[] of packed RGB into an sd_image_t the caller owns. */
struct owned_image {
    std::vector<uint8_t> data;
    sd_image_t           image{0, 0, 0, nullptr};
};

/** The name the enum uses for "whatever the file already is". */
constexpr const char * WTYPE_UNCHANGED = "as-is";

/**
 * Every weight type this build can *produce*, which is a shorter list than the
 * ones it can read.
 *
 * This is the set of cases `ggml_quantize_chunk` has (ggml.c), plus f32, which
 * `convert_tensor` dequantises to itself without going through it. The rest of
 * `sd_type_t` — q8_1, q8_K, the integer types, f64 — are types a file may
 * legitimately contain and nothing here can write, and asking for one is not an
 * error anybody sees: the switch falls through to `GGML_ABORT`, which ends the
 * process from four minutes into a load, with no Java exception and nothing in
 * the crash buffer naming the setting that did it.
 *
 * The IQ family is here because it cannot abort, not because it is a good idea.
 * `convert_tensor` supplies an importance matrix of all ones, and an IQ quant
 * against a flat imatrix is worse than its size suggests. The manifest offers
 * none of them; this list only decides what is survivable, not what is sensible.
 */
const char * const CONVERTIBLE_TYPES[] = {
    "f32", "f16", "bf16",
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q1_0",
    "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
    "tq1_0", "tq2_0", "mxfp4", "nvfp4",
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s",
    "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs",
};

/**
 * The type to quantise to at load, read off the name the app sends.
 *
 * `sd_ctx_params_init` starts this at `SD_TYPE_COUNT`, which is upstream's way
 * of saying "leave every tensor as the file wrote it", and that is what an
 * empty name keeps. Anything else is matched against ggml's own registry rather
 * than a table typed here, case-insensitively — upstream's help says `q3_K` and
 * a manifest edited by hand will sooner or later say `q3_k`.
 *
 * @param refusal set when the name is one this build cannot honour; empty
 *   otherwise. It is filled rather than logged because the alternative to
 *   refusing is worse than a failed load — see CONVERTIBLE_TYPES.
 */
sd_type_t parse_wtype(const std::string & name, std::string & refusal) {
    refusal.clear();
    if (name.empty() || jni_iequals(name, WTYPE_UNCHANGED)) return SD_TYPE_COUNT;

    for (const char * candidate : CONVERTIBLE_TYPES) {
        if (!jni_iequals(name, candidate)) continue;
        const auto type = str_to_sd_type(candidate);
        if (type != SD_TYPE_COUNT) return type;
    }

    refusal = "This build cannot convert weights to \"" + name + "\" while loading. " +
              "Choose one of: ";
    bool first = true;
    for (const char * candidate : CONVERTIBLE_TYPES) {
        if (str_to_sd_type(candidate) == SD_TYPE_COUNT) continue;
        if (!first) refusal += ", ";
        refusal += candidate;
        first = false;
    }
    refusal += ".";
    return SD_TYPE_COUNT;
}

/**
 * An LTX-AV soundtrack, as a 16-bit WAV beside its frames.
 *
 * Upstream hands back floats in [-1, 1]; nothing on Android plays those, and a
 * WAV header is 44 bytes against pulling in an encoder. Clamped rather than
 * scaled, because a sample outside the range is a defect in the model's output
 * and quietly rescaling the whole track would hide it.
 */
bool write_wav(const std::string & path, const sd_audio_t & audio) {
    if (audio.data == nullptr || audio.sample_count == 0) return false;
    const uint32_t channels    = audio.channels > 0 ? audio.channels : 1;
    const uint32_t rate        = audio.sample_rate > 0 ? audio.sample_rate : 44100;
    const uint64_t total       = audio.sample_count * channels;
    const uint32_t data_bytes  = (uint32_t) (total * 2);
    const uint32_t byte_rate   = rate * channels * 2;

    FILE * file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return false;

    const auto u32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, file); };
    const auto u16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, file); };
    std::fwrite("RIFF", 1, 4, file);
    u32(36 + data_bytes);
    std::fwrite("WAVEfmt ", 1, 8, file);
    u32(16);                              // PCM header size
    u16(1);                               // PCM, uncompressed
    u16((uint16_t) channels);
    u32(rate);
    u32(byte_rate);
    u16((uint16_t) (channels * 2));       // block align
    u16(16);                              // bits per sample
    std::fwrite("data", 1, 4, file);
    u32(data_bytes);

    for (uint64_t i = 0; i < total; ++i) {
        float sample = audio.data[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        const int16_t pcm = (int16_t) std::lround(sample * 32767.0f);
        std::fwrite(&pcm, 2, 1, file);
    }
    std::fclose(file);
    return true;
}

/**
 * Fill the hi-res stage, which both stills and clips take.
 *
 * Generate small, enlarge, denoise again at the larger size. Distinct from the
 * ESRGAN path, which runs over a *finished* picture in its own context and has
 * no idea what the picture was meant to be. On a clip the difference is not
 * subtle: this scales the whole sequence's latent and re-samples it together,
 * so frames stay consistent with one another, where running an upscaler over
 * each frame independently makes flat areas shimmer.
 *
 * @param upscaler_model the ESRGAN to use when the mode is `model`, or empty.
 */
/**
 * The skip-layer guidance settings, all four of them.
 *
 * Shared because the video path had only the first three and upstream needs
 * all four — `layers` is what turns the feature on, and it is the one the
 * struct's own initialiser leaves empty. The vector lives on `od_sd`, which
 * outlives the call the params struct is passed to.
 */
/** A negative dial means "leave upstream's INFINITY, which resolves per model". */
float or_model_default(float value) {
    return value < 0.0f ? INFINITY : value;
}

/**
 * The cache mode and its options, in upstream's own two-flag shape.
 *
 * The option names are the ones `--cache-option` documents, and which one a
 * name lands on depends on the mode: `threshold` is the reuse threshold for
 * the EasyCache family and the residual-difference threshold for the
 * block-level ones. Anything unrecognised is left alone rather than guessed at.
 */
void apply_cache(const od_sd & e, sd_cache_params_t & cache) {
    sd_cache_params_init(&cache);
    const std::string & mode = e.cache_mode;
    if (mode.empty() || mode == "disabled") return;

    if      (mode == "easycache")  cache.mode = SD_CACHE_EASYCACHE;
    else if (mode == "ucache")     cache.mode = SD_CACHE_UCACHE;
    else if (mode == "dbcache")    cache.mode = SD_CACHE_DBCACHE;
    else if (mode == "taylorseer") cache.mode = SD_CACHE_TAYLORSEER;
    else if (mode == "cache_dit")  cache.mode = SD_CACHE_CACHE_DIT;
    else if (mode == "spectrum")   cache.mode = SD_CACHE_SPECTRUM;
    else {
        SLOGE("unknown cache mode '%s'; caching stays off", mode.c_str());
        return;
    }

    const bool block_level = cache.mode == SD_CACHE_DBCACHE ||
                             cache.mode == SD_CACHE_TAYLORSEER ||
                             cache.mode == SD_CACHE_CACHE_DIT;

    size_t at = 0;
    const std::string & text = e.cache_option;
    while (at < text.size()) {
        size_t comma = text.find(',', at);
        if (comma == std::string::npos) comma = text.size();
        const std::string pair = text.substr(at, comma - at);
        at = comma + 1;
        const size_t eq = pair.find('=');
        if (eq == std::string::npos) continue;
        auto trim = [](std::string s) {
            while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
            while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
            return s;
        };
        const std::string key   = trim(pair.substr(0, eq));
        const std::string value = trim(pair.substr(eq + 1));
        if (key.empty() || value.empty()) continue;

        float number = 0.0f;
        try { number = std::stof(value); } catch (...) { continue; }
        const bool flag = number != 0.0f;

        if      (key == "threshold" && block_level) cache.residual_diff_threshold = number;
        else if (key == "threshold")                cache.reuse_threshold = number;
        else if (key == "start")                    cache.start_percent = number;
        else if (key == "end")                      cache.end_percent = number;
        else if (key == "decay")                    cache.error_decay_rate = number;
        else if (key == "relative")                 cache.use_relative_threshold = flag;
        else if (key == "reset")                    cache.reset_error_on_compute = flag;
        else if (key == "Fn")                       cache.Fn_compute_blocks = (int) number;
        else if (key == "Bn")                       cache.Bn_compute_blocks = (int) number;
        else if (key == "warmup")                   cache.max_warmup_steps = (int) number;
        else if (key == "w")                        cache.spectrum_w = number;
        else if (key == "m")                        cache.spectrum_m = (int) number;
        else if (key == "lam")                      cache.spectrum_lam = number;
        else if (key == "window")                   cache.spectrum_window_size = (int) number;
        else if (key == "flex")                     cache.spectrum_flex_window = number;
        else if (key == "stop")                     cache.spectrum_stop_percent = number;
        else SLOGE("cache option '%s' is not one this mode takes; ignored", key.c_str());
    }
}

/** The parts of a sampler pass both outputs share, sentinels intact. */
void apply_sample_params(const od_sd & e, sd_sample_params_t & sample) {
    sample.sample_steps                = e.steps;
    sample.guidance.txt_cfg            = e.cfg_scale;
    sample.guidance.img_cfg            = or_model_default(e.img_cfg);
    sample.guidance.distilled_guidance = e.guidance;
    sample.flow_shift                  = or_model_default(e.flow_shift);
    sample.eta                         = or_model_default(e.eta);
    sample.shifted_timestep            = e.shifted_timestep;
    sample.sample_method               = str_to_sample_method(e.sampling_method.c_str());
    sample.scheduler                   = str_to_scheduler(e.schedule.c_str());
}

void apply_slg(const od_sd & e, sd_slg_params_t & slg) {
    slg.scale       = e.slg_scale;
    slg.layer_start = e.skip_layer_start;
    slg.layer_end   = e.skip_layer_end;
    slg.layers      = e.skip_layers.empty() ? nullptr : const_cast<int *>(e.skip_layers.data());
    slg.layer_count = e.skip_layers.size();
}

/** The architecture the loader announced for the resident checkpoint. */
std::string detected_version() {
    std::lock_guard<std::mutex> lock(g_version_mutex);
    return g_version;
}

void apply_hires(const od_sd & e, sd_hires_params_t & hires, const std::string & upscaler_model) {
    hires.enabled = e.hires_enabled;
    if (!hires.enabled) return;
    const auto parsed = str_to_sd_hires_upscaler(e.hires_upscaler.c_str());
    hires.upscaler = parsed == SD_HIRES_UPSCALER_COUNT ? SD_HIRES_UPSCALER_LATENT : parsed;
    hires.scale    = e.hires_scale;
    hires.steps    = e.hires_steps > 0 ? e.hires_steps : e.steps;
    hires.denoising_strength = e.hires_denoise;
    if (hires.upscaler == SD_HIRES_UPSCALER_MODEL) {
        // Asking for the model mode with no model is asking for nothing; fall
        // back to the latent one rather than passing a null it would read.
        if (upscaler_model.empty()) {
            hires.upscaler = SD_HIRES_UPSCALER_LATENT;
        } else {
            hires.model_path = upscaler_model.c_str();
        }
    }
}

/**
 * The hi-res stage on a clip, which is not the one on a still.
 *
 * On an image, "hi-res" is generate-small-then-denoise-larger and every
 * upscaler mode works. On a clip it is LTX-AV's own latent spatial upsampler,
 * which is a separate model file, and `generate_video` refuses outright —
 * `return false` — for any other architecture, for any upscaler but MODEL, and
 * for a missing model path. The app turned all three of those into "The run
 * produced no frames. This is usually memory", which is a diagnosis of a
 * problem nobody had, delivered after several minutes of sampling.
 *
 * So the three cases are answered here, where the answer is knowable, and the
 * stage is dropped rather than the run. @return why it was dropped, or empty.
 */
std::string apply_video_hires(const od_sd & e, sd_hires_params_t & hires,
                              const std::string & upscaler_model) {
    apply_hires(e, hires, upscaler_model);
    if (!hires.enabled) return {};

    const std::string version = detected_version();
    if (version.find("LTX") == std::string::npos && version.find("ltx") == std::string::npos) {
        hires.enabled = false;
        return "the hi-res stage on a clip is LTX-AV's latent upsampler, and " +
               (version.empty() ? std::string("this checkpoint") : version) + " has none";
    }
    if (upscaler_model.empty()) {
        hires.enabled = false;
        return "LTX-AV's hi-res stage needs its latent upsampler attached, and none was";
    }
    // apply_hires downgrades an upscaler it cannot serve to the latent one,
    // which is right for a still and fatal here.
    hires.upscaler   = SD_HIRES_UPSCALER_MODEL;
    hires.model_path = upscaler_model.c_str();
    return {};
}

owned_image take_image(JNIEnv * env, jbyteArray bytes, jint width, jint height) {
    owned_image out;
    if (bytes == nullptr || width <= 0 || height <= 0) return out;
    const jsize n = env->GetArrayLength(bytes);
    out.data.resize((size_t) n);
    env->GetByteArrayRegion(bytes, 0, n, reinterpret_cast<jbyte *>(out.data.data()));
    out.image.width   = (uint32_t) width;
    out.image.height  = (uint32_t) height;
    out.image.channel = 3;
    out.image.data    = out.data.data();
    return out;
}

} // namespace

extern "C" {

/** The keys this binary will act on. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_SdBridge_nativeSupportedParams(JNIEnv * env, jobject) {
    const od_sd defaults = {};
    const auto & readers = default_table();

    json out = json::object();
    for (const auto & entry : table()) {
        json row{ { "reload", false } };
        const auto reader = readers.find(entry.first);
        if (reader != readers.end()) {
            row["default"] = reader->second(defaults);
        }
        out[entry.first] = row;
    }
    return jni_from_string(env, dump_json(out));
}

/**
 * What stable-diffusion.cpp decided this checkpoint is.
 *
 * Not a guess from a filename or a repo tag: the loader works the version out
 * from the tensors it finds and announces it, and one of the few things that
 * has to follow from it is the sampler settings a family expects.
 */
/** True when the loaded file is the denoiser alone, so its parts must be supplied. */
JNIEXPORT jboolean JNICALL
Java_ai_ondevice_engine_SdBridge_nativeIsBareDiffusion(JNIEnv *, jobject) {
    return g_bare_diffusion.load() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_SdBridge_nativeDetectedVersion(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_version_mutex);
    return jni_from_string(env, g_version);
}

/** Applied/total per LoRA, as sd.cpp counted them during the last run. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_SdBridge_nativeLoraReport(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_lora_mutex);
    return jni_from_string(env, g_lora_report.empty() ? "[]" : g_lora_report);
}

/**
 * The components the loader actually took, as `[{"role","path"}]`.
 *
 * What is resident, rather than what was asked for. The two differ whenever a
 * checkpoint carries its own encoders, whenever a file is declined, and
 * whenever the loader never reaches one — and the screen was reporting the
 * second while calling it the first.
 */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_SdBridge_nativeLoadedComponents(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_loaded_mutex);
    json out = json::array();
    for (const auto & [role, path] : g_loaded) {
        out.push_back(json{ { "role", role }, { "path", path } });
    }
    return jni_from_string(env, dump_json(out));
}

JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_SdBridge_nativeLoadStage(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_stage_mutex);
    return jni_from_string(env, g_stage);
}

JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_SdBridge_nativeSystemInfo(JNIEnv * env, jobject) {
    json info;
    info["system"] = std::string(sd_get_system_info());
    info["cores"]  = sd_get_num_physical_cores();
    return jni_from_string(env, dump_json(info));
}

/**
 * The whole of `sd_ctx_params_t`, as two JSON objects rather than as arguments.
 *
 * This took eleven positional strings and grew by one whenever a field turned
 * out to matter. What that cost was not typing: the fields it did *not* take
 * were invisible. `uncond_diffusion_model_path` unset meant Ideogram 4 could
 * not run, `high_noise_diffusion_model_path` unset meant Wan 2.2 could not, and
 * neither said so — the app simply had no way to name the file, so the
 * architecture looked unsupported when it was unplumbed. `max_vram` and
 * `stream_layers`, upstream's answer to a model larger than memory, were
 * unreachable for the same reason on the device that needs them most.
 *
 * So: components keyed by `AttachmentRole.paramKey`, settings keyed by the
 * struct's own field names, and a new field costs one line on each side. This
 * is the shape `LlamaBridge.nativeLoad` has always had.
 */
JNIEXPORT jlong JNICALL
Java_ai_ondevice_engine_SdBridge_nativeLoad(
        JNIEnv * env, jobject, jstring jmodel, jstring jcomponents, jstring jsettings,
        jint threads) {
    const auto model = jni_to_string(env, jmodel);

    json components = json::object();
    json settings   = json::object();
    try {
        const auto components_text = jni_to_string(env, jcomponents);
        const auto settings_text   = jni_to_string(env, jsettings);
        if (!components_text.empty()) components = json::parse(components_text);
        if (!settings_text.empty())   settings   = json::parse(settings_text);
    } catch (const std::exception & ex) {
        jni_throw(env, std::string("The load request could not be read: ") + ex.what());
        return 0;
    }

    // Every string the params struct points at lives here until new_sd_ctx has
    // returned. `sd_ctx_params_t` holds borrowed `const char *`, and a temporary
    // that dies at the end of its statement is a dangling pointer read minutes
    // later, inside the loader, with no symptom that names this function.
    // A list rather than a map: `push_back` never invalidates a reference to an
    // element already in it, and there is no key to collide — "vae" arrives in
    // one object and "vae_format" in the other, and a keyed store would have to
    // be right about that forever.
    std::list<std::string> held;

    auto text_of = [&](const json & source, const char * key) -> const char * {
        const auto found = source.find(key);
        if (found == source.end() || !found->is_string()) return nullptr;
        auto value = found->get<std::string>();
        if (value.empty()) return nullptr;
        return held.emplace_back(std::move(value)).c_str();
    };
    auto bool_of = [&](const char * key, bool fallback) {
        const auto found = settings.find(key);
        return (found != settings.end() && found->is_boolean()) ? found->get<bool>() : fallback;
    };

    // Quantise on the way in, which is what decides whether a file can run here
    // at all. A 6 GB fp16 safetensors is not a large model badly packaged; it is
    // the only packaging most architectures ever get, and no phone loads it.
    // Asked for q4_0 it becomes about 2 GB before it reaches memory, so the
    // question stops being "is there a GGUF of this" — the answer to which is
    // no, for most of the list — and becomes one about time and quality.
    //
    // Refused before anything is allocated, because the failure it prevents is
    // an abort rather than an error.
    std::string wtype_refusal;
    const char * wtype_name = text_of(settings, "wtype");
    const auto wtype = parse_wtype(wtype_name == nullptr ? std::string() : wtype_name, wtype_refusal);
    if (!wtype_refusal.empty()) {
        jni_throw(env, wtype_refusal);
        return 0;
    }

    // A version left over from the last checkpoint would be read as this one's.
    { std::lock_guard<std::mutex> lock(g_version_mutex); g_version.clear(); }
    { std::lock_guard<std::mutex> lock(g_stage_mutex); g_stage.clear(); }
    { std::lock_guard<std::mutex> lock(g_loaded_mutex); g_loaded.clear(); }
    g_bare_diffusion.store(false);

    static std::once_flag once;
    std::call_once(once, [] {
        sd_set_log_callback([](sd_log_level_t level, const char * text, void *) {
            if (text == nullptr) return;
            if (level >= SD_LOG_ERROR) {
                set_last_error(text);
            }
            note_version(text);
            note_stage(text);
            if (level >= SD_LOG_WARN) {
                __android_log_write(level >= SD_LOG_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_WARN,
                                    "ondevice.sd", text);
            }
        }, nullptr);

        // Where Cancel lands, and where it deliberately does not.
        //
        // Upstream reads its own cancel flag between denoising steps, between
        // batches and between latents. On a phone one step of a 4B transformer
        // is nearer two minutes than a moment, so a press could sit unnoticed
        // for the length of the thing it was meant to stop. This callback runs
        // before every ggml node instead.
        //
        // Two things it has to get right. The `ask` pass is upstream asking
        // whether we want to observe a node, not whether to continue — answering
        // false there makes it skip ahead rather than stop, so it always gets
        // true. And the stop only applies once sampling has started: a graph
        // abandoned during the prompt encode returns nullopt to
        // LLMEmbedder::encode_prompt, which does not expect one and calls
        // ggml_abort, taking the process with it. Prompt encoding is seconds
        // and upstream's own check covers the gap; sampling and the VAE decode
        // are where the minutes are, and that is what this reaches.
        sd_set_backend_eval_callback([](ggml_tensor *, bool ask, void *) -> bool {
            if (ask) return true;
            auto * running = g_current.load();
            if (running == nullptr) return true;
            return !(running->cancelled.load() && running->sampling_started.load());
        }, nullptr);
    });

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path       = model.c_str();

    // Components, keyed as the app names roles. Absent means absent: a null
    // leaves upstream's own default, which for every one of these is "not
    // supplied", and is a different thing from an empty string.
    params.vae_path         = text_of(components, "vae");
    params.control_net_path = text_of(components, "control_net");
    params.clip_l_path      = text_of(components, "clip_l");
    params.clip_g_path      = text_of(components, "clip_g");
    params.t5xxl_path       = text_of(components, "t5xxl");
    params.ip_adapter_path  = text_of(components, "ip_adapter");
    params.embeddings_connectors_path = text_of(components, "embd_dir");
    params.clip_vision_path = text_of(components, "clip_vision");
    // FLUX.2 reads its prompt with a language model rather than with CLIP and
    // T5 — Qwen3 for Klein, Mistral Small for dev — so the text encoder is a
    // GGUF the size of a chat model and arrives through its own path.
    params.llm_path         = text_of(components, "llm");
    params.llm_vision_path  = text_of(components, "llm_vision");

    // The companion denoisers. Each of these is the model, published in more
    // than one piece, and each was a whole architecture the app could not run.
    params.uncond_diffusion_model_path    = text_of(components, "uncond_diffusion_model");
    params.high_noise_diffusion_model_path = text_of(components, "high_noise_diffusion_model");
    params.motion_module_path             = text_of(components, "motion_module");

    params.audio_vae_path     = text_of(components, "audio_vae");
    params.photo_maker_path   = text_of(components, "photo_maker");
    params.pulid_weights_path = text_of(components, "pulid");

    // Textual inversions, which upstream takes as an array of name/path pairs
    // rather than as a directory. The vectors outlive the call for the same
    // reason `held` does.
    std::vector<sd_embedding_t> embeddings;
    std::vector<std::string>    embedding_text;
    if (const auto listed = components.find("embeddings");
        listed != components.end() && listed->is_array()) {
        embedding_text.reserve(listed->size() * 2);
        for (const auto & entry : *listed) {
            if (!entry.is_object()) continue;
            const auto name = entry.value("name", std::string());
            const auto path = entry.value("path", std::string());
            if (name.empty() || path.empty()) continue;
            embedding_text.push_back(name);
            embedding_text.push_back(path);
        }
        for (size_t i = 0; i + 1 < embedding_text.size(); i += 2) {
            embeddings.push_back({ embedding_text[i].c_str(), embedding_text[i + 1].c_str() });
        }
    }
    params.embeddings      = embeddings.empty() ? nullptr : embeddings.data();
    params.embedding_count = (uint32_t) embeddings.size();

    params.n_threads = threads > 0 ? threads : od_default_threads();
    params.wtype     = wtype;

    // Per-tensor precision, which `wtype` cannot express: upstream takes rules
    // like `^vae\.=f16,model\.=q8_0`, so the decoder — where quantisation shows
    // up as visible blotching — can stay at a higher precision than the
    // denoiser that dominates the size.
    params.tensor_type_rules = text_of(settings, "tensor_type_rules");

    params.enable_mmap = bool_of("enable_mmap", true);

    // Attention without materialising the N-by-N matrix. Every video example
    // upstream ships passes --diffusion-fa, and the saving is memory as much as
    // time, which is the constraint that binds on a phone. Still unmeasured.
    params.flash_attn           = bool_of("flash_attn", true);
    params.diffusion_flash_attn = bool_of("diffusion_flash_attn", false);

    params.diffusion_conv_direct       = bool_of("diffusion_conv_direct", false);
    params.vae_conv_direct             = bool_of("vae_conv_direct", false);
    params.force_sdxl_vae_conv_scale   = bool_of("force_sdxl_vae_conv_scale", false);

    // A model larger than memory, held mostly on disk.
    //
    // `max_vram` is a budget in GiB; `stream_layers` pages layers in and out
    // against it; `eager_load` does the opposite and pulls everything in at
    // load. These are the only knobs upstream offers for the case where the
    // weights do not fit, which on a phone is most of the interesting cases —
    // and none of them was reachable.
    params.max_vram      = text_of(settings, "max_vram");
    params.stream_layers = bool_of("stream_layers", false);
    params.eager_load    = bool_of("eager_load", false);
    params.auto_fit      = bool_of("auto_fit", false);

    params.backend        = "CPU";
    params.params_backend = "CPU";

    // The enum settings, each refused by name rather than silently ignored.
    // upstream returns its `_COUNT` sentinel for a string it does not know, and
    // passing that through would select whichever behaviour the sentinel
    // happens to land on.
    std::string enum_refusal;
    auto parse_enum = [&](const char * key, auto parser, auto sentinel, auto assign) {
        const char * name = text_of(settings, key);
        if (name == nullptr || !enum_refusal.empty()) return;
        const auto parsed = parser(name);
        if (parsed == sentinel) {
            enum_refusal = std::string("\"") + name + "\" is not a value " + key + " accepts.";
            return;
        }
        assign(parsed);
    };
    parse_enum("rng_type", str_to_rng_type, RNG_TYPE_COUNT,
               [&](rng_type_t v) { params.rng_type = v; });
    parse_enum("sampler_rng_type", str_to_rng_type, RNG_TYPE_COUNT,
               [&](rng_type_t v) { params.sampler_rng_type = v; });
    parse_enum("prediction", str_to_prediction, PREDICTION_COUNT,
               [&](prediction_t v) { params.prediction = v; });
    parse_enum("lora_apply_mode", str_to_lora_apply_mode, LORA_APPLY_MODE_COUNT,
               [&](lora_apply_mode_t v) { params.lora_apply_mode = v; });
    // No `str_to_vae_format` is exported, so this one is spelled out. The names
    // are upstream's own `sd_vae_format_t` spellings.
    if (const char * format = text_of(settings, "vae_format"); format != nullptr) {
        const std::pair<const char *, sd_vae_format_t> formats[] = {
            { "auto",  SD_VAE_FORMAT_AUTO },  { "flux",  SD_VAE_FORMAT_FLUX },
            { "sd3",   SD_VAE_FORMAT_SD3 },   { "flux2", SD_VAE_FORMAT_FLUX2 },
            { "wan",   SD_VAE_FORMAT_WAN },
        };
        bool matched = false;
        for (const auto & [name, value] : formats) {
            if (!jni_iequals(format, name)) continue;
            params.vae_format = value;
            matched = true;
            break;
        }
        if (!matched && enum_refusal.empty()) {
            enum_refusal = std::string("\"") + format + "\" is not a value vae_format accepts.";
        }
    }
    if (!enum_refusal.empty()) {
        jni_throw(env, enum_refusal);
        return 0;
    }

    {
        std::lock_guard<std::mutex> lock(g_last_error_mutex);
        g_last_error.clear();
    }

    auto * engine = new od_sd();
    engine->ctx = new_sd_ctx(&params);

    // A checkpoint and a bare diffusion model go through different doors.
    //
    // `model_path` is for a full checkpoint — the denoiser, its text encoders
    // and a VAE in one file. Most quantised releases are not that: leejet's
    // FLUX.2 Klein and every SDXL GGUF are the denoiser alone, and upstream
    // takes those on `diffusion_model_path`, where it prefixes their tensors
    // with `model.diffusion_model.` before matching. Handed to `model_path`
    // instead, the names match no layout it knows and it stops at
    // "get sd version from file failed" — which reads like a corrupt download
    // and is not one.
    //
    // Which kind a file is cannot be read off its name, and the app has no
    // business guessing from tensor names what upstream's loader is about to
    // decide for itself. So: ask, and if the answer is no, ask the other way.
    // The first attempt fails at version detection, before any weights are
    // allocated, so the retry costs a header read.
    if (engine->ctx == nullptr) {
        {
            std::lock_guard<std::mutex> lock(g_last_error_mutex);
            g_last_error.clear();
        }
        SLOGI("not a full checkpoint; retrying as a bare diffusion model");
        g_bare_diffusion.store(true);
        params.model_path           = nullptr;
        params.diffusion_model_path = model.c_str();
        engine->ctx = new_sd_ctx(&params);
    }

    if (engine->ctx == nullptr) {
        const auto reason = take_last_error();
        delete engine;
        jni_throw(env, "stable-diffusion.cpp could not load this model" +
                       (reason.empty() ? std::string() : ": " + reason));
        return 0;
    }

    // A context can come back non-null with its weights half-built: the tensor load fails, the graph does not, and `new_sd_ctx` still hands back a pointer.
    const auto load_error = take_last_error();
    if (!load_error.empty()) {
        delete engine;
        jni_throw(env, "This model's weights could not be read: " + load_error);
        return 0;
    }

    // What this context can make, asked of the context rather than assumed.
    //
    // This used to demand image generation and refuse anything else, which
    // rejected every video architecture at load — after its weights were read,
    // so the cost was paid and then thrown away. `supports_image_generation` is
    // upstream's `!supports_video_generation`, so the two are exclusive for
    // every checkpoint *except* an SD 1.x carrying a motion module, which
    // answers yes to both and is the one thing here that can do either.
    //
    // The refusal that is left is the honest one: a context that can make
    // neither is a load with nothing to do afterwards.
    engine->supports_image.store(sd_ctx_supports_image_generation(engine->ctx));
    engine->supports_video.store(sd_ctx_supports_video_generation(engine->ctx));
    if (!engine->supports_image.load() && !engine->supports_video.load()) {
        delete engine;
        jni_throw(env, "This model loaded but can generate neither images nor video.");
        return 0;
    }
    SLOGI("can generate: %s%s",
          engine->supports_image.load() ? "images " : "",
          engine->supports_video.load() ? "video" : "");

    sd_set_progress_callback(progress_cb, nullptr);

    // The preview is a linear projection of the latent, which costs nothing and
    // needs no second decoder. sd.cpp can also decode it properly with a TAESD
    // file, and that path is gone: it is another model to find, download and
    // keep in memory, for a thumbnail that is discarded the moment the real
    // decoder runs.
    sd_set_preview_callback(preview_cb, PREVIEW_PROJ,
                            /* interval */ 1, /* denoised */ true, /* noisy */ false, nullptr);

    return reinterpret_cast<jlong>(engine);
}

JNIEXPORT void JNICALL
Java_ai_ondevice_engine_SdBridge_nativeFree(JNIEnv *, jobject, jlong handle) {
    auto * e = as_sd(handle);
    if (g_current.load() == e) g_current.store(nullptr);
    delete e;
}

JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_SdBridge_nativeApplyParams(JNIEnv * env, jobject, jlong handle, jstring jparams) {
    auto * e = as_sd(handle);
    if (e == nullptr) return jni_from_string(env, R"({"applied":[],"rejected":[]})");

    std::lock_guard<std::mutex> lock(e->mutex);
    json applied  = json::array();
    json rejected = json::array();
    try {
        const auto values = json::parse(jni_to_string(env, jparams));
        for (auto it = values.begin(); it != values.end(); ++it) {
            const auto found = table().find(it.key());
            if (found == table().end()) { rejected.push_back(it.key()); continue; }
            if (it.value().is_null()) continue;
            found->second.apply(*e, it.value());
            applied.push_back(it.key());
        }
    } catch (const std::exception & ex) {
        return jni_from_string(env, dump_json(json{ { "applied", applied }, { "rejected", rejected },
                                          { "error", ex.what() } }));
    }
    return jni_from_string(env, dump_json(json{ { "applied", applied }, { "rejected", rejected } }));
}

/** Step, total and s/it, for the polling coroutine that drives the readout. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_SdBridge_nativeProgress(JNIEnv * env, jobject, jlong handle) {
    auto * e = as_sd(handle);
    if (e == nullptr) return jni_from_string(env, "{}");
    const int phase = e->phase.load();
    std::string stage;
    { std::lock_guard<std::mutex> lock(g_stage_mutex); stage = g_stage; }
    return jni_from_string(env, dump_json(json{
        { "step",           e->step.load() },
        { "steps",          e->total_steps.load() },
        { "secondsPerStep", e->seconds_per_step.load() },
        { "generating",     e->generating.load() },
        { "previewSerial",  e->preview_serial.load() },
        { "phase",          phase == PHASE_SAMPLING ? "sampling"
                            : phase == PHASE_DECODING ? "decoding" : "preparing" },
        // The runtime's own last word, so a phase that lasts minutes says which
        // minutes-long thing it is: encoding a prompt through a 4B language
        // model and decoding a latent are both "not sampling" and nothing else
        // alike.
        { "stage",          stage },
    }));
}

/** The latest intermediate latent, decoded, as packed RGB. */
JNIEXPORT jbyteArray JNICALL
Java_ai_ondevice_engine_SdBridge_nativePreview(JNIEnv * env, jobject, jlong handle) {
    auto * e = as_sd(handle);
    if (e == nullptr) return nullptr;
    std::lock_guard<std::mutex> lock(e->preview_mutex);
    if (e->preview_rgb.empty()) return nullptr;

    // Two ints of header so one call carries the dimensions with the pixels;
    // a preview whose size the caller has to guess is a decode waiting to fail.
    const jsize total = (jsize) (e->preview_rgb.size() + 8);
    jbyteArray out = env->NewByteArray(total);
    if (out == nullptr) return nullptr;
    uint8_t header[8];
    const int32_t w = e->preview_width;
    const int32_t h = e->preview_height;
    std::memcpy(header, &w, 4);
    std::memcpy(header + 4, &h, 4);
    env->SetByteArrayRegion(out, 0, 8, reinterpret_cast<const jbyte *>(header));
    env->SetByteArrayRegion(out, 8, (jsize) e->preview_rgb.size(),
                            reinterpret_cast<const jbyte *>(e->preview_rgb.data()));
    return out;
}

JNIEXPORT void JNICALL
Java_ai_ondevice_engine_SdBridge_nativeCancel(JNIEnv *, jobject, jlong handle) {
    auto * e = as_sd(handle);
    if (e == nullptr || e->ctx == nullptr) return;
    // Both, and the order matters only in that neither is enough alone.
    //
    // sd_cancel_generation is upstream's flag, and upstream reads it between
    // denoising steps, between batches and between latents. On a phone one
    // step of a 4B transformer is closer to two minutes than to a moment, and
    // the VAE decode that follows is a single op with no check inside it — so
    // a press could sit unnoticed for the length of the thing it was meant to
    // stop. The atomic below is read by the graph callback, which ggml asks
    // before every node.
    e->cancelled.store(true);
    // Only once sampling is running. Before that, upstream's cancel abandons
    // the conditioner's graph and the assert on its empty result is an abort,
    // not an error — the app dies rather than stopping. The press is remembered
    // and applied the moment sampling begins; the graph callback below already
    // covers everything after that point.
    if (e->sampling_started.load()) {
        sd_cancel_generation(e->ctx, SD_CANCEL_ALL);
    } else {
        e->cancel_deferred.store(true);
    }
}

/** What this context can make, so the app offers the screen that fits it. */
JNIEXPORT jboolean JNICALL
Java_ai_ondevice_engine_SdBridge_nativeSupportsImage(JNIEnv *, jobject, jlong handle) {
    auto * e = as_sd(handle);
    return e != nullptr && e->supports_image.load();
}

JNIEXPORT jboolean JNICALL
Java_ai_ondevice_engine_SdBridge_nativeSupportsVideo(JNIEnv *, jobject, jlong handle) {
    auto * e = as_sd(handle);
    return e != nullptr && e->supports_video.load();
}

/**
 * Generate a clip, and leave it on disk.
 *
 * Returns a manifest — `{"dir","frames":[…],"width","height","fps","audio"}` —
 * rather than pixels. A five-second 480p clip is about 147 MB of raw RGB and
 * upstream returns every frame at once; handing that back as a `byte[]` would
 * hold it three times over, and the third copy is whatever the screen decodes
 * it into. Frames are written as they are converted and released immediately.
 *
 * The audio is the LTX-AV case: it is the only architecture here that returns a
 * soundtrack, and it comes back as floats that are written as a WAV beside the
 * frames.
 */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_SdBridge_nativeGenerateVideo(
        JNIEnv * env, jobject, jlong handle,
        jbyteArray jinit, jint initW, jint initH,
        jbyteArray jend, jint endW, jint endH,
        jbyteArray jcontrol, jint controlW, jint controlH,
        jstring jattachments, jstring joutputDir) {
    auto * e = as_sd(handle);
    if (e == nullptr || e->ctx == nullptr) {
        jni_throw(env, "No diffusion model is loaded.");
        return nullptr;
    }
    if (!e->supports_video.load()) {
        jni_throw(env, "This model does not generate video. An SD 1.x checkpoint needs a "
                       "motion module attached before it can.");
        return nullptr;
    }

    const auto out_dir = jni_to_string(env, joutputDir);
    if (out_dir.empty()) {
        jni_throw(env, "No output directory was given for the frames.");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(e->mutex);
    e->cancelled.store(false);
    g_current.store(e);
    e->generating.store(true);
    e->step.store(0);
    e->phase.store(PHASE_PREPARING);
    e->sampling_started.store(false);
    e->sampler_steps.store(0);
    e->cancel_deferred.store(false);
    { std::lock_guard<std::mutex> lora_lock(g_lora_mutex); g_lora_report.clear(); }
    {
        std::lock_guard<std::mutex> preview_lock(e->preview_mutex);
        e->preview_rgb.clear();
    }

    // The first frame, and the last.
    //
    // `end_image` has no counterpart in image generation: given both, the model
    // is asked to travel from one still to the other, which is a different
    // request from "animate this" and the one most worth having.
    owned_image init    = take_image(env, jinit, initW, initH);
    owned_image last    = take_image(env, jend, endW, endH);
    owned_image control = take_image(env, jcontrol, controlW, controlH);

    std::vector<sd_lora_t>   loras;
    std::vector<std::string> lora_paths;
    std::string              control_net_path;
    /** Only read when the hi-res stage is set to the `model` upscaler. */
    std::string              upscaler_model;
    try {
        const auto attachments_str = jni_to_string(env, jattachments);
        if (!attachments_str.empty()) {
            const auto parsed = json::parse(attachments_str);
            for (const auto & item : parsed) {
                const auto role = item.value("role", "");
                const auto path = item.value("path", "");
                if (path.empty()) continue;
                if (role == "LORA") {
                    lora_paths.push_back(path);
                    loras.push_back(sd_lora_t{
                        // Wan 2.2 runs two denoisers and a LoRA is trained for
                        // one of them; upstream takes the flag per LoRA.
                        /* is_high_noise */ item.value("highNoise", false),
                        /* multiplier    */ item.value("weight", 1.0f),
                        /* path          */ nullptr,
                    });
                } else if (role == "CONTROLNET") {
                    control_net_path = path;
                } else if (role == "UPSCALER") {
                    upscaler_model = path;
                }
            }
        }
    } catch (const std::exception & ex) {
        SLOGE("attachment list rejected: %s", ex.what());
    }
    for (size_t i = 0; i < loras.size(); ++i) {
        loras[i].path = lora_paths[i].c_str();
    }

    // Not loaded, and dropped if a still left one resident.
    //
    // `generate_video` passes an empty control image and a strength of zero to
    // the sampler; a clip's control map goes to VACE instead, whose blocks live
    // inside the checkpoint and whose hold is `vace_strength`. So a ControlNet
    // here is several hundred megabytes that nothing will read.
    if (!control_net_path.empty()) {
        SLOGI("control net ignored on a clip: video control runs through VACE");
    }
    if (sd_ctx_has_control_net(e->ctx)) {
        sd_ctx_unload_control_net(e->ctx);
    }

    sd_vid_gen_params_t params;
    sd_vid_gen_params_init(&params);
    if (!loras.empty()) {
        params.loras      = loras.data();
        params.lora_count = (uint32_t) loras.size();
    }
    params.prompt          = e->prompt.c_str();
    params.negative_prompt = e->negative_prompt.c_str();
    params.width           = e->width;
    params.height          = e->height;
    params.clip_skip       = e->clip_skip;
    params.seed            = e->seed;
    params.strength        = e->strength;
    params.video_frames    = e->video_frames;
    params.fps             = e->fps;
    params.vace_strength   = e->vace_strength;
    params.moe_boundary    = e->moe_boundary;

    apply_sample_params(*e, params.sample_params);
    apply_slg(*e, params.sample_params.guidance.slg);
    // Wan 2.2 is two experts either side of a noise boundary, and each takes
    // its own step count. Defaulting the second to the first keeps one dial
    // meaningful for the architectures that have only one denoiser.
    params.high_noise_sample_params = params.sample_params;
    params.high_noise_sample_params.sample_steps = e->high_noise_steps > 0
                                                       ? e->high_noise_steps
                                                       : e->steps;

    params.vae_tiling_params.enabled = e->vae_tiling;
    apply_cache(*e, params.cache);
    const std::string hires_dropped = apply_video_hires(*e, params.hires, upscaler_model);
    if (!hires_dropped.empty()) {
        SLOGI("hi-res stage skipped: %s", hires_dropped.c_str());
    }

    if (init.image.data != nullptr) {
        params.init_image = init.image;
        params.width  = (int) init.image.width;
        params.height = (int) init.image.height;
    }
    if (last.image.data != nullptr) params.end_image = last.image;
    // One control frame, applied to the sequence. Upstream takes an array —
    // a different map per frame — which nothing in the app can author yet.
    if (control.image.data != nullptr) {
        params.control_frames      = &control.image;
        params.control_frames_size = 1;
    }

    sd_image_t * frames    = nullptr;
    int          frame_count = 0;
    sd_audio_t * audio     = nullptr;
    const bool   ok = generate_video(e->ctx, &params, &frames, &frame_count, &audio);

    const bool cancelled = e->cancelled.load();
    e->generating.store(false);
    g_current.store(nullptr);

    if (!ok || frames == nullptr || frame_count <= 0) {
        if (frames != nullptr) free(frames);
        if (audio != nullptr) { free(audio->data); free(audio); }
        if (cancelled) return nullptr;
        jni_throw(env, "The run produced no frames. This is usually memory — lower the "
                       "size, ask for fewer frames, or enable vae_tiling.");
        return nullptr;
    }

    // Written one at a time and freed as we go, so peak memory is upstream's
    // buffer plus one encoded frame rather than plus a copy of all of them.
    json listed = json::array();
    int  width  = 0;
    int  height = 0;
    for (int i = 0; i < frame_count; ++i) {
        const sd_image_t & frame = frames[i];
        if (frame.data == nullptr) continue;
        width  = (int) frame.width;
        height = (int) frame.height;
        char name[32];
        std::snprintf(name, sizeof(name), "frame_%04d.png", i);
        const std::string path = out_dir + "/" + name;
        const int written = stbi_write_png(path.c_str(), (int) frame.width, (int) frame.height,
                                           (int) frame.channel, frame.data,
                                           (int) frame.width * (int) frame.channel);
        if (written == 0) {
            SLOGE("could not write %s", path.c_str());
        } else {
            listed.push_back(name);
        }
        free(frame.data);
        frames[i].data = nullptr;
    }
    free(frames);

    std::string audio_name;
    if (audio != nullptr) {
        if (audio->data != nullptr && audio->sample_count > 0) {
            audio_name = "audio.wav";
            if (!write_wav(out_dir + "/" + audio_name, *audio)) {
                SLOGE("could not write the audio track");
                audio_name.clear();
            }
        }
        free(audio->data);
        free(audio);
    }

    if (listed.empty()) {
        jni_throw(env, "The frames could not be written. Check free space.");
        return nullptr;
    }

    json manifest{
        { "dir",    out_dir },
        { "frames", listed },
        { "width",  width },
        { "height", height },
        { "fps",    e->fps },
    };
    if (!audio_name.empty()) manifest["audio"] = audio_name;
    SLOGI("wrote %zu frames to %s", listed.size(), out_dir.c_str());
    return jni_from_string(env, dump_json(manifest));
}

/** Run one generation. */
JNIEXPORT jbyteArray JNICALL
Java_ai_ondevice_engine_SdBridge_nativeGenerate(
        JNIEnv * env, jobject, jlong handle,
        jbyteArray jinit, jint initW, jint initH,
        jbyteArray jmask, jint maskW, jint maskH,
        jbyteArray jcontrol, jint controlW, jint controlH,
        jbyteArray jref, jint refW, jint refH,
        jbyteArray jstyle, jint styleW, jint styleH,
        jbyteArray jidentity, jint identityW, jint identityH,
        jstring jattachments) {
    auto * e = as_sd(handle);
    if (e == nullptr || e->ctx == nullptr) {
        jni_throw(env, "No diffusion model is loaded.");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(e->mutex);
    // Cleared before anything a Cancel is meant to interrupt, not after.
    e->cancelled.store(false);
    g_current.store(e);
    e->generating.store(true);
    e->step.store(0);
    e->phase.store(PHASE_PREPARING);
    e->sampling_started.store(false);
    e->sampler_steps.store(0);
    e->cancel_deferred.store(false);
    { std::lock_guard<std::mutex> lock(g_lora_mutex); g_lora_report.clear(); }
    {
        std::lock_guard<std::mutex> preview_lock(e->preview_mutex);
        e->preview_rgb.clear();
    }

    owned_image init    = take_image(env, jinit, initW, initH);
    owned_image mask    = take_image(env, jmask, maskW, maskH);
    owned_image control = take_image(env, jcontrol, controlW, controlH);
    // The picture an edit model is *shown*, as distinct from the one an
    // img2img run starts from. Kontext and FLUX.2 read this one; it is `-r` on
    // upstream's command line and it does not go through denoising strength.
    owned_image reference = take_image(env, jref, refW, refH);
    // The IP-Adapter's own picture, and a third distinct thing: not the map a
    // ControlNet reads, not the picture an edit model is shown. sd.cpp gives it
    // its own field, and it was the one field nothing here ever filled — so an
    // IP-Adapter loaded, cost its weights and a 2.4 GB encoder, and was handed
    // nothing to look at.
    owned_image style = take_image(env, jstyle, styleW, styleH);
    // The face PhotoMaker and PuLID are asked to keep — a fourth distinct
    // picture, and the last field of the five that nothing filled. Both
    // adapters could be attached, both loaded their weights, and neither was
    // ever shown a person, so the run was the run without them.
    owned_image identity = take_image(env, jidentity, identityW, identityH);

    // Attachments arrive as a role-tagged list, not as named arguments.
    std::vector<sd_lora_t>   loras;
    std::vector<std::string> lora_paths;
    std::string              control_net_path;
    /** Only read when the hi-res stage is set to the `model` upscaler. */
    std::string              upscaler_model;
    try {
        const auto attachments_str = jni_to_string(env, jattachments);
        if (!attachments_str.empty()) {
            const auto parsed = json::parse(attachments_str);
            for (const auto & item : parsed) {
                const auto role = item.value("role", "");
                const auto path = item.value("path", "");
                if (path.empty()) continue;
                if (role == "LORA") {
                    lora_paths.push_back(path);
                    loras.push_back(sd_lora_t{
                        /* is_high_noise */ false,
                        /* multiplier    */ item.value("weight", 1.0f),
                        /* path          */ nullptr, // filled after the vector settles
                    });
                } else if (role == "CONTROLNET") {
                    control_net_path = path;
                } else if (role == "UPSCALER") {
                    upscaler_model = path;
                }
            }
        }
    } catch (const std::exception & ex) {
        SLOGE("attachment list rejected: %s", ex.what());
    }
    // The paths are bound only now: `lora_paths` reallocates as it grows, so a
    // c_str() taken while pushing would dangle by the time the vector is done.
    for (size_t i = 0; i < loras.size(); ++i) {
        loras[i].path = lora_paths[i].c_str();
    }

    // ControlNet is hot-swappable, and the header warns it is unsafe during a
    // run — which is why it happens here, inside the lock, before generating.
    if (!control_net_path.empty()) {
        if (!sd_ctx_load_control_net(e->ctx, control_net_path.c_str())) {
            SLOGE("control net refused: %s", control_net_path.c_str());
        }
    } else if (sd_ctx_has_control_net(e->ctx)) {
        sd_ctx_unload_control_net(e->ctx);
    }

    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    if (!loras.empty()) {
        params.loras      = loras.data();
        params.lora_count = (uint32_t) loras.size();
    }
    params.prompt          = e->prompt.c_str();
    params.negative_prompt = e->negative_prompt.c_str();
    params.width           = e->width;
    params.height          = e->height;
    params.clip_skip       = e->clip_skip;
    params.batch_count     = e->batch_count;
    params.seed            = e->seed;
    params.strength        = e->strength;

    apply_sample_params(*e, params.sample_params);
    apply_slg(*e, params.sample_params.guidance.slg);

    params.vae_tiling_params.enabled = e->vae_tiling;
    apply_cache(*e, params.cache);
    apply_hires(*e, params.hires, upscaler_model);

    // The generate-time half of the two identity adapters. The weights were
    // already resident; without these they were resident and idle.
    params.pm_params.style_strength = e->pm_style_strength;
    params.pulid_params.id_weight   = e->pulid_id_weight;
    if (identity.image.data != nullptr) {
        params.pm_params.id_images       = &identity.image;
        params.pm_params.id_images_count = 1;
    }

    if (reference.image.data != nullptr) {
        params.ref_images       = &reference.image;
        params.ref_images_count = 1;
        // An edit model is told what to change, not how far to travel from
        // where it started, so the strength dial has nothing to act on.
        params.width  = (int) reference.image.width;
        params.height = (int) reference.image.height;
    }
    if (init.image.data != nullptr) {
        params.init_image = init.image;
        // Only the source's true size is meaningful for img2img; generating at
        // a different size than the input would silently rescale the result.
        params.width  = (int) init.image.width;
        params.height = (int) init.image.height;
    }
    if (mask.image.data != nullptr)    params.mask_image = mask.image;
    if (control.image.data != nullptr) {
        params.control_image    = control.image;
        params.control_strength = e->control_strength;
    }
    if (style.image.data != nullptr) {
        params.ip_adapter_image    = style.image;
        params.ip_adapter_strength = e->ip_adapter_strength;
    }

    sd_image_t * images = nullptr;
    int          count  = 0;
    const bool   ok     = generate_image(e->ctx, &params, &images, &count);

    const bool cancelled = e->cancelled.load();
    e->generating.store(false);
    g_current.store(nullptr);

    if (!ok || images == nullptr || count <= 0) {
        if (images != nullptr) free(images);
        // A cancelled run has no image either, and telling someone who just
        // pressed Cancel to lower the resolution is a diagnosis of a problem
        // they do not have.
        if (cancelled) {
            return nullptr;
        }
        jni_throw(env, "The diffusion run produced no image. This is usually memory — "
                       "lower the size or enable vae_tiling.");
        return nullptr;
    }

    const sd_image_t & first = images[0];
    const size_t pixels = (size_t) first.width * first.height;
    jbyteArray out = env->NewByteArray((jsize) (pixels * 3 + 8));
    if (out != nullptr) {
        uint8_t header[8];
        const int32_t w = (int32_t) first.width;
        const int32_t h = (int32_t) first.height;
        std::memcpy(header, &w, 4);
        std::memcpy(header + 4, &h, 4);
        env->SetByteArrayRegion(out, 0, 8, reinterpret_cast<const jbyte *>(header));

        std::vector<uint8_t> rgb(pixels * 3);
        for (size_t i = 0; i < pixels; ++i) {
            rgb[i * 3 + 0] = first.data[i * first.channel + 0];
            rgb[i * 3 + 1] = first.data[i * first.channel + 1];
            rgb[i * 3 + 2] = first.data[i * first.channel + 2];
        }
        env->SetByteArrayRegion(out, 8, (jsize) rgb.size(), reinterpret_cast<const jbyte *>(rgb.data()));
    }

    for (int i = 0; i < count; ++i) {
        free(images[i].data);
    }
    free(images);
    return out;
}

/** ESRGAN upscaling. */
JNIEXPORT jbyteArray JNICALL
Java_ai_ondevice_engine_SdBridge_nativeUpscale(
        JNIEnv * env, jobject, jstring jesrgan, jbyteArray jrgb, jint width, jint height,
        jint factor, jint threads, jint tileSize) {
    const auto esrgan = jni_to_string(env, jesrgan);
    if (esrgan.empty()) {
        jni_throw(env, "No upscaler model is installed. Add an ESRGAN model and attach it.");
        return nullptr;
    }

    owned_image src = take_image(env, jrgb, width, height);
    if (src.image.data == nullptr) {
        jni_throw(env, "There is no image to upscale.");
        return nullptr;
    }

    upscaler_ctx_t * upscaler = new_upscaler_ctx(
        esrgan.c_str(),
        /* direct         */ false,
        /* n_threads      */ threads > 0 ? threads : od_default_threads(),
        /* tile_size      */ tileSize,
        /* backend        */ "CPU",
        /* params_backend */ "CPU");
    if (upscaler == nullptr) {
        jni_throw(env, "The upscaler model could not be loaded. ESRGAN weights are expected.");
        return nullptr;
    }

    const uint32_t requested =
        factor > 0 ? (uint32_t) factor : (uint32_t) std::max(1, get_upscale_factor(upscaler));

    sd_image_t * images = nullptr;
    int          count  = 0;
    const bool   ok     = upscale(upscaler, src.image, requested, &images, &count);
    free_upscaler_ctx(upscaler);

    if (!ok || images == nullptr || count <= 0) {
        if (images != nullptr) free(images);
        jni_throw(env, "Upscaling produced no image. This is usually memory — try a smaller "
                       "tile size or a lower factor.");
        return nullptr;
    }

    const sd_image_t & first  = images[0];
    const size_t       pixels = (size_t) first.width * first.height;
    jbyteArray         out    = env->NewByteArray((jsize) (pixels * 3 + 8));
    if (out != nullptr) {
        uint8_t       header[8];
        const int32_t w = (int32_t) first.width;
        const int32_t h = (int32_t) first.height;
        std::memcpy(header, &w, 4);
        std::memcpy(header + 4, &h, 4);
        env->SetByteArrayRegion(out, 0, 8, reinterpret_cast<const jbyte *>(header));

        std::vector<uint8_t> rgb(pixels * 3);
        for (size_t i = 0; i < pixels; ++i) {
            rgb[i * 3 + 0] = first.data[i * first.channel + 0];
            rgb[i * 3 + 1] = first.data[i * first.channel + 1];
            rgb[i * 3 + 2] = first.data[i * first.channel + 2];
        }
        env->SetByteArrayRegion(out, 8, (jsize) rgb.size(), reinterpret_cast<const jbyte *>(rgb.data()));
    }

    for (int i = 0; i < count; ++i) {
        free(images[i].data);
    }
    free(images);
    return out;
}

} // extern "C"
