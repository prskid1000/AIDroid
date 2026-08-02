// stable-diffusion.cpp behind the string-keyed contract (SPEC §16.7).

#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <android/log.h>

#include "stable-diffusion.h"
#include "nlohmann/json.hpp"

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
    float flow_shift  = 0.0f;
    float slg_scale   = 0.0f;
    float skip_layer_start = 0.01f;
    float skip_layer_end   = 0.2f;
    int   width       = 512;
    int   height      = 512;
    int   clip_skip   = -1;
    int   batch_count = 1;
    float strength    = 0.75f;
    float control_strength = 0.9f;
    float ip_adapter_strength = 1.0f;
    int64_t seed      = -1;
    bool  vae_tiling  = false;

    // Published by the callbacks, read by a polling Kotlin coroutine.
    std::atomic<int>   step{0};
    std::atomic<int>   total_steps{0};
    std::atomic<float> seconds_per_step{0.0f};
    std::atomic<bool>  generating{false};
    /** The step count we asked for — used to tell sampling from other phases. */
    std::atomic<int>   expected_steps{0};
    std::atomic<int>   phase{0};
    std::atomic<bool>  sampling_started{false};
    /** Set by Cancel, read by the graph callback before every ggml node. */
    std::atomic<bool>  cancelled{false};

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
    std::lock_guard<std::mutex> lock(g_stage_mutex);
    g_stage = line;
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

constexpr int PHASE_PREPARING = 0;
constexpr int PHASE_SAMPLING  = 1;
constexpr int PHASE_DECODING  = 2;

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

struct row { void (*apply)(od_sd &, const json &); };

const std::map<std::string, row> & table() {
    static const std::map<std::string, row> t = {
        { "prompt",           { [](od_sd & e, const json & v) { e.prompt = as_string(v); } } },
        { "negative_prompt",  { [](od_sd & e, const json & v) { e.negative_prompt = as_string(v); } } },
        { "steps",            { [](od_sd & e, const json & v) { e.steps = std::max(1, as_int(v, e.steps)); } } },
        { "cfg_scale",        { [](od_sd & e, const json & v) { e.cfg_scale = as_float(v, e.cfg_scale); } } },
        { "guidance",         { [](od_sd & e, const json & v) { e.guidance = as_float(v, e.guidance); } } },
        { "flow_shift",       { [](od_sd & e, const json & v) { e.flow_shift = as_float(v, e.flow_shift); } } },
        { "slg_scale",        { [](od_sd & e, const json & v) { e.slg_scale = as_float(v, e.slg_scale); } } },
        { "skip_layer_start", { [](od_sd & e, const json & v) { e.skip_layer_start = as_float(v, e.skip_layer_start); } } },
        { "skip_layer_end",   { [](od_sd & e, const json & v) { e.skip_layer_end = as_float(v, e.skip_layer_end); } } },
        { "width",            { [](od_sd & e, const json & v) { e.width = as_int(v, e.width); } } },
        { "height",           { [](od_sd & e, const json & v) { e.height = as_int(v, e.height); } } },
        { "seed",             { [](od_sd & e, const json & v) { e.seed = as_long(v, e.seed); } } },
        { "clip_skip",        { [](od_sd & e, const json & v) { e.clip_skip = as_int(v, e.clip_skip); } } },
        { "batch_count",      { [](od_sd & e, const json & v) { e.batch_count = std::max(1, as_int(v, e.batch_count)); } } },
        { "strength",         { [](od_sd & e, const json & v) { e.strength = as_float(v, e.strength); } } },
        { "control_strength", { [](od_sd & e, const json & v) { e.control_strength = as_float(v, e.control_strength); } } },
        { "ip_adapter_strength", { [](od_sd & e, const json & v) { e.ip_adapter_strength = as_float(v, e.ip_adapter_strength); } } },
        { "vae_tiling",       { [](od_sd & e, const json & v) { e.vae_tiling = as_bool(v, false); } } },
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
        { "slg_scale",        [](const od_sd & e) { return json(e.slg_scale); } },
        { "skip_layer_start", [](const od_sd & e) { return json(e.skip_layer_start); } },
        { "skip_layer_end",   [](const od_sd & e) { return json(e.skip_layer_end); } },
        { "width",            [](const od_sd & e) { return json(e.width); } },
        { "height",           [](const od_sd & e) { return json(e.height); } },
        { "seed",             [](const od_sd & e) { return json(e.seed); } },
        { "clip_skip",        [](const od_sd & e) { return json(e.clip_skip); } },
        { "batch_count",      [](const od_sd & e) { return json(e.batch_count); } },
        { "strength",         [](const od_sd & e) { return json(e.strength); } },
        { "control_strength", [](const od_sd & e) { return json(e.control_strength); } },
        { "ip_adapter_strength", [](const od_sd & e) { return json(e.ip_adapter_strength); } },
        { "vae_tiling",       [](const od_sd & e) { return json(e.vae_tiling); } },
        { "sampling_method",  [](const od_sd & e) { return json(e.sampling_method); } },
        { "schedule",         [](const od_sd & e) { return json(e.schedule); } },
    };
    return t;
}

void progress_cb(int step, int steps, float time, void *) {
    od_sd * e = g_current.load();
    if (e == nullptr) return;

    if (steps == e->expected_steps.load() && steps > 0) {
        e->step.store(step);
        e->total_steps.store(steps);
        e->phase.store(PHASE_SAMPLING);
        if (time > 0.0f) {
            e->seconds_per_step.store(time);
        }
    } else if (!e->sampling_started.load()) {
        // Before sampling begins this is the loader; afterwards it is the VAE
        // decoding tiles. Both are honest to name and neither is a step.
        e->phase.store(PHASE_PREPARING);
    } else {
        e->phase.store(PHASE_DECODING);
    }

    if (e->phase.load() == PHASE_SAMPLING) {
        e->sampling_started.store(true);
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

JNIEXPORT jlong JNICALL
Java_ai_ondevice_engine_SdBridge_nativeLoad(
        JNIEnv * env, jobject, jstring jmodel, jstring jvae, jstring jcontrolNet,
        jstring jclipL, jstring jclipG, jstring jt5xxl, jstring jipAdapter, jstring jembeddings,
        jstring jclipVision, jstring jllm, jint threads) {
    const auto model      = jni_to_string(env, jmodel);
    const auto vae        = jni_to_string(env, jvae);
    const auto controlNet = jni_to_string(env, jcontrolNet);
    // The rest of sd_ctx_params_t's auxiliary paths.
    const auto clipL      = jni_to_string(env, jclipL);
    const auto clipG      = jni_to_string(env, jclipG);
    const auto t5xxl      = jni_to_string(env, jt5xxl);
    const auto ipAdapter  = jni_to_string(env, jipAdapter);
    const auto embeddings = jni_to_string(env, jembeddings);
    // An IP-Adapter cannot work without this.
    const auto clipVision = jni_to_string(env, jclipVision);
    // FLUX.2 reads its prompt with a language model rather than with CLIP and
    // T5 — Qwen3 for Klein, Mistral Small for dev — so the text encoder is a
    // GGUF the size of a chat model and arrives through its own path.
    const auto llm        = jni_to_string(env, jllm);

    // A version left over from the last checkpoint would be read as this one's.
    { std::lock_guard<std::mutex> lock(g_version_mutex); g_version.clear(); }
    { std::lock_guard<std::mutex> lock(g_stage_mutex); g_stage.clear(); }
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
    params.vae_path         = vae.empty() ? nullptr : vae.c_str();
    params.control_net_path = controlNet.empty() ? nullptr : controlNet.c_str();
    params.clip_l_path      = clipL.empty() ? nullptr : clipL.c_str();
    params.clip_g_path      = clipG.empty() ? nullptr : clipG.c_str();
    params.t5xxl_path       = t5xxl.empty() ? nullptr : t5xxl.c_str();
    params.ip_adapter_path  = ipAdapter.empty() ? nullptr : ipAdapter.c_str();
    params.embeddings_connectors_path = embeddings.empty() ? nullptr : embeddings.c_str();
    params.clip_vision_path = clipVision.empty() ? nullptr : clipVision.c_str();
    params.llm_path         = llm.empty() ? nullptr : llm.c_str();
    params.n_threads        = threads > 0 ? threads : od_default_threads();
    params.enable_mmap      = true;
    // Flash attention stays off.
    params.flash_attn       = false;

    params.backend        = "CPU";
    params.params_backend = "CPU";

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

    if (!sd_ctx_supports_image_generation(engine->ctx)) {
        delete engine;
        jni_throw(env, "This model loaded but does not support image generation.");
        return 0;
    }

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
    return jni_from_string(env, dump_json(json{
        { "step",           e->step.load() },
        { "steps",          e->total_steps.load() },
        { "secondsPerStep", e->seconds_per_step.load() },
        { "generating",     e->generating.load() },
        { "previewSerial",  e->preview_serial.load() },
        { "phase",          phase == PHASE_SAMPLING ? "sampling"
                            : phase == PHASE_DECODING ? "decoding" : "preparing" },
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
    sd_cancel_generation(e->ctx, SD_CANCEL_ALL);
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
    e->expected_steps.store(e->steps);
    e->phase.store(PHASE_PREPARING);
    e->sampling_started.store(false);
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

    // Attachments arrive as a role-tagged list, not as named arguments.
    std::vector<sd_lora_t>   loras;
    std::vector<std::string> lora_paths;
    std::string              control_net_path;
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

    params.sample_params.sample_steps         = e->steps;
    params.sample_params.guidance.txt_cfg     = e->cfg_scale;
    params.sample_params.guidance.distilled_guidance = e->guidance;
    params.sample_params.guidance.slg.scale   = e->slg_scale;
    params.sample_params.guidance.slg.layer_start = e->skip_layer_start;
    params.sample_params.guidance.slg.layer_end   = e->skip_layer_end;
    params.sample_params.flow_shift           = e->flow_shift;
    params.sample_params.sample_method        = str_to_sample_method(e->sampling_method.c_str());
    params.sample_params.scheduler            = str_to_scheduler(e->schedule.c_str());

    params.vae_tiling_params.enabled = e->vae_tiling;

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
