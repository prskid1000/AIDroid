// stable-diffusion.cpp behind the string-keyed contract (SPEC §16.7).
//
// Diffusion differs from the other two engines in one way that shapes the whole
// file: `generate_image` blocks for the entire run. There is no token loop to
// pull from. So progress and the live preview are published *into* this struct
// by sd.cpp's own callbacks, and Kotlin polls them from a second coroutine —
// which keeps the rule that no native thread ever calls into the JVM, and means
// cancellation is a flag rather than a race between two runtimes.
//
// SPEC §5.4 asks for intermediate latents rather than a spinner. That is what
// the preview callback is for, and it is why TAESD is worth loading: it decodes
// a latent to a viewable image cheaply enough to do every few steps.

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

using json = nlohmann::ordered_json;

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

// sd.cpp's callbacks carry a `void* data`, and the app only ever holds one
// diffusion context, so a single current pointer is enough and avoids handing
// raw pointers through a C callback that outlives a JNI frame.
std::atomic<od_sd *> g_current{nullptr};

// The last error sd.cpp logged. Kept so a load failure can be reported with the
// runtime's own words instead of a generic "it didn't work" — the difference
// between "this GGUF uses tensor names longer than ggml supports" and a shrug.
std::mutex  g_last_error_mutex;
std::string g_last_error;

void set_last_error(const char * text) {
    if (text == nullptr) return;
    std::string line(text);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
    }
    // Upstream prefixes every line with `file.hpp:72   - `. That is useful in a
    // terminal and noise in a sentence shown to someone deciding whether to
    // delete a 1.6 GB download, so the file:line is dropped and the message
    // kept. The full line still goes to logcat untouched.
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
    return v.is_string() ? v.get<std::string>() : v.dump();
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
        { "vae_tiling",       { [](od_sd & e, const json & v) { e.vae_tiling = as_bool(v, false); } } },
        { "sampling_method",  { [](od_sd & e, const json & v) { e.sampling_method = as_string(v); } } },
        { "schedule",         { [](od_sd & e, const json & v) { e.schedule = as_string(v); } } },
    };
    return t;
}

/**
 * sd.cpp funnels three unrelated things through one progress callback:
 * weight loading (`pretty_bytes_progress`), VAE tile decoding, and the actual
 * sampling loop. They all arrive as `(step, steps, time)` with no tag.
 *
 * Reporting them all as "step X/Y" is how the screen ended up saying
 * "step 686/686" for a three-step run — 686 was the loader counting tensors.
 * The only signal available is the *total*: the sampling loop is the one whose
 * total matches the step count we asked for. Anything else is a different
 * phase, and is reported as one rather than mislabelled as progress.
 */
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

JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_SdBridge_nativeSystemInfo(JNIEnv * env, jobject) {
    json info;
    info["system"] = std::string(sd_get_system_info());
    info["cores"]  = sd_get_num_physical_cores();
    return jni_from_string(env, info.dump());
}

JNIEXPORT jlong JNICALL
Java_ai_ondevice_engine_SdBridge_nativeLoad(
        JNIEnv * env, jobject, jstring jmodel, jstring jvae, jstring jtaesd, jstring jcontrolNet,
        jstring jclipL, jstring jclipG, jstring jt5xxl, jstring jipAdapter, jstring jembeddings,
        jint threads) {
    const auto model      = jni_to_string(env, jmodel);
    const auto vae        = jni_to_string(env, jvae);
    const auto taesd      = jni_to_string(env, jtaesd);
    const auto controlNet = jni_to_string(env, jcontrolNet);
    // The rest of sd_ctx_params_t's auxiliary paths. These were classified,
    // offered in the UI, sent across as role-tagged attachments — and then
    // dropped, because the per-run attachment loop only understands LORA and
    // CONTROLNET and every other role fell through its if/else. They are
    // load-time fields in sd.cpp, so this is where they belong.
    const auto clipL      = jni_to_string(env, jclipL);
    const auto clipG      = jni_to_string(env, jclipG);
    const auto t5xxl      = jni_to_string(env, jt5xxl);
    const auto ipAdapter  = jni_to_string(env, jipAdapter);
    const auto embeddings = jni_to_string(env, jembeddings);

    static std::once_flag once;
    std::call_once(once, [] {
        sd_set_log_callback([](sd_log_level_t level, const char * text, void *) {
            if (text == nullptr) return;
            if (level >= SD_LOG_ERROR) {
                set_last_error(text);
            }
            if (level >= SD_LOG_WARN) {
                __android_log_write(level >= SD_LOG_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_WARN,
                                    "ondevice.sd", text);
            }
        }, nullptr);
    });

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path       = model.c_str();
    params.vae_path         = vae.empty() ? nullptr : vae.c_str();
    params.taesd_path       = taesd.empty() ? nullptr : taesd.c_str();
    params.control_net_path = controlNet.empty() ? nullptr : controlNet.c_str();
    params.clip_l_path      = clipL.empty() ? nullptr : clipL.c_str();
    params.clip_g_path      = clipG.empty() ? nullptr : clipG.c_str();
    params.t5xxl_path       = t5xxl.empty() ? nullptr : t5xxl.c_str();
    params.ip_adapter_path  = ipAdapter.empty() ? nullptr : ipAdapter.c_str();
    params.embeddings_connectors_path = embeddings.empty() ? nullptr : embeddings.c_str();
    params.n_threads        = threads > 0 ? threads : sd_get_num_physical_cores();
    params.enable_mmap      = true;
    // The GPU backends are not compiled in on this platform, so asking for
    // flash attention would be a claim the build cannot honour.
    params.flash_attn       = false;

    {
        std::lock_guard<std::mutex> lock(g_last_error_mutex);
        g_last_error.clear();
    }

    auto * engine = new od_sd();
    engine->ctx = new_sd_ctx(&params);
    if (engine->ctx == nullptr) {
        const auto reason = take_last_error();
        delete engine;
        jni_throw(env, "stable-diffusion.cpp could not load this model" +
                       (reason.empty() ? std::string() : ": " + reason));
        return 0;
    }

    // A context can come back non-null with its weights half-built: the tensor
    // load fails, the graph does not, and `new_sd_ctx` still hands back a
    // pointer. `sd_ctx_supports_image_generation` believes it too — it reports
    // what the *architecture* can do, not whether the weights arrived. The
    // first honest signal is that ggml logged an error while reading the file,
    // and the next thing that happens otherwise is `ggml_abort` inside the
    // conditioner, which kills the process with no message the user can act on.
    //
    // So: any error logged while building the context disqualifies it. That is
    // deliberately blunt. Refusing a model that would have worked is a bad day;
    // vanishing mid-tap is not something the user can even report.
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

    // TAESD decodes a latent to something that looks like the final image, but
    // it is a *separate model file*. Asking for PREVIEW_TAE without one means
    // the callback never fires and the preview stays empty for the whole run —
    // which is worse than no preview, because the screen sits on "warming up"
    // while the engine is plainly working. PREVIEW_PROJ is a cheap linear
    // projection of the latent: blurry and colour-shifted, but real, and it
    // needs no extra weights. So the mode follows what is actually installed.
    const bool has_taesd = !taesd.empty();
    sd_set_preview_callback(preview_cb, has_taesd ? PREVIEW_TAE : PREVIEW_PROJ,
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
        return jni_from_string(env, json{ { "applied", applied }, { "rejected", rejected },
                                          { "error", ex.what() } }.dump());
    }
    return jni_from_string(env, json{ { "applied", applied }, { "rejected", rejected } }.dump());
}

/** Step, total and s/it, for the polling coroutine that drives the readout. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_SdBridge_nativeProgress(JNIEnv * env, jobject, jlong handle) {
    auto * e = as_sd(handle);
    if (e == nullptr) return jni_from_string(env, "{}");
    const int phase = e->phase.load();
    return jni_from_string(env, json{
        { "step",           e->step.load() },
        { "steps",          e->total_steps.load() },
        { "secondsPerStep", e->seconds_per_step.load() },
        { "generating",     e->generating.load() },
        { "previewSerial",  e->preview_serial.load() },
        { "phase",          phase == PHASE_SAMPLING ? "sampling"
                            : phase == PHASE_DECODING ? "decoding" : "preparing" },
    }.dump());
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
    sd_cancel_generation(e->ctx, SD_CANCEL_ALL);
}

/**
 * Run one generation. Blocks for its whole duration — the caller runs it on a
 * background dispatcher and polls [nativeProgress] meanwhile.
 *
 * Returns packed RGB with an 8-byte width/height header, matching the preview.
 */
JNIEXPORT jbyteArray JNICALL
Java_ai_ondevice_engine_SdBridge_nativeGenerate(
        JNIEnv * env, jobject, jlong handle,
        jbyteArray jinit, jint initW, jint initH,
        jbyteArray jmask, jint maskW, jint maskH,
        jbyteArray jcontrol, jint controlW, jint controlH,
        jstring jattachments) {
    auto * e = as_sd(handle);
    if (e == nullptr || e->ctx == nullptr) {
        jni_throw(env, "No diffusion model is loaded.");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(e->mutex);
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

    // Attachments arrive as a role-tagged list, not as named arguments. That
    // is what keeps this generic: a new auxiliary kind is a new role string and
    // a manifest key, never another parameter on this function and another
    // branch below. The runtime decides whether a given file is usable with the
    // loaded base — the app does not carry a compatibility table it would have
    // to keep correct for every architecture ever released.
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

    sd_image_t * images = nullptr;
    int          count  = 0;
    const bool   ok     = generate_image(e->ctx, &params, &images, &count);

    e->generating.store(false);
    g_current.store(nullptr);

    if (!ok || images == nullptr || count <= 0) {
        if (images != nullptr) free(images);
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

/**
 * ESRGAN upscaling.
 *
 * A separate context from the diffusion one, because that is how sd.cpp models
 * it: `upscaler_ctx_t` owns its own weights and is not part of `sd_ctx_t`. It is
 * built and freed per call rather than held, since an upscale is a deliberate
 * one-off on a finished picture and keeping a second set of weights resident for
 * the rest of the session would cost memory the generator wants.
 *
 * Same wire format as nativeGenerate: an 8-byte header carrying the output
 * width and height, then packed RGB. The dimensions have to be returned rather
 * than inferred, because the model's own factor wins when the caller passes 0.
 */
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

    // Take the backend strings from a default-initialised params struct rather
    // than passing nullptr: the upscaler forwards them to the same backend
    // resolution the diffusion context uses, and this build has no GPU backend
    // compiled in either way.
    sd_ctx_params_t defaults;
    sd_ctx_params_init(&defaults);

    upscaler_ctx_t * upscaler = new_upscaler_ctx(
        esrgan.c_str(),
        /* direct         */ false,
        /* n_threads      */ threads > 0 ? threads : sd_get_num_physical_cores(),
        /* tile_size      */ tileSize,
        defaults.backend,
        defaults.params_backend);
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
