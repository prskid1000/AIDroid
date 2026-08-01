// The llama.cpp side of SPEC §16.7.

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

#include "llama.h"
#include "ggml-backend.h"
#include "common.h"
#include "sampling.h"
#include "chat.h"
#include "log.h"
#include "build-info.h"

#include "mtmd.h"
#include "mtmd-helper.h"

#include "nlohmann/json.hpp"

#include "jni_util.h"

using json = nlohmann::ordered_json;

/** JSON on its way out to Kotlin, with invalid UTF-8 replaced rather than thrown at. */
static std::string dump_json(const json & value) {
    return value.dump(-1, ' ', false, json::error_handler_t::replace);
}

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "ondevice.llama", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ondevice.llama", __VA_ARGS__)

namespace {

struct od_engine {
    common_params params;

    llama_model *      model = nullptr;
    llama_context *    ctx   = nullptr;
    const llama_vocab * vocab = nullptr;

    common_chat_templates_ptr templates;
    common_sampler *          smpl = nullptr;

    // The vision projector, when the model was installed with one. It is a
    // separate GGUF beside the weights, so it is a load-time parameter like the
    // model path itself — `mmproj` in the table below.
    mtmd_context * mctx = nullptr;

    // What the Jinja template is handed besides the messages — upstream's
    // `--chat-template-kwargs`, verbatim. Values are raw JSON text, so `true`,
    // `"high"` and `2` are all spelled the way they would be in the flag.
    //
    // This is where a hybrid-reasoning model's switch lives: Qwen 3.5 reads
    // `enable_thinking` out of here and emits a different prompt suffix for
    // it. Anything else a template reads goes through the same door rather
    // than through a growing list of named parameters.
    std::map<std::string, std::string> chat_template_kwargs;

    /** The `enable_thinking` kwarg, or upstream's default when unset. */
    bool thinking_enabled() const {
        const auto it = chat_template_kwargs.find("enable_thinking");
        return it == chat_template_kwargs.end() || it->second != "false";
    }

    std::mutex mutex;

    // Prompt cache. The tokens currently resident in the KV, so a follow-up
    // turn only has to decode what actually changed (SPEC §4.3).
    std::vector<llama_token> cached;

    // Generation state.
    bool                     generating   = false;
    std::atomic<bool>        cancelled{false};
    std::string              generated;
    std::vector<llama_token> generated_tokens;
    int                      n_prompt      = 0;
    int                      n_cache_hit   = 0;
    int                      n_generated   = 0;
    int                      n_predict     = -1;
    int64_t                  t_prompt_us   = 0;
    int64_t                  t_gen_start   = 0;
    std::vector<std::string> stops;
    common_chat_params       chat_params;
    // The compiled form of `chat_params.parser`.
    common_peg_arena         chat_parser;
    common_chat_msg          last_msg;
    bool                     saw_stop      = false;
    std::string              stop_reason;

    ~od_engine() {
        if (mctx) {
            mtmd_free(mctx);
        }
        if (smpl) {
            common_sampler_free(smpl);
        }
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }
};

od_engine * as_engine(jlong handle) {
    return reinterpret_cast<od_engine *>(handle);
}

// -------------------------------------------------------------------------- Value coercion.

bool as_bool(const json & v, bool fallback) {
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number())  return v.get<double>() != 0.0;
    if (v.is_string()) {
        const auto s = v.get<std::string>();
        return s == "true" || s == "1" || s == "on" || s == "yes";
    }
    return fallback;
}

int32_t as_int(const json & v, int32_t fallback) {
    if (v.is_number_integer()) return v.get<int32_t>();
    if (v.is_number())         return static_cast<int32_t>(std::llround(v.get<double>()));
    if (v.is_boolean())        return v.get<bool>() ? 1 : 0;
    if (v.is_string()) {
        try { return std::stoi(v.get<std::string>()); } catch (...) { return fallback; }
    }
    return fallback;
}

float as_float(const json & v, float fallback) {
    if (v.is_number())  return v.get<float>();
    if (v.is_boolean()) return v.get<bool>() ? 1.0f : 0.0f;
    if (v.is_string()) {
        try { return std::stof(v.get<std::string>()); } catch (...) { return fallback; }
    }
    return fallback;
}

std::string as_string(const json & v) {
    return v.is_string() ? v.get<std::string>() : dump_json(v);
}

std::vector<std::string> as_string_list(const json & v) {
    std::vector<std::string> out;
    if (v.is_array()) {
        for (const auto & item : v) {
            out.push_back(as_string(item));
        }
    } else if (v.is_string() && !v.get<std::string>().empty()) {
        out.push_back(v.get<std::string>());
    }
    return out;
}

ggml_type as_cache_type(const json & v, ggml_type fallback) {
    const auto name = as_string(v);
    if (name == "f32")  return GGML_TYPE_F32;
    if (name == "f16")  return GGML_TYPE_F16;
    if (name == "bf16") return GGML_TYPE_BF16;
    if (name == "q8_0") return GGML_TYPE_Q8_0;
    if (name == "q5_1") return GGML_TYPE_Q5_1;
    if (name == "q5_0") return GGML_TYPE_Q5_0;
    if (name == "q4_1") return GGML_TYPE_Q4_1;
    if (name == "q4_0") return GGML_TYPE_Q4_0;
    return fallback;
}

// -------------------------------------------------------------------------- The dispatch table.

struct param_row {
    bool reload;
    void (*apply)(od_engine &, const json &);
};

/** Merge a JSON object into the template kwargs, each value kept as JSON text. */
void apply_template_kwargs(od_engine & engine, const json & object) {
    if (!object.is_object()) return;
    for (auto it = object.begin(); it != object.end(); ++it) {
        engine.chat_template_kwargs[it.key()] = dump_json(it.value());
    }
}

const std::map<std::string, param_row> & param_table() {
    static const std::map<std::string, param_row> table = {
        // — load-time —
        { "n_ctx",           { true,  [](od_engine & e, const json & v) { e.params.n_ctx   = as_int(v, e.params.n_ctx);   } } },
        { "n_batch",         { true,  [](od_engine & e, const json & v) { e.params.n_batch = as_int(v, e.params.n_batch); } } },
        { "n_ubatch",        { true,  [](od_engine & e, const json & v) { e.params.n_ubatch = as_int(v, e.params.n_ubatch); } } },
        // No n_gpu_layers, main_gpu or no_kv_offload: this build has one device
        // and offers no way to act on them, so listing them would put three
        // knobs on the parameter screen that read back whatever they are set to
        // and change nothing.
        { "n_threads",       { true,  [](od_engine & e, const json & v) { e.params.cpuparams.n_threads = as_int(v, e.params.cpuparams.n_threads); } } },
        { "n_threads_batch", { true,  [](od_engine & e, const json & v) { e.params.cpuparams_batch.n_threads = as_int(v, e.params.cpuparams_batch.n_threads); } } },
        { "n_parallel",      { true,  [](od_engine & e, const json & v) { e.params.n_parallel = as_int(v, e.params.n_parallel); } } },
        { "use_mmap",        { true,  [](od_engine & e, const json & v) {
              const bool mlock = e.params.load_mode == LLAMA_LOAD_MODE_MLOCK ||
                                 e.params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
              const bool mmap  = as_bool(v, true);
              e.params.load_mode = mmap ? (mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MMAP)
                                        : (mlock ? LLAMA_LOAD_MODE_MLOCK      : LLAMA_LOAD_MODE_NONE);
          } } },
        { "use_mlock",       { true,  [](od_engine & e, const json & v) {
              const bool mmap  = e.params.load_mode == LLAMA_LOAD_MODE_MMAP ||
                                 e.params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
              const bool mlock = as_bool(v, false);
              e.params.load_mode = mmap ? (mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MMAP)
                                        : (mlock ? LLAMA_LOAD_MODE_MLOCK      : LLAMA_LOAD_MODE_NONE);
          } } },
        { "flash_attn",      { true,  [](od_engine & e, const json & v) {
              e.params.flash_attn_type = as_bool(v, false) ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                                           : LLAMA_FLASH_ATTN_TYPE_DISABLED;
          } } },
        { "cache_type_k",    { true,  [](od_engine & e, const json & v) { e.params.cache_type_k = as_cache_type(v, e.params.cache_type_k); } } },
        { "cache_type_v",    { true,  [](od_engine & e, const json & v) { e.params.cache_type_v = as_cache_type(v, e.params.cache_type_v); } } },
        { "check_tensors",   { true,  [](od_engine & e, const json & v) { e.params.check_tensors = as_bool(v, false); } } },
        { "rope_freq_base",  { true,  [](od_engine & e, const json & v) { e.params.rope_freq_base = as_float(v, e.params.rope_freq_base); } } },
        { "rope_freq_scale", { true,  [](od_engine & e, const json & v) { e.params.rope_freq_scale = as_float(v, e.params.rope_freq_scale); } } },
        { "rope_scaling_type", { true, [](od_engine & e, const json & v) {
              const auto name = as_string(v);
              if (name == "none")   e.params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
              if (name == "linear") e.params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
              if (name == "yarn")   e.params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
          } } },
        { "yarn_ext_factor",  { true, [](od_engine & e, const json & v) { e.params.yarn_ext_factor  = as_float(v, e.params.yarn_ext_factor);  } } },
        { "yarn_attn_factor", { true, [](od_engine & e, const json & v) { e.params.yarn_attn_factor = as_float(v, e.params.yarn_attn_factor); } } },
        { "yarn_beta_fast",   { true, [](od_engine & e, const json & v) { e.params.yarn_beta_fast   = as_float(v, e.params.yarn_beta_fast);   } } },
        { "yarn_beta_slow",   { true, [](od_engine & e, const json & v) { e.params.yarn_beta_slow   = as_float(v, e.params.yarn_beta_slow);   } } },
        { "yarn_orig_ctx",    { true, [](od_engine & e, const json & v) { e.params.yarn_orig_ctx    = as_int(v, e.params.yarn_orig_ctx);      } } },

        // — live —
        { "temp",             { false, [](od_engine & e, const json & v) { e.params.sampling.temp = as_float(v, e.params.sampling.temp); } } },
        { "top_k",            { false, [](od_engine & e, const json & v) { e.params.sampling.top_k = as_int(v, e.params.sampling.top_k); } } },
        { "top_p",            { false, [](od_engine & e, const json & v) { e.params.sampling.top_p = as_float(v, e.params.sampling.top_p); } } },
        { "min_p",            { false, [](od_engine & e, const json & v) { e.params.sampling.min_p = as_float(v, e.params.sampling.min_p); } } },
        { "typical_p",        { false, [](od_engine & e, const json & v) { e.params.sampling.typ_p = as_float(v, e.params.sampling.typ_p); } } },
        { "top_n_sigma",      { false, [](od_engine & e, const json & v) { e.params.sampling.top_n_sigma = as_float(v, e.params.sampling.top_n_sigma); } } },
        { "min_keep",         { false, [](od_engine & e, const json & v) { e.params.sampling.min_keep = as_int(v, e.params.sampling.min_keep); } } },
        { "repeat_penalty",   { false, [](od_engine & e, const json & v) { e.params.sampling.penalty_repeat = as_float(v, e.params.sampling.penalty_repeat); } } },
        { "repeat_last_n",    { false, [](od_engine & e, const json & v) { e.params.sampling.penalty_last_n = as_int(v, e.params.sampling.penalty_last_n); } } },
        { "presence_penalty", { false, [](od_engine & e, const json & v) { e.params.sampling.penalty_present = as_float(v, e.params.sampling.penalty_present); } } },
        { "frequency_penalty",{ false, [](od_engine & e, const json & v) { e.params.sampling.penalty_freq = as_float(v, e.params.sampling.penalty_freq); } } },
        { "dry_multiplier",   { false, [](od_engine & e, const json & v) { e.params.sampling.dry_multiplier = as_float(v, e.params.sampling.dry_multiplier); } } },
        { "dry_base",         { false, [](od_engine & e, const json & v) { e.params.sampling.dry_base = as_float(v, e.params.sampling.dry_base); } } },
        { "dry_allowed_length", { false, [](od_engine & e, const json & v) { e.params.sampling.dry_allowed_length = as_int(v, e.params.sampling.dry_allowed_length); } } },
        { "dry_penalty_last_n", { false, [](od_engine & e, const json & v) { e.params.sampling.dry_penalty_last_n = as_int(v, e.params.sampling.dry_penalty_last_n); } } },
        { "dry_sequence_breakers", { false, [](od_engine & e, const json & v) {
              auto list = as_string_list(v);
              if (!list.empty()) e.params.sampling.dry_sequence_breakers = list;
          } } },
        { "xtc_probability",  { false, [](od_engine & e, const json & v) { e.params.sampling.xtc_probability = as_float(v, e.params.sampling.xtc_probability); } } },
        { "xtc_threshold",    { false, [](od_engine & e, const json & v) { e.params.sampling.xtc_threshold = as_float(v, e.params.sampling.xtc_threshold); } } },
        { "mirostat",         { false, [](od_engine & e, const json & v) { e.params.sampling.mirostat = as_int(v, e.params.sampling.mirostat); } } },
        { "mirostat_tau",     { false, [](od_engine & e, const json & v) { e.params.sampling.mirostat_tau = as_float(v, e.params.sampling.mirostat_tau); } } },
        { "mirostat_eta",     { false, [](od_engine & e, const json & v) { e.params.sampling.mirostat_eta = as_float(v, e.params.sampling.mirostat_eta); } } },
        { "dynatemp_range",   { false, [](od_engine & e, const json & v) { e.params.sampling.dynatemp_range = as_float(v, e.params.sampling.dynatemp_range); } } },
        { "dynatemp_exponent",{ false, [](od_engine & e, const json & v) { e.params.sampling.dynatemp_exponent = as_float(v, e.params.sampling.dynatemp_exponent); } } },
        { "seed",             { false, [](od_engine & e, const json & v) {
              const auto seed = as_int(v, -1);
              e.params.sampling.seed = seed < 0 ? LLAMA_DEFAULT_SEED : static_cast<uint32_t>(seed);
          } } },
        { "ignore_eos",       { false, [](od_engine & e, const json & v) { e.params.sampling.ignore_eos = as_bool(v, false); } } },
        { "n_probs",          { false, [](od_engine & e, const json & v) { e.params.sampling.n_probs = as_int(v, e.params.sampling.n_probs); } } },
        { "n_predict",        { false, [](od_engine & e, const json & v) { e.params.n_predict = as_int(v, e.params.n_predict); } } },
        { "n_keep",           { false, [](od_engine & e, const json & v) { e.params.n_keep = as_int(v, e.params.n_keep); } } },
        { "context_shift",    { false, [](od_engine & e, const json & v) { e.params.ctx_shift = as_bool(v, e.params.ctx_shift); } } },
        { "grammar",          { false, [](od_engine & e, const json & v) {
              const auto text = as_string(v);
              e.params.sampling.grammar = text.empty()
                  ? common_grammar()
                  : common_grammar(COMMON_GRAMMAR_TYPE_USER, text);
          } } },
        { "stop",             { false, [](od_engine & e, const json & v) { e.params.antiprompt = as_string_list(v); } } },
        { "chat_template",    { true,  [](od_engine & e, const json & v) { e.params.chat_template = as_string(v); } } },
        // The projector file that lets a text model be shown a picture. Set
        // from the model's VISION_PROJECTOR companion, and only readable at
        // load: mtmd builds its own graph against the weights.
        { "mmproj",           { true,  [](od_engine & e, const json & v) { e.params.mmproj.path = as_string(v); } } },
        // The one kwarg with a switch of its own, because it is the one users
        // reach for. It writes into the same map as the general form below, so
        // whichever is set last wins and there is no second source of truth.
        { "enable_thinking",  { false, [](od_engine & e, const json & v) {
              e.chat_template_kwargs["enable_thinking"] = as_bool(v, true) ? "true" : "false";
          } } },
        { "chat_template_kwargs", { false, [](od_engine & e, const json & v) {
              if (v.is_string()) {
                  // Accepted as a string too, so the value can be pasted
                  // straight from a model card's --chat-template-kwargs line.
                  try {
                      apply_template_kwargs(e, json::parse(as_string(v)));
                  } catch (const std::exception &) { /* not JSON; ignored, and reported as rejected */ }
              } else if (v.is_object()) {
                  apply_template_kwargs(e, v);
              }
          } } },
        { "samplers",         { false, [](od_engine & e, const json & v) {
              // §4.2 — the chain order is the user's, verbatim. An unrecognised
              // name is dropped by upstream's own parser, not by us.
              const auto names = as_string_list(v);
              if (!names.empty()) {
                  e.params.sampling.samplers = common_sampler_types_from_names(names);
              }
          } } },
        { "logit_bias",       { false, [](od_engine & e, const json & v) {
              if (!v.is_object()) return;
              e.params.sampling.logit_bias.clear();
              for (auto it = v.begin(); it != v.end(); ++it) {
                  try {
                      e.params.sampling.logit_bias.push_back({ std::stoi(it.key()), as_float(it.value(), 0.0f) });
                  } catch (...) { /* a non-numeric key is not a token id */ }
              }
          } } },
    };
    return table;
}

// -------------------------------------------------------------------------- What upstream's own defaults are.

const std::map<std::string, json (*)(const common_params &)> & default_table() {
    static const std::map<std::string, json (*)(const common_params &)> table = {
        { "n_ctx",              [](const common_params & p) { return json(p.n_ctx); } },
        { "n_batch",            [](const common_params & p) { return json(p.n_batch); } },
        { "n_ubatch",           [](const common_params & p) { return json(p.n_ubatch); } },
        { "n_threads",          [](const common_params & p) { return json(p.cpuparams.n_threads); } },
        { "n_threads_batch",    [](const common_params & p) { return json(p.cpuparams_batch.n_threads); } },
        { "n_parallel",         [](const common_params & p) { return json(p.n_parallel); } },
        { "check_tensors",      [](const common_params & p) { return json(p.check_tensors); } },
        { "enable_thinking",    [](const common_params &)   { return json(true); } },
        { "chat_template_kwargs", [](const common_params &) { return json::object(); } },
        { "rope_freq_base",     [](const common_params & p) { return json(p.rope_freq_base); } },
        { "rope_freq_scale",    [](const common_params & p) { return json(p.rope_freq_scale); } },
        { "yarn_ext_factor",    [](const common_params & p) { return json(p.yarn_ext_factor); } },
        { "yarn_attn_factor",   [](const common_params & p) { return json(p.yarn_attn_factor); } },
        { "yarn_beta_fast",     [](const common_params & p) { return json(p.yarn_beta_fast); } },
        { "yarn_beta_slow",     [](const common_params & p) { return json(p.yarn_beta_slow); } },
        { "yarn_orig_ctx",      [](const common_params & p) { return json(p.yarn_orig_ctx); } },

        { "temp",               [](const common_params & p) { return json(p.sampling.temp); } },
        { "top_k",              [](const common_params & p) { return json(p.sampling.top_k); } },
        { "top_p",              [](const common_params & p) { return json(p.sampling.top_p); } },
        { "min_p",              [](const common_params & p) { return json(p.sampling.min_p); } },
        { "typical_p",          [](const common_params & p) { return json(p.sampling.typ_p); } },
        { "top_n_sigma",        [](const common_params & p) { return json(p.sampling.top_n_sigma); } },
        { "min_keep",           [](const common_params & p) { return json(p.sampling.min_keep); } },
        { "repeat_penalty",     [](const common_params & p) { return json(p.sampling.penalty_repeat); } },
        { "repeat_last_n",      [](const common_params & p) { return json(p.sampling.penalty_last_n); } },
        { "presence_penalty",   [](const common_params & p) { return json(p.sampling.penalty_present); } },
        { "frequency_penalty",  [](const common_params & p) { return json(p.sampling.penalty_freq); } },
        { "dry_multiplier",     [](const common_params & p) { return json(p.sampling.dry_multiplier); } },
        { "dry_base",           [](const common_params & p) { return json(p.sampling.dry_base); } },
        { "dry_allowed_length", [](const common_params & p) { return json(p.sampling.dry_allowed_length); } },
        { "dry_penalty_last_n", [](const common_params & p) { return json(p.sampling.dry_penalty_last_n); } },
        { "xtc_probability",    [](const common_params & p) { return json(p.sampling.xtc_probability); } },
        { "xtc_threshold",      [](const common_params & p) { return json(p.sampling.xtc_threshold); } },
        { "mirostat",           [](const common_params & p) { return json(p.sampling.mirostat); } },
        { "mirostat_tau",       [](const common_params & p) { return json(p.sampling.mirostat_tau); } },
        { "mirostat_eta",       [](const common_params & p) { return json(p.sampling.mirostat_eta); } },
        { "dynatemp_range",     [](const common_params & p) { return json(p.sampling.dynatemp_range); } },
        { "dynatemp_exponent",  [](const common_params & p) { return json(p.sampling.dynatemp_exponent); } },
        { "ignore_eos",         [](const common_params & p) { return json(p.sampling.ignore_eos); } },
        { "n_probs",            [](const common_params & p) { return json(p.sampling.n_probs); } },

        { "n_predict",          [](const common_params & p) { return json(p.n_predict); } },
        { "n_keep",             [](const common_params & p) { return json(p.n_keep); } },
        { "context_shift",      [](const common_params & p) { return json(p.ctx_shift); } },
    };
    return table;
}

/** Apply a JSON object of parameters. */
json apply_params(od_engine & engine, const json & values, bool * needs_reload) {
    json applied  = json::array();
    json rejected = json::array();

    for (auto it = values.begin(); it != values.end(); ++it) {
        const auto & table = param_table();
        const auto   row   = table.find(it.key());
        if (row == table.end()) {
            rejected.push_back(it.key());
            continue;
        }
        if (it.value().is_null()) {
            continue;
        }
        row->second.apply(engine, it.value());
        applied.push_back(it.key());
        if (row->second.reload && needs_reload != nullptr) {
            *needs_reload = true;
        }
    }

    return json{ { "applied", applied }, { "rejected", rejected } };
}

void rebuild_sampler(od_engine & engine) {
    if (engine.smpl != nullptr) {
        common_sampler_free(engine.smpl);
        engine.smpl = nullptr;
    }
    engine.smpl = common_sampler_init(engine.model, engine.params.sampling);
}

std::string model_meta(llama_model * model, const char * key) {
    char buf[4096];
    const int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    return n < 0 ? std::string() : std::string(buf, n);
}

/**
 * Decode `tokens` in `n_batch` chunks, continuing from whatever is resident.
 *
 * Stop is checked between batches, not only between generated tokens: a long
 * prompt is seconds of work with nothing to show for it, and a Stop pressed
 * during it used to be noticed only once the whole prompt was through.
 */
bool decode_tokens(od_engine & engine, const std::vector<llama_token> & tokens, size_t from) {
    const int n_batch = std::max(1, engine.params.n_batch);
    for (size_t i = from; i < tokens.size(); i += n_batch) {
        if (engine.cancelled) {
            return false;
        }
        const int n = static_cast<int>(std::min<size_t>(n_batch, tokens.size() - i));
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data() + i), n);
        if (llama_decode(engine.ctx, batch) != 0) {
            return false;
        }
    }
    return true;
}

common_chat_msg to_chat_msg(const json & item) {
    common_chat_msg msg;
    msg.role    = item.value("role", "user");
    msg.content = item.value("content", "");
    if (item.contains("tool_calls") && item["tool_calls"].is_array()) {
        for (const auto & call : item["tool_calls"]) {
            common_chat_tool_call tc;
            tc.name      = call.value("name", "");
            tc.arguments = call.value("arguments", "");
            tc.id        = call.value("id", "");
            msg.tool_calls.push_back(tc);
        }
    }
    if (item.contains("tool_name"))    msg.tool_name    = item.value("tool_name", "");
    if (item.contains("tool_call_id")) msg.tool_call_id = item.value("tool_call_id", "");
    if (item.contains("reasoning"))    msg.reasoning_content = item.value("reasoning", "");
    return msg;
}

json from_tool_call(const common_chat_tool_call & call) {
    return json{ { "name", call.name }, { "arguments", call.arguments }, { "id", call.id } };
}

} // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeInit(JNIEnv *, jobject) {
    static std::once_flag once;
    std::call_once(once, [] {
        // Upstream logs every tensor at load; on a phone that is thousands of
        // lines per model and it is not information the user or logcat needs.
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (text == nullptr) {
                return;
            }
            // The loader emits a progress bar as a stream of single-character continuation writes.
            if (level == GGML_LOG_LEVEL_CONT) {
                return;
            }
            const char * trimmed = text;
            while (*trimmed == '.' || *trimmed == ' ' || *trimmed == '\n' || *trimmed == '\r') {
                ++trimmed;
            }
            if (*trimmed == '\0') {
                return;
            }
            if (level >= GGML_LOG_LEVEL_WARN) {
                __android_log_write(level >= GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_WARN,
                                    "ondevice.llama", text);
                return;
            }
            // Two INFO lines are worth the exception, because they answer a question nothing else can: where the weights actually went.
            if (std::strstr(trimmed, "offloade") != nullptr ||
                std::strstr(trimmed, "buffer size") != nullptr ||
                std::strncmp(trimmed, "ggml-hex", 8) == 0) {
                __android_log_write(ANDROID_LOG_INFO, "ondevice.llama", trimmed);
            }
        }, nullptr);
        common_log_pause(common_log_main());
        llama_backend_init();
    });
}

JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeSystemInfo(JNIEnv * env, jobject) {
    json info;
    info["build"]  = llama_build_number();
    info["commit"] = std::string(llama_commit());

    // The backends this build actually registered, read back from ggml rather
    // than asserted. SPEC §8.2 — do not assume what the hardware offers.
    json backends = json::array();
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        backends.push_back(ggml_backend_reg_name(reg));
    }
    info["backends"] = backends;

    json devices = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        devices.push_back(json{
            { "name",        ggml_backend_dev_name(dev) },
            { "description", ggml_backend_dev_description(dev) },
            { "freeBytes",   static_cast<int64_t>(free_bytes) },
            { "totalBytes",  static_cast<int64_t>(total_bytes) },
        });
    }
    info["devices"] = devices;

    return jni_from_string(env, dump_json(info));
}

/** Every parameter key this binary will actually act on, with the reload flag that only the table knows. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeSupportedParams(JNIEnv * env, jobject) {
    // Aggregate-initialised: common_params has no user-provided default constructor, so `const common_params defaults;` will not compile.
    const common_params defaults = {};
    const auto & readers = default_table();

    json out = json::object();
    for (const auto & entry : param_table()) {
        json row{ { "reload", entry.second.reload } };
        const auto reader = readers.find(entry.first);
        if (reader != readers.end()) {
            row["default"] = reader->second(defaults);
        }
        out[entry.first] = row;
    }
    return jni_from_string(env, dump_json(out));
}

/** Load a model. */
JNIEXPORT jlong JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeLoad(JNIEnv * env, jobject, jstring jpath, jstring jparams) {
    const auto path       = jni_to_string(env, jpath);
    const auto params_str = jni_to_string(env, jparams);

    auto * engine = new od_engine();

    // The app has already run §3.3's fit arithmetic and shown the user the numbers.
    engine->params.fit_params = false;
    engine->params.n_ctx      = 4096;
    engine->params.n_batch    = 512;
    engine->params.n_ubatch   = 256;
    engine->params.n_gpu_layers = 0;
    // Every core but one.
    engine->params.cpuparams.n_threads =
        std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
    engine->params.cpuparams_batch.n_threads = engine->params.cpuparams.n_threads;
    engine->params.warmup     = false;

    try {
        if (!params_str.empty()) {
            const auto values = json::parse(params_str);
            apply_params(*engine, values, nullptr);
        }
    } catch (const std::exception & e) {
        LOGE("parameter JSON rejected: %s", e.what());
    }

    llama_model_params mparams = common_model_params_to_llama(engine->params);
    engine->model = llama_model_load_from_file(path.c_str(), mparams);
    if (engine->model == nullptr) {
        delete engine;
        jni_throw(env, "llama.cpp could not load " + path +
                       " — the file is not a GGUF this build understands.");
        return 0;
    }

    // A quantized KV cache without flash attention is a trap, not a trade.
    // ggml's ordinary attention path has no quantized kernels, so it
    // dequantizes K and V on every op of every layer of every token: the phone
    // pegs every core and produces nothing. Upstream pairs the two for this
    // reason. Asking for one here turns the other on rather than leaving the
    // user with a setting that only looks like it saved memory.
    const bool quantized_kv =
        (engine->params.cache_type_k != GGML_TYPE_F16 && engine->params.cache_type_k != GGML_TYPE_F32) ||
        (engine->params.cache_type_v != GGML_TYPE_F16 && engine->params.cache_type_v != GGML_TYPE_F32);
    if (quantized_kv && engine->params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        LOGI("quantized KV cache requested; enabling flash attention, which it requires");
        engine->params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    llama_context_params cparams = common_context_params_to_llama(engine->params);
    engine->ctx = llama_init_from_model(engine->model, cparams);
    if (engine->ctx == nullptr) {
        const auto n_ctx = std::to_string(engine->params.n_ctx);
        delete engine;
        jni_throw(env, quantized_kv
            ? "Model loaded but the context could not be created at n_ctx=" + n_ctx +
              " with a quantized KV cache. Not every model's attention can be flash-attended, "
              "and a quantized cache needs it. Set cache_type_k and cache_type_v back to f16."
            : "Model loaded but the context could not be created at n_ctx=" + n_ctx +
              " — there was not enough memory for the KV cache.");
        return 0;
    }

    engine->vocab     = llama_model_get_vocab(engine->model);
    engine->templates = common_chat_templates_init(engine->model, engine->params.chat_template);
    rebuild_sampler(*engine);

    // A projector that will not load is not a reason to refuse the model: the
    // text half is intact and every text conversation still works. It is a
    // reason to say so, because the alternative is a model that silently
    // ignores every picture it is shown.
    if (!engine->params.mmproj.path.empty()) {
        mtmd_context_params mparams_mtmd = mtmd_context_params_default();
        mparams_mtmd.use_gpu       = false;
        mparams_mtmd.print_timings = false;
        mparams_mtmd.n_threads     = engine->params.cpuparams.n_threads;
        mparams_mtmd.warmup        = false;
        // Encoding an image is seconds of uninterruptible work otherwise.
        // ggml asks before each node and abandons the graph on false, which is
        // the only place a Stop can land inside the projector.
        mparams_mtmd.cb_eval = [](ggml_tensor *, bool, void * user_data) -> bool {
            return !static_cast<od_engine *>(user_data)->cancelled;
        };
        mparams_mtmd.cb_eval_user_data = engine;
        engine->mctx = mtmd_init_from_file(engine->params.mmproj.path.c_str(),
                                           engine->model, mparams_mtmd);
        if (engine->mctx == nullptr) {
            LOGE("vision projector %s would not load; this model stays text-only",
                 engine->params.mmproj.path.c_str());
        } else {
            LOGI("vision projector loaded: vision=%d audio=%d",
                 mtmd_support_vision(engine->mctx) ? 1 : 0,
                 mtmd_support_audio(engine->mctx) ? 1 : 0);
        }
    }

    return reinterpret_cast<jlong>(engine);
}

JNIEXPORT void JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeFree(JNIEnv *, jobject, jlong handle) {
    delete as_engine(handle);
}

/** What the model actually says about itself. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeInfo(JNIEnv * env, jobject, jlong handle) {
    auto * e = as_engine(handle);
    if (e == nullptr) return jni_from_string(env, "{}");

    char desc[512] = {0};
    llama_model_desc(e->model, desc, sizeof(desc));

    const int32_t n_head    = llama_model_n_head(e->model);
    const int32_t n_head_kv = llama_model_n_head_kv(e->model);
    const int32_t n_embd    = llama_model_n_embd(e->model);

    json info;
    info["description"]   = std::string(desc);
    info["architecture"]  = model_meta(e->model, "general.architecture");
    info["contextTrain"]  = llama_model_n_ctx_train(e->model);
    info["contextLoaded"] = static_cast<int32_t>(llama_n_ctx(e->ctx));
    info["layers"]        = llama_model_n_layer(e->model);
    info["embeddingLength"]   = n_embd;
    info["embeddingLengthKv"] = n_head > 0 ? (n_embd / n_head) * std::max(1, n_head_kv) : n_embd;
    info["heads"]         = n_head;
    info["headsKv"]       = n_head_kv;
    info["parameters"]    = static_cast<int64_t>(llama_model_n_params(e->model));
    info["sizeBytes"]     = static_cast<int64_t>(llama_model_size(e->model));
    info["chatTemplate"]  = common_chat_templates_source(e->templates.get());
    info["templateSource"] = e->params.chat_template.empty() ? "gguf.chat_template" : "override";
    info["threads"]       = e->params.cpuparams.n_threads;

    json kwargs = json::object();
    for (const auto & [key, value] : e->chat_template_kwargs) {
        try {
            kwargs[key] = json::parse(value);
        } catch (const std::exception &) { /* written by us; unparseable means dropped */ }
    }
    info["chatTemplateKwargs"] = kwargs;

    // What the projector turned out to be, rather than what the file was named.
    info["vision"]      = e->mctx != nullptr && mtmd_support_vision(e->mctx);
    info["mediaMarker"] = e->mctx != nullptr ? mtmd_get_marker(e->mctx) : mtmd_default_marker();

    json eog = json::array();
    for (llama_token token = 0; token < llama_vocab_n_tokens(e->vocab); ++token) {
        if (llama_vocab_is_eog(e->vocab, token)) {
            eog.push_back(common_token_to_piece(e->vocab, token, true));
        }
    }
    info["eogTokens"] = eog;

    return jni_from_string(env, dump_json(info));
}

JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeApplyParams(JNIEnv * env, jobject, jlong handle, jstring jparams) {
    auto * e = as_engine(handle);
    if (e == nullptr) return jni_from_string(env, R"({"applied":[],"rejected":[]})");

    std::lock_guard<std::mutex> lock(e->mutex);
    bool needs_reload = false;
    json report;
    try {
        report = apply_params(*e, json::parse(jni_to_string(env, jparams)), &needs_reload);
    } catch (const std::exception & ex) {
        return jni_from_string(env, dump_json(json{ { "applied", json::array() },
                                          { "rejected", json::array() },
                                          { "error", ex.what() } }));
    }
    report["needsReload"] = needs_reload;
    rebuild_sampler(*e);
    return jni_from_string(env, dump_json(report));
}

/** Render the chat template. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeFormatPrompt(
        JNIEnv * env, jobject, jlong handle, jstring jmessages, jstring jtools, jboolean addGenerationPrompt) {
    auto * e = as_engine(handle);
    if (e == nullptr) return jni_from_string(env, "{}");

    try {
        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = addGenerationPrompt == JNI_TRUE;
        // Must match what the parser is given, and it was not: this defaults to NONE, so the applied template resolved to a format with no notion of reasoning.
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
        // Both, and not redundantly: the kwargs reach the Jinja template, and
        // `enable_thinking` additionally tells upstream's parser whether to
        // expect a reasoning block back.
        inputs.chat_template_kwargs  = e->chat_template_kwargs;
        inputs.enable_thinking       = e->thinking_enabled();

        for (const auto & item : json::parse(jni_to_string(env, jmessages))) {
            inputs.messages.push_back(to_chat_msg(item));
        }

        const auto tools_str = jni_to_string(env, jtools);
        if (!tools_str.empty() && tools_str != "[]") {
            for (const auto & tool : json::parse(tools_str)) {
                common_chat_tool t;
                t.name        = tool.value("name", "");
                t.description = tool.value("description", "");
                t.parameters  = tool.contains("parameters") ? dump_json(tool["parameters"]) : "{}";
                inputs.tools.push_back(t);
            }
        }

        const auto params = common_chat_templates_apply(e->templates.get(), inputs);

        std::lock_guard<std::mutex> lock(e->mutex);
        e->chat_params = params;
        e->chat_parser = common_peg_arena();
        if (!params.parser.empty()) {
            try {
                e->chat_parser.load(params.parser);
            } catch (const std::exception & ex) {
                // Content-only is the honest fallback, and it is what the app
                // gets anyway; say so rather than leaving a silent mystery.
                LOGE("chat parser for format %s would not load: %s",
                     common_chat_format_name(params.format), ex.what());
            }
        }

        LOGI("chat format=%s tools=%zu parser=%zub loaded=%d",
             common_chat_format_name(params.format), inputs.tools.size(),
             params.parser.size(), e->chat_parser.empty() ? 0 : 1);

        json out;
        out["prompt"]           = params.prompt;
        out["format"]           = common_chat_format_name(params.format);
        out["grammar"]          = params.grammar;
        out["supportsThinking"] = params.supports_thinking;
        out["thinkingStart"]    = params.thinking_start_tag;
        out["additionalStops"]  = params.additional_stops;
        out["templateSource"]   = common_chat_templates_source(e->templates.get());
        return jni_from_string(env, dump_json(out));
    } catch (const std::exception & ex) {
        jni_throw(env, std::string("The chat template could not be rendered: ") + ex.what());
        return jni_from_string(env, "{}");
    }
}

/** Token boundaries for the inspector, with the special ones flagged. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeTokenize(JNIEnv * env, jobject, jlong handle, jstring jtext) {
    auto * e = as_engine(handle);
    if (e == nullptr) return jni_from_string(env, "[]");

    const auto text   = jni_to_string(env, jtext);
    const auto tokens = common_tokenize(e->vocab, text, true, true);

    json out = json::array();
    for (const auto token : tokens) {
        out.push_back(json{
            { "id",      token },
            { "text",    common_token_to_piece(e->vocab, token, true) },
            { "special", llama_vocab_is_eog(e->vocab, token) || llama_vocab_get_attr(e->vocab, token) & LLAMA_TOKEN_ATTR_CONTROL },
        });
    }
    return jni_from_string(env, dump_json(out));
}

JNIEXPORT jint JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeTokenCount(JNIEnv * env, jobject, jlong handle, jstring jtext) {
    auto * e = as_engine(handle);
    if (e == nullptr) return 0;
    return static_cast<jint>(common_tokenize(e->vocab, jni_to_string(env, jtext), true, true).size());
}

/** Begin a generation. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeStartGeneration(
        JNIEnv * env, jobject, jlong handle, jstring jprompt, jstring jstops, jstring jimages) {
    auto * e = as_engine(handle);
    if (e == nullptr) return jni_from_string(env, "{}");

    std::lock_guard<std::mutex> lock(e->mutex);

    // Cleared here rather than after the prompt is in, because everything
    // between the two is work a Stop is meant to interrupt. Clearing it later
    // erased a cancel that arrived while the image was still encoding.
    e->cancelled = false;

    const auto prompt = jni_to_string(env, jprompt);

    std::vector<std::string> image_paths;
    try {
        for (const auto & item : json::parse(jni_to_string(env, jimages))) {
            if (item.is_string()) image_paths.push_back(item.get<std::string>());
        }
    } catch (const std::exception &) { /* no images is the ordinary case */ }

    const uint32_t n_ctx = llama_n_ctx(e->ctx);

    size_t n_common   = 0;
    size_t n_prompt   = 0;
    const int64_t t0  = ggml_time_us();

    if (!image_paths.empty()) {
        if (e->mctx == nullptr) {
            return jni_from_string(env, dump_json(json{
                { "error", "This model was installed without a vision projector, so it cannot be shown a picture." },
                { "suggestion", "Install the model's mmproj companion, or send the message without the image." },
            }));
        }

        // An image turn always starts from an empty context. The image lives in
        // the KV as embeddings, not as tokens, so there is nothing to compare a
        // later prompt against and the prefix cache cannot be trusted.
        llama_memory_seq_rm(llama_get_memory(e->ctx), 0, 0, -1);
        e->cached.clear();

        std::vector<mtmd_bitmap *>       owned;
        std::vector<const mtmd_bitmap *> bitmaps;
        for (const auto & path : image_paths) {
            auto wrapper = mtmd_helper_bitmap_init_from_file(e->mctx, path.c_str(), false);
            if (wrapper.bitmap == nullptr) {
                for (auto * bitmap : owned) mtmd_bitmap_free(bitmap);
                return jni_from_string(env, dump_json(json{
                    { "error", "The image could not be read: " + path.substr(path.find_last_of('/') + 1) },
                    { "suggestion", "Attach a JPEG or PNG." },
                }));
            }
            owned.push_back(wrapper.bitmap);
            bitmaps.push_back(wrapper.bitmap);
        }

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_input_text text{};
        text.text          = prompt.c_str();
        text.text_len      = prompt.size();
        text.add_special   = true;
        text.parse_special = true;

        const int32_t tokenized = mtmd_tokenize(e->mctx, chunks, &text, bitmaps.data(), bitmaps.size());
        if (tokenized != 0) {
            mtmd_input_chunks_free(chunks);
            for (auto * bitmap : owned) mtmd_bitmap_free(bitmap);
            return jni_from_string(env, dump_json(json{
                { "error", tokenized == 1
                    ? "The prompt carries a different number of image markers than images."
                    : "The image could not be pre-processed for this projector." },
                { "suggestion", "Start a new conversation and attach the image again." },
            }));
        }

        n_prompt = mtmd_helper_get_n_tokens(chunks);
        if (n_prompt >= n_ctx) {
            mtmd_input_chunks_free(chunks);
            for (auto * bitmap : owned) mtmd_bitmap_free(bitmap);
            return jni_from_string(env, dump_json(json{
                { "error", "The prompt is " + std::to_string(n_prompt) + " tokens with the image and the context is " +
                           std::to_string(n_ctx) + "." },
                { "suggestion", "Raise n_ctx and reload, or start a new conversation." },
            }));
        }

        llama_pos n_past = 0;
        const int32_t evaluated = mtmd_helper_eval_chunks(
            e->mctx, e->ctx, chunks, 0, 0, std::max(1, e->params.n_batch), true, &n_past);

        mtmd_input_chunks_free(chunks);
        for (auto * bitmap : owned) mtmd_bitmap_free(bitmap);

        if (evaluated != 0) {
            // A Stop pressed mid-encode arrives here as a failed graph, and
            // saying "out of memory" to someone who just pressed Stop would be
            // a lie with a suggestion attached.
            if (e->cancelled) {
                // Half a prompt is in the KV and `cached` still describes the
                // turn before it. Dropping both is the only honest state.
                llama_memory_seq_rm(llama_get_memory(e->ctx), 0, 0, -1);
                e->cached.clear();
                return jni_from_string(env, dump_json(json{ { "cancelled", true } }));
            }
            return jni_from_string(env, dump_json(json{
                { "error", "The image could not be evaluated." },
                { "suggestion", "This is usually memory. Lower n_ctx and try again." },
            }));
        }
        LOGI("vision prompt images=%zu tokens=%zu positions=%d",
             image_paths.size(), n_prompt, static_cast<int>(n_past));
    } else {
        const auto tokens = common_tokenize(e->vocab, prompt, true, true);
        if (tokens.size() >= n_ctx) {
            return jni_from_string(env, dump_json(json{
                { "error", "The prompt is " + std::to_string(tokens.size()) +
                           " tokens and the context is " + std::to_string(n_ctx) + "." },
                { "suggestion", "Raise n_ctx and reload, or start a new conversation." },
            }));
        }

        // How much of the KV we can keep. Stopping one short of the shared prefix
        // guarantees there is always a token to decode and therefore fresh logits.
        while (n_common < e->cached.size() && n_common < tokens.size() &&
               e->cached[n_common] == tokens[n_common]) {
            ++n_common;
        }
        if (n_common == tokens.size() && n_common > 0) {
            --n_common;
        }

        llama_memory_seq_rm(llama_get_memory(e->ctx), 0, static_cast<llama_pos>(n_common), -1);

        if (!decode_tokens(*e, tokens, n_common)) {
            if (e->cancelled) {
                // Half a prompt is in the KV and `cached` still describes the
                // turn before it. Dropping both is the only honest state.
                llama_memory_seq_rm(llama_get_memory(e->ctx), 0, 0, -1);
                e->cached.clear();
                return jni_from_string(env, dump_json(json{ { "cancelled", true } }));
            }
            return jni_from_string(env, dump_json(json{
                { "error", "llama_decode failed while processing the prompt." },
                { "suggestion", "This is usually memory. Lower n_ctx or n_batch and try again." },
            }));
        }
        e->cached = tokens;
        n_prompt  = tokens.size();
    }
    e->t_prompt_us = ggml_time_us() - t0;

    e->n_prompt    = static_cast<int>(n_prompt);
    e->n_cache_hit = static_cast<int>(n_common);
    e->n_generated = 0;
    e->generated.clear();
    e->generated_tokens.clear();
    e->generating   = true;
    e->cancelled    = false;
    e->saw_stop     = false;
    e->stop_reason.clear();
    e->last_msg     = common_chat_msg();
    e->n_predict    = e->params.n_predict;
    e->stops        = e->params.antiprompt;
    for (const auto & stop : e->chat_params.additional_stops) {
        e->stops.push_back(stop);
    }

    common_sampler_reset(e->smpl);
    e->t_gen_start = ggml_time_us();

    const float prompt_per_second = e->t_prompt_us > 0
        ? static_cast<float>(n_prompt - n_common) * 1e6f / static_cast<float>(e->t_prompt_us)
        : 0.0f;

    return jni_from_string(env, dump_json(json{
        { "promptTokens",  e->n_prompt },
        { "cachedTokens",  e->n_cache_hit },
        { "promptPerSecond", prompt_per_second },
        { "contextLimit",  static_cast<int32_t>(n_ctx) },
    }));
}

/** One step. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeNextToken(JNIEnv * env, jobject, jlong handle) {
    auto * e = as_engine(handle);
    if (e == nullptr || !e->generating) {
        return jni_from_string(env, R"({"done":true,"stopReason":"CANCELLED"})");
    }

    std::lock_guard<std::mutex> lock(e->mutex);

    if (e->cancelled) {
        e->generating = false;
        return jni_from_string(env, R"({"done":true,"stopReason":"CANCELLED"})");
    }

    const llama_token token = common_sampler_sample(e->smpl, e->ctx, -1);
    common_sampler_accept(e->smpl, token, true);

    json out;

    if (llama_vocab_is_eog(e->vocab, token)) {
        e->generating   = true;
        out["done"]     = true;
        out["stopReason"] = "EOS";
    } else {
        const auto piece = common_token_to_piece(e->ctx, token, false);
        e->generated += piece;
        e->generated_tokens.push_back(token);
        e->n_generated++;

        // Parse the whole partial reply each step.
        common_chat_parser_params pparams(e->chat_params);
        pparams.parser           = e->chat_parser;
        pparams.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        common_chat_msg msg;
        try {
            msg = common_chat_parse(e->generated, true, pparams);
        } catch (const std::exception &) {
            msg.content = e->generated;
        }

        const auto diffs = common_chat_msg_diff::compute_diffs(e->last_msg, msg);
        std::string content_delta;
        std::string reasoning_delta;
        json        tool_deltas = json::array();
        for (const auto & diff : diffs) {
            content_delta   += diff.content_delta;
            reasoning_delta += diff.reasoning_content_delta;
            if (diff.tool_call_index != std::string::npos) {
                tool_deltas.push_back(json{
                    { "index", static_cast<int>(diff.tool_call_index) },
                    { "call",  from_tool_call(diff.tool_call_delta) },
                });
            }
        }
        e->last_msg = msg;

        out["piece"]          = piece;
        out["contentDelta"]   = content_delta;
        out["reasoningDelta"] = reasoning_delta;
        if (!tool_deltas.empty()) {
            out["toolCallDeltas"] = tool_deltas;
        }
        out["done"] = false;

        // A stop string may straddle tokens, so the check is against the whole
        // reply rather than the piece.
        for (const auto & stop : e->stops) {
            if (!stop.empty() && e->generated.size() >= stop.size() &&
                e->generated.compare(e->generated.size() - stop.size(), stop.size(), stop) == 0) {
                out["done"]       = true;
                out["stopReason"] = "STOP_SEQUENCE";
                break;
            }
        }

        if (!out["done"].get<bool>() && e->n_predict > 0 && e->n_generated >= e->n_predict) {
            out["done"]       = true;
            out["stopReason"] = "MAX_TOKENS";
        }

        if (!out["done"].get<bool>() &&
            static_cast<uint32_t>(e->n_prompt + e->n_generated) >= llama_n_ctx(e->ctx) - 1) {
            out["done"]       = true;
            out["stopReason"] = "CONTEXT_FULL";
        }

        if (!out["done"].get<bool>()) {
            std::vector<llama_token> one{ token };
            if (llama_decode(e->ctx, llama_batch_get_one(one.data(), 1)) != 0) {
                out["done"]       = true;
                out["stopReason"] = "CONTEXT_FULL";
            } else {
                e->cached.push_back(token);
            }
        }
    }

    const int64_t elapsed_us = ggml_time_us() - e->t_gen_start;
    out["generated"]       = e->n_generated;
    out["elapsedMillis"]   = static_cast<int64_t>(elapsed_us / 1000);
    out["tokensPerSecond"] = elapsed_us > 0 ? static_cast<float>(e->n_generated) * 1e6f / static_cast<float>(elapsed_us) : 0.0f;
    out["contextUsed"]     = e->n_prompt + e->n_generated;

    if (out.value("done", false)) {
        e->generating = false;
        // The final parse is non-partial, so a tool call that was still being
        // written is now either complete or was never one.
        common_chat_parser_params pparams(e->chat_params);
        pparams.parser           = e->chat_parser;
        pparams.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        try {
            const auto msg = common_chat_parse(e->generated, false, pparams);
            out["content"]   = msg.content;
            out["reasoning"] = msg.reasoning_content;
            json calls = json::array();
            for (const auto & call : msg.tool_calls) {
                calls.push_back(from_tool_call(call));
            }
            out["toolCalls"] = calls;
        } catch (const std::exception & ex) {
            LOGE("%s parse failed, falling back to raw text: %s",
                 common_chat_format_name(e->chat_params.format), ex.what());
            out["content"] = e->generated;
        }
    }

    return jni_from_string(env, dump_json(out));
}

JNIEXPORT void JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeCancel(JNIEnv *, jobject, jlong handle) {
    auto * e = as_engine(handle);
    if (e != nullptr) {
        e->cancelled = true;
    }
}

/** Drop the KV cache without unloading — §8.3's memory-pressure response. */
JNIEXPORT void JNICALL
Java_ai_ondevice_engine_LlamaBridge_nativeClearCache(JNIEnv *, jobject, jlong handle) {
    auto * e = as_engine(handle);
    if (e == nullptr) return;
    std::lock_guard<std::mutex> lock(e->mutex);
    llama_memory_clear(llama_get_memory(e->ctx), true);
    e->cached.clear();
}

} // extern "C"
