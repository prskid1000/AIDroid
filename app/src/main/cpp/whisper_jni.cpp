// whisper.cpp behind the same string-keyed contract as llama (SPEC §16.7).

#include <jni.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <android/log.h>

#include "whisper.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include "jni_util.h"

using json = nlohmann::ordered_json;

/** JSON on its way out to Kotlin, with invalid UTF-8 replaced rather than thrown at. */
static std::string dump_json(const json & value) {
    return value.dump(-1, ' ', false, json::error_handler_t::replace);
}

#define WLOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ondevice.whisper", __VA_ARGS__)
#define WLOGI(...) __android_log_print(ANDROID_LOG_INFO,  "ondevice.whisper", __VA_ARGS__)

namespace {

struct od_whisper {
    whisper_context * ctx = nullptr;
    std::mutex        mutex;

    // Held so a string parameter outlives the whisper_full call that reads it.
    std::string language      = "auto";
    std::string initial_prompt;

    bool    translate;
    bool    token_timestamps;
    bool    single_segment;
    bool    suppress_blank;
    bool    suppress_nst;
    bool    detect_language;
    bool    split_on_word;
    bool    diarize;
    int32_t beam_size;
    int32_t best_of;
    int32_t audio_ctx;
    int32_t max_len;
    int32_t max_context;
    int32_t offset_ms;
    int32_t duration_ms;
    float   temperature;
    float   temperature_inc;
    float   entropy_thold;
    float   logprob_thold;
    float   no_speech_thold;
    float   word_thold;

    /**
     * Two switches whisper has no field for; they are how this app spells a
     * thing it does to whisper's parameters rather than a thing whisper holds.
     * `no_fallback` becomes a negative `temperature_inc`, and `no_timestamps`
     * is applied after the fact to the segments.
     */
    bool    no_timestamps   = false;
    bool    no_fallback     = false;

    /** Every core but one — about the phone, not about whisper. */
    int32_t threads = std::max(1, (int) std::thread::hardware_concurrency() - 1);

    /**
     * Everything else, taken from whisper rather than written here.
     *
     * Copies drift. `best_of` was 2 where whisper's greedy default is 5, and
     * `token_timestamps` was on where whisper leaves it off — two settings the
     * app had quietly changed for every transcription, with nothing saying so
     * and no way to notice. `whisper_full_default_params` is the only
     * definition of a default, so it is the one read.
     */
    od_whisper() {
        const whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        translate        = p.translate;
        token_timestamps = p.token_timestamps;
        single_segment   = p.single_segment;
        suppress_blank   = p.suppress_blank;
        suppress_nst     = p.suppress_nst;
        detect_language  = p.detect_language;
        split_on_word    = p.split_on_word;
        diarize          = p.tdrz_enable;
        beam_size        = p.beam_search.beam_size;
        best_of          = p.greedy.best_of;
        audio_ctx        = p.audio_ctx;
        max_len          = p.max_len;
        max_context      = p.n_max_text_ctx;
        offset_ms        = p.offset_ms;
        duration_ms      = p.duration_ms;
        temperature      = p.temperature;
        temperature_inc  = p.temperature_inc;
        entropy_thold    = p.entropy_thold;
        logprob_thold    = p.logprob_thold;
        no_speech_thold  = p.no_speech_thold;
        word_thold       = p.thold_pt;
    }

    ~od_whisper() {
        if (ctx) {
            whisper_free(ctx);
        }
    }
};

od_whisper * as_whisper(jlong handle) {
    return reinterpret_cast<od_whisper *>(handle);
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

float as_float(const json & v, float fallback) {
    if (v.is_number()) return v.get<float>();
    if (v.is_string()) { try { return std::stof(v.get<std::string>()); } catch (...) { return fallback; } }
    return fallback;
}

std::string as_string(const json & v) {
    return v.is_string() ? v.get<std::string>() : dump_json(v);
}

struct row { void (*apply)(od_whisper &, const json &); };

const std::map<std::string, row> & table() {
    static const std::map<std::string, row> t = {
        { "language",         { [](od_whisper & e, const json & v) { e.language = as_string(v); } } },
        { "translate",        { [](od_whisper & e, const json & v) { e.translate = as_bool(v, false); } } },
        { "detect_language",  { [](od_whisper & e, const json & v) { e.detect_language = as_bool(v, false); } } },
        { "beam_size",        { [](od_whisper & e, const json & v) { e.beam_size = as_int(v, e.beam_size); } } },
        { "best_of",          { [](od_whisper & e, const json & v) { e.best_of = as_int(v, e.best_of); } } },
        { "audio_ctx",        { [](od_whisper & e, const json & v) { e.audio_ctx = as_int(v, e.audio_ctx); } } },
        { "threads",          { [](od_whisper & e, const json & v) { e.threads = std::max(1, as_int(v, e.threads)); } } },
        { "max_len",          { [](od_whisper & e, const json & v) { e.max_len = as_int(v, e.max_len); } } },
        { "max_context",      { [](od_whisper & e, const json & v) { e.max_context = as_int(v, e.max_context); } } },
        { "offset_t",         { [](od_whisper & e, const json & v) { e.offset_ms = as_int(v, e.offset_ms); } } },
        { "duration",         { [](od_whisper & e, const json & v) { e.duration_ms = as_int(v, e.duration_ms); } } },
        { "split_on_word",    { [](od_whisper & e, const json & v) { e.split_on_word = as_bool(v, false); } } },
        { "word_thold",       { [](od_whisper & e, const json & v) { e.word_thold = as_float(v, e.word_thold); } } },
        { "entropy_thold",    { [](od_whisper & e, const json & v) { e.entropy_thold = as_float(v, e.entropy_thold); } } },
        { "logprob_thold",    { [](od_whisper & e, const json & v) { e.logprob_thold = as_float(v, e.logprob_thold); } } },
        { "no_speech_thold",  { [](od_whisper & e, const json & v) { e.no_speech_thold = as_float(v, e.no_speech_thold); } } },
        { "temperature",      { [](od_whisper & e, const json & v) { e.temperature = as_float(v, e.temperature); } } },
        { "temperature_inc",  { [](od_whisper & e, const json & v) { e.temperature_inc = as_float(v, e.temperature_inc); } } },
        { "no_fallback",      { [](od_whisper & e, const json & v) { e.no_fallback = as_bool(v, false); } } },
        { "prompt",           { [](od_whisper & e, const json & v) { e.initial_prompt = as_string(v); } } },
        { "diarize",          { [](od_whisper & e, const json & v) { e.diarize = as_bool(v, false); } } },
        { "suppress_blank",   { [](od_whisper & e, const json & v) { e.suppress_blank = as_bool(v, true); } } },
        { "suppress_nst",     { [](od_whisper & e, const json & v) { e.suppress_nst = as_bool(v, false); } } },
        { "single_segment",   { [](od_whisper & e, const json & v) { e.single_segment = as_bool(v, false); } } },

    };
    return t;
}

// Every setter above reads the live value as its fallback, so a default-constructed od_whisper holds this build's default for each key.
const std::map<std::string, json (*)(const od_whisper &)> & default_table() {
    static const std::map<std::string, json (*)(const od_whisper &)> t = {
        { "language",         [](const od_whisper & e) { return json(e.language); } },
        { "translate",        [](const od_whisper & e) { return json(e.translate); } },
        { "detect_language",  [](const od_whisper & e) { return json(e.detect_language); } },
        { "beam_size",        [](const od_whisper & e) { return json(e.beam_size); } },
        { "best_of",          [](const od_whisper & e) { return json(e.best_of); } },
        { "audio_ctx",        [](const od_whisper & e) { return json(e.audio_ctx); } },
        { "threads",          [](const od_whisper & e) { return json(e.threads); } },
        { "max_len",          [](const od_whisper & e) { return json(e.max_len); } },
        { "max_context",      [](const od_whisper & e) { return json(e.max_context); } },
        { "offset_t",         [](const od_whisper & e) { return json(e.offset_ms); } },
        { "duration",         [](const od_whisper & e) { return json(e.duration_ms); } },
        { "split_on_word",    [](const od_whisper & e) { return json(e.split_on_word); } },
        { "word_thold",       [](const od_whisper & e) { return json(e.word_thold); } },
        { "entropy_thold",    [](const od_whisper & e) { return json(e.entropy_thold); } },
        { "logprob_thold",    [](const od_whisper & e) { return json(e.logprob_thold); } },
        { "no_speech_thold",  [](const od_whisper & e) { return json(e.no_speech_thold); } },
        { "temperature",      [](const od_whisper & e) { return json(e.temperature); } },
        { "temperature_inc",  [](const od_whisper & e) { return json(e.temperature_inc); } },
        { "no_fallback",      [](const od_whisper & e) { return json(e.no_fallback); } },
        { "prompt",           [](const od_whisper & e) { return json(e.initial_prompt); } },
        { "diarize",          [](const od_whisper & e) { return json(e.diarize); } },
        { "suppress_blank",   [](const od_whisper & e) { return json(e.suppress_blank); } },
        { "suppress_nst",     [](const od_whisper & e) { return json(e.suppress_nst); } },
        { "single_segment",   [](const od_whisper & e) { return json(e.single_segment); } },

    };
    return t;
}

whisper_full_params build_params(od_whisper & e) {
    // Beam search when a beam size was asked for, greedy otherwise.
    whisper_full_params p = whisper_full_default_params(
        e.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

    p.print_realtime   = false;
    p.print_progress   = false;
    p.print_timestamps = false;
    p.print_special    = false;

    p.translate        = e.translate;
    p.detect_language  = e.detect_language;
    p.language         = e.language.empty() || e.language == "auto" ? nullptr : e.language.c_str();
    p.n_threads        = e.threads;
    p.audio_ctx        = e.audio_ctx;
    p.max_len          = e.max_len;
    p.n_max_text_ctx   = e.max_context >= 0 ? e.max_context : p.n_max_text_ctx;
    p.offset_ms        = e.offset_ms;
    p.duration_ms      = e.duration_ms;
    p.split_on_word    = e.split_on_word;
    p.token_timestamps = e.token_timestamps;
    p.thold_pt         = e.word_thold;
    p.entropy_thold    = e.entropy_thold;
    p.logprob_thold    = e.logprob_thold;
    p.no_speech_thold  = e.no_speech_thold;
    p.temperature      = e.temperature;
    // A negative increment disables the temperature fallback entirely, which is
    // what "no_fallback" means upstream.
    p.temperature_inc  = e.no_fallback ? -1.0f : e.temperature_inc;
    p.suppress_blank   = e.suppress_blank;
    p.suppress_nst     = e.suppress_nst;
    p.single_segment   = e.single_segment;
    p.tdrz_enable      = e.diarize;
    if (!e.initial_prompt.empty()) {
        p.initial_prompt = e.initial_prompt.c_str();
    }
    if (e.beam_size > 1) {
        p.beam_search.beam_size = e.beam_size;
    } else {
        p.greedy.best_of = e.best_of;
    }

    return p;
}

/** whisper's log, forwarded to logcat — which llama has had and whisper has not. */
void forward_log(ggml_log_level level, const char * text, void *) {
    if (text == nullptr) return;
    const char * trimmed = text;
    while (*trimmed == ' ' || *trimmed == '\t' || *trimmed == '\n') ++trimmed;
    if (*trimmed == '\0') return;

    int priority;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: priority = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  priority = ANDROID_LOG_WARN;  break;
        case GGML_LOG_LEVEL_INFO:
        case GGML_LOG_LEVEL_CONT:  priority = ANDROID_LOG_INFO;  break;
        default: return;
    }
    __android_log_write(priority, "ondevice.whisper", trimmed);
}

} // namespace

extern "C" {

/** The keys this binary will act on. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_WhisperBridge_nativeSupportedParams(JNIEnv * env, jobject) {
    const od_whisper defaults = {};
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

/** What ggml registered in *this* binary — the same question llama and sd are asked, and it was the one runtime with no way to answer. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_WhisperBridge_nativeSystemInfo(JNIEnv * env, jobject) {
    json backends = json::array();
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        backends.push_back(ggml_backend_reg_name(ggml_backend_reg_get(i)));
    }
    return jni_from_string(env, dump_json(json{ { "backends", backends } }));
}

JNIEXPORT jlong JNICALL
Java_ai_ondevice_engine_WhisperBridge_nativeLoad(JNIEnv * env, jobject, jstring jpath) {
    const auto path = jni_to_string(env, jpath);

    // Before the first model, so the load's own placement lines are the first
    // thing it says rather than the first thing it misses.
    static std::once_flag logging_once;
    std::call_once(logging_once, [] { whisper_log_set(forward_log, nullptr); });

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    WLOGI("load %s", path.c_str());

    // A failed load is an exception on the Kotlin side, never an abort.
    auto * engine = new od_whisper();
    try {
        engine->ctx = whisper_init_from_file_with_params(path.c_str(), cparams);
    } catch (const std::exception & ex) {
        delete engine;
        jni_throw(env, "whisper.cpp could not load " + path + " — " + ex.what());
        return 0;
    } catch (...) {
        delete engine;
        jni_throw(env, "whisper.cpp could not load " + path + ".");
        return 0;
    }
    if (engine->ctx == nullptr) {
        delete engine;
        jni_throw(env, "whisper.cpp could not load " + path +
                       " — it is not a GGML model this build understands.");
        return 0;
    }
    return reinterpret_cast<jlong>(engine);
}

JNIEXPORT void JNICALL
Java_ai_ondevice_engine_WhisperBridge_nativeFree(JNIEnv *, jobject, jlong handle) {
    delete as_whisper(handle);
}

JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_WhisperBridge_nativeApplyParams(JNIEnv * env, jobject, jlong handle, jstring jparams) {
    auto * e = as_whisper(handle);
    if (e == nullptr) return jni_from_string(env, R"({"applied":[],"rejected":[]})");

    std::lock_guard<std::mutex> lock(e->mutex);
    json applied  = json::array();
    json rejected = json::array();
    try {
        const auto values = json::parse(jni_to_string(env, jparams));
        for (auto it = values.begin(); it != values.end(); ++it) {
            const auto found = table().find(it.key());
            if (found == table().end()) {
                rejected.push_back(it.key());
                continue;
            }
            if (it.value().is_null()) continue;
            found->second.apply(*e, it.value());
            applied.push_back(it.key());
        }
    } catch (const std::exception & ex) {
        return jni_from_string(env, dump_json(json{ { "applied", applied }, { "rejected", rejected },
                                          { "error", ex.what() } }));
    } catch (...) {
        return jni_from_string(env, dump_json(json{ { "applied", applied }, { "rejected", rejected },
                                          { "error", "the runtime rejected these parameters" } }));
    }
    return jni_from_string(env, dump_json(json{ { "applied", applied }, { "rejected", rejected } }));
}

JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_WhisperBridge_nativeInfo(JNIEnv * env, jobject, jlong handle) {
    auto * e = as_whisper(handle);
    if (e == nullptr) return jni_from_string(env, "{}");
    json info;
    info["multilingual"] = whisper_is_multilingual(e->ctx) != 0;
    info["vocabSize"]    = whisper_model_n_vocab(e->ctx);
    info["audioLayers"]  = whisper_model_n_audio_layer(e->ctx);
    info["textLayers"]   = whisper_model_n_text_layer(e->ctx);
    info["type"]         = whisper_model_type_readable(e->ctx);
    info["threads"]      = e->threads;
    return jni_from_string(env, dump_json(info));
}

/** Transcribe a block of mono 16 kHz float samples. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_WhisperBridge_nativeTranscribe(
        JNIEnv * env, jobject, jlong handle, jfloatArray jsamples) {
    auto * e = as_whisper(handle);
    if (e == nullptr) return jni_from_string(env, R"({"error":"No model is loaded."})");

    const jsize n = env->GetArrayLength(jsamples);
    if (n <= 0) {
        return jni_from_string(env, R"({"segments":[]})");
    }
    std::vector<float> samples(n);
    env->GetFloatArrayRegion(jsamples, 0, n, samples.data());

    std::lock_guard<std::mutex> lock(e->mutex);
    whisper_full_params params = build_params(*e);

    // A decode that throws is a failed transcription, not a dead app.
    int64_t t0 = 0;
    try {
        t0 = whisper_full(e->ctx, params, samples.data(), (int) samples.size());
    } catch (const std::exception & ex) {
        WLOGE("whisper_full threw: %s", ex.what());
        return jni_from_string(env, dump_json(json{ { "error", std::string(ex.what()) } }));
    } catch (...) {
        WLOGE("whisper_full threw");
        return jni_from_string(env, dump_json(json{ { "error", "the decode failed" } }));
    }
    if (t0 != 0) {
        return jni_from_string(env, dump_json(json{ { "error", "whisper_full failed" } }));
    }

    json segments = json::array();
    const int count = whisper_full_n_segments(e->ctx);
    for (int i = 0; i < count; ++i) {
        const char * text = whisper_full_get_segment_text(e->ctx, i);

        float sum = 0.0f;
        const int tokens = whisper_full_n_tokens(e->ctx, i);
        int counted = 0;
        for (int j = 0; j < tokens; ++j) {
            const auto id = whisper_full_get_token_id(e->ctx, i, j);
            // Special tokens carry no useful probability for this purpose.
            if (id >= whisper_token_eot(e->ctx)) continue;
            sum += whisper_full_get_token_p(e->ctx, i, j);
            counted++;
        }

        segments.push_back(json{
            // whisper reports centiseconds; the app works in milliseconds.
            { "startMillis", (int64_t) whisper_full_get_segment_t0(e->ctx, i) * 10 },
            { "endMillis",   (int64_t) whisper_full_get_segment_t1(e->ctx, i) * 10 },
            { "text",        text == nullptr ? "" : std::string(text) },
            { "confidence",  counted > 0 ? sum / (float) counted : 1.0f },
            { "speakerTurn", whisper_full_get_segment_speaker_turn_next(e->ctx, i) },
        });
    }

    json out;
    out["segments"] = segments;
    out["language"] = whisper_lang_str(whisper_full_lang_id(e->ctx));
    return jni_from_string(env, dump_json(out));
}

} // extern "C"
