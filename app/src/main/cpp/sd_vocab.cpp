// Replaces stable-diffusion.cpp's src/tokenizers/vocab/vocab.cpp.
//
// Same twelve functions, but only three of them carry data. See the note in
// CMakeLists.txt for why: the other nine vocabularies belong to text encoders
// that only ship beside diffusion models an order of magnitude too large for
// this device, and together they were 49 MB of the APK.
//
// Qwen 2 was among the dropped nine and is back, because the reasoning that
// dropped it stopped being true. FLUX.2 Klein is 1.4 GB at Q4 and reads its
// prompt with Qwen3, whose tokenizer is Qwen 2's; without these merges it
// loaded, allocated four gigabytes and could not turn the prompt into tokens.
//
// A dropped vocabulary returns an empty string and says so in the log. sd.cpp
// builds a tokenizer with no merges from that, which would encode a prompt to
// nonsense — but reaching this at all means a model whose weights could not
// have been loaded in the first place, so the log line is the diagnosis rather
// than a fallback.

#include "tokenizers/vocab/vocab.h"

#include <android/log.h>

#include "tokenizers/vocab/clip_merges.hpp"
#include "tokenizers/vocab/qwen_merges.hpp"
#include "tokenizers/vocab/t5.hpp"

namespace {

std::string embedded(const unsigned char * data, size_t size) {
    return std::string(reinterpret_cast<const char *>(data), size);
}

std::string not_built_in(const char * encoder) {
    __android_log_print(
        ANDROID_LOG_ERROR, "ondevice.sd",
        "no built-in %s vocabulary: this build carries CLIP, Qwen 2 and T5 only", encoder);
    return {};
}

}  // namespace

std::string load_clip_merges() {
    return embedded(clip_merges_utf8_c_str, sizeof(clip_merges_utf8_c_str));
}

std::string load_t5_tokenizer_json() {
    return embedded(t5_tokenizer_json_str, sizeof(t5_tokenizer_json_str));
}

/** FLUX.2's encoder is a Qwen3, and Qwen3 tokenises as Qwen 2 does. */
std::string load_qwen2_merges() {
    return embedded(qwen2_merges_utf8_c_str, sizeof(qwen2_merges_utf8_c_str));
}

std::string load_umt5_tokenizer_json() { return not_built_in("UMT5"); }
std::string load_mistral_merges() { return not_built_in("Mistral"); }
std::string load_mistral_vocab_json() { return not_built_in("Mistral"); }
std::string load_gemma_merges() { return not_built_in("Gemma"); }
std::string load_gemma_vocab_json() { return not_built_in("Gemma"); }
std::string load_gemma2_merges() { return not_built_in("Gemma 2"); }
std::string load_gemma2_vocab_json() { return not_built_in("Gemma 2"); }
std::string load_gpt_oss_merges() { return not_built_in("GPT-OSS"); }
std::string load_gpt_oss_vocab_json() { return not_built_in("GPT-OSS"); }
