// Grapheme-to-phoneme, via espeak-ng.

#include <jni.h>

#include <mutex>
#include <string>
#include <vector>

#include <espeak-ng/speak_lib.h>

#include "jni_util.h"

namespace {

std::mutex        g_mutex;
bool              g_ready = false;
std::string       g_voice;

/** Punctuation espeak consumes but Kokoro wants to see. */
const char * const KEPT_PUNCTUATION = ";:,.!?";

/** The last kept punctuation mark in [begin, end), or 0 if there is none. */
char trailing_punctuation(const char * begin, const char * end) {
    for (const char * p = end; p > begin; --p) {
        const char c = *(p - 1);
        for (const char * k = KEPT_PUNCTUATION; *k != '\0'; ++k) {
            if (c == *k) return c;
        }
    }
    return '\0';
}

} // namespace

extern "C" {

/** Point espeak at its data and load a voice. */
JNIEXPORT void JNICALL
Java_ai_ondevice_speech_PhonemizerBridge_nativeInit(
        JNIEnv * env, jobject, jstring jdataParent, jstring jvoice) {
    std::lock_guard<std::mutex> lock(g_mutex);

    const auto data_parent = jni_to_string(env, jdataParent);
    const auto voice       = jni_to_string(env, jvoice);

    if (!g_ready) {
        // DONT_EXIT matters: espeak's default failure mode is to call exit(), which on Android takes the whole app down with no message.
        const int rate = espeak_Initialize(
            AUDIO_OUTPUT_SYNCHRONOUS, 0, data_parent.c_str(), espeakINITIALIZE_DONT_EXIT);
        if (rate == EE_INTERNAL_ERROR) {
            jni_throw(env, "espeak-ng could not read its data tables from " + data_parent +
                           ". The unpacked copy may be incomplete.");
            return;
        }
        g_ready = true;
    }

    if (voice != g_voice) {
        if (espeak_SetVoiceByName(voice.c_str()) != EE_OK) {
            jni_throw(env, "espeak-ng has no voice called \"" + voice +
                           "\". Its language data may not be installed in this build.");
            return;
        }
        g_voice = voice;
    }
}

/** Text in, IPA out. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_speech_PhonemizerBridge_nativePhonemize(
        JNIEnv * env, jobject, jstring jtext) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_ready) {
        jni_throw(env, "The phonemiser was used before it was initialised.");
        return jni_from_string(env, "");
    }

    const auto   text = jni_to_string(env, jtext);
    const char * cursor = text.c_str();
    const void * pointer = cursor;

    std::string out;
    while (pointer != nullptr) {
        const char * const before = static_cast<const char *>(pointer);
        const char * phonemes = espeak_TextToPhonemes(&pointer, espeakCHARS_UTF8, 0x02);

        if (phonemes != nullptr && *phonemes != '\0') {
            if (!out.empty() && out.back() != ' ') out += ' ';
            out += phonemes;
        }

        // Whatever espeak just consumed, ending at the new cursor (or the end
        // of the string once it reports completion).
        const char * const after = pointer != nullptr
            ? static_cast<const char *>(pointer)
            : text.c_str() + text.size();
        const char mark = trailing_punctuation(before, after);
        if (mark != '\0') out += mark;
    }

    return jni_from_string(env, out);
}

JNIEXPORT jstring JNICALL
Java_ai_ondevice_speech_PhonemizerBridge_nativeVersion(JNIEnv * env, jobject) {
    const char * path = nullptr;
    const char * version = espeak_Info(&path);
    return jni_from_string(env, version != nullptr ? version : "unknown");
}

JNIEXPORT void JNICALL
Java_ai_ondevice_speech_PhonemizerBridge_nativeFree(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ready) {
        espeak_Terminate();
        g_ready = false;
        g_voice.clear();
    }
}

} // extern "C"
