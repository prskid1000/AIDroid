#include "jni_util.h"

#include <algorithm>
#include <cctype>

std::string jni_to_string(JNIEnv * env, jstring value) {
    if (value == nullptr) {
        return {};
    }
    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return {};
    }
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

jstring jni_from_string(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
}

bool jni_iequals(const std::string & a, const std::string & b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](unsigned char x, unsigned char y) {
               return std::tolower(x) == std::tolower(y);
           });
}

void jni_throw(JNIEnv * env, const std::string & message) {
    jclass clazz = env->FindClass("java/lang/IllegalStateException");
    if (clazz != nullptr) {
        env->ThrowNew(clazz, message.c_str());
    }
}
