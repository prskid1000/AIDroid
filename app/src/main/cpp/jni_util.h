#pragma once

#include <jni.h>
#include <string>

// Small helpers shared by every JNI translation unit in this app.

std::string jni_to_string(JNIEnv * env, jstring value);

jstring jni_from_string(JNIEnv * env, const std::string & value);

/** Throws java.lang.IllegalStateException with [message]. */
void jni_throw(JNIEnv * env, const std::string & message);

/** ASCII case-insensitive compare, for matching ggml's registry names. */
bool jni_iequals(const std::string & a, const std::string & b);
