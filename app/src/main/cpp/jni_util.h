#pragma once

#include <jni.h>
#include <string>

// Small helpers shared by every JNI translation unit in this app.
//
// The rule they exist to enforce: a JNI boundary crossing either produces a
// value or throws a Java exception with a message a human can act on. Nothing
// here ever returns a silent default on failure, because SPEC §1.2 says a
// refusal has to say what went wrong.

std::string jni_to_string(JNIEnv * env, jstring value);

jstring jni_from_string(JNIEnv * env, const std::string & value);

/** Throws java.lang.IllegalStateException with [message]. */
void jni_throw(JNIEnv * env, const std::string & message);

/**
 * ASCII case-insensitive compare, for matching ggml's registry names.
 *
 * The names come from two directions that will never agree on case: a Kotlin
 * enum spells the NPU "HEXAGON", ggml registers it as "HTP", and sd.cpp lowers
 * everything before comparing. Rather than pick a canonical case and hope, the
 * comparison ignores it.
 */
bool jni_iequals(const std::string & a, const std::string & b);
