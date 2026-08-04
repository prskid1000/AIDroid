/*
 * QuickJS, for the workflow Script step.
 *
 * A real language, with a fence around it.
 *
 * The alternative that was nearly taken is a template with a small expression
 * evaluator, and it would have covered most of what people write. What it
 * cannot do is anything with a loop or a branch in it, and "reshape whatever
 * the last model said" turns out to want both often enough that the ceiling
 * would have been hit and then argued about.
 *
 * What makes this safe is not what is written here but what is *not linked*.
 * QuickJS keeps its standard library — files, processes, sockets, the module
 * loader — in quickjs-libc.c, a separate translation unit that this build does
 * not compile (see CMakeLists.txt). The globals a script sees are the ones the
 * language itself defines: Object, Array, String, Math, JSON, RegExp, Date.
 * There is no `require`, no `open`, no `fetch`, and no way to write one,
 * because the C functions that would implement them do not exist in this
 * binary.
 *
 * Three further bounds, all of them enforced here:
 *
 *   - a memory limit on the runtime, so a script that allocates without end
 *     fails as an exception rather than as the kernel killing a process that
 *     is holding ten gigabytes of somebody's model weights;
 *   - a stack limit, so runaway recursion is caught rather than smashing;
 *   - an interrupt handler QuickJS calls between opcodes, which is what makes
 *     `while (true) {}` stoppable at all. Without it the only way out of a
 *     spinning script is to kill the app.
 *
 * The runtime is built and destroyed per evaluation. A script cannot leave
 * anything behind for the next one, which costs about a millisecond and buys
 * the guarantee that two steps in a workflow cannot interfere.
 */

#include <jni.h>
#include <android/log.h>

#include <chrono>
#include <string>

#include "quickjs.h"
#include "jni_util.h"

namespace {

constexpr const char * TAG = "ondevice.quickjs";

/**
 * Sixty-four megabytes.
 *
 * Generous for reshaping text and small beside anything else this app holds —
 * a diffusion checkpoint is three gigabytes. The number exists so that a
 * mistake in a script is an error message rather than a process death, and it
 * is that rather than a considered budget.
 */
constexpr size_t MEMORY_LIMIT = 64u * 1024u * 1024u;

/** One megabyte of JS stack. Deep recursion is a bug, not a workload. */
constexpr size_t STACK_LIMIT = 1u * 1024u * 1024u;

/** How long a script may run before it is stopped. */
constexpr int64_t DEFAULT_TIMEOUT_MS = 2000;

struct Deadline {
    std::chrono::steady_clock::time_point at;
};

/**
 * Called by QuickJS between opcodes; returning non-zero unwinds the script.
 *
 * This is the whole of the answer to a script that never finishes. It is
 * cooperative in the sense that QuickJS chooses when to ask, but it asks often
 * and it asks inside loops, which is where the problem lives.
 */
int interrupt(JSRuntime *, void * opaque) {
    const auto * deadline = static_cast<const Deadline *>(opaque);
    return std::chrono::steady_clock::now() >= deadline->at ? 1 : 0;
}

/** An exception, rendered the way a person reading it would want. */
std::string describe(JSContext * ctx) {
    JSValue error = JS_GetException(ctx);
    const char * text = JS_ToCString(ctx, error);
    std::string message = text ? text : "the script failed";
    if (text) JS_FreeCString(ctx, text);

    // A stack, when there is one, says which line — worth more than the
    // message alone for anything longer than an expression.
    JSValue stack = JS_GetPropertyStr(ctx, error, "stack");
    if (!JS_IsUndefined(stack) && !JS_IsException(stack)) {
        const char * trace = JS_ToCString(ctx, stack);
        if (trace && *trace) {
            message += "\n";
            message += trace;
        }
        if (trace) JS_FreeCString(ctx, trace);
    }
    JS_FreeValue(ctx, stack);
    JS_FreeValue(ctx, error);
    return message;
}

}  // namespace

extern "C" {

/**
 * Evaluate [source], with [inputsJson] bound to the global `steps`.
 *
 * Returns the result as a string. An object or array comes back as JSON, so
 * that a script can hand a list to the next step without inventing a
 * convention; everything else comes back as its text.
 *
 * The result is `{"ok":true,"value":…}` or `{"ok":false,"error":…}` rather
 * than a thrown Java exception, because a script failing is an ordinary thing
 * for a workflow to report on a step and not an exceptional one for the app.
 */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_QuickJsBridge_nativeEval(
    JNIEnv * env,
    jobject,
    jstring jsource,
    jstring jinputs,
    jlong timeoutMs) {

    const std::string source = jni_to_string(env, jsource);
    const std::string inputs = jni_to_string(env, jinputs);
    const int64_t budget = timeoutMs > 0 ? timeoutMs : DEFAULT_TIMEOUT_MS;

    JSRuntime * runtime = JS_NewRuntime();
    if (runtime == nullptr) {
        return jni_from_string(env, R"({"ok":false,"error":"no script runtime"})");
    }

    JS_SetMemoryLimit(runtime, MEMORY_LIMIT);
    JS_SetMaxStackSize(runtime, STACK_LIMIT);

    Deadline deadline{
        std::chrono::steady_clock::now() + std::chrono::milliseconds(budget),
    };
    JS_SetInterruptHandler(runtime, interrupt, &deadline);

    JSContext * ctx = JS_NewContext(runtime);
    if (ctx == nullptr) {
        JS_FreeRuntime(runtime);
        return jni_from_string(env, R"({"ok":false,"error":"no script context"})");
    }

    std::string result;
    bool ok = false;

    // The earlier steps, as one global. Parsed by the engine's own JSON rather
    // than built field by field through the C API: the shapes are arbitrary
    // and this is the one place both sides already agree on a format.
    JSValue global = JS_GetGlobalObject(ctx);
    JSValue steps = JS_ParseJSON(ctx, inputs.c_str(), inputs.size(), "<steps>");
    if (JS_IsException(steps)) {
        JS_FreeValue(ctx, steps);
        steps = JS_NewObject(ctx);
    }
    JS_SetPropertyStr(ctx, global, "steps", steps);
    JS_FreeValue(ctx, global);

    JSValue value = JS_Eval(ctx, source.c_str(), source.size(), "<script>", JS_EVAL_TYPE_GLOBAL);

    if (JS_IsException(value)) {
        result = describe(ctx);
        // Distinguish the two ways a script stops without finishing, because
        // the remedies are different: one is a slow script, the other is a
        // script that will never finish however long it is given.
        if (std::chrono::steady_clock::now() >= deadline.at) {
            result = "The script ran for longer than " + std::to_string(budget) +
                     "ms and was stopped. If it is meant to loop, give it a bound.";
        }
    } else if (JS_IsUndefined(value) || JS_IsNull(value)) {
        // A script whose last statement is an assignment evaluates to nothing.
        // Empty is the honest answer, and the step above reports it.
        ok = true;
    } else if (JS_IsObject(value) && !JS_IsFunction(ctx, value)) {
        JSValue json = JS_JSONStringify(ctx, value, JS_UNDEFINED, JS_UNDEFINED);
        const char * text = JS_ToCString(ctx, json);
        if (text) {
            result = text;
            JS_FreeCString(ctx, text);
            ok = true;
        } else {
            result = "the script returned something that cannot be written down";
        }
        JS_FreeValue(ctx, json);
    } else {
        const char * text = JS_ToCString(ctx, value);
        if (text) {
            result = text;
            JS_FreeCString(ctx, text);
            ok = true;
        } else {
            result = "the script returned something that cannot be read as text";
        }
    }

    JS_FreeValue(ctx, value);
    JS_FreeContext(ctx);
    JS_FreeRuntime(runtime);

    // Hand-built rather than through a JSON library: two fields, one of them
    // needing escaping, and adding a dependency to this shim for that would be
    // the wrong trade.
    std::string out = ok ? R"({"ok":true,"value":)" : R"({"ok":false,"error":)";
    out += '"';
    for (char c : result) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    out += "\"}";

    if (!ok) {
        __android_log_print(ANDROID_LOG_WARN, TAG, "script failed: %s", result.c_str());
    }
    return jni_from_string(env, out);
}

/** What this build embeds, for the About screen and for a bug report. */
JNIEXPORT jstring JNICALL
Java_ai_ondevice_engine_QuickJsBridge_nativeVersion(JNIEnv * env, jobject) {
    return jni_from_string(env, JS_GetVersion());
}

}  // extern "C"
