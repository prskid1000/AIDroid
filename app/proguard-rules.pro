# kotlinx.serialization
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.**
-keepclassmembers class ai.ondevice.** {
    *** Companion;
}
-keepclasseswithmembers class ai.ondevice.** {
    kotlinx.serialization.KSerializer serializer(...);
}

# JNI entry points for the native runtimes (SPEC 16.7 — the boundary is a
# string-keyed map, so only the bridge classes need keeping).
#
# This used to read `ai.ondevice.engine.jni.**`, a package that has never
# existed: the bridges are in `ai.ondevice.engine` and `ai.ondevice.speech`, so
# the rule kept nothing. Nothing broke, because proguard-android-optimize.txt
# carries `-keepclasseswithmembernames class * { native <methods>; }` and every
# one of these is a class of native methods — verified against mapping.txt,
# where all four appear unrenamed. Naming them anyway: a rule whose protection
# is entirely accidental is one nobody can reason about, and the symbol the
# linker looks for is spelled out of the class's fully-qualified name.
-keepclasseswithmembernames,includedescriptorclasses class ai.ondevice.engine.LlamaBridge,
    ai.ondevice.engine.WhisperBridge,
    ai.ondevice.engine.SdBridge,
    ai.ondevice.speech.PhonemizerBridge {
    native <methods>;
}

# Tink, which androidx.security.crypto pulls in for the encrypted preferences,
# is annotated with Error Prone's compile-time annotations. Those are
# compileOnly for Tink and so are not on the runtime classpath, which R8 reports
# as missing classes and treats as an error. They are erased annotations that
# nothing reads at runtime, so not warning about them is the whole fix.
#
# Worth knowing why this went unnoticed: minification is off for debug and on
# only for release, so every build anyone had run was a debug one and the
# release variant had never got as far as R8.
-dontwarn com.google.errorprone.annotations.**

# Ktor, which serves the proxy (SPEC 18).
#
# Three separate things, and only the first is obvious.
#
# The CIO engine is named directly at the `embeddedServer(CIO, ...)` call, so
# it needs no ServiceLoader rule — but Ktor's own class graph reaches optional
# integrations it does not ship, and R8 reports every one as a missing class
# rather than as the dead branch it is. slf4j is the loud one: Ktor logs
# through it, Android has no binding, and the no-op fallback is reached by a
# reflective lookup R8 cannot see through.
-dontwarn io.ktor.**
-dontwarn org.slf4j.**
-dontwarn kotlinx.coroutines.debug.**

# Ktor reads its own version out of the jar at startup and identifies internal
# classes by name in a couple of places. Keeping the names is cheaper than
# finding out which ones at runtime, and this package is small.
-keep class io.ktor.util.debug.** { *; }
-keepclassmembers class io.ktor.** {
    volatile <fields>;
}

# The wire types the codecs serialise. They are ai.ondevice classes and so are
# already covered by the serializer rule above; named here because the proxy's
# config document is the one thing that is read back from storage written by an
# older build, and a renamed field there is a configuration silently reset to
# defaults rather than an error anybody would see.
-keep,allowobfuscation @kotlinx.serialization.Serializable class ai.ondevice.proxy.** { *; }
