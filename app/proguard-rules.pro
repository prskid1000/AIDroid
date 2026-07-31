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
-keep class ai.ondevice.engine.jni.** { *; }

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
