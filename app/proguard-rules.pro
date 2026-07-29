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
