import java.io.FileInputStream
import java.util.Properties

/**
 * The host python, as python itself reports it.
 *
 * `sys.executable` rather than `where python`: the thing on PATH is often a
 * launcher, and the launcher is not what CMake can run from a build rule.
 * Returns null when there is no python, which is not an error here — only the
 * OpenCL backend needs one, and CMake refuses with its own message.
 */
fun hostPython(): String? = listOf("python", "python3").firstNotNullOfOrNull { name ->
    runCatching {
        val process = ProcessBuilder(
            if (System.getProperty("os.name").startsWith("Windows")) {
                listOf("cmd", "/c", name, "-c", "import sys; print(sys.executable)")
            } else {
                listOf(name, "-c", "import sys; print(sys.executable)")
            },
        ).redirectErrorStream(true).start()
        val path = process.inputStream.bufferedReader().readText().trim()
        if (process.waitFor() == 0 && File(path).isFile) path.replace('\\', '/') else null
    }.getOrNull()
}

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.ksp)
    alias(libs.plugins.hilt)
}

android {
    namespace = "ai.ondevice"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.ondevice"
        minSdk = 31
        targetSdk = 35
        // Bump both on every release. Android refuses an update whose
        // versionCode is not higher than the installed one, and the only way
        // past that refusal is an uninstall — which takes the model files and
        // the database with it.
        versionCode = 2
        versionName = "0.2.0"

        // SPEC 2.2 — arm64 is the only shipping ABI; armeabi-v7a would double
        // the APK for devices that could never load these models anyway.
        //
        // x86_64 is added by the debug build type rather than listed here.
        // Build-type `abiFilters` are *merged* with defaultConfig's, not
        // substituted for them, so the release block's `abiFilters.clear()`
        // cleared its own empty set and shipped x86_64 anyway — 63 MB of
        // emulator-only native code in every release APK, silently, for as long
        // as the comment claiming otherwise had been there.
        ndk { abiFilters += "arm64-v8a" }

        externalNativeBuild {
            cmake {
                // -DNDEBUG turns off ggml's assertions in the shipped build.
                cppFlags += listOf("-std=c++17", "-fexceptions", "-frtti", "-O3", "-DNDEBUG")

                // cppFlags reaches CMAKE_CXX_FLAGS only, so for a long time
                // every .c file in this project compiled with no -O at all:
                // the NDK clears CMAKE_C_FLAGS_DEBUG, CMAKE_C_FLAGS was empty,
                // and clang's default is -O0. C++ got -O3 and C got nothing.
                //
                // That is not a rounding error, because ggml splits along the
                // same line the flags did. Its quantised dot products live in
                // ggml-cpu/quants.c and ggml-cpu/arch/<arch>/quants.c, the op
                // dispatcher in ggml-cpu.c — six C files that hold essentially
                // all of token-generation time, and every one of them was
                // un-optimised while the 200-odd C++ files around them were
                // not. Those kernels are hand-written NEON/AVX intrinsics, and
                // -O0 spills every intrinsic to the stack instead of keeping it
                // in a vector register, which is why the cost is a multiple and
                // not a few per cent. It hit all three ggml runtimes at once —
                // chat, transcribe and image share one ggml — plus espeak-ng
                // and libwebp, which are C throughout.
                //
                // Release builds escaped it (RelWithDebInfo supplies -O2 for C),
                // so this only ever showed up in the debug builds we measure on.
                cFlags += listOf("-O3", "-DNDEBUG")

                arguments += listOf("-DANDROID_STL=c++_shared")

                // ggml's OpenCL backend embeds its kernels by running a python
                // script, and CMake has to be told which interpreter — its own
                // search looks for `python3.exe` and finds nothing behind a
                // pyenv shim, a Windows Store alias or a venv launcher, all of
                // which are `python` on PATH and work perfectly well.
                //
                // So PATH is asked the question directly, in the shell that
                // understands those, and the answer is the real executable.
                // When there is no python at all this passes nothing and CMake
                // says so itself — with a message naming python, which is the
                // part that was worth the glue.
                hostPython()?.let { arguments += "-DPython3_EXECUTABLE=$it" }
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // The Hexagon DSP images, as assets rather than jniLibs.
    //
    // They are .so files but nothing in this process ever dlopens them: they
    // run on the NPU, and the DSP's own loader finds them by *path*, from
    // ADSP_LIBRARY_PATH. jniLibs would put them somewhere that path cannot
    // name — with extractNativeLibs off, which is the default here, the
    // library directory is a location inside the APK that only the dynamic
    // linker knows how to open. So they are shipped as assets and unpacked to
    // filesDir on first run (see HexagonSkels).
    //
    // Absent — no tools/build-hexagon.sh has been run — this is simply an
    // assets directory that does not exist, which AGP accepts.
    sourceSets.getByName("main").assets.srcDir(rootProject.file("native/hexagon/assets"))

    // SPEC 17.2 — two channels built from one source. `sideload` may self-update
    // its runtime bundles via PackageInstaller; `play` degrades the updater to a
    // store link because Play policy forbids downloading native code.
    flavorDimensions += "channel"
    productFlavors {
        create("sideload") {
            dimension = "channel"
            buildConfigField("String", "UPDATE_CHANNEL", "\"SIDELOAD\"")
            buildConfigField("boolean", "CAN_SELF_UPDATE_RUNTIMES", "true")
        }
        create("play") {
            dimension = "channel"
            buildConfigField("String", "UPDATE_CHANNEL", "\"PLAY\"")
            buildConfigField("boolean", "CAN_SELF_UPDATE_RUNTIMES", "false")
        }
    }

    // The release key is never in the repository. It is read from
    // `keystore.properties` beside this file, or from the environment for a
    // build machine that has no checkout-local file. If neither is present the
    // release build still runs and simply produces an unsigned APK, so a fresh
    // clone can verify that R8 and the shrinker are happy without holding a
    // signing key — which is the only part of the release build most changes
    // can actually break.
    val keystoreProperties = Properties()
    rootProject.file("keystore.properties").takeIf { it.exists() }?.let { file ->
        FileInputStream(file).use { keystoreProperties.load(it) }
    }

    fun secret(key: String, env: String): String? =
        keystoreProperties.getProperty(key) ?: System.getenv(env)

    // Resolved against the repo root, not this module. `file(...)` inside an
    // `android {}` block is relative to app/, so a `storeFile=release.jks`
    // sitting beside keystore.properties silently did not exist and the release
    // came out unsigned — with no error, because an unsigned release is a valid
    // thing to produce.
    val releaseStore = secret("storeFile", "ANDROID_KEYSTORE")
        ?.let { rootProject.file(it) }
        ?.takeIf { it.exists() }

    signingConfigs {
        if (releaseStore != null) {
            create("release") {
                storeFile = releaseStore
                storePassword = secret("storePassword", "ANDROID_KEYSTORE_PASSWORD")
                keyAlias = secret("keyAlias", "ANDROID_KEY_ALIAS")
                keyPassword = secret("keyPassword", "ANDROID_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
            // The emulator, and only here. No emulator ships to users.
            ndk { abiFilters += "x86_64" }
            // Signed with the release key when there is one, so debug and
            // release are mutually installable. Android refuses to update a
            // package with a differently-signed APK, so with the default debug
            // key the only way to move between the two on a phone is to
            // uninstall — which takes the model files and the whole database
            // with it. That is a real cost (gigabytes of downloads and every
            // conversation) paid to keep a key distinction that buys nothing:
            // both builds carry the same applicationId and neither is
            // distributed. Falls back to the debug key when no keystore is
            // configured, so a fresh clone still builds.
            signingConfigs.findByName("release")?.let { signingConfig = it }
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.findByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    packaging {
        resources.excludes += setOf(
            "/META-INF/{AL2.0,LGPL2.1}",
            "META-INF/DEPENDENCY",
        )
        // The Khronos ICD loader is a link target and must not ship.
        //
        // The phone's own libOpenCL.so is the one that has to run: it is the
        // only loader allowed to open the Adreno driver, since an app's linker
        // namespace permits /data and nothing else. Shipping ours would win the
        // soname and hand ggml a loader that can find no driver — which is
        // exactly what it did when tried, one dlopen refusal per candidate
        // path. What makes the platform's reachable at all is the
        // <uses-native-library> declaration in AndroidManifest.xml.
        jniLibs.excludes += "**/libOpenCL.so"
    }
}

ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.service)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.navigation.compose)

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons.core)
    implementation(libs.compose.ui.tooling.preview)
    debugImplementation(libs.compose.ui.tooling)

    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)

    implementation(libs.androidx.datastore.preferences)
    implementation(libs.androidx.security.crypto)

    implementation(libs.hilt.android)
    implementation(libs.hilt.navigation.compose)
    ksp(libs.hilt.compiler)

    implementation(libs.okhttp)
    implementation(libs.kotlinx.serialization.json)

    // Kokoro's weights are an ONNX graph. ggml cannot load it, so this is the
    // one place the app runs a second inference runtime.
    implementation(libs.onnxruntime.android)

    implementation(libs.coil.compose)
    implementation(libs.coil.network.okhttp)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
}
