cmake_minimum_required(VERSION 3.22.1)
project(ondevice_native CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../native")

# ---------------------------------------------------------------------------
# One ggml, shared by all three runtimes — and it has to be built for the
# strictest of them.
#
# ggml sizes `ggml_tensor::name` at compile time and defaults to 64 bytes.
# Diffusion tensor names are far longer than language-model ones
# ("model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff…" is 76),
# which is why stable-diffusion.cpp's own CMakeLists raises the limit to 160.
#
# It raises it with `add_definitions`, which only reaches targets declared
# after that line — and here ggml is declared earlier, by llama.cpp. The result
# was a ggml compiled at 64 that every diffusion model in the world fails to
# load with "tensor name is too long", *including* ones built specifically for
# sd.cpp. It reads exactly like a bad model file and is not one.
#
# So the definition is hoisted to the top, before any runtime is added. A
# larger name buffer costs a few bytes per tensor and is otherwise invisible to
# llama.cpp and whisper.cpp.
add_compile_definitions(GGML_MAX_NAME=160)

# ---------------------------------------------------------------------------
# OpenCL — the GPU ggml can actually reach on this class of device.
#
# arm64 only, and not because of taste: the x86_64 target is the emulator,
# whose GPU is the host's through a translation layer with no OpenCL at all.
#
# Two things are needed and neither comes from the NDK. The headers are
# vendored (native/VERSIONS). The *library* is the interesting part: Android
# ships no OpenCL, but a device that has one exposes its own libOpenCL.so as a
# vendor public library — on this phone that file is itself a Khronos ICD
# loader, and it is the only thing that knows where Adreno's driver lives. So
# the Khronos loader is built here purely so the linker has symbols to bind
# against, and deliberately *not* packaged: an APK-local libOpenCL.so would
# shadow the platform's with a loader that can find no driver at all. See the
# jniLibs exclusion in app/build.gradle.kts, which is the half of this that
# actually keeps it out.
# ---------------------------------------------------------------------------

set(ONDEVICE_OPENCL OFF)
if (ANDROID_ABI STREQUAL "arm64-v8a")
    set(ONDEVICE_OPENCL ON)
endif()

if (ONDEVICE_OPENCL)
    set(ONDEVICE_OPENCL_HEADERS "${VENDOR_DIR}/OpenCL-Headers")

    # Shared for this subdirectory only. The loader's own CMake picks STATIC or
    # SHARED off BUILD_SHARED_LIBS, and a static one would be *linked into* our
    # .so — which is the one outcome to avoid, since then the driverless loader
    # is what runs instead of the phone's. Restored immediately: llama.cpp and
    # everything after it are still built static.
    set(BUILD_SHARED_LIBS ON)
    set(OPENCL_ICD_LOADER_HEADERS_DIR "${ONDEVICE_OPENCL_HEADERS}" CACHE PATH "" FORCE)
    set(OPENCL_ICD_LOADER_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(OPENCL_ICD_LOADER_PIC ON CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory("${VENDOR_DIR}/OpenCL-ICD-Loader"
                     "${CMAKE_CURRENT_BINARY_DIR}/opencl-icd" EXCLUDE_FROM_ALL)
    set(BUILD_SHARED_LIBS OFF)

    # Out of the directory Gradle collects .so files from. Belt; the exclusion
    # in build.gradle.kts is braces.
    set_target_properties(OpenCL PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/link-only")

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

    # ggml's OpenCL backend runs a Python script at build time to turn each .cl
    # kernel into a header. The interpreter it needs is the *build host's*, and
    # the NDK toolchain points every find_package at the Android sysroot, where
    # there is obviously no python — so the search is widened for programs only
    # and resolved once here. Resolving it here rather than inside ggml also
    # means the failure, if python really is missing, names python.
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    find_package(Python3 COMPONENTS Interpreter)
    if (NOT Python3_FOUND)
        message(FATAL_ERROR
            "OpenCL needs a host python3 to embed ggml's kernels. Install one, "
            "or build without it — the CPU backend does not need python.")
    endif()
    message(STATUS "ondevice: OpenCL kernels will be embedded using ${Python3_EXECUTABLE}")

    set(GGML_OPENCL                    ON  CACHE BOOL "" FORCE)
    # Embedded, not shipped beside the library: a .cl file the app has to find
    # at runtime is one more thing that can go missing from an APK.
    set(GGML_OPENCL_EMBED_KERNELS      ON  CACHE BOOL "" FORCE)
    set(GGML_OPENCL_USE_ADRENO_KERNELS ON  CACHE BOOL "" FORCE)
    set(GGML_OPENCL_TARGET_VERSION     "300" CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# llama.cpp
#
# SPEC §17.5 pins the runtime to an upstream tag rather than tracking a branch,
# because the parameter manifest is generated from that exact commit — a
# floating dependency would let the manifest and the engine disagree about what
# `sinceBuild` means. The tag lives in native/VERSIONS.
# ---------------------------------------------------------------------------

set(BUILD_SHARED_LIBS       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON      ON  CACHE BOOL "" FORCE)  # chat templates, samplers, tool-call parsing
set(LLAMA_BUILD_TESTS       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP         OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI          OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI   OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL           OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS        OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL              OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS      OFF CACHE BOOL "" FORCE)

# The app decides its own CPU features per ABI (see below); ggml must not probe
# the *build host*, which is an x86 desktop and has nothing to do with a phone.
set(GGML_NATIVE             OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP             OFF CACHE BOOL "" FORCE)  # no libomp in the NDK sysroot
set(GGML_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)

if (ANDROID_ABI STREQUAL "x86_64")
    # The emulator's CPU is whatever the host exposes through the hypervisor, and
    # a SIGILL on an unsupported instruction is indistinguishable from a bug in
    # our own code. x86_64 is a development target, so it trades throughput for
    # never being the reason a run dies.
    set(GGML_AVX       OFF CACHE BOOL "" FORCE)
    set(GGML_AVX2      OFF CACHE BOOL "" FORCE)
    set(GGML_FMA       OFF CACHE BOOL "" FORCE)
    set(GGML_F16C      OFF CACHE BOOL "" FORCE)
    set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory("${VENDOR_DIR}/llama.cpp" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Hexagon — the NPU, added by hand rather than with GGML_HEXAGON=ON.
#
# Upstream's own ggml-hexagon/CMakeLists.txt cannot run here, and not for a
# reason a flag can fix: it drives two compilers this machine does not have.
# `qaic` generates the fastRPC stub from an IDL file, and hexagon-clang builds
# libggml-htp-v<NN>.so, which is DSP code and shares no instruction set with
# anything the NDK targets. Both ship only inside Qualcomm's Hexagon SDK, and
# that SDK is behind a developer account.
#
# tools/build-hexagon.sh runs those two compilers once, inside llama.cpp's
# public toolchain image, and leaves the results in native/hexagon/. What is
# left over — one .cpp, one .c and a stub — is ordinary arm64 code, so it is
# compiled here with everything else. The block below is `ggml_add_backend` and
# `ggml_add_backend_library` written out longhand for that one backend; the
# definition on `ggml` is what makes ggml-backend-reg.cpp call
# ggml_backend_hexagon_reg(), and without it the sources would compile into a
# library nothing ever asks for.
#
# Absent native/hexagon/, the build is exactly what it was before: NPU stays
# unregistered, Settings keeps saying so, and nothing else changes.
# ---------------------------------------------------------------------------

set(ONDEVICE_HEXAGON_DIR "${VENDOR_DIR}/hexagon")

if (ANDROID_ABI STREQUAL "arm64-v8a" AND EXISTS "${ONDEVICE_HEXAGON_DIR}/generated/htp_iface_stub.c")
    set(GGML_HEXAGON_SRC "${VENDOR_DIR}/llama.cpp/ggml/src/ggml-hexagon")

    add_library(ggml-hexagon STATIC
        "${GGML_HEXAGON_SRC}/ggml-hexagon.cpp"
        "${GGML_HEXAGON_SRC}/htp-drv.cpp"
        "${ONDEVICE_HEXAGON_DIR}/generated/htp_iface_stub.c"
    )

    target_include_directories(ggml-hexagon PRIVATE
        "${ONDEVICE_HEXAGON_DIR}/incs"
        "${ONDEVICE_HEXAGON_DIR}/incs/stddef"
        "${ONDEVICE_HEXAGON_DIR}/generated"
        "${GGML_HEXAGON_SRC}"
        "${GGML_HEXAGON_SRC}/htp"
        "${VENDOR_DIR}/llama.cpp/ggml/src"
    )

    set_property(TARGET ggml-hexagon PROPERTY POSITION_INDEPENDENT_CODE ON)
    # Vendored code built under our warning set, not upstream's.
    target_compile_options(ggml-hexagon PRIVATE -Wno-everything)
    # libcdsprpc.so is opened by name at runtime rather than linked: a phone
    # with no DSP simply fails to register the backend, which is the outcome
    # this build wants on every other device.
    target_link_libraries(ggml-hexagon PRIVATE ggml-base log dl)

    target_link_libraries(ggml PUBLIC ggml-hexagon)
    target_compile_definitions(ggml PUBLIC GGML_USE_HEXAGON)

    message(STATUS "ondevice: Hexagon backend from ${ONDEVICE_HEXAGON_DIR}")
else()
    message(STATUS "ondevice: no Hexagon backend (run tools/build-hexagon.sh to add one)")
endif()

# ---------------------------------------------------------------------------
# whisper.cpp
#
# It vendors its own ggml. Pointing it at llama.cpp's copy instead avoids
# compiling ggml twice and — more importantly — avoids two ggml backend
# registries in one process, which would each think they owned the CPU device.
# ---------------------------------------------------------------------------

set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL           OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2           OFF CACHE BOOL "" FORCE)
set(WHISPER_ALL_WARNINGS   OFF CACHE BOOL "" FORCE)
set(WHISPER_USE_SYSTEM_GGML ON  CACHE BOOL "" FORCE)

add_subdirectory("${VENDOR_DIR}/whisper.cpp" "${CMAKE_CURRENT_BINARY_DIR}/whisper.cpp" EXCLUDE_FROM_ALL)

add_library(ondevice_llama SHARED llama_jni.cpp jni_util.cpp)

target_include_directories(ondevice_llama PRIVATE
    "${VENDOR_DIR}/llama.cpp/include"
    "${VENDOR_DIR}/llama.cpp/ggml/include"
    "${VENDOR_DIR}/llama.cpp/common"
    "${VENDOR_DIR}/llama.cpp/vendor"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(ondevice_llama PRIVATE llama llama-common android log)

# Keep the shipped .so as small as the C++ allows — a runtime bundle is
# downloaded on the sideload channel and its size is quoted to the user.
target_link_options(ondevice_llama PRIVATE -Wl,--gc-sections)

# ---------------------------------------------------------------------------
# stable-diffusion.cpp
# ---------------------------------------------------------------------------

set(SD_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SD_BUILD_TESTS     OFF CACHE BOOL "" FORCE)

add_subdirectory("${VENDOR_DIR}/stable-diffusion.cpp" "${CMAKE_CURRENT_BINARY_DIR}/sd.cpp" EXCLUDE_FROM_ALL)

add_library(ondevice_sd SHARED sd_jni.cpp jni_util.cpp)

target_include_directories(ondevice_sd PRIVATE
    "${VENDOR_DIR}/stable-diffusion.cpp/include"
    "${VENDOR_DIR}/llama.cpp/vendor"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(ondevice_sd PRIVATE stable-diffusion android log)
target_link_options(ondevice_sd PRIVATE -Wl,--gc-sections)

add_library(ondevice_whisper SHARED whisper_jni.cpp jni_util.cpp)

target_include_directories(ondevice_whisper PRIVATE
    "${VENDOR_DIR}/whisper.cpp/include"
    "${VENDOR_DIR}/whisper.cpp/ggml/include"
    "${VENDOR_DIR}/llama.cpp/vendor"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(ondevice_whisper PRIVATE whisper android log)
target_link_options(ondevice_whisper PRIVATE -Wl,--gc-sections)

# ---------------------------------------------------------------------------
# espeak-ng — the grapheme-to-phoneme front end Kokoro needs.
#
# Upstream's own CMake is not usable here. Its `data.cmake` compiles the
# phoneme tables by *running* the espeak-ng binary it just built, which in a
# cross-compile is an arm64 executable the build host cannot execute. It also
# builds a CLI, tests, docs and CPack packaging, none of which a library
# consumer wants.
#
# So the two libraries are declared here directly, from upstream's own source
# lists. The compiled data tables are staged into assets instead
# (tools/stage-espeak-data.py) and their format version is asserted there
# against the pinned source, because espeak refuses tables it did not compile.
# ---------------------------------------------------------------------------

set(ESPEAK_DIR "${VENDOR_DIR}/espeak-ng")

# espeak's build knobs, resolved here rather than probed. Everything switched
# off is a synthesiser feature: this app only ever calls
# espeak_TextToPhonemes, and Kokoro produces the audio.
set(HAVE_MKSTEMP     1)  # bionic has it
set(USE_ASYNC        0)  # no worker thread; phonemisation is synchronous
set(USE_KLATT        1)  # referenced unconditionally from synthesize.c
set(USE_LIBPCAUDIO   0)  # no audio output — Kokoro is the voice
set(USE_LIBSONIC     0)
set(USE_MBROLA       0)
set(USE_SPEECHPLAYER 0)
set(PROJECT_VERSION "1.52.0")
configure_file(
    "${ESPEAK_DIR}/src/libespeak-ng/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/espeak-config/config.h"
)

add_library(ucd STATIC
    "${ESPEAK_DIR}/src/ucd-tools/src/case.c"
    "${ESPEAK_DIR}/src/ucd-tools/src/categories.c"
    "${ESPEAK_DIR}/src/ucd-tools/src/ctype.c"
    "${ESPEAK_DIR}/src/ucd-tools/src/proplist.c"
    "${ESPEAK_DIR}/src/ucd-tools/src/scripts.c"
    "${ESPEAK_DIR}/src/ucd-tools/src/tostring.c"
)
target_include_directories(ucd PUBLIC "${ESPEAK_DIR}/src/ucd-tools/src/include")
set_property(TARGET ucd PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(espeak-ng STATIC
    "${ESPEAK_DIR}/src/libespeak-ng/common.c"
    "${ESPEAK_DIR}/src/libespeak-ng/mnemonics.c"
    "${ESPEAK_DIR}/src/libespeak-ng/error.c"
    "${ESPEAK_DIR}/src/libespeak-ng/ieee80.c"
    "${ESPEAK_DIR}/src/libespeak-ng/compiledata.c"
    "${ESPEAK_DIR}/src/libespeak-ng/compiledict.c"
    "${ESPEAK_DIR}/src/libespeak-ng/dictionary.c"
    "${ESPEAK_DIR}/src/libespeak-ng/encoding.c"
    "${ESPEAK_DIR}/src/libespeak-ng/intonation.c"
    "${ESPEAK_DIR}/src/libespeak-ng/langopts.c"
    "${ESPEAK_DIR}/src/libespeak-ng/numbers.c"
    "${ESPEAK_DIR}/src/libespeak-ng/phoneme.c"
    "${ESPEAK_DIR}/src/libespeak-ng/phonemelist.c"
    "${ESPEAK_DIR}/src/libespeak-ng/readclause.c"
    "${ESPEAK_DIR}/src/libespeak-ng/setlengths.c"
    "${ESPEAK_DIR}/src/libespeak-ng/soundicon.c"
    "${ESPEAK_DIR}/src/libespeak-ng/spect.c"
    "${ESPEAK_DIR}/src/libespeak-ng/ssml.c"
    "${ESPEAK_DIR}/src/libespeak-ng/synthdata.c"
    "${ESPEAK_DIR}/src/libespeak-ng/synthesize.c"
    "${ESPEAK_DIR}/src/libespeak-ng/tr_languages.c"
    "${ESPEAK_DIR}/src/libespeak-ng/translate.c"
    "${ESPEAK_DIR}/src/libespeak-ng/translateword.c"
    "${ESPEAK_DIR}/src/libespeak-ng/voices.c"
    "${ESPEAK_DIR}/src/libespeak-ng/wavegen.c"
    "${ESPEAK_DIR}/src/libespeak-ng/speech.c"
    "${ESPEAK_DIR}/src/libespeak-ng/espeak_api.c"
    "${ESPEAK_DIR}/src/libespeak-ng/klatt.c"
)
target_include_directories(espeak-ng
    PUBLIC  "${ESPEAK_DIR}/src/include"
    PRIVATE "${ESPEAK_DIR}/src/include/compat"
            "${ESPEAK_DIR}/src/libespeak-ng"
            "${CMAKE_CURRENT_BINARY_DIR}/espeak-config"
)
# The library is linked into a .so, so every object in it has to be PIC.
set_property(TARGET espeak-ng PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(espeak-ng
    PUBLIC  LIBESPEAK_NG_EXPORT=1
    # espeak falls back to this when given no path. The app always passes one
    # (its unpacked assets), so the value only has to be well-formed.
    PRIVATE PATH_ESPEAK_DATA="/system/share/espeak-ng-data"
)
target_link_libraries(espeak-ng PRIVATE ucd m)
# Upstream builds clean under its own warning set; ours is stricter and this is
# vendored code we do not patch.
target_compile_options(espeak-ng PRIVATE -Wno-everything)

add_library(ondevice_phonemizer SHARED phonemizer_jni.cpp jni_util.cpp)

target_include_directories(ondevice_phonemizer PRIVATE
    "${ESPEAK_DIR}/src/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(ondevice_phonemizer PRIVATE espeak-ng android log)
target_link_options(ondevice_phonemizer PRIVATE -Wl,--gc-sections)
