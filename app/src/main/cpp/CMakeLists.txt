cmake_minimum_required(VERSION 3.22.1)
project(ondevice_native CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# REALPATH, not the literal "cpp/../../../../native". Both name the same
# directory, but only the resolved form compares equal to the absolute paths
# CMake's own file(GLOB) produces — which matters below, where one of
# stable-diffusion.cpp's source files is swapped out by name.
get_filename_component(VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../native" REALPATH)

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
# There is one compute device: the CPU.
#
# ggml's OpenCL backend was built here for a while and is gone on evidence, not
# taste. llama.cpp ran on Adreno; the other two did not. sd.cpp fell back to
# CPU for most of the graph and came out slower than plain CPU, and whisper.cpp
# segfaulted at model load — its weight_buft_supported() answers "supported"
# for any GPU device without probing it, so every weight is assigned the OpenCL
# buffer type and the allocation dies. A GPU that works for one of three
# runtimes is not a compute device the app can offer.
#
# What it cost to carry: ggml's Adreno kernels are embedded as source strings
# in every .so, the Khronos ICD loader had to be vendored and built purely for
# link symbols, and the build needed a host python to run ggml's kernel-
# embedding script. All of that is gone with it.
# ---------------------------------------------------------------------------

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

# The vision projector. Upstream keeps mtmd under tools/, but exposes this hook
# for exactly this case: the library on its own, without the CLI binaries that
# LLAMA_BUILD_TOOLS would also produce. mtmd's video support turns itself off
# because LLAMA_SUBPROCESS is OFF — it shells out to ffmpeg, which no phone has.
set(LLAMA_BUILD_MTMD        ON  CACHE BOOL "" FORCE)

# The app decides its own CPU features per ABI (see below); ggml must not probe
# the *build host*, which is an x86 desktop and has nothing to do with a phone.
set(GGML_NATIVE             OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP             OFF CACHE BOOL "" FORCE)  # no libomp in the NDK sysroot
set(GGML_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)

if (ANDROID_ABI STREQUAL "arm64-v8a")
    # Without this, ggml gets the NDK's baseline -march=armv8-a and every
    # quantised dot product falls back to the plain-NEON path: __ARM_FEATURE_DOTPROD
    # is undefined, so `vdotq_s32` is compiled out of ggml_vec_dot_q4_K_q8_K and
    # the whole q4_K x q8_K matmul runs on 16-bit multiply-accumulates.
    #
    # It costs generation almost nothing — that is bandwidth-bound, waiting on
    # the weights either way — and costs prompt processing a factor of four,
    # which is compute-bound. Measured on this device with a 9B at Q4_K_M:
    # prompt 7.6 t/s against generation 5.3 t/s, a ratio that should be five to
    # twenty to one and was instead one and a half.
    #
    # armv8.2 with dotprod and fp16 arithmetic, not armv8.6 with i8mm: dotprod
    # has been in every ARM core since the A55 and A75 in 2017, so every phone
    # that can run Android 12 has it, while i8mm arrived with the A710 in 2021
    # and a Snapdragon 888 or 870 would take a SIGILL on the first matmul.
    # Reaching i8mm safely means GGML_CPU_ALL_VARIANTS, which needs
    # GGML_BACKEND_DL and a shared-library ggml — a different build shape.
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
endif()

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
    "${VENDOR_DIR}/llama.cpp/tools/mtmd"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(ondevice_llama PRIVATE llama llama-common mtmd android log)

# Keep the shipped .so as small as the C++ allows — a runtime bundle is
# downloaded on the sideload channel and its size is quoted to the user.
target_link_options(ondevice_llama PRIVATE -Wl,--gc-sections)

# ---------------------------------------------------------------------------
# stable-diffusion.cpp
# ---------------------------------------------------------------------------

set(SD_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SD_BUILD_TESTS     OFF CACHE BOOL "" FORCE)

# PNG is what the app writes and JPEG is what it reads, both through the stb
# headers sd.cpp already carries. WebP and WebM are encoders for outputs
# nothing in this app produces.
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)

add_subdirectory("${VENDOR_DIR}/stable-diffusion.cpp" "${CMAKE_CURRENT_BINARY_DIR}/sd.cpp" EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Only the tokenizer vocabularies a phone can reach.
#
# sd.cpp compiles in the vocabulary of every text encoder it supports, as
# source arrays: UMT5, Gemma, Gemma 2, GPT-OSS, Mistral, Qwen 2, T5 and CLIP.
# That is 232 MB of headers and it landed as 49 MB of .rodata in
# libondevice_sd.so — on its own, most of the shipped APK, and far more than
# ggml and sd.cpp's own code put together. --gc-sections cannot touch it
# because each vocabulary is reachable from its tokenizer and every tokenizer
# is reachable from the model factory.
#
# The ones dropped belong to encoders that only appear beside 12–20B diffusion
# models — Wan, Chroma, Qwen-Image. Their *weights* are tens of gigabytes; the
# device runs out of memory long before it runs out of tokenizer. CLIP is kept
# because it is what SD 1.x, SD 2 and SDXL use, and T5 because SD 3 pairs it
# with CLIP and is the one such model that is nearly mobile-sized.
#
# Done by swapping one file out of the target rather than by patching the
# checkout: native/ is fetched by tools/fetch-native.sh and gitignored, so an
# edit in there would vanish on the next fetch.
# ---------------------------------------------------------------------------

set(SD_UPSTREAM_VOCAB "${VENDOR_DIR}/stable-diffusion.cpp/src/tokenizers/vocab/vocab.cpp")

get_target_property(SD_SOURCES stable-diffusion SOURCES)
if (NOT SD_UPSTREAM_VOCAB IN_LIST SD_SOURCES)
    # A silent miss here costs 49 MB and nothing says so: both files would
    # compile, the linker would pull upstream's out of the archive first
    # because it resolves the symbols, and ours would simply never be
    # extracted. So it is an error rather than a fallback.
    message(FATAL_ERROR
        "stable-diffusion.cpp no longer lists ${SD_UPSTREAM_VOCAB} — its source "
        "layout moved, and sd_vocab.cpp needs to follow it.")
endif()
list(REMOVE_ITEM SD_SOURCES "${SD_UPSTREAM_VOCAB}")
list(APPEND SD_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/sd_vocab.cpp")
set_target_properties(stable-diffusion PROPERTIES SOURCES "${SD_SOURCES}")
target_include_directories(stable-diffusion PRIVATE "${VENDOR_DIR}/stable-diffusion.cpp/src")

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
