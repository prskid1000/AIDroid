cmake_minimum_required(VERSION 3.22.1)
project(ondevice_native CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../native")

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
