# Resolve ggml's `find_package(OpenCL REQUIRED)` to the vendored Khronos pair.
#
# CMake's own FindOpenCL searches the toolchain sysroot, and the NDK has no
# OpenCL in it — Android does not promise a device has one, so the platform
# does not ship the API. Every device that *does* have one exposes it as a
# vendor public library instead, resolved by soname at load time.
#
# So the search is answered here rather than performed: the headers are the
# checkout pinned in native/VERSIONS, and the library is the Khronos ICD loader
# target built beside it, which exists only to give the linker the symbols. The
# note in CMakeLists.txt explains why that loader is deliberately not shipped.
#
# Declared as a module rather than by pre-seeding OpenCL_LIBRARY because that
# cache variable has to name a file that exists when CMake configures, and the
# loader is not built until afterwards. A target name works; a path does not.

if (NOT TARGET OpenCL)
    message(FATAL_ERROR
        "FindOpenCL shim: the OpenCL target does not exist yet. "
        "OpenCL-ICD-Loader must be added before anything that looks for OpenCL.")
endif()

set(OpenCL_FOUND TRUE)
set(OpenCL_LIBRARIES OpenCL)
set(OpenCL_LIBRARY OpenCL)
set(OpenCL_INCLUDE_DIRS "${ONDEVICE_OPENCL_HEADERS}")
set(OpenCL_INCLUDE_DIR "${ONDEVICE_OPENCL_HEADERS}")
set(OpenCL_VERSION_STRING "3.0")
set(OpenCL_VERSION_MAJOR 3)
set(OpenCL_VERSION_MINOR 0)
