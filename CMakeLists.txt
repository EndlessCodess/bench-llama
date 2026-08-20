cmake_minimum_required(VERSION 3.16)

project(bench_run CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Path to the llama.cpp source tree; the tool links against an existing
# llama.cpp build (shared libraries) and does not build llama.cpp itself.
set(LLAMA_CPP_ROOT "/data1/xuyang/projects/llama.cpp" CACHE PATH
    "root of the llama.cpp source tree to build against")

# Build directory of the llama.cpp build whose shared libraries we link.
# Defaults to the CUDA build that produced the current results.
set(LLAMA_CPP_BUILD "${LLAMA_CPP_ROOT}/build-cuda13" CACHE PATH
    "llama.cpp build directory containing bin/{libllama,libggml,...}.so")

add_executable(bench_run bench_run.cpp)

target_include_directories(bench_run PRIVATE
    ${LLAMA_CPP_ROOT}/include
    ${LLAMA_CPP_ROOT}/common
    ${LLAMA_CPP_ROOT}/ggml/include
    ${LLAMA_CPP_ROOT}/vendor
)

target_link_directories(bench_run PRIVATE ${LLAMA_CPP_BUILD}/bin)

target_link_libraries(bench_run PRIVATE
    llama
    llama-common
    ggml
    ggml-base
)

# resolve libllama.so etc. from the llama.cpp bin directory at runtime
set_target_properties(bench_run PROPERTIES BUILD_RPATH "$ORIGIN")
