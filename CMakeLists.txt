cmake_minimum_required(VERSION 3.16)

project(bench_run CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(LLAMA_CPP_ROOT_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" ABSOLUTE)

# Path to the llama.cpp source tree; the tool links against an existing
# llama.cpp build (shared libraries) and does not build llama.cpp itself.
set(LLAMA_CPP_ROOT "${LLAMA_CPP_ROOT_DEFAULT}" CACHE PATH
    "root of the llama.cpp source tree to build against")

# Build directory of the llama.cpp build whose shared libraries we link.
set(LLAMA_CPP_BUILD "${LLAMA_CPP_ROOT}/build" CACHE PATH
    "llama.cpp build directory containing bin/{libllama,libggml,...}")

if(NOT EXISTS "${LLAMA_CPP_ROOT}/include/llama.h")
    message(FATAL_ERROR "LLAMA_CPP_ROOT is not a llama.cpp source tree: ${LLAMA_CPP_ROOT}")
endif()

set(LLAMA_CPP_LIBRARY_DIR "${LLAMA_CPP_BUILD}/bin")
foreach(LIBRARY_NAME llama llama-common ggml ggml-base)
    string(REPLACE "-" "_" LIBRARY_VAR "${LIBRARY_NAME}")
    string(TOUPPER "${LIBRARY_VAR}" LIBRARY_VAR)
    find_library(${LIBRARY_VAR}_LIBRARY
        NAMES ${LIBRARY_NAME}
        PATHS "${LLAMA_CPP_LIBRARY_DIR}"
        NO_DEFAULT_PATH
    )
    if(NOT ${LIBRARY_VAR}_LIBRARY)
        message(FATAL_ERROR "Cannot find ${LIBRARY_NAME} in ${LLAMA_CPP_LIBRARY_DIR}. Build llama.cpp with BUILD_SHARED_LIBS=ON first.")
    endif()
endforeach()

add_executable(bench_run bench_run.cpp)

target_include_directories(bench_run PRIVATE
    ${LLAMA_CPP_ROOT}/include
    ${LLAMA_CPP_ROOT}/common
    ${LLAMA_CPP_ROOT}/ggml/include
    ${LLAMA_CPP_ROOT}/vendor
)

target_link_libraries(bench_run PRIVATE
    ${LLAMA_LIBRARY}
    ${LLAMA_COMMON_LIBRARY}
    ${GGML_LIBRARY}
    ${GGML_BASE_LIBRARY}
)

if(APPLE)
    set(LOCAL_RPATH "@loader_path")
else()
    set(LOCAL_RPATH "$ORIGIN")
endif()

set_target_properties(bench_run PROPERTIES
    BUILD_RPATH "${LLAMA_CPP_LIBRARY_DIR};${LOCAL_RPATH}"
)
