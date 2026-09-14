#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT:-"$(cd "${PROJECT_ROOT}/.." && pwd)/llama.cpp"}
LLAMA_CPP_BUILD=${LLAMA_CPP_BUILD:-"${LLAMA_CPP_ROOT}/build-metal"}
BENCH_BUILD=${BENCH_BUILD:-"${PROJECT_ROOT}/build-metal"}
MODE=smoke
MODEL=

usage() {
    echo "Usage: $0 --model /path/to/model.gguf [--full] [--llama-root /path/to/llama.cpp]"
    echo "  default: build Metal binaries and run LC-001 once (1 warmup + 1 formal)"
    echo "  --full:  run the formal LC-001 through LC-004 baseline"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL=${2:-}
            shift 2
            ;;
        --llama-root)
            LLAMA_CPP_ROOT=${2:-}
            LLAMA_CPP_BUILD="${LLAMA_CPP_ROOT}/build-metal"
            shift 2
            ;;
        --full)
            MODE=full
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ $(uname -s) != Darwin ]]; then
    echo "This helper must run on macOS." >&2
    exit 1
fi
if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then
    echo "A local GGUF file is required: --model /path/to/model.gguf" >&2
    exit 2
fi
if [[ ! -f "${LLAMA_CPP_ROOT}/CMakeLists.txt" || ! -f "${LLAMA_CPP_ROOT}/include/llama.h" ]]; then
    echo "llama.cpp was not found at ${LLAMA_CPP_ROOT}. Use --llama-root." >&2
    exit 2
fi

for command in cmake python3; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Missing command: ${command}" >&2
        exit 1
    fi
done

JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

cmake -S "${LLAMA_CPP_ROOT}" -B "${LLAMA_CPP_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=Apple \
    -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF
cmake --build "${LLAMA_CPP_BUILD}" --config Release --target llama-common -j "${JOBS}"

cmake -S "${PROJECT_ROOT}" -B "${BENCH_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CPP_ROOT="${LLAMA_CPP_ROOT}" \
    -DLLAMA_CPP_BUILD="${LLAMA_CPP_BUILD}"
cmake --build "${BENCH_BUILD}" --config Release --target bench_run -j "${JOBS}"

ARGS=(
    "${PROJECT_ROOT}/bench_test.py"
    --platform metal
    --bench-run "${BENCH_BUILD}/bench_run"
    --model "${MODEL}"
)

if [[ ${MODE} == smoke ]]; then
    ARGS+=(--case LC-001 --warmup-runs 1 --formal-runs 1 --skip-model-hash)
    echo "Running the LC-001 smoke test. Use --full for the formal LC-001 through LC-004 baseline."
else
    echo "Running the formal LC-001 through LC-004 Metal baseline."
fi

python3 "${ARGS[@]}"
