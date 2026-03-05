#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_LIB_DIR="/usr/local/cuda/targets/x86_64-linux/lib"
COMPAT_DIR="${ROOT_DIR}/.cuda-compat-lib"
TARGET="//max/examples/diffusion:simple_offline_generation"

if [[ ! -f "${CUDA_LIB_DIR}/libcublasLt.so.13" || ! -f "${CUDA_LIB_DIR}/libcublas.so.13" ]]; then
  echo "CUDA 13 cuBLAS libraries were not found in: ${CUDA_LIB_DIR}" >&2
  exit 1
fi

mkdir -p "${COMPAT_DIR}"
ln -sfn "${CUDA_LIB_DIR}/libcublasLt.so.13" "${COMPAT_DIR}/libcublasLt.so.12"
ln -sfn "${CUDA_LIB_DIR}/libcublas.so.13" "${COMPAT_DIR}/libcublas.so.12"

export LD_LIBRARY_PATH="${COMPAT_DIR}:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"

if [[ $# -eq 0 ]]; then
  exec "${ROOT_DIR}/bazelw" run "${TARGET}" -- \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "A cat holding a sign that says hello world" \
    --num-inference-steps 50 \
    --guidance-scale 4.0 \
    --seed 42
fi

if [[ "${1}" == "--" ]]; then
  shift
fi

exec "${ROOT_DIR}/bazelw" run "${TARGET}" -- "$@"
