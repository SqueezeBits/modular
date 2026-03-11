#!/usr/bin/env bash

set -euo pipefail

PROFILE_DIR="${PROFILE_DIR:-profiles}"
mkdir -p "${PROFILE_DIR}"

T2I_STEPS="${T2I_STEPS:-2}"
EDIT_STEPS="${EDIT_STEPS:-2}"

T2I_WARM_OUTPUT="${T2I_WARM_OUTPUT:-/tmp/qwen_t2i_warmup.png}"
EDIT_WARM_OUTPUT="${EDIT_WARM_OUTPUT:-/tmp/qwen_edit_warmup.png}"

T2I_OUTPUT="${T2I_OUTPUT:-${PROFILE_DIR}/qwen_t2i_warm.png}"
EDIT_OUTPUT="${EDIT_OUTPUT:-${PROFILE_DIR}/qwen_edit_warm.png}"

EDIT_INPUT_WOMAN="${EDIT_INPUT_WOMAN:-woman_e2e.png}"
EDIT_INPUT_MAN="${EDIT_INPUT_MAN:-man_e2e.png}"

T2I_BASE="${PROFILE_DIR}/qwen_t2i_warm"
EDIT_BASE="${PROFILE_DIR}/qwen_edit_warm"

run_generation() {
  local name="$1"
  shift
  echo "[warmup] starting ${name}"
  "$@"
  echo "[warmup] finished ${name}"
}

run_profile() {
  local name="$1"
  local base="$2"
  shift 2

  echo "[profile] starting ${name}"
  rm -f "${base}.nsys-rep" "${base}.sqlite" "${base}.perfetto.json"

  env MODULAR_ENABLE_PROFILING=detailed nsys profile \
    --force-overwrite=true \
    -o "${base}" \
    "$@"

  nsys export \
    --force-overwrite=true \
    --type sqlite \
    --output "${base}.sqlite" \
    "${base}.nsys-rep"

  python3 tools/export_qwen_profile_perfetto.py \
    --sqlite "${base}.sqlite" \
    --output "${base}.perfetto.json"

  echo "[profile] finished ${name}"
  echo "  rep: ${base}.nsys-rep"
  echo "  sqlite: ${base}.sqlite"
  echo "  perfetto: ${base}.perfetto.json"
}

run_generation \
  "qwen_t2i_warmup" \
  ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
  --model Qwen/Qwen-Image-2512 \
  --prompt "A young woman with long brown curly pigtails, pastel striped shirt, standing outdoors, photorealistic" \
  --negative-prompt "low quality" \
  --width 768 \
  --height 1024 \
  --num-inference-steps "${T2I_STEPS}" \
  --guidance-scale 1.0 \
  --true-cfg-scale 4.0 \
  --seed 0 \
  --output "${T2I_WARM_OUTPUT}"

run_profile \
  "qwen_t2i_warm" \
  "${T2I_BASE}" \
  ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
  --model Qwen/Qwen-Image-2512 \
  --prompt "A young woman with long brown curly pigtails, pastel striped shirt, standing outdoors, photorealistic" \
  --negative-prompt "low quality" \
  --width 768 \
  --height 1024 \
  --num-inference-steps "${T2I_STEPS}" \
  --guidance-scale 1.0 \
  --true-cfg-scale 4.0 \
  --seed 0 \
  --output "${T2I_OUTPUT}"

run_generation \
  "qwen_edit_warmup" \
  ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
  --model Qwen/Qwen-Image-Edit-2511 \
  --prompt "Place these two people side by side in a natural outdoor portrait photo." \
  --negative-prompt "duplicate person, extra limbs, distorted body" \
  --input-image "${EDIT_INPUT_WOMAN}" \
  --input-image "${EDIT_INPUT_MAN}" \
  --width 1024 \
  --height 1024 \
  --num-inference-steps "${EDIT_STEPS}" \
  --guidance-scale 1.0 \
  --true-cfg-scale 4.0 \
  --seed 0 \
  --output "${EDIT_WARM_OUTPUT}"

run_profile \
  "qwen_edit_warm" \
  "${EDIT_BASE}" \
  ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
  --model Qwen/Qwen-Image-Edit-2511 \
  --prompt "Place these two people side by side in a natural outdoor portrait photo." \
  --negative-prompt "duplicate person, extra limbs, distorted body" \
  --input-image "${EDIT_INPUT_WOMAN}" \
  --input-image "${EDIT_INPUT_MAN}" \
  --width 1024 \
  --height 1024 \
  --num-inference-steps "${EDIT_STEPS}" \
  --guidance-scale 1.0 \
  --true-cfg-scale 4.0 \
  --seed 0 \
  --output "${EDIT_OUTPUT}"
