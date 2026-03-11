#!/usr/bin/env bash

set -euo pipefail

WOMAN_OUTPUT="${WOMAN_OUTPUT:-woman_e2e.png}"
MAN_OUTPUT="${MAN_OUTPUT:-man_e2e.png}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-output_combined_e2e.png}"

WOMAN_PROMPT="${WOMAN_PROMPT:-A young woman with long brown curly pigtails, pastel striped shirt, standing outdoors, photorealistic}"
MAN_PROMPT="${MAN_PROMPT:-A man wearing a plain white t-shirt and jeans, standing outdoors, photorealistic}"
EDIT_PROMPT="${EDIT_PROMPT:-Place these two people side by side in a natural outdoor portrait photo.}"
EDIT_NEGATIVE_PROMPT="${EDIT_NEGATIVE_PROMPT:-duplicate person, extra limbs, distorted body}"

WOMAN_STEPS="${WOMAN_STEPS:-50}"
MAN_STEPS="${MAN_STEPS:-50}"
EDIT_STEPS="${EDIT_STEPS:-40}"

WOMAN_SEED="${WOMAN_SEED:-0}"
MAN_SEED="${MAN_SEED:-1}"
EDIT_SEED="${EDIT_SEED:-0}"

run_stage() {
  local name="$1"
  shift

  local start_ts
  local end_ts
  start_ts="$(date +%s)"

  echo "[e2e] starting ${name}"
  "$@"

  end_ts="$(date +%s)"
  echo "[e2e] finished ${name} in $((end_ts - start_ts))s"
}

TOTAL_START_TS="$(date +%s)"

run_stage "woman_t2i" \
  ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
  --model Qwen/Qwen-Image-2512 \
  --prompt "${WOMAN_PROMPT}" \
  --negative-prompt "low quality" \
  --width 768 \
  --height 1024 \
  --num-inference-steps "${WOMAN_STEPS}" \
  --guidance-scale 1.0 \
  --true-cfg-scale 4.0 \
  --seed "${WOMAN_SEED}" \
  --output "${WOMAN_OUTPUT}" \
  --profile-timings

run_stage "man_t2i" \
  ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
  --model Qwen/Qwen-Image-2512 \
  --prompt "${MAN_PROMPT}" \
  --negative-prompt "low quality" \
  --width 768 \
  --height 1024 \
  --num-inference-steps "${MAN_STEPS}" \
  --guidance-scale 1.0 \
  --true-cfg-scale 4.0 \
  --seed "${MAN_SEED}" \
  --output "${MAN_OUTPUT}" \
  --profile-timings

run_stage "combined_edit" \
  ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
  --model Qwen/Qwen-Image-Edit-2511 \
  --prompt "${EDIT_PROMPT}" \
  --negative-prompt "${EDIT_NEGATIVE_PROMPT}" \
  --input-image "${WOMAN_OUTPUT}" \
  --input-image "${MAN_OUTPUT}" \
  --width 1024 \
  --height 1024 \
  --num-inference-steps "${EDIT_STEPS}" \
  --guidance-scale 1.0 \
  --true-cfg-scale 4.0 \
  --seed "${EDIT_SEED}" \
  --output "${COMBINED_OUTPUT}" \
  --profile-timings

TOTAL_END_TS="$(date +%s)"

echo "[e2e] done in $((TOTAL_END_TS - TOTAL_START_TS))s"
echo "[e2e] outputs:"
echo "  - ${WOMAN_OUTPUT}"
echo "  - ${MAN_OUTPUT}"
echo "  - ${COMBINED_OUTPUT}"
