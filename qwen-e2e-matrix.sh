#!/usr/bin/env bash

set -euo pipefail

OUT_DIR="${OUT_DIR:-/tmp/qwen_e2e_matrix}"
SHAPES="${SHAPES:-768x1024 1024x1024}"
STEP_COUNTS="${STEP_COUNTS:-1 3}"
SOURCE_STEPS="${SOURCE_STEPS:-12}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
SEED="${SEED:-0}"
OUTPUT_EXT="${OUTPUT_EXT:-jpg}"

WOMAN_PROMPT="${WOMAN_PROMPT:-A young woman with long brown curly pigtails, pastel striped shirt, standing outdoors, photorealistic}"
MAN_PROMPT="${MAN_PROMPT:-A man wearing a plain white t-shirt and jeans, standing outdoors, photorealistic}"
T2I_PROMPT="${T2I_PROMPT:-${WOMAN_PROMPT}}"
EDIT_PROMPT="${EDIT_PROMPT:-Place these two people side by side in a natural outdoor portrait photo.}"
EDIT_NEGATIVE_PROMPT="${EDIT_NEGATIVE_PROMPT:-duplicate person, extra limbs, distorted body}"

WOMAN_SRC="${OUT_DIR}/woman_src.${OUTPUT_EXT}"
MAN_SRC="${OUT_DIR}/man_src.${OUTPUT_EXT}"
T2I_PREFIX="${OUT_DIR}/qwen_t2i"
EDIT_PREFIX="${OUT_DIR}/qwen_edit"
T2I_LOG="${OUT_DIR}/qwen_t2i_matrix.log"
EDIT_LOG="${OUT_DIR}/qwen_edit_matrix.log"

mkdir -p "${OUT_DIR}"

read -r -a SHAPE_LIST <<< "${SHAPES}"
read -r -a STEP_LIST <<< "${STEP_COUNTS}"

run_stage() {
  local name="$1"
  shift

  local start_ts end_ts
  start_ts="$(date +%s)"
  echo "[matrix] starting ${name}"
  "$@"
  end_ts="$(date +%s)"
  echo "[matrix] finished ${name} in $((end_ts - start_ts))s"
}

run_and_log() {
  local log_path="$1"
  shift

  rm -f "${log_path}"
  "$@" 2>&1 | tee "${log_path}"
}

append_matrix_flags() {
  local -n out_ref="$1"
  local shape
  local step_count

  for shape in "${SHAPE_LIST[@]}"; do
    out_ref+=(--shape "${shape}")
  done

  for step_count in "${STEP_LIST[@]}"; do
    out_ref+=(--step-count "${step_count}")
  done
}

print_case_summary() {
  local title="$1"
  local log_path="$2"
  echo
  echo "=== ${title} summary ==="
  grep -E '^\[summary\]|^\[case ' "${log_path}" || true
}

ensure_edit_sources() {
  if [[ -f "${WOMAN_SRC}" && -f "${MAN_SRC}" ]]; then
    echo "[matrix] reusing existing edit source images"
    return
  fi

  run_stage "woman_source" \
    ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
    --model Qwen/Qwen-Image-2512 \
    --prompt "${WOMAN_PROMPT}" \
    --negative-prompt "low quality" \
    --width 768 \
    --height 1024 \
    --num-inference-steps "${SOURCE_STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --true-cfg-scale "${TRUE_CFG_SCALE}" \
    --seed "${SEED}" \
    --output "${WOMAN_SRC}"

  run_stage "man_source" \
    ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
    --model Qwen/Qwen-Image-2512 \
    --prompt "${MAN_PROMPT}" \
    --negative-prompt "low quality" \
    --width 768 \
    --height 1024 \
    --num-inference-steps "${SOURCE_STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --true-cfg-scale "${TRUE_CFG_SCALE}" \
    --seed "$((SEED + 1))" \
    --output "${MAN_SRC}"
}

main() {
  local t2i_cmd edit_cmd

  ensure_edit_sources

  t2i_cmd=(
    ./bazelw run //max/examples/diffusion:same_process_multi_shape_runner --
    --model Qwen/Qwen-Image-2512
    --prompt "${T2I_PROMPT}"
    --negative-prompt "low quality"
    --guidance-scale "${GUIDANCE_SCALE}"
    --true-cfg-scale "${TRUE_CFG_SCALE}"
    --seed "${SEED}"
    --output-ext "${OUTPUT_EXT}"
    --output-prefix "${T2I_PREFIX}"
  )
  append_matrix_flags t2i_cmd

  run_stage "qwen_t2i_matrix" run_and_log "${T2I_LOG}" "${t2i_cmd[@]}"
  print_case_summary "Qwen T2I" "${T2I_LOG}"

  edit_cmd=(
    ./bazelw run //max/examples/diffusion:same_process_multi_shape_runner --
    --model Qwen/Qwen-Image-Edit-2511
    --prompt "${EDIT_PROMPT}"
    --negative-prompt "${EDIT_NEGATIVE_PROMPT}"
    --input-image "${WOMAN_SRC}"
    --input-image "${MAN_SRC}"
    --guidance-scale "${GUIDANCE_SCALE}"
    --true-cfg-scale "${TRUE_CFG_SCALE}"
    --seed "${SEED}"
    --output-ext "${OUTPUT_EXT}"
    --output-prefix "${EDIT_PREFIX}"
  )
  append_matrix_flags edit_cmd

  run_stage "qwen_edit_matrix" run_and_log "${EDIT_LOG}" "${edit_cmd[@]}"
  print_case_summary "Qwen Edit" "${EDIT_LOG}"

  echo
  echo "[matrix] outputs saved under ${OUT_DIR}/"
  echo "[matrix] logs:"
  echo "  - ${T2I_LOG}"
  echo "  - ${EDIT_LOG}"
}

main "$@"
