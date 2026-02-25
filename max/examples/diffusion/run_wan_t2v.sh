#!/bin/bash
# ===----------------------------------------------------------------------=== #
# Wan 2.2 T2V End-to-End Generation Script
#
# Runs text-to-video generation at 480p and 720p resolutions.
# Requires CUDA 12.9+ and an NVIDIA GPU with sufficient VRAM.
#
# Usage:
#   bash max/examples/diffusion/run_wan_t2v.sh [480p|720p|both]
# ===----------------------------------------------------------------------=== #

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
MODEL="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
NEGATIVE_PROMPT="low quality, blurry, distorted"
NUM_INFERENCE_STEPS=40
GUIDANCE_SCALE=4.0
GUIDANCE_SCALE_2=3.0
FPS=16
SEED=42
OUTPUT_DIR="/tmp/wan_t2v_outputs"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Build command
BAZEL_CMD="MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas ${REPO_ROOT}/bazelw run -c opt //max/examples/diffusion:simple_offline_generation --"

run_generation() {
    local resolution="$1"
    local height width num_frames output_file

    case "$resolution" in
        480p)
            height=480
            width=832
            num_frames=81
            output_file="${OUTPUT_DIR}/wan_t2v_480p.mp4"
            ;;
        720p)
            height=720
            width=1280
            num_frames=81
            output_file="${OUTPUT_DIR}/wan_t2v_720p.mp4"
            ;;
        *)
            echo "ERROR: Unknown resolution '$resolution'. Use 480p, 720p, or both."
            exit 1
            ;;
    esac

    echo "============================================================"
    echo " Wan 2.2 T2V - ${resolution} (${width}x${height}, ${num_frames} frames)"
    echo "============================================================"
    echo "Model: ${MODEL}"
    echo "Steps: ${NUM_INFERENCE_STEPS}"
    echo "Guidance: ${GUIDANCE_SCALE} / ${GUIDANCE_SCALE_2}"
    echo "Seed: ${SEED}"
    echo "Output: ${output_file}"
    echo "------------------------------------------------------------"

    time eval "$BAZEL_CMD" \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --negative-prompt "$NEGATIVE_PROMPT" \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --guidance-scale "$GUIDANCE_SCALE" \
        --guidance-scale-2 "$GUIDANCE_SCALE_2" \
        --num-frames "$num_frames" \
        --height "$height" \
        --width "$width" \
        --fps "$FPS" \
        --seed "$SEED" \
        --output "$output_file"

    if [ -f "$output_file" ]; then
        local filesize
        filesize=$(stat -c%s "$output_file" 2>/dev/null || stat -f%z "$output_file" 2>/dev/null || echo "unknown")
        echo ""
        echo "SUCCESS: ${output_file} (${filesize} bytes)"
    else
        echo ""
        echo "ERROR: Output file not generated!"
        exit 1
    fi
    echo ""
}

# Parse arguments
RESOLUTION="${1:-both}"

case "$RESOLUTION" in
    480p)
        run_generation 480p
        ;;
    720p)
        run_generation 720p
        ;;
    both)
        run_generation 480p
        run_generation 720p
        ;;
    *)
        echo "Usage: $0 [480p|720p|both]"
        echo ""
        echo "Runs Wan 2.2 T2V end-to-end video generation."
        echo "  480p: 832x480, 81 frames"
        echo "  720p: 1280x720, 81 frames"
        echo "  both: Run both resolutions (default)"
        exit 1
        ;;
esac

echo "============================================================"
echo " All generation complete!"
echo " Output files in: ${OUTPUT_DIR}/"
echo "============================================================"
