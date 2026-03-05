#!/bin/bash

MODEL="black-forest-labs/FLUX.2-dev"

PROMPT="The image show the fourth elements, each one in a part of the picture, first part is at top left and show a splashing multicolor water text with many water reflections, the text is made of water, the water word is WATER, the background is splashing water, the second part of the image is a top right and show a soil rounded text, the word made of soil is EARTH, the background is planet earth, the third part of the image is at bottom left and show a cloud multicolor rounded text, the word is AIR made of colorfull cloud the background is a sunset, and the last part of the image in the bottom right shows a red fire rounded text made of lava, the colorfull big word made of fire is FIRE, the background is the closeup eruptive sun"

rm -rf ./cache_tmp

MODULAR_CACHE_DIR=./cache_tmp ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
    --model $MODEL \
    --prompt "${PROMPT}" \
    --num-inference-steps 16 \
    --height 1024 \
    --width 1024 \
    --warmup-prompt "warmup run" \
    --warmup-height 512 \
    --warmup-width 512 \
    --warmup-num-inference-steps 4 \
    --profile-timings \
    --num-warmups 2
