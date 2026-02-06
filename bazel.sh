#!/bin/bash

CACHE_DIR="/home/byungchul_cache/bazel-cache-$(whoami)"
OUTPUT_BASE="/home/byungchul_cache/bazel-output-$(whoami)"

mkdir -p "$CACHE_DIR" "$OUTPUT_BASE"

# Set MODULAR_MOJO_MAX_IMPORT_PATH if not already set
if [ -z "$MODULAR_MOJO_MAX_IMPORT_PATH" ]; then
    export MODULAR_MOJO_MAX_IMPORT_PATH="$(pwd)/bazel-bin"
fi

# Get the first argument (bazel command like build, test, run, etc.)
CMD="$1"
shift

exec ./bazelw \
    --output_base="$OUTPUT_BASE" \
    "$CMD" \
    --repository_cache="$CACHE_DIR/repository" \
    --disk_cache="$CACHE_DIR/disk" \
    --config=disable-mypy \
    "$@"
