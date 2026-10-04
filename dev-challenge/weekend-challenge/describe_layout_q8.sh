#!/usr/bin/env bash
# Run independent single-image descriptions with the local Q8 model.
set -euo pipefail

challenge_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$challenge_dir/../.." && pwd)"
binary="${LLAMA_MTMD_CLI:-$HOME/tools/llama.cpp/build/bin/llama-mtmd-cli}"
model_dir="$repo_dir/models/SmolVLM2-2.2B-Instruct-GGUF"
sample_dir="$challenge_dir/data/C1-template_00000-traj_00000"

if [[ ! -x "$binary" ]]; then
    printf 'Missing executable: %s\n' "$binary" >&2
    exit 1
fi
for required in \
    "$model_dir/SmolVLM2-2.2B-Instruct-Q8_0.gguf" \
    "$model_dir/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf" \
    "$sample_dir/frame000.png" "$sample_dir/frame004.png"; do
    if [[ ! -f "$required" ]]; then
        printf 'Missing file: %s\n' "$required" >&2
        exit 1
    fi
done

# Raw runtime logs may contain local paths; keep them outside the repository.
log_dir="$(mktemp -d /tmp/workbench-q8-descriptions.XXXXXX)"
prompt="Describe each tabletop object in this image. For each object, give its name, position in the picture (left/right and upper/middle/lower), and visible orientation. For a cup, describe where its handle points. For an elongated object, describe its long-axis direction. If an orientation is unclear, say uncertain. Use one short item per object. Describe only this image; do not suggest movements."

printf 'Raw logs: %s\n' "$log_dir"
for frame in frame000 frame004; do
    printf '\nAnalyzing %s independently...\n' "$frame"
    "$binary" \
        -m "$model_dir/SmolVLM2-2.2B-Instruct-Q8_0.gguf" \
        --mmproj "$model_dir/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf" \
        --image "$sample_dir/$frame.png" \
        --no-mmproj-offload \
        -t 8 -c 8192 -n 192 --temp 0 --repeat-penalty 1.1 \
        -p "$prompt" 2>&1 | tee "$log_dir/$frame.log"
done
