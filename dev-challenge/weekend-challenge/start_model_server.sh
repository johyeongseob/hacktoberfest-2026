#!/usr/bin/env bash
# Keep the open-weight model loaded for the local browser demo.
set -euo pipefail
challenge_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$challenge_dir/../.." && pwd)"
binary="${LLAMA_SERVER:-$HOME/tools/llama.cpp/build/bin/llama-server}"
model_dir="$repo_dir/models/SmolVLM2-2.2B-Instruct-GGUF"
if [[ ! -x "$binary" ]]; then
    printf 'Build llama-server first. See README.md.\n' >&2
    exit 1
fi
exec "$binary" \
    -m "$model_dir/SmolVLM2-2.2B-Instruct-Q8_0.gguf" \
    --mmproj "$model_dir/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf" \
    --alias smolvlm2-workbench --no-mmproj-offload \
    -t 8 -c 8192 --parallel 1 \
    --host 127.0.0.1 --port 8080
