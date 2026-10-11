#!/usr/bin/env bash
# Smoke test: one ketchup episode (task 4) on CPU, with inference timing.
# Run from anywhere:
#   bash baseline/run_smoke_test.sh
# Writes eval_info.json, inference_timing.json, a video, and run.log to OUT
# (default outputs/smoke-ketchup-timing). Set OUT to write somewhere else.

# Run from dev-challenge/week-1 so the relative paths below work.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The Week 1 environment also sets LIBERO_CONFIG_PATH and MUJOCO_GL.
source ~/.venvs/hf26-week1/bin/activate
set -euo pipefail
# Models and assets are cached after the first run; skip Hub checks.
export HF_HUB_OFFLINE=1

OUT=${OUT:-outputs/smoke-ketchup-timing}
if [ -e "$OUT/eval_info.json" ]; then
  echo "$OUT already has results. Set OUT to a new folder, e.g. OUT=outputs/smoke-2 bash $0" >&2
  exit 1
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

python baseline/benchmark_eval.py \
  --policy.path=lerobot/smolvla_libero \
  --policy.device=cpu \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[4]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --output_dir="$OUT"
