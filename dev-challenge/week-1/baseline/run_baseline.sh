#!/usr/bin/env bash
# Baseline: all ten LIBERO-Object tasks with their original instructions,
# 10 episodes each (initial states 0-9, seeds 1000-1009), on CPU.
# Run from anywhere:
#   bash baseline/run_baseline.sh
# Writes eval_info.json, inference_timing.json, 100 videos, and run.log to OUT
# (default outputs/m2-object). Takes about 85 minutes on the laptop CPU.

# Run from dev-challenge/week-1 so the relative paths below work.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The Week 1 environment also sets LIBERO_CONFIG_PATH and MUJOCO_GL.
source ~/.venvs/hf26-week1/bin/activate
set -euo pipefail
# Models and assets are cached after the first run; skip Hub checks.
export HF_HUB_OFFLINE=1

OUT=${OUT:-outputs/m2-object}
if [ -e "$OUT/eval_info.json" ]; then
  echo "$OUT already has results. Set OUT to a new folder, e.g. OUT=outputs/baseline-2 bash $0" >&2
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
  --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --output_dir="$OUT"
