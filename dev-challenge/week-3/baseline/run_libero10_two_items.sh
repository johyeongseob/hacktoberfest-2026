#!/usr/bin/env bash
# Reference run: lerobot/smolvla_libero on the LIBERO-10 tasks that put two
# items in the basket, on CPU, with inference timing.
#   bash baseline/run_libero10_two_items.sh
# Smoke test (cream cheese and butter, one episode):
#   TASK_IDS='[1]' EPISODES=1 OUT=outputs/libero10-smoke bash baseline/run_libero10_two_items.sh
# LIBERO-10 task ids (benchmark order, not the dataset's task_index):
#   0  put both the alphabet soup and the tomato sauce in the basket
#   1  put both the cream cheese box and the butter in the basket
#   7  put both the alphabet soup and the cream cheese box in the basket
# Writes eval_info.json, inference_timing.json, videos, and run.log to OUT
# (default outputs/libero10-two-items).

# Run from dev-challenge/week-3 so the relative paths below work.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The environment also sets LIBERO_CONFIG_PATH and MUJOCO_GL.
source ~/.venvs/hf26-week1/bin/activate
set -euo pipefail

TASK_IDS=${TASK_IDS:-'[0,1,7]'}
EPISODES=${EPISODES:-10}
OUT=${OUT:-outputs/libero10-two-items}
if [ -e "$OUT/eval_info.json" ]; then
  echo "$OUT already has results. Set OUT to a new folder." >&2
  exit 1
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "tasks=$TASK_IDS episodes per task=$EPISODES out=$OUT"

# --rename_map: LeRobot 0.6.1 does not apply the checkpoint's camera names
# (huggingface/lerobot#4578). --policy.n_action_steps=10: the checkpoint ships
# 50, which replays a whole chunk blind (huggingface/lerobot#4614).

python baseline/benchmark_eval.py \
  --policy.path=lerobot/smolvla_libero \
  --policy.device=cpu \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids="$TASK_IDS" \
  --eval.batch_size=1 \
  --eval.n_episodes="$EPISODES" \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --output_dir="$OUT"
