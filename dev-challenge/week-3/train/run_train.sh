#!/usr/bin/env bash
# Fine-tune lerobot/smolvla_libero on a Week 3 training set (scripted chains
# plus one original demo per item, from train/prepare_data.py). Same recipe
# as Week 2 (week-2/train/run_train.sh): whole model, batch 32, learning rate
# 2.5e-5 with 200 warmup steps and cosine decay, a checkpoint every 1000
# steps (about ten per run) for a success-versus-step curve.
# The step instructions are stored in the data, so no relabeling here.
#   DATA=/workspace/outputs/week3/train-data/chains-120 RUN=chains-120 bash train/run_train.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "${VENV:-/workspace/venvs/hf26-week3/bin/activate}"
set -euo pipefail

DATA=${DATA:?set DATA to a training set folder}
RUN=${RUN:?set RUN to a run name}
SEED=${SEED:-1000}
STEPS=${STEPS:-3000}
SAVE_FREQ=${SAVE_FREQ:-1000}
LR=${LR:-2.5e-5}
WARMUP=${WARMUP:-200}
BATCH=${BATCH:-32}
WORKERS=${WORKERS:-6}
OUT_ROOT=${OUT_ROOT:-/workspace/outputs/week3/train}
OUT="$OUT_ROOT/$RUN"
if [ -e "$OUT" ]; then
  echo "$OUT already exists. Remove it or choose another RUN." >&2
  exit 1
fi
mkdir -p "$OUT_ROOT"
echo "run=$RUN data=$DATA steps=$STEPS lr=$LR batch=$BATCH seed=$SEED"

lerobot-train \
  --policy.path=lerobot/smolvla_libero \
  --policy.push_to_hub=false \
  --policy.optimizer_lr="$LR" \
  --policy.scheduler_warmup_steps="$WARMUP" \
  --policy.scheduler_decay_steps="$STEPS" \
  --dataset.repo_id="local/$(basename "$DATA")" \
  --dataset.root="$DATA" \
  --dataset.video_backend=torchcodec \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --batch_size="$BATCH" \
  --num_workers="$WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=10 \
  --seed="$SEED" \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  2>&1 | tee "$OUT_ROOT/$RUN.log"
