#!/usr/bin/env bash
# Fine-tune smolvla_libero on the first N demos per item of the four Week 1
# items, with the five-wording relabeling (train_relabel.py).
#   bash train/run_train.sh N [SEED]
# e.g. bash train/run_train.sh 5        -> OUT_ROOT/n5-s1000
#      bash train/run_train.sh 5 1001   -> OUT_ROOT/n5-s1001
# STEPS (default 3000) is the same for every N. A checkpoint is saved every
# SAVE_FREQ (default 1000) steps, and the loss is logged every 10 steps.
# watch_eval.py scores each checkpoint on the validation set while training
# continues.
# The learning rate is lower than the smolvla_libero recipe (1e-4) because this
# is a short fine-tune on a few demos.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /workspace/venvs/hf26-week2/bin/activate
set -euo pipefail

N=${1:?usage: bash train/run_train.sh N [SEED]}
SEED=${2:-1000}
STEPS=${STEPS:-3000}
SAVE_FREQ=${SAVE_FREQ:-1000}
LR=${LR:-2.5e-5}
WARMUP=${WARMUP:-200}
BATCH=${BATCH:-32}
OUT_ROOT=${OUT_ROOT:-/workspace/outputs/week2/train}
OUT="$OUT_ROOT/n$N-s$SEED"
if [ -e "$OUT" ]; then
  echo "$OUT already exists. Remove it or set OUT_ROOT to a new folder." >&2
  exit 1
fi

REVISION=$(python -c "import json; print(json.load(open('data/subsets.json'))['revision'])")
EPISODES=$(python -c "
import json
items = json.load(open('data/subsets.json'))['items'].values()
print(sorted(e for item in items for e in item['subsets']['$N']))
")
echo "N=$N seed=$SEED steps=$STEPS lr=$LR batch=$BATCH episodes=$EPISODES"

mkdir -p "$OUT_ROOT"
python train/train_relabel.py \
  --policy.path=lerobot/smolvla_libero \
  --policy.push_to_hub=false \
  --policy.optimizer_lr="$LR" \
  --policy.scheduler_warmup_steps="$WARMUP" \
  --policy.scheduler_decay_steps="$STEPS" \
  --dataset.repo_id=lerobot/libero \
  --dataset.revision="$REVISION" \
  --dataset.episodes="$EPISODES" \
  --dataset.video_backend=torchcodec \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=10 \
  --seed="$SEED" \
  --output_dir="$OUT" \
  --job_name="n$N-s$SEED" \
  2>&1 | tee "$OUT_ROOT/n$N-s$SEED.log"
