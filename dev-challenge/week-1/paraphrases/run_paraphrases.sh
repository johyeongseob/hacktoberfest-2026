#!/usr/bin/env bash
# Paraphrased instructions: the original instruction and four paraphrases for
# alphabet soup (task 0), cream cheese (task 1), butter (task 6), and orange
# juice (task 9), the four items that succeeded 10/10 in the baseline.
# Run from anywhere:
#   bash paraphrases/run_paraphrases.sh
# Videos go to OUT/videos, one line per episode is appended to
# OUT/results.jsonl, and the terminal output to OUT/run.log
# (default OUT=outputs/m3-paraphrase).

# Run from dev-challenge/week-1 so the relative paths below work.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The Week 1 environment also sets LIBERO_CONFIG_PATH and MUJOCO_GL.
source ~/.venvs/hf26-week1/bin/activate
set -euo pipefail
# Models and assets are cached after the first run; skip Hub checks.
export HF_HUB_OFFLINE=1

OUT=${OUT:-outputs/m3-paraphrase}
EPISODES=${EPISODES:-10}
# task_id:item name
ITEMS=("0:alphabet soup" "1:cream cheese" "6:butter" "9:orange juice")

for entry in "${ITEMS[@]}"; do
  id=${entry%%:*}
  name=${entry#*:}
  if [ -e "$OUT/videos/task${id}_${name// /-}" ]; then
    echo "$OUT already has $name results. Set OUT to a new folder, e.g. OUT=outputs/paraphrases-2 bash $0" >&2
    exit 1
  fi
done
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

run() {
  python paraphrases/picnic_rollout.py --task-id "$1" --episodes "$EPISODES" --output-dir "$OUT" "${@:2}"
}

for entry in "${ITEMS[@]}"; do
  id=${entry%%:*}
  name=${entry#*:}
  run "$id"  # original instruction, through the same loop as the control
  run "$id" --instruction "grab the $name and put it in the basket"
  run "$id" --instruction "$name in the basket"
  run "$id" --instruction "pack the $name for our picnic"
  run "$id" --instruction "could you put the $name in the picnic basket?"
done
