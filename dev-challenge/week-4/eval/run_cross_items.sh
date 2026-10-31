#!/usr/bin/env bash
# Test 1: every item in every scene. Every LIBERO-Object scene holds six
# items; ask for each one on its own ("pick up the {item} and place it in
# the basket"), one episode each, from initial state 0 (no training episode
# uses it). The scene's own item is the control.
#   OUT=/workspace/outputs/week4/eval/base/single bash eval/run_cross_items.sh
#   POLICY=<checkpoint>/pretrained_model OUT=... bash eval/run_cross_items.sh
# Each scene writes results.jsonl and videos to OUT/scene<N> and its log to
# OUT/scene<N>.log.

# Run from dev-challenge/week-4 so the relative paths below work.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The environment also sets LIBERO_CONFIG_PATH and MUJOCO_GL.
source "${VENV:-/workspace/venvs/hf26-week3/bin/activate}"
set -euo pipefail

SCENES=${SCENES:-"0 1 2 3 4 5 6 7 8 9"}
INIT=${INIT:-0}
DEVICE=${DEVICE:-cuda}
PARALLEL=${PARALLEL:-10}
OUT=${OUT:?set OUT to an output folder}
POLICY=${POLICY:-lerobot/smolvla_libero}
EPISODES=${EPISODES:-1}
# Step limit per item; the human demos take about 146 steps per item.
STEPS_PER_ITEM=${STEPS_PER_ITEM:-300}

# Items in each LIBERO-Object scene (from the scene BDDL files).
ITEMS[0]="alphabet soup|butter|cream cheese|milk|salad dressing|tomato sauce"
ITEMS[1]="alphabet soup|butter|cream cheese|milk|orange juice|tomato sauce"
ITEMS[2]="alphabet soup|cream cheese|ketchup|milk|salad dressing|tomato sauce"
ITEMS[3]="alphabet soup|bbq sauce|chocolate pudding|cream cheese|ketchup|salad dressing"
ITEMS[4]="alphabet soup|bbq sauce|cream cheese|ketchup|milk|salad dressing"
ITEMS[5]="bbq sauce|butter|chocolate pudding|milk|orange juice|tomato sauce"
ITEMS[6]="bbq sauce|butter|chocolate pudding|ketchup|orange juice|tomato sauce"
ITEMS[7]="butter|chocolate pudding|cream cheese|milk|orange juice|tomato sauce"
ITEMS[8]="alphabet soup|bbq sauce|chocolate pudding|ketchup|orange juice|salad dressing"
ITEMS[9]="bbq sauce|butter|chocolate pudding|ketchup|orange juice|salad dressing"

mkdir -p "$OUT"
for scene in $SCENES; do
  # Keep at most PARALLEL scenes running.
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
  IFS='|' read -ra items <<< "${ITEMS[$scene]}"
  echo "scene $scene: ${ITEMS[$scene]}"
  python eval/chain_rollout.py --policy-path "$POLICY" --each --task-id "$scene" \
    --items "${items[@]}" --init-state "$INIT" --episodes "$EPISODES" --steps-per-item "$STEPS_PER_ITEM" --device "$DEVICE" --output-dir "$OUT/scene$scene" \
    > "$OUT/scene$scene.log" 2>&1 &
done
wait
grep -h "all items in the basket" "$OUT"/scene*.log
echo "All scenes done: $OUT/scene*/results.jsonl"
