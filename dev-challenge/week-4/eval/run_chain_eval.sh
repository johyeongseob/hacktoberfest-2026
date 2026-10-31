#!/usr/bin/env bash
# Tests 2 and 3: two items, step by step (the instruction switches to the
# next item once an item is in the basket), all ten scenes in parallel, from
# initial state 0. SPLIT=test_seen: trained orders; SPLIT=test: held-out orders.
#   SPLIT=test_seen VIDEO=1 OUT=/workspace/outputs/week4/eval/base/seen bash eval/run_chain_eval.sh
#   SPLIT=test VIDEO=1 OUT=/workspace/outputs/week4/eval/base/heldout bash eval/run_chain_eval.sh
# POLICY=<checkpoint>/pretrained_model tests a fine-tuned model instead.
# Writes OUT/scene<N>/results.jsonl, a log per scene, and a summary.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "${VENV:-/workspace/venvs/hf26-week3/bin/activate}"
set -euo pipefail

POLICY=${POLICY:-lerobot/smolvla_libero}
PLAN=${PLAN:-data/pair_plan.json}
SPLIT=${SPLIT:-test}
LENGTHS=${LENGTHS:-2}  # the task is two items
INIT=${INIT:-0}
EPISODES=${EPISODES:-1}
# Step limit per item; the human demos take about 146 steps per item.
STEPS_PER_ITEM=${STEPS_PER_ITEM:-300}
PARALLEL=${PARALLEL:-10}
VIDEO=${VIDEO:-0}
OUT=${OUT:?set OUT to an output folder}

mkdir -p "$OUT"
for scene in 0 1 2 3 4 5 6 7 8 9; do
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
  # shellcheck disable=SC2086  # LENGTHS holds several numbers
  python eval/chain_rollout.py --policy-path "$POLICY" --task-id "$scene" \
    --plan "$PLAN" --split "$SPLIT" --lengths $LENGTHS --init-state "$INIT" --episodes "$EPISODES" \
    --steps-per-item "$STEPS_PER_ITEM" \
    --device cuda $([ "$VIDEO" = 1 ] || echo --no-video) --output-dir "$OUT/scene$scene" \
    > "$OUT/scene$scene.log" 2>&1 &
done
wait
python - "$OUT" <<'EOF'
import json, sys
from collections import defaultdict
from pathlib import Path

ok = defaultdict(lambda: [0, 0, 0])
for f in Path(sys.argv[1]).glob("scene*/results.jsonl"):
    for line in open(f):
        r = json.loads(line)
        n = len(r["items"])
        ok[n][0] += r["success"]
        ok[n][1] += 1
        ok[n][2] += len(r["items_in_basket_at_end"])
for n, (s, total, items_in) in sorted(ok.items()):
    print(f"{n} items: all in {s}/{total}, items in basket {items_in}/{n * total}")
EOF
