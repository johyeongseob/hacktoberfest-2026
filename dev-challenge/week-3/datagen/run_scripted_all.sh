#!/usr/bin/env bash
# Scripted pick of every item in every LIBERO-Object scene, each item in its
# own episode, from training initial states INIT, INIT+1, ... (EPISODES of
# them). No policy and no GPU model; scenes run in parallel.
#   bash datagen/run_scripted_all.sh
#   PARALLEL=10 EPISODES=5 VENV=/workspace/venvs/hf26-week3/bin/activate OUT=/workspace/outputs/week3/scripted-all bash datagen/run_scripted_all.sh
# Each scene writes results.jsonl to OUT/scene<N> and its log to OUT/scene<N>.log.

# Run from dev-challenge/week-3 so the relative paths below work.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The environment also sets LIBERO_CONFIG_PATH and MUJOCO_GL.
source "${VENV:-$HOME/.venvs/hf26-week1/bin/activate}"
set -euo pipefail

SCENES=${SCENES:-"0 1 2 3 4 5 6 7 8 9"}
INIT=${INIT:-10}
EPISODES=${EPISODES:-5}
PARALLEL=${PARALLEL:-4}
OUT=${OUT:-outputs/scripted-all}

mkdir -p "$OUT"
for scene in $SCENES; do
  # Keep at most PARALLEL scenes running.
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
  echo "scene $scene"
  python datagen/scripted_pick.py --task-id "$scene" --each --init-state "$INIT" --episodes "$EPISODES" \
    --no-video --output-dir "$OUT/scene$scene" > "$OUT/scene$scene.log" 2>&1 &
done
wait
python - "$OUT" <<'EOF'
import json, sys
from collections import defaultdict
from pathlib import Path

rows = [json.loads(line) for f in sorted(Path(sys.argv[1]).glob("scene*/results.jsonl")) for line in open(f)]
by_item = defaultdict(lambda: [0, 0])
for r in rows:
    by_item[r["items"][0]][0] += r["success"]
    by_item[r["items"][0]][1] += 1
print(f"all: {sum(r['success'] for r in rows)}/{len(rows)}")
for item, (ok, n) in sorted(by_item.items()):
    print(f"{item}: {ok}/{n}")
EOF
