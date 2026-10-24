#!/usr/bin/env bash
# Compare scripted grasp settings on the tall bottles that fail most
# (ketchup, salad dressing): gripper yaw 0 or 90 degrees, release height
# 0.12 or 0.20 m over the basket. Every scene that holds the item, EPISODES
# training initial states each; runs in parallel.
#   PARALLEL=16 VENV=/workspace/venvs/hf26-week3/bin/activate OUT=/workspace/outputs/week3/sweep-grasp bash datagen/sweep_grasp.sh
# Results: OUT/<setting>/<item>-scene<N>/results.jsonl, then a summary per setting and item.

# Run from dev-challenge/week-3 so the relative paths below work.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The environment also sets LIBERO_CONFIG_PATH and MUJOCO_GL.
source "${VENV:-$HOME/.venvs/hf26-week1/bin/activate}"
set -euo pipefail

EPISODES=${EPISODES:-5}
INIT=${INIT:-10}
PARALLEL=${PARALLEL:-4}
OUT=${OUT:-outputs/sweep-grasp}

# Scenes that hold each item (from the scene BDDL files).
declare -A SCENES=(["ketchup"]="2 3 4 6 8 9" ["salad dressing"]="0 2 3 4 8 9")

mkdir -p "$OUT"
for yaw in 0 1.5708; do
  for release in 0.12 0.20; do
    setting="yaw${yaw}_release${release}"
    mkdir -p "$OUT/$setting"  # the logs below go here before Python creates anything
    for item in "ketchup" "salad dressing"; do
      for scene in ${SCENES[$item]}; do
        while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
        dir="$OUT/$setting/${item// /-}-scene$scene"
        python datagen/scripted_pick.py --task-id "$scene" --items "$item" --init-state "$INIT" \
          --episodes "$EPISODES" --yaw "$yaw" --release-height "$release" --no-video --output-dir "$dir" \
          > "$dir.log" 2>&1 &
      done
    done
  done
done
wait
python - "$OUT" <<'EOF'
import json, sys
from collections import defaultdict
from pathlib import Path

ok = defaultdict(lambda: [0, 0])
for f in Path(sys.argv[1]).glob("*/*/results.jsonl"):
    for line in open(f):
        r = json.loads(line)
        key = (f.parent.parent.name, r["items"][0])
        ok[key][0] += r["success"]
        ok[key][1] += 1
for (setting, item), (s, n) in sorted(ok.items()):
    print(f"{setting:24s} {item:15s} {s}/{n}")
EOF
