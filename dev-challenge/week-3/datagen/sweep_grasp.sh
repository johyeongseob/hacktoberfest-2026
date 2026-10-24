#!/usr/bin/env bash
# Compare scripted pick settings on every item of every LIBERO-Object scene,
# EPISODES training initial states each, in parallel.
# SETTINGS holds name=arguments pairs for datagen/scripted_pick.py, separated by ";".
#   PARALLEL=16 VENV=/workspace/venvs/hf26-week3/bin/activate OUT=/workspace/outputs/week3/sweep-center bash datagen/sweep_grasp.sh
# First run (2026-10-24): gripper yaw 90 degrees and a 0.20 m release did not
# help the tall bottles (ketchup 8/30 -> 4/30, salad dressing 11/30 -> 5/30).
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
SETTINGS=${SETTINGS:-"gripper=;item=--center-object --travel-height 0.35 --release-height 0.10"}

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
IFS=";" read -ra pairs <<< "$SETTINGS"
for pair in "${pairs[@]}"; do
  setting=${pair%%=*}
  args=${pair#*=}
  mkdir -p "$OUT/$setting"  # the logs below go here before Python creates anything
  for scene in 0 1 2 3 4 5 6 7 8 9; do
    IFS='|' read -ra items <<< "${ITEMS[$scene]}"
    for item in "${items[@]}"; do
      while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
      dir="$OUT/$setting/${item// /-}-scene$scene"
      # shellcheck disable=SC2086  # args holds zero or more flags
      python datagen/scripted_pick.py --task-id "$scene" --items "$item" --init-state "$INIT" \
        --episodes "$EPISODES" $args --no-video --output-dir "$dir" > "$dir.log" 2>&1 &
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
    setting = f.parent.parent.name
    for line in open(f):
        r = json.loads(line)
        for key in ((setting, r["items"][0]), (setting, "ALL")):
            ok[key][0] += r["success"]
            ok[key][1] += 1
for (setting, item), (s, n) in sorted(ok.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    print(f"{item:18s} {setting:10s} {s}/{n}")
EOF
