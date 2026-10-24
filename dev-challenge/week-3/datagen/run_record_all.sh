#!/usr/bin/env bash
# Record the training chains of data/chain_plan.json for all ten scenes in
# parallel (PARTS datasets per scene), then merge them into one LeRobot dataset.
#   PARALLEL=16 VENV=/workspace/venvs/hf26-week3/bin/activate OUT=/workspace/outputs/week3/datasets bash datagen/run_record_all.sh
# The next 120 orders of the large plan (orders 5-8 of each scene and length):
#   PLAN=data/chain_plan_large.json BATCH=1 OUT=/workspace/outputs/week3/datasets-batch1 ... bash datagen/run_record_all.sh
# Writes OUT/scene<N>-part<i> (with chains.jsonl and a log) and OUT/scripted-chains (merged).

# Run from dev-challenge/week-3 so the relative paths below work.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The environment also sets LIBERO_CONFIG_PATH and MUJOCO_GL.
source "${VENV:-$HOME/.venvs/hf26-week1/bin/activate}"
set -euo pipefail

SCENES=${SCENES:-"0 1 2 3 4 5 6 7 8 9"}
PARALLEL=${PARALLEL:-4}
MAX_TRIES=${MAX_TRIES:-10}
PARTS=${PARTS:-2}
PLAN=${PLAN:-data/chain_plan.json}
BATCH=${BATCH:-}
OUT=${OUT:-outputs/datasets}

mkdir -p "$OUT"
for scene in $SCENES; do
  for ((part = 0; part < PARTS; part++)); do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
    name="scene$scene-part$part"
    echo "$name"
    python datagen/record_scripted.py --task-id "$scene" --plan "$PLAN" --split train ${BATCH:+--batch "$BATCH"} \
      --part "$part/$PARTS" --max-tries "$MAX_TRIES" --root "$OUT/$name" > "$OUT/$name.log" 2>&1 &
  done
done
wait
grep -h "^saved" "$OUT"/scene*.log

python - "$OUT" <<'EOF'
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

out = Path(sys.argv[1])
# Parts that saved no episode have no data to merge.
roots = sorted(p for p in out.glob("scene*-part*") if (p / "data").exists())
aggregate_datasets(
    repo_ids=[f"local/{root.name}" for root in roots],
    roots=roots,
    aggr_repo_id="local/scripted-chains",
    aggr_root=out / "scripted-chains",
)
print(f"merged {len(roots)} parts into {out / 'scripted-chains'}")
EOF
