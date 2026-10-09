#!/usr/bin/env bash
# M3: paraphrased instructions for cream cheese (task 1) and butter (task 6).
# Run from dev-challenge/week-1 with the hf26-week1 environment active:
#   mkdir -p outputs/m3-paraphrase
#   bash run_m3.sh 2>&1 | tee outputs/m3-paraphrase/run.log
# Each line appends to outputs/m3-paraphrase/results.jsonl.

set -euo pipefail

OUT=outputs/m3-paraphrase
EPISODES=${EPISODES:-10}
mkdir -p "$OUT"

run() {
  python picnic_rollout.py --task-id "$1" --episodes "$EPISODES" --output-dir "$OUT" "${@:2}"
}

# Original instruction through our own loop (M2 baseline: 10/10 for both).
run 1
run 6

# Cream cheese
run 1 --instruction "grab the cream cheese and put it in the basket"
run 1 --instruction "cream cheese in the basket"
run 1 --instruction "pack the cream cheese for our picnic"
run 1 --instruction "could you put the cream cheese in the picnic basket?"

# Butter
run 6 --instruction "grab the butter and put it in the basket"
run 6 --instruction "butter in the basket"
run 6 --instruction "pack the butter for our picnic"
run 6 --instruction "could you put the butter in the picnic basket?"
