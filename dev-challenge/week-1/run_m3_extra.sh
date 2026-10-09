#!/usr/bin/env bash
# M3 extension: the same five instructions for orange juice (task 9) and
# alphabet soup (task 0). Both succeeded 10/10 in M2.
# Run from dev-challenge/week-1 with the hf26-week1 environment active:
#   bash run_m3_extra.sh 2>&1 | tee -a outputs/m3-paraphrase/run.log
# Results append to outputs/m3-paraphrase/results.jsonl.

set -euo pipefail

# Models and assets are cached; skip Hugging Face Hub checks.
export HF_HUB_OFFLINE=1

OUT=outputs/m3-paraphrase
EPISODES=${EPISODES:-10}
mkdir -p "$OUT"

run() {
  python picnic_rollout.py --task-id "$1" --episodes "$EPISODES" --output-dir "$OUT" "${@:2}"
}

# Orange juice
run 9
run 9 --instruction "grab the orange juice and put it in the basket"
run 9 --instruction "orange juice in the basket"
run 9 --instruction "pack the orange juice for our picnic"
run 9 --instruction "could you put the orange juice in the picnic basket?"

# Alphabet soup
run 0
run 0 --instruction "grab the alphabet soup and put it in the basket"
run 0 --instruction "alphabet soup in the basket"
run 0 --instruction "pack the alphabet soup for our picnic"
run 0 --instruction "could you put the alphabet soup in the picnic basket?"
