#!/usr/bin/env bash
# Start the whole Week 2 experiment in one tmux session and return.
#   bash run_all.sh
# Windows in the tmux session "week2":
#   n0        test set on lerobot/smolvla_libero (N = 0)
#   train     run_train.sh for each N in RUNS, one after another
#   n<N>      watch_eval.py for each run: validation at 1000/2000/3000 steps,
#             then the test set on the best checkpoint
#   done      waits for every test summary and writes ALL_DONE.txt
# RUNS defaults to the core runs 1 5 40 followed by the extras 10 3 20.
# Attach with `tmux attach -t week2`; detach with Ctrl-b d.

cd "$(dirname "${BASH_SOURCE[0]}")"
set -euo pipefail

RUNS=${RUNS:-"1 5 40 10 3 20"}
SEED=${SEED:-1000}
SESSION=week2
VENV=${VENV:-/workspace/venvs/hf26-week2/bin/activate}
EVAL_ROOT=${EVAL_ROOT:-/workspace/outputs/week2/eval}
HERE=$(pwd)

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists. Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi
mkdir -p "$EVAL_ROOT"

# Each window gets a shell with the environment, then its command, so the
# output stays on screen after the command ends.
start() {
  local name=$1 cmd=$2
  if [ "$name" = n0 ]; then
    tmux new-session -d -s "$SESSION" -n "$name"
  else
    tmux new-window -t "$SESSION" -n "$name"
  fi
  tmux send-keys -t "$SESSION:$name" "source $VENV && cd $HERE && $cmd" Enter
}

start n0 "python eval/run_eval.py --policy-path lerobot/smolvla_libero --set test --output-dir $EVAL_ROOT/n0-test"

train_cmd=""
for n in $RUNS; do
  train_cmd+="${train_cmd:+ && }bash train/run_train.sh $n $SEED"
done
start train "$train_cmd"

summaries="$EVAL_ROOT/n0-test/summary.txt"
for n in $RUNS; do
  run="n$n-s$SEED"
  start "n$n" "python eval/watch_eval.py --run $run 2>&1 | tee $EVAL_ROOT/watch-$run.log"
  summaries+=" $EVAL_ROOT/$run-test/summary.txt"
done

start done "for f in $summaries; do while [ ! -f \$f ]; do sleep 60; done; done; for f in $summaries; do echo \"== \$f\"; cat \$f; done > $EVAL_ROOT/ALL_DONE.txt; echo 'All runs finished:'; cat $EVAL_ROOT/ALL_DONE.txt"

tmux select-window -t "$SESSION:train"
echo "Started tmux session $SESSION with runs: $RUNS (seed $SEED)."
echo "Attach: tmux attach -t $SESSION    Switch windows: Ctrl-b n / Ctrl-b p    Detach: Ctrl-b d"
echo "When everything is done: $EVAL_ROOT/ALL_DONE.txt"
