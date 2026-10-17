"""Validate every checkpoint of one training run as it appears, then test the best.

Start it next to `run_train.sh` (or after training). For each checkpoint step
(every SAVE_FREQ steps up to STEPS) it waits until the checkpoint is fully
written, runs the validation set (A-D, initial states 10-14, 80 episodes,
no videos), and records the score. When all checkpoints are scored, it picks
the best (ties go to the earlier step) and runs the test set on it.

Outputs, under EVAL_ROOT (default /workspace/outputs/week2/eval):
    <run>-val-<step>/     validation results per checkpoint
    <run>-val.json        validation curve and the chosen step
    <run>-test/           test results of the chosen checkpoint

Example (in its own tmux window, for run n1-s1000):
    python watch_eval.py --run n1-s1000
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Training run name, e.g. n1-s1000")
    parser.add_argument("--train-root", type=Path, default=Path("/workspace/outputs/week2/train"))
    parser.add_argument("--eval-root", type=Path, default=Path("/workspace/outputs/week2/eval"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--parallel", type=int, default=4, help="Rollout processes per validation")
    args = parser.parse_args()

    checkpoints = args.train_root / args.run / "checkpoints"
    curve = []
    for step in range(args.save_freq, args.steps + 1, args.save_freq):
        step_dir = checkpoints / f"{step:06d}"
        # LeRobot writes pretrained_model first and training_state second, so
        # training_state means the model files are complete.
        while not (step_dir / "training_state").exists():
            time.sleep(30)
        time.sleep(10)
        out = args.eval_root / f"{args.run}-val-{step:06d}"
        if not (out / "summary.txt").exists():
            subprocess.run(
                [
                    sys.executable, str(HERE / "run_eval.py"),
                    "--policy-path", str(step_dir / "pretrained_model"),
                    "--set", "val", "--no-video",
                    "--parallel", str(args.parallel),
                    "--output-dir", str(out),
                ],
                check=True,
            )
        rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
        successes = sum(row["success"] for row in rows)
        curve.append({"step": step, "successes": successes, "episodes": len(rows)})
        print(f"[{args.run}] step {step}: validation {successes}/{len(rows)}", flush=True)

    best = max(curve, key=lambda point: (point["successes"], -point["step"]))
    (args.eval_root / f"{args.run}-val.json").write_text(
        json.dumps({"run": args.run, "curve": curve, "chosen_step": best["step"]}, indent=2) + "\n"
    )
    test_dir = args.eval_root / f"{args.run}-test"
    if (test_dir / "summary.txt").exists():
        print(f"[{args.run}] chosen step {best['step']}; test already done: {test_dir / 'summary.txt'}", flush=True)
        return
    print(f"[{args.run}] chosen step {best['step']} ({best['successes']}/{best['episodes']}); running the test set", flush=True)
    subprocess.run(
        [
            sys.executable, str(HERE / "run_eval.py"),
            "--policy-path", str(checkpoints / f"{best['step']:06d}" / "pretrained_model"),
            "--set", "test",
            "--output-dir", str(test_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
