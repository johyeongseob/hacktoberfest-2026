"""Run the Week 2 validation or test set for one checkpoint, in parallel.

Sets (item task IDs: 0 alphabet soup, 1 cream cheese, 6 butter, 9 orange juice):
    val   4 items x A-D x initial states 10-14            =  80 episodes
    test  4 items x original, A-H x initial states 0-9    = 360 episodes
          + the other 6 items x original x states 0-9     =  60 episodes

Each job is one `rollout.py` process (policy loaded once per job); jobs run
side by side on one GPU. Each job writes OUTPUT_DIR/parts/<job>.jsonl and
OUTPUT_DIR/logs/<job>.log. Only when every job succeeds are the parts merged
into OUTPUT_DIR/results.jsonl and a per-wording OUTPUT_DIR/summary.txt
written; after a failure, rerunning the same command redoes all jobs.

Examples:
    python run_eval.py --policy-path lerobot/smolvla_libero --set test --output-dir outputs/eval/n0-test
    python run_eval.py --policy-path outputs/train/n5/checkpoints/001000/pretrained_model \
        --set val --output-dir outputs/eval/n5-val-001000 --no-video
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINED_ITEMS = [0, 1, 6, 9]
OTHER_ITEMS = [2, 3, 4, 5, 7, 8]


def jobs_for(set_name: str) -> list[dict]:
    if set_name == "val":
        return [
            {"task_id": task_id, "wordings": ["A", "B", "C", "D"], "init_state": 10, "episodes": 5}
            for task_id in TRAINED_ITEMS
        ]
    jobs = []
    for task_id in TRAINED_ITEMS:
        # Two jobs per trained item to balance the load.
        jobs.append({"task_id": task_id, "wordings": ["original", "A", "B", "C", "D"], "init_state": 0, "episodes": 10})
        jobs.append({"task_id": task_id, "wordings": ["E", "F", "G", "H"], "init_state": 0, "episodes": 10})
    for task_id in OTHER_ITEMS:
        jobs.append({"task_id": task_id, "wordings": ["original"], "init_state": 0, "episodes": 10})
    return jobs


def job_name(job: dict) -> str:
    return f"task{job['task_id']}_{'-'.join(job['wordings'])}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--set", choices=["val", "test"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=8, help="Rollout processes at the same time")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    if (args.output_dir / "results.jsonl").exists():
        raise SystemExit(f"{args.output_dir} already has results; choose a new --output-dir")
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)

    def run(job: dict) -> int:
        name = job_name(job)
        cmd = [
            sys.executable, str(HERE / "rollout.py"),
            "--policy-path", args.policy_path,
            "--task-id", str(job["task_id"]),
            "--wordings", *job["wordings"],
            "--init-state", str(job["init_state"]),
            "--episodes", str(job["episodes"]),
            "--device", args.device,
            "--output-dir", str(args.output_dir),
            "--results-file", str(args.output_dir / "parts" / f"{name}.jsonl"),
        ]
        if args.no_video:
            cmd.append("--no-video")
        with open(args.output_dir / "logs" / f"{name}.log", "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{'done' if code == 0 else f'FAILED ({code})'}: {name}", flush=True)
        return code

    jobs = jobs_for(args.set)
    with ThreadPoolExecutor(args.parallel) as pool:
        codes = list(pool.map(run, jobs))

    # No summary.txt after a failure: watch_eval.py and run_all.sh treat
    # summary.txt as "this set is complete".
    failed = [job_name(job) for job, code in zip(jobs, codes) if code != 0]
    if failed:
        raise SystemExit(f"Failed jobs (see {args.output_dir / 'logs'}): {', '.join(failed)}")

    rows = []
    for job in jobs:
        part = args.output_dir / "parts" / f"{job_name(job)}.jsonl"
        rows += [json.loads(line) for line in part.read_text().splitlines()]
    with open(args.output_dir / "results.jsonl", "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    summary = defaultdict(lambda: [0, 0])
    for row in rows:
        group = row["wording"] if row["task_id"] in TRAINED_ITEMS else "other items"
        summary[group][0] += row["success"]
        summary[group][1] += 1
    calls = sum(r["inference_calls"] for r in rows)
    order = ["original", *"ABCDEFGH", "other items"]
    lines = [f"{group}: {summary[group][0]}/{summary[group][1]}" for group in order if group in summary]
    lines.append(f"inference: {calls} calls, mean {sum(r['inference_total_s'] for r in rows) / calls:.3f} s per call")
    (args.output_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
