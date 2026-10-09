"""Tile the 10 episodes of one item (and one instruction) into a 2x5 video grid.

Each tile is labeled with its episode number and outcome. Videos are played at
the 20 Hz control rate, and shorter episodes freeze on their last frame until
the longest one ends.

    python make_grid.py m2 --task-id 3
    python make_grid.py m3 --task-id 1 --instruction "grab the cream cheese and put it in the basket"
    python make_grid.py m3 --task-id 6            # original instruction
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

OUTPUTS = Path("outputs")
CONTROL_FPS = 20
TILE = 256
COLUMNS, ROWS = 5, 2


def m2_episodes(task_id: int) -> list[tuple[Path, bool]]:
    # lerobot-eval writes videos at 80 fps and success flags to eval_info.json.
    info = json.loads((OUTPUTS / "m2-object" / "eval_info.json").read_text())
    task = next(t for t in info["per_task"] if t["task_id"] == task_id)
    return [(Path(path), ok) for path, ok in zip(task["metrics"]["video_paths"], task["metrics"]["successes"])]


def m3_episodes(task_id: int, instruction: str | None) -> list[tuple[Path, bool]]:
    # picnic_rollout.py writes videos at 20 fps and one JSON line per episode.
    label = "original" if instruction is None else re.sub(r"[^a-z0-9]+", "-", instruction.lower()).strip("-")
    results = [json.loads(line) for line in (OUTPUTS / "m3-paraphrase" / "results.jsonl").read_text().splitlines()]
    rows = {
        r["episode"]: r
        for r in results
        if r["task_id"] == task_id and Path(r["video"]).name.startswith(f"task{task_id}_{label}_ep")
    }
    return [(Path(rows[ep]["video"]), rows[ep]["success"]) for ep in sorted(rows)]


def duration_s(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["m2", "m3"])
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--instruction", default=None, help="M3 only; omit for the original instruction")
    parser.add_argument("--no-labels", action="store_true", help="skip text labels if ffmpeg has no font support")
    args = parser.parse_args()

    if args.source == "m2":
        episodes = m2_episodes(args.task_id)
        speed = 4  # 80 fps file -> 20 Hz playback
        name = f"m2_task{args.task_id}"
    else:
        episodes = m3_episodes(args.task_id, args.instruction)
        speed = 1
        label = "original" if args.instruction is None else re.sub(r"[^a-z0-9]+", "-", args.instruction.lower()).strip("-")
        name = f"m3_task{args.task_id}_{label}"
    if len(episodes) != COLUMNS * ROWS:
        raise SystemExit(f"Expected {COLUMNS * ROWS} episodes, found {len(episodes)}")

    length = max(duration_s(path) * speed for path, _ in episodes)
    filters, layout = [], []
    for i, (_, ok) in enumerate(episodes):
        chain = f"[{i}:v]setpts={speed}*(PTS-STARTPTS),fps={CONTROL_FPS},scale={TILE}:{TILE},tpad=stop_mode=clone:stop_duration={length:.2f}"
        if not args.no_labels:
            text = f"ep{i} {'success' if ok else 'fail'}"
            color = "lime" if ok else "red"
            chain += f",drawtext=text='{text}':x=6:y=6:fontsize=18:fontcolor={color}:box=1:boxcolor=black@0.5"
        filters.append(f"{chain}[v{i}]")
        layout.append(f"{(i % COLUMNS) * TILE}_{(i // COLUMNS) * TILE}")
    inputs = "".join(f"[v{i}]" for i in range(len(episodes)))
    filters.append(f"{inputs}xstack=inputs={len(episodes)}:layout={'|'.join(layout)}[out]")

    grid_dir = OUTPUTS / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    output = grid_dir / f"{name}.mp4"
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for path, _ in episodes:
        command += ["-i", str(path)]
    command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-t", f"{length:.2f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
    subprocess.run(command, check=True)
    successes = sum(ok for _, ok in episodes)
    print(f"{output}  ({successes}/{len(episodes)} success, {length:.1f} s)")


if __name__ == "__main__":
    main()
