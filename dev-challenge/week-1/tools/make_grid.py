"""Tile 10 episode videos into a 2x5 video grid.

Each tile is labeled with what it shows and its outcome. Videos are played at
the 20 Hz control rate, all tiles start together, and shorter episodes freeze
on their last frame until the longest one ends.

    python tools/make_grid.py m2 --task-id 3                 # one item, episodes 0-9
    python tools/make_grid.py m2-items --episode 0           # items 0-9, one episode each
    python tools/make_grid.py m3 --task-id 1 --instruction "grab the cream cheese and put it in the basket"
    python tools/make_grid.py m3 --task-id 6                 # original instruction
    python tools/make_grid.py m3-row --task-id 0             # one item, original + A-D in one row
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
LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
LABEL_SIZE = 13
MONO_ADVANCE = 0.602  # DejaVu Sans Mono glyph width as a fraction of the font size
ITEMS = [
    "alphabet soup", "cream cheese", "salad dressing", "bbq sauce", "ketchup",
    "tomato sauce", "butter", "milk", "chocolate pudding", "orange juice",
]


def m2_tasks() -> dict[int, dict]:
    # lerobot-eval writes videos at 80 fps and success flags to eval_info.json.
    info = json.loads((OUTPUTS / "m2-object" / "eval_info.json").read_text())
    return {t["task_id"]: t["metrics"] for t in info["per_task"]}


def m2_episodes(task_id: int) -> list[tuple[Path, bool, str]]:
    metrics = m2_tasks()[task_id]
    return [(Path(p), ok, f"ep{i}") for i, (p, ok) in enumerate(zip(metrics["video_paths"], metrics["successes"]))]


def m2_items(episode: int) -> list[tuple[Path, bool, str]]:
    tasks = m2_tasks()
    return [
        (Path(tasks[t]["video_paths"][episode]), tasks[t]["successes"][episode], f"{t}: {ITEMS[t]}")
        for t in sorted(tasks)
    ]


def m3_episodes(task_id: int, instruction: str | None) -> list[tuple[Path, bool, str]]:
    # picnic_rollout.py writes videos at 20 fps and one JSON line per episode.
    label = "original" if instruction is None else re.sub(r"[^a-z0-9]+", "-", instruction.lower()).strip("-")
    results = [json.loads(line) for line in (OUTPUTS / "m3-paraphrase" / "results.jsonl").read_text().splitlines()]
    rows = {
        r["episode"]: r
        for r in results
        if r["task_id"] == task_id and Path(r["video"]).parent.name == label
    }
    return [(Path(rows[ep]["video"]), rows[ep]["success"], f"ep{ep}") for ep in sorted(rows)]


# (short description shown above the instruction, instruction template)
M3_STYLES = [
    ("Original: training template", "pick up the {item} and place it in the basket"),
    ("A: both verbs swapped", "grab the {item} and put it in the basket"),
    ("B: no verb", "{item} in the basket"),
    ("C: new verb, no basket", "pack the {item} for our picnic"),
    ("D: question, one verb", "could you put the {item} in the picnic basket?"),
]


def m3_row(task_id: int) -> list[tuple[Path, bool, str, str, str]]:
    # Original and A-D for one item: the first successful episode of each, or episode 0 if none succeeded.
    results = [json.loads(line) for line in (OUTPUTS / "m3-paraphrase" / "results.jsonl").read_text().splitlines()]
    row = []
    for title, style in M3_STYLES:
        instruction = style.format(item=ITEMS[task_id])
        runs = sorted((r for r in results if r["task_id"] == task_id and r["instruction"] == instruction),
                      key=lambda r: r["episode"])
        pick = next((r for r in runs if r["success"]), runs[0])
        row.append((Path(pick["video"]), pick["success"], f"{task_id}: {ITEMS[task_id]}", instruction, title))
    return row


def duration_s(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["m2", "m2-items", "m3", "m3-row"])
    parser.add_argument("--task-id", type=int, help="m2 and m3: which item")
    parser.add_argument("--episode", type=int, default=0, help="m2-items: which episode of every item")
    parser.add_argument("--instruction", default=None, help="m3 only; omit for the original instruction")
    parser.add_argument("--no-labels", action="store_true", help="skip text labels if ffmpeg has no font support")
    args = parser.parse_args()

    columns, rows, tile = COLUMNS, ROWS, TILE
    if args.source == "m2":
        episodes = m2_episodes(args.task_id)
        speed = 4  # 80 fps file -> 20 Hz playback
        name = f"m2_task{args.task_id}"
    elif args.source == "m2-items":
        episodes = m2_items(args.episode)
        speed = 4
        name = f"m2_items_ep{args.episode}"
    elif args.source == "m3-row":
        episodes = m3_row(args.task_id)
        speed = 1
        columns, rows, tile = 5, 1, 384  # larger tiles so the longest instruction fits on one line
        name = f"m3_row_task{args.task_id}"
    else:
        episodes = m3_episodes(args.task_id, args.instruction)
        speed = 1
        label = "original" if args.instruction is None else re.sub(r"[^a-z0-9]+", "-", args.instruction.lower()).strip("-")
        name = f"m3_task{args.task_id}_{label}"
    if len(episodes) != columns * rows:
        raise SystemExit(f"Expected {columns * rows} episodes, found {len(episodes)}")
    # Optional fourth and fifth fields: a caption (the instruction) and a title above it,
    # both shown at the bottom center.
    episodes = [(e + (None, None))[:5] for e in episodes]
    captions = [e[3] for e in episodes if e[3]]
    # One font size for every caption, chosen so the longest one fits on one line.
    caption_size = min(LABEL_SIZE, int((tile - 16) / (max(map(len, captions), default=1) * MONO_ADVANCE)))

    length = max(duration_s(path) * speed for path, *_ in episodes)
    filters, layout = [], []
    for i, (_, ok, tag, caption, title) in enumerate(episodes):
        chain = f"[{i}:v]setpts={speed}*(PTS-STARTPTS),fps={CONTROL_FPS},scale={tile}:{tile},tpad=stop_mode=clone:stop_duration={length:.2f}"
        if not args.no_labels:
            # Top left: name or episode in white, then the outcome in green or red.
            # A monospace font makes the name width predictable, so the outcome can follow it.
            box = f"fontfile={LABEL_FONT}:fontsize={LABEL_SIZE}:box=1:boxcolor=black@0.5:boxborderw=3:y=6:y_align=font"
            outcome, color = ("success", "lime") if ok else ("fail", "red")
            outcome_x = 6 + round(len(tag) * LABEL_SIZE * MONO_ADVANCE) + 12
            safe_tag = tag.replace(":", r"\:")  # ":" separates drawtext options
            chain += f",drawtext=text='{safe_tag}':x=6:fontcolor=white:{box}"
            chain += f",drawtext=text='{outcome}':x={outcome_x}:fontcolor={color}:{box}"
            if caption:
                caption_y = tile - 8 - round(caption_size * 1.3)
                chain += (f",drawtext=text='{caption}':fontfile={LABEL_FONT}:fontsize={caption_size}:fontcolor=white"
                          f":box=1:boxcolor=black@0.5:boxborderw=3:x=(w-tw)/2:y={caption_y}:y_align=font")
            if title:
                title_y = caption_y - 8 - round(LABEL_SIZE * 1.3)
                safe_title = title.replace(":", r"\:")  # ":" separates drawtext options
                chain += (f",drawtext=text='{safe_title}':fontfile={LABEL_FONT}:fontsize={LABEL_SIZE}:fontcolor=white"
                          f":box=1:boxcolor=black@0.5:boxborderw=3:x=(w-tw)/2:y={title_y}:y_align=font")
        filters.append(f"{chain}[v{i}]")
        layout.append(f"{(i % columns) * tile}_{(i // columns) * tile}")
    inputs = "".join(f"[v{i}]" for i in range(len(episodes)))
    filters.append(f"{inputs}xstack=inputs={len(episodes)}:layout={'|'.join(layout)}[out]")

    # Grids live next to the run they show.
    grid_dir = OUTPUTS / "m2-object" if args.source.startswith("m2") else OUTPUTS / "m3-paraphrase" / "videos"
    grid_dir.mkdir(parents=True, exist_ok=True)
    output = grid_dir / f"{name}.mp4"
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for path, *_ in episodes:
        command += ["-i", str(path)]
    command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-t", f"{length:.2f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
    subprocess.run(command, check=True)
    successes = sum(ok for _, ok, *_ in episodes)
    print(f"{output}  ({successes}/{len(episodes)} success, {length:.1f} s)")


if __name__ == "__main__":
    main()
