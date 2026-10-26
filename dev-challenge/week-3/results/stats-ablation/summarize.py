"""Recompute the table in README.md from the per-scene results in this folder.

Also writes episodes.csv, one readable row per episode.

Usage:
    python summarize.py
"""

import csv
import json
from pathlib import Path

# LIBERO-Object task i places this item; scene i is task i at initial state 46.
OWN_ITEM = [
    "alphabet soup",
    "cream cheese",
    "salad dressing",
    "bbq sauce",
    "ketchup",
    "tomato sauce",
    "butter",
    "milk",
    "chocolate pudding",
    "orange juice",
]

RUNS = [
    "published",
    "dataset-stats-001000",
    "pretrained-stats-001000",
    "dataset-stats-002000",
    "pretrained-stats-002000",
    "dataset-stats-003000",
    "pretrained-stats-003000",
]


def main() -> None:
    here = Path(__file__).parent
    rows = []
    for run in RUNS:
        episodes = [
            json.loads(line)
            for path in sorted((here / run).glob("scene*.jsonl"))
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        own = [ep for ep in episodes if ep["items"] == [OWN_ITEM[ep["task_id"]]]]
        other = [ep for ep in episodes if ep["items"] != [OWN_ITEM[ep["task_id"]]]]
        own_ok = sum(ep["success"] for ep in own)
        other_ok = sum(ep["success"] for ep in other)
        print(f"{run:24s} own {own_ok}/{len(own)}  other {other_ok}/{len(other)}")

        stats, _, steps = run.rpartition("-")
        for ep in episodes:
            scene = ep["task_id"]
            rows.append(
                {
                    "stats": stats if stats else "published",
                    "steps": 0 if run == "published" else int(steps),
                    "scene": scene,
                    "scene_item": OWN_ITEM[scene],
                    "asked_item": ep["items"][0],
                    "own_item": ep["items"] == [OWN_ITEM[scene]],
                    "success": ep["success"],
                }
            )

    with open(here / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
