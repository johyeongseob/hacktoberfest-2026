"""Build the nested demo subsets for the four Week 1 items.

For each item, the demos are the LIBERO-Object episodes of `lerobot/libero`
whose task is "pick up the {item} and place it in the basket", sorted by
episode index. The subset for N keeps the first N, so 1 is inside 3, 3 inside
5, and so on. Writes data/subsets.json next to this script.

Run with the Week 2 environment (pyarrow and huggingface_hub come with
LeRobot):
    python data/make_subsets.py
"""

import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO_ID = "lerobot/libero"
REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
ITEMS = ["alphabet soup", "cream cheese", "butter", "orange juice"]
NS = [1, 3, 5, 10, 20, 40]


def format_list(match: re.Match) -> str:
    # Pad the key so every list starts in the same column, and continue long
    # lists in rows of 10 under the first number.
    indent, key, body = match.groups()
    numbers = [n.strip() for n in body.split(",")]
    key = key.ljust(len(f'"{max(NS)}": '))
    rows = [", ".join(numbers[i : i + 10]) for i in range(0, len(numbers), 10)]
    continuation = ",\n" + " " * len(f"{indent}{key}[")
    return f"{indent}{key}[{continuation.join(rows)}]"


def main():
    # Only the metadata and the small state/action parquet files, not the videos.
    root = Path(
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=["meta/tasks.parquet", "data/*/*.parquet"],
        )
    )
    tasks = {
        row["task_index"]: row["__index_level_0__"]
        for row in pq.read_table(root / "meta/tasks.parquet").to_pylist()
    }
    episode_task = {}
    for path in sorted(root.glob("data/*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        for episode, task in zip(
            table.column("episode_index").to_pylist(),
            table.column("task_index").to_pylist(),
        ):
            episode_task[episode] = task

    items = {}
    for item in ITEMS:
        instruction = f"pick up the {item} and place it in the basket"
        episodes = sorted(e for e, t in episode_task.items() if tasks[t] == instruction)
        items[item] = {
            "instruction": instruction,
            "n_available": len(episodes),
            "subsets": {str(n): episodes[:n] for n in NS},
        }
        print(f"{item}: {len(episodes)} demos, first 3 = {episodes[:3]}")

    text = json.dumps({"repo_id": REPO_ID, "revision": REVISION, "items": items}, indent=2)
    # Aligned episode lists in rows of 10, so the nested subsets line up.
    text = re.sub(r"(?m)^( *)(\"\d+\": )\[\s+([\d,\s]+?)\s+\]", format_list, text)
    out = Path(__file__).with_name("subsets.json")
    out.write_text(text + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
