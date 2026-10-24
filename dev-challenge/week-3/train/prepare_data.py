"""Build the Week 3 training set: scripted chains plus one original demo per item.

The scripted chains teach new behavior (other items, the next item after
one is in the basket). Fine-tuning on them alone could make the robot
forget its original single-item demos, as the six untrained items did in
Week 2, so the first LIBERO-Object demo of each of the ten items from
lerobot/libero (pinned revision, as in Week 2) is copied in as well. The
copies are written with the same features as the scripted recordings, then
everything is merged into one LeRobot dataset.

Usage (on the pod, from dev-challenge/week-3):
    python train/prepare_data.py --chains /workspace/outputs/week3/datasets/scripted-chains \
        --out /workspace/outputs/week3/train-data/chains-120
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datagen"))
from record_scripted import FEATURES, FPS  # noqa: E402

REPO_ID = "lerobot/libero"
REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"  # Week 2's pinned revision
OBJECT_SENTENCE = "pick up the {} and place it in the basket"
ITEMS = [
    "alphabet soup", "cream cheese", "salad dressing", "bbq sauce", "ketchup",
    "tomato sauce", "butter", "milk", "chocolate pudding", "orange juice",
]


def first_demo_per_item() -> dict[str, int]:
    # Episode index of the first LIBERO-Object demo of each item, found as in
    # week-2/data/make_subsets.py: task of each episode from the small data
    # parquet files (no videos).
    root = Path(snapshot_download(
        REPO_ID, repo_type="dataset", revision=REVISION,
        allow_patterns=["meta/tasks.parquet", "data/*/*.parquet"],
    ))
    tasks = {r["task_index"]: r["__index_level_0__"] for r in pq.read_table(root / "meta/tasks.parquet").to_pylist()}
    first = {}
    for path in sorted(root.glob("data/*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        for episode, task in zip(table.column("episode_index").to_pylist(), table.column("task_index").to_pylist()):
            for item in ITEMS:
                if tasks[task] == OBJECT_SENTENCE.format(item):
                    first[item] = min(first.get(item, episode), episode)
    missing = set(ITEMS) - set(first)
    if missing:
        raise SystemExit(f"No LIBERO-Object demo found for {missing}")
    return first


def to_uint8_hwc(image) -> np.ndarray:
    return (image.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chains", type=Path, nargs="+", required=True, help="Merged scripted chain data sets")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    first = first_demo_per_item()
    print("original demos:", first)
    source = LeRobotDataset(REPO_ID, revision=REVISION, episodes=sorted(first.values()))
    originals_root = args.out.parent / f"{args.out.name}-originals"
    if originals_root.exists():
        shutil.rmtree(originals_root)
    originals = LeRobotDataset.create(
        repo_id=f"local/{originals_root.name}", fps=FPS, features=FEATURES, root=originals_root, robot_type="panda"
    )
    current = None
    for i in range(len(source)):
        frame = source[i]
        episode = int(frame["episode_index"])
        if current is not None and episode != current:
            originals.save_episode()
        current = episode
        originals.add_frame({
            "observation.images.image": to_uint8_hwc(frame["observation.images.image"]),
            "observation.images.image2": to_uint8_hwc(frame["observation.images.image2"]),
            "observation.state": frame["observation.state"].numpy().astype(np.float32),
            "action": frame["action"].numpy().astype(np.float32),
            "task": frame["task"],
        })
    originals.save_episode()
    originals.finalize()

    roots = [*args.chains, originals_root]
    aggregate_datasets(
        repo_ids=[f"local/{root.name}" for root in roots], roots=roots,
        aggr_repo_id=f"local/{args.out.name}", aggr_root=args.out,
    )
    merged = LeRobotDataset(f"local/{args.out.name}", root=args.out)
    print(f"{args.out}: {merged.num_episodes} episodes, {merged.num_frames} frames, {len(merged.meta.tasks)} tasks")


if __name__ == "__main__":
    main()
