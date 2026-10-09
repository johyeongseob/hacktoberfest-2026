"""Look inside the `lerobot/libero` training dataset without downloading all of it.

Downloads the metadata and the tabular data (states, actions, task indices;
about 20 MB), summarizes the LIBERO-Object tasks, then downloads only the
video files that hold one ketchup episode (one file per camera, about 75 MB)
and saves sample frames.

    python inspect_dataset.py
    python inspect_dataset.py --item butter --nth 3
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from lerobot.utils.io_utils import write_video
from PIL import Image

REPO_ID = "lerobot/libero"
CAMERAS = ["observation.images.image", "observation.images.image2"]


def download(path: str) -> Path:
    return Path(hf_hub_download(REPO_ID, path, repo_type="dataset"))


def load_tasks() -> dict[int, str]:
    tasks = pd.read_parquet(download("meta/tasks.parquet"))
    # v3.0 stores the task text as the index and task_index as a column.
    if "task_index" in tasks.columns:
        return {int(row.task_index): str(text) for text, row in tasks.iterrows()}
    return {int(index): str(row.task) for index, row in tasks.iterrows()}


def read_frames(video: Path, start_s: float, end_s: float) -> list[np.ndarray]:
    frames = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        container.seek(int(start_s / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start_s - 1e-3:
                continue
            if t >= end_s - 1e-3:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", default="ketchup", help="LIBERO-Object item name")
    parser.add_argument("--nth", type=int, default=0, help="which episode of that item to inspect")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset-inspect"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads(download("meta/info.json").read_text())
    print("== Dataset")
    for key in ["codebase_version", "robot_type", "total_episodes", "total_frames", "total_tasks", "fps"]:
        print(f"{key}: {info[key]}")
    print("features:")
    for name, feature in info["features"].items():
        print(f"  {name}: shape={feature['shape']} dtype={feature['dtype']}")

    tasks = load_tasks()
    episodes = pd.read_parquet(download("meta/episodes/chunk-000/file-000.parquet"))
    # The episode metadata has no task column, so read task_index from the frame data.
    data_root = Path(snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=["data/*"]))
    data = pd.concat(pd.read_parquet(path) for path in sorted(data_root.glob("data/*/*.parquet")))
    episode_tasks = data.groupby("episode_index")["task_index"].first()
    episodes["task"] = episodes["episode_index"].map(episode_tasks).map(tasks)

    print("\n== LIBERO-Object tasks in the dataset")
    object_tasks = sorted(t for t in tasks.values() if t.startswith("pick up the") and t.endswith("basket"))
    for task in object_tasks:
        subset = episodes[episodes["task"] == task]
        print(f"{len(subset):3d} episodes, mean {subset['length'].mean():5.1f} frames  {task}")
    object_episodes = episodes[episodes["task"].isin(object_tasks)]
    print(f"Total: {len(object_episodes)} episodes, {int(object_episodes['length'].sum())} frames")

    task = next(t for t in object_tasks if f"the {args.item} and" in t)
    episode = episodes[episodes["task"] == task].iloc[args.nth]
    index = int(episode["episode_index"])
    print(f"\n== Episode {index}: {task}")
    print(f"length: {int(episode['length'])} frames ({int(episode['length']) / info['fps']:.1f} s at {info['fps']} fps)")

    frames = data[data["episode_index"] == index].sort_values("frame_index")
    states = np.stack(frames["observation.state"].to_numpy())
    actions = np.stack(frames["action"].to_numpy())
    np.set_printoptions(precision=3, suppress=True)
    print("state  (eef pos x3, eef axis-angle x3, gripper x2), first 3 frames:")
    print(states[:3])
    print("action (eef delta x6, gripper x1), first 3 frames:")
    print(actions[:3])
    print("action min:", actions.min(axis=0))
    print("action max:", actions.max(axis=0))
    gripper_changes = np.flatnonzero(np.diff(np.sign(actions[:, 6]))) + 1
    print("gripper command flips at frames:", gripper_changes.tolist())

    summary = {"episode_index": index, "task": task, "length": int(episode["length"]), "fps": info["fps"]}
    for camera in CAMERAS:
        video_path = info["video_path"].format(
            video_key=camera,
            chunk_index=int(episode[f"videos/{camera}/chunk_index"]),
            file_index=int(episode[f"videos/{camera}/file_index"]),
        )
        start = float(episode[f"videos/{camera}/from_timestamp"])
        end = float(episode[f"videos/{camera}/to_timestamp"])
        images = read_frames(download(video_path), start, end)
        name = camera.split(".")[-1]
        for position, frame in zip(["first", "middle", "last"], [0, len(images) // 2, len(images) - 1]):
            Image.fromarray(images[frame]).save(args.output_dir / f"ep{index}_{name}_{position}.png")
        write_video(args.output_dir / f"ep{index}_{name}.mp4", images, fps=int(info["fps"]))
        summary[f"{name}_frames"] = len(images)

    (args.output_dir / f"ep{index}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved frames and videos to {args.output_dir}")


if __name__ == "__main__":
    main()
