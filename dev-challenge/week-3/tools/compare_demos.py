"""Compare the scripted chain demos with the human LIBERO-Object demos.

Reads only the state/action parquet files (no videos) of lerobot/libero (Week 2's
pinned revision) and of the scripted chain data set on the Hub, and prints,
per data set: steps per picked item, action size, how often an action hits the
limit, rotation actions, and gripper switching. A policy imitates these
numbers, so large gaps point at what the fine-tune learned.

Usage (from dev-challenge/week-3, about 35 MB of parquet files):
    python tools/compare_demos.py
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

HUMAN = ("lerobot/libero", "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
SCRIPTED = ("johyeongseob725/libero-object-scripted-chains", None)


def load(repo, revision, prefix=""):
    root = Path(snapshot_download(
        repo, repo_type="dataset", revision=revision,
        allow_patterns=[f"{prefix}meta/tasks.parquet", f"{prefix}data/*/*.parquet"],
    ))
    # The task text is the parquet index; its column name depends on the writer.
    tasks = {
        r["task_index"]: r.get("__index_level_0__", r.get("task"))
        for r in pq.read_table(root / prefix / "meta/tasks.parquet").to_pylist()
    }
    tables = [pq.read_table(p, columns=["episode_index", "task_index", "action"]) for p in sorted((root / prefix).glob("data/*/*.parquet"))]
    episodes = {}
    for t in tables:
        ep, task, act = t.column("episode_index").to_pylist(), t.column("task_index").to_pylist(), t.column("action").to_pylist()
        for e, k, a in zip(ep, task, act):
            episodes.setdefault(e, {"tasks": [], "actions": []})
            episodes[e]["tasks"].append(tasks[k])
            episodes[e]["actions"].append(a)
    return episodes


def summary(name, episodes):
    actions = np.concatenate([np.asarray(e["actions"]) for e in episodes.values()])
    # One "pick" = a run of frames with the same instruction.
    pick_lengths, closes = [], []
    for e in episodes.values():
        runs, start = [], 0
        for i in range(1, len(e["tasks"]) + 1):
            if i == len(e["tasks"]) or e["tasks"][i] != e["tasks"][start]:
                runs.append(i - start)
                start = i
        pick_lengths += runs
        g = np.asarray(e["actions"])[:, 6]
        closes.append(int(np.sum((g[1:] > 0) & (g[:-1] <= 0))))
    xyz = np.linalg.norm(actions[:, :3], axis=1)
    rot = np.linalg.norm(actions[:, 3:6], axis=1)
    print(f"\n== {name}: {len(episodes)} episodes, {len(actions)} frames, {len(pick_lengths)} picks")
    print(f"steps per pick: median {np.median(pick_lengths):.0f}, 10-90% {np.percentile(pick_lengths, 10):.0f}-{np.percentile(pick_lengths, 90):.0f}")
    print(f"|xyz action|: mean {xyz.mean():.2f}, median {np.median(xyz):.2f}; frames with |xyz| < 0.05: {np.mean(xyz < 0.05):.0%}")
    print(f"frames with any x/y/z at the limit (|a| >= 0.93): {np.mean(np.any(np.abs(actions[:, :3]) >= 0.93, axis=1)):.0%}")
    print(f"|rotation action|: mean {rot.mean():.3f}, frames > 0.05: {np.mean(rot > 0.05):.0%}")
    print(f"gripper closed (a > 0): {np.mean(actions[:, 6] > 0):.0%} of frames; closes per episode: {np.mean(closes):.1f}")
    for d, label in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
        print(f"  {label}: mean {actions[:, d].mean():+.3f}  std {actions[:, d].std():.3f}")


def main():
    human = load(*HUMAN)
    object_episodes = {k: v for k, v in human.items() if v["tasks"][0].startswith("pick up the") and v["tasks"][0].endswith("in the basket")}
    summary("human LIBERO-Object demos", object_episodes)
    for batch in ("batch0", "batch1"):
        summary(f"scripted chains {batch}", load(*SCRIPTED, prefix=f"{batch}/"))


if __name__ == "__main__":
    main()
