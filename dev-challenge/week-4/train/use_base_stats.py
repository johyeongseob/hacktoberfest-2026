"""Make a training set normalize state and action the way smolvla_libero does.

lerobot-train replaces the pretrained model's normalization statistics with
the training set's own (dataset.meta.stats). smolvla_libero was trained on
all four LIBERO suites; the scripted LIBERO-Object chains have very
different statistics (one held gripper orientation, actions that often hit
the limit), so after the swap the same network outputs mean different
actions.

This writes smolvla_libero's mean and std for observation.state and action
into the training set's meta/stats.json (the old file is kept as
meta/stats_own.json), so fine-tuning keeps the pretrained normalization.

Usage (on the pod, after train/prepare_data.py, before fine-tuning):
    python train/use_base_stats.py /workspace/outputs/week4/train-data/pairs30
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

KEYS = ["observation.state", "action"]  # SmolVLA normalizes both with MEAN_STD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="LeRobot dataset folder (with meta/stats.json)")
    parser.add_argument("--policy", default="lerobot/smolvla_libero")
    args = parser.parse_args()

    folder = Path(snapshot_download(args.policy, allow_patterns=["policy_preprocessor*"]))
    (normalizer,) = folder.glob("policy_preprocessor*normalizer*.safetensors")
    base = load_file(normalizer)

    stats_path = args.data / "meta/stats.json"
    backup = args.data / "meta/stats_own.json"
    if not backup.exists():
        shutil.copy(stats_path, backup)
    stats = json.loads(backup.read_text())
    for key in KEYS:
        for stat in ("mean", "std"):
            own = stats[key][stat]
            stats[key][stat] = base[f"{key}.{stat}"].tolist()
            print(f"{key}.{stat}: {[round(v, 3) for v in own]} -> {[round(v, 3) for v in stats[key][stat]]}")
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"wrote {stats_path} (own statistics kept in {backup.name})")


if __name__ == "__main__":
    main()
