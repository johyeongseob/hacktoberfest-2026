"""Make a training set normalize state and action the way smolvla_libero does.

lerobot-train replaces the pretrained model's normalization statistics with
the training set's own (dataset.meta.stats). smolvla_libero was trained on
all four LIBERO suites, and the scripted LIBERO-Object chains have very
different statistics (one held gripper orientation, actions that often hit
the limit), so after the swap the same network outputs mean different actions.

This writes smolvla_libero's mean and std for observation.state and action
into the training set's meta/stats.json (the old file is kept as
meta/stats_own.json), so fine-tuning keeps the pretrained normalization.

I first suspected the swap caused the drop on each scene's own item
(10/10 -> 0/10 at step 1,000). An A/B run with and without this script ruled
that out: both fell to about 2/10 by step 3,000. See
../results/stats-ablation/README.md. The script is kept for that comparison.

Usage (on the pod, after train/prepare_data.py, before train/run_train.sh):
    python train/use_base_stats.py /workspace/outputs/week3/train-data/fast-singles
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
            new = base[f"{key}.{stat}"].tolist()
            if len(new) != len(own):
                raise ValueError(f"{key}.{stat}: policy has {len(new)} dims, dataset has {len(own)}")
            stats[key][stat] = new
            print(f"{key}.{stat}: {[round(v, 3) for v in own]} -> {[round(v, 3) for v in stats[key][stat]]}")
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"wrote {stats_path} (own statistics kept in {backup.name})")


if __name__ == "__main__":
    main()
