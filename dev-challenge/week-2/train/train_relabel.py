"""`lerobot-train` with Week 2 instruction relabeling.

Takes exactly the same arguments as `lerobot-train`. Before each batch reaches
the policy, every sample's instruction is replaced by one of five wordings for
its item, chosen uniformly at random: the original LIBERO instruction or one
of the taught wordings A to D in data/wordings.json. The unseen wordings E to
H are never used here.

The replacement happens in the main process (on the batch returned by the
data loader), so it works with any number of data loader workers.

Example:
    python train/train_relabel.py --policy.path=lerobot/smolvla_libero ... --seed=1000
"""

import json
import random
import sys
from pathlib import Path

import lerobot.scripts.lerobot_train as lerobot_train

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WORDINGS = json.loads((DATA_DIR / "wordings.json").read_text())
ITEMS = list(json.loads((DATA_DIR / "subsets.json").read_text())["items"])
TEMPLATES = [WORDINGS["original"], *WORDINGS["taught"].values()]
# Dataset instruction -> the five wordings for that item.
CHOICES = {
    WORDINGS["original"].format(item=item): [t.format(item=item) for t in TEMPLATES] for item in ITEMS
}


def training_seed() -> int:
    # Follow the run's --seed so that extra seeds also change the wording draws.
    for arg in sys.argv[1:]:
        if arg.startswith("--seed="):
            return int(arg.split("=", 1)[1])
    return 1000


rng = random.Random(training_seed())
original_cycle = lerobot_train.cycle
logged_batches = 0


def relabel(task: str) -> str:
    if task not in CHOICES:
        raise ValueError(f"Unexpected training instruction {task!r}; check --dataset.episodes.")
    return rng.choice(CHOICES[task])


def relabeling_cycle(iterable):
    global logged_batches
    for batch in original_cycle(iterable):
        batch["task"] = [relabel(task) for task in batch["task"]]
        if logged_batches < 2:
            # A quick check in the log that the relabeling is active.
            print(f"[relabel] batch {logged_batches} sample tasks: {batch['task'][:5]}", flush=True)
            logged_batches += 1
        yield batch


lerobot_train.cycle = relabeling_cycle

if __name__ == "__main__":
    lerobot_train.main()
