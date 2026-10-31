"""Fix which items and item orders are used for training and which only for testing.

Per LIBERO-Object scene, with its six items A to F in SCENE_ITEMS order:
- train: every item on its own (6) and the 24 ordered pairs outside the
  cycle A->B, B->C, ..., F->A
- test: that cycle, six orders the training set never shows
- test_seen: the reverse cycle B->A, C->B, ..., A->F, six trained orders
  tested as a control
In both tests every item is asked for first once and second once. Written
once, before any data is made, to data/pair_plan.json.

Usage (from dev-challenge/week-4):
    python data/make_chain_plan.py
"""

import argparse
import json
from pathlib import Path

# Items in each LIBERO-Object scene (from the scene BDDL files).
SCENE_ITEMS = {
    0: ["alphabet soup", "butter", "cream cheese", "milk", "salad dressing", "tomato sauce"],
    1: ["alphabet soup", "butter", "cream cheese", "milk", "orange juice", "tomato sauce"],
    2: ["alphabet soup", "cream cheese", "ketchup", "milk", "salad dressing", "tomato sauce"],
    3: ["alphabet soup", "bbq sauce", "chocolate pudding", "cream cheese", "ketchup", "salad dressing"],
    4: ["alphabet soup", "bbq sauce", "cream cheese", "ketchup", "milk", "salad dressing"],
    5: ["bbq sauce", "butter", "chocolate pudding", "milk", "orange juice", "tomato sauce"],
    6: ["bbq sauce", "butter", "chocolate pudding", "ketchup", "orange juice", "tomato sauce"],
    7: ["butter", "chocolate pudding", "cream cheese", "milk", "orange juice", "tomato sauce"],
    8: ["alphabet soup", "bbq sauce", "chocolate pudding", "ketchup", "orange juice", "salad dressing"],
    9: ["bbq sauce", "butter", "chocolate pudding", "ketchup", "orange juice", "salad dressing"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("pair_plan.json"))
    args = parser.parse_args()

    plan = {"pairs": True, "lengths": [1, 2], "scenes": {}}
    for scene, items in SCENE_ITEMS.items():
        cycle = [[items[i], items[(i + 1) % len(items)]] for i in range(len(items))]
        reverse = [[b, a] for a, b in cycle]
        ordered = [[a, b] for a in items for b in items if a != b]
        plan["scenes"][str(scene)] = {
            "train": [[item] for item in items] + [pair for pair in ordered if pair not in cycle],
            "test": cycle,
            "test_seen": reverse,
        }
    args.out.write_text(json.dumps(plan, indent=1) + "\n")
    n = {split: sum(len(e[split]) for e in plan["scenes"].values()) for split in ("train", "test", "test_seen")}
    print(f"{args.out}: {n['train']} training chains (60 single items, 240 pairs), "
          f"{n['test']} held-out test pairs, {n['test_seen']} trained test pairs")


if __name__ == "__main__":
    main()
