"""Fix which item orders are used for training and which only for testing.

For every LIBERO-Object scene, draws random orders of two, three, and four
of the scene's six items. Training orders become scripted demos; test
orders are held out: no test order appears in training, so the test shows
whether the fine-tuned robot handles orders it never practiced. Written once,
before any data is made, to data/chain_plan.json.

Usage (from dev-challenge/week-3):
    python data/make_chain_plan.py
    python data/make_chain_plan.py --singles 3 --out data/single_plan.json
"""

import argparse
import json
import random
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


def extend(args) -> None:
    # A larger training set that contains the smaller one: the same test
    # orders, the same training orders first, then new orders that appear in
    # neither, so the two training sets differ only in size.
    base = json.loads(args.extend.read_text())
    rng = random.Random(args.seed + 1)
    plan = {"seed": base["seed"], "extended_from": args.extend.name, "lengths": base["lengths"], "scenes": {}}
    for scene, items in SCENE_ITEMS.items():
        old = base["scenes"][str(scene)]
        train = [list(order) for order in old["train"]]
        seen = {tuple(order) for order in old["train"] + old["test"]}
        for length in base["lengths"]:
            while sum(len(c) == length for c in train) < args.train:
                order = tuple(rng.sample(items, length))
                if order not in seen:
                    seen.add(order)
                    train.append(list(order))
        plan["scenes"][str(scene)] = {"train": train, "test": old["test"]}
    args.out.write_text(json.dumps(plan, indent=1) + "\n")
    n = sum(len(e["train"]) for e in plan["scenes"].values())
    print(f"{args.out}: {n} training orders (extends {args.extend.name}), test orders unchanged")


def singles(args) -> None:
    # Every item of every scene on its own, --singles times: the chains come
    # from fixed scenes, so a scene's first item could be guessed from the
    # table alone; single picks of all six items make the instruction matter.
    # Copies of an item get different initial states in record_scripted.py.
    plan = {"singles": args.singles, "lengths": [1], "scenes": {}}
    for scene, items in SCENE_ITEMS.items():
        plan["scenes"][str(scene)] = {"train": [[item] for item in items for _ in range(args.singles)], "test": []}
    args.out.write_text(json.dumps(plan, indent=1) + "\n")
    print(f"{args.out}: {args.singles} single picks of each of the 60 scene items")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, default=4, help="Training orders per scene and length")
    parser.add_argument("--test", type=int, default=2, help="Held-out test orders per scene and length")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("chain_plan.json"))
    parser.add_argument(
        "--extend", type=Path, default=None,
        help="Keep this plan's training and test orders and add training orders up to --train per length",
    )
    parser.add_argument(
        "--singles", type=int, default=None,
        help="Instead: single picks of every scene item, this many times each (data/single_plan.json)",
    )
    args = parser.parse_args()

    if args.singles is not None:
        singles(args)
        return
    if args.extend is not None:
        extend(args)
        return
    rng = random.Random(args.seed)
    plan = {"seed": args.seed, "lengths": args.lengths, "scenes": {}}
    for scene, items in SCENE_ITEMS.items():
        entry = {"train": [], "test": []}
        for length in args.lengths:
            seen = set()
            for split, count in (("train", args.train), ("test", args.test)):
                while sum(len(c) == length for c in entry[split]) < count:
                    order = tuple(rng.sample(items, length))
                    if order not in seen:  # every order is used once, in one split
                        seen.add(order)
                        entry[split].append(list(order))
        plan["scenes"][str(scene)] = entry
    args.out.write_text(json.dumps(plan, indent=1) + "\n")
    n = {split: sum(len(e[split]) for e in plan["scenes"].values()) for split in ("train", "test")}
    print(f"{args.out}: {n['train']} training orders, {n['test']} test orders")


if __name__ == "__main__":
    main()
