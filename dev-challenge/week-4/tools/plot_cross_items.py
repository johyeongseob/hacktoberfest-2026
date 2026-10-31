"""Draw the "every item in every scene" test as a scene x item grid.

Each cell is one LIBERO-Object scene (row) asked for one item (column), from
a results file of eval/run_cross_items.sh (results/<model>/single.jsonl); the
number is successes over episodes. Items that are not in a scene are left
blank, and the scene's own item (the one LIBERO-Object trains) has a dark
outline. Several files give one panel each, side by side, to compare models.

Runs on the laptop CPU (needs matplotlib), from dev-challenge/week-4:
    python tools/plot_cross_items.py results/before-fine-tuning/single.jsonl results/fine-tuned/single.jsonl         --labels "before fine-tuning" "fine-tuned (no mark)" --out assets/cross_items_before_after.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from make_chain_plan import SCENE_ITEMS  # noqa: E402

# LIBERO-Object task i trains item ITEMS[i] in scene i.
ITEMS = [
    "alphabet soup", "cream cheese", "salad dressing", "bbq sauce", "ketchup",
    "tomato sauce", "butter", "milk", "chocolate pudding", "orange juice",
]
# One-hue sequential ramp (blue, light to dark) and the page inks.
RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SURFACE, INK, MUTED, BLANK = "#fcfcfb", "#0b0b0b", "#52514e", "#f0efec"


def load(path: Path) -> dict:
    cells = {}
    for line in open(path):
        r = json.loads(line)
        ok, n = cells.get((r["task_id"], r["items"][0]), (0, 0))
        cells[(r["task_id"], r["items"][0])] = (ok + r["success"], n + 1)
    return cells


def totals(cells: dict) -> str:
    own = [v for (scene, item), v in cells.items() if ITEMS[scene] == item]
    other = [v for (scene, item), v in cells.items() if ITEMS[scene] != item]
    s = lambda vs: f"{sum(v[0] for v in vs)}/{sum(v[1] for v in vs)}"
    return f"own item {s(own)}, other items {s(other)}"


def panel(ax, cells: dict, label: str) -> None:
    ax.set_facecolor(SURFACE)
    for scene in range(10):
        for col, item in enumerate(ITEMS):
            if item not in SCENE_ITEMS[scene]:
                ax.add_patch(plt.Rectangle((col, scene), 1, 1, color=BLANK, lw=0))
                continue
            ok, n = cells.get((scene, item), (0, 0))
            rate = ok / n if n else 0
            # 2px surface gap between cells: inset each fill a little.
            ax.add_patch(plt.Rectangle((col + 0.04, scene + 0.04), 0.92, 0.92, color=RAMP(rate), lw=0))
            if ITEMS[scene] == item:
                ax.add_patch(plt.Rectangle((col + 0.04, scene + 0.04), 0.92, 0.92, fill=False, ec=INK, lw=2))
            text = f"{ok}/{n}" if n else "-"
            ax.text(col + 0.5, scene + 0.45, text, ha="center", va="center", fontsize=12,
                    color="white" if rate > 0.55 else INK)
    ax.set_xlim(0, 10)
    ax.set_ylim(10, 0)
    ax.set_xticks([c + 0.5 for c in range(10)], ITEMS, rotation=45, ha="right", fontsize=12, color=MUTED)
    ax.set_yticks([s + 0.5 for s in range(10)], [f"scene {s}" for s in range(10)], fontsize=12, color=MUTED)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{label}\n{totals(cells)}", fontsize=14, color=INK, loc="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", type=Path, nargs="+", help="single.jsonl results files")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [path.parent.name for path in args.files]
    if len(labels) != len(args.files):
        raise SystemExit("Give one label per file")

    fig, axes = plt.subplots(1, len(args.files), figsize=(7.2 * len(args.files), 7.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, path, label in zip(axes[0], args.files, labels):
        cells = load(path)
        panel(ax, cells, label)
        print(f"{label}: {totals(cells)}")
    fig.text(0.01, 0.01, "Outlined: the scene's own LIBERO-Object item. Blank: item not in the scene.",
             fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
