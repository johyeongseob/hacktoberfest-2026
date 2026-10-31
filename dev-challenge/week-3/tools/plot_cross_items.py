"""Draw the "every item in every scene" test as a scene x item grid.

Each cell is one LIBERO-Object scene (row) asked for one item (column), from
eval/run_cross_items.sh results; the number is successes over episodes.
Items that are not in a scene are left blank, the scene's own item (the one
LIBERO-Object trains) has a dark outline, and items with no single-pick demo
in the training data (the scripted pick failed there) can be marked "no demo"
(--no-demo, for models trained on the single picks).
Several result folders give one panel each, side by side, to compare models.

Runs on the laptop CPU (needs matplotlib):
    python tools/plot_cross_items.py outputs/pod1/cross-items --labels "smolvla_libero" --out results/cross_items.png
    python tools/plot_cross_items.py outputs/val-cross/base outputs/val-cross/009000 \
        --labels "before" "after" --no-demo 1 --singles outputs/steady/singles8 --out results/cross_items_before_after.png
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


def load(folder: Path) -> dict:
    cells = {}
    for path in folder.glob("scene*/results.jsonl"):
        for line in open(path):
            r = json.loads(line)
            ok, n = cells.get((r["task_id"], r["items"][0]), (0, 0))
            cells[(r["task_id"], r["items"][0])] = (ok + r["success"], n + 1)
    return cells


def missing_singles(folder: Path) -> set:
    # Scene items with no successful single pick in a recording folder
    # (record_scripted.py's scene*/chains.jsonl): the training set has no
    # single-pick demo of them.
    recorded = set()
    for path in folder.glob("scene*/chains.jsonl"):
        for line in open(path):
            r = json.loads(line)
            if len(r["items"]) == 1 and r["success"]:
                recorded.add((r["task_id"], r["items"][0]))
    return {(scene, item) for scene, items in SCENE_ITEMS.items() for item in items} - recorded


def totals(cells: dict) -> str:
    own = [v for (scene, item), v in cells.items() if ITEMS[scene] == item]
    other = [v for (scene, item), v in cells.items() if ITEMS[scene] != item]
    s = lambda vs: f"{sum(v[0] for v in vs)}/{sum(v[1] for v in vs)}"
    return f"own item {s(own)}, other items {s(other)}"


def panel(ax, cells: dict, label: str, no_demo: set) -> None:
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
            if (scene, item) in no_demo:
                ax.text(col + 0.5, scene + 0.8, "no demo", ha="center", va="center", fontsize=7.5,
                        color="white" if rate > 0.55 else MUTED)
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
    parser.add_argument("folders", type=Path, nargs="+", help="run_cross_items.sh output folders")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--no-demo", nargs="*", type=int, default=[],
                        help="Panel numbers (0-based) to mark the cells without a single-pick demo")
    parser.add_argument("--singles", type=Path, default=None,
                        help="The single-pick recording folder the --no-demo panels were trained on")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [folder.name for folder in args.folders]
    if len(labels) != len(args.folders):
        raise SystemExit("Give one label per folder")
    if args.no_demo and args.singles is None:
        raise SystemExit("--no-demo needs --singles")
    missing = missing_singles(args.singles) if args.singles else set()

    fig, axes = plt.subplots(1, len(args.folders), figsize=(7.2 * len(args.folders), 7.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, (ax, folder, label) in enumerate(zip(axes[0], args.folders, labels)):
        cells = load(folder)
        panel(ax, cells, label, missing if i in args.no_demo else set())
        print(f"{label}: {totals(cells)}")
    fig.text(0.01, 0.01, "Outlined: the scene's own LIBERO-Object item. Blank: item not in the scene.",
             fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
