"""Tile Week 2 test videos into a grid: the same scene across models (N).

Adapted from the Week 1 tools/make_grid.py: tiles play at the 20 Hz control
rate, start together, and shorter episodes freeze on their last frame until
the longest one ends. Each tile shows N and the outcome (success or fail) at
the top left and the item at the bottom. A white band under each wording's
rows gives the wording, centered.

    python tools/make_grid.py items     # wordings C and F, N = 0 and 1, the four trained items
    python tools/make_grid.py trend --item "orange juice" --wording E --init 8 --ns 1 10 20 40
    python tools/make_grid.py forget --item butter --other ketchup --init 6
    python tools/make_grid.py pair --item ketchup --wording original --init 6 --ns 0 1

Writes outputs/grids/<name>.mp4 and assets/<name>.webp (for the docs).
"""

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # the week-2 folder
EVAL = HERE / "outputs" / "eval"
CONTROL_FPS = 20
LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
LABEL_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
LABEL_SIZE = 13
MONO_ADVANCE = 0.602  # DejaVu Sans Mono glyph width as a fraction of the font size
BAND, BAND_SIZE = 32, 16  # white band naming the wording: height and font size
SEPARATOR = 6  # dark line between the taught and the unseen pairs of rows
BAND_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BAND_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TRAINED_ITEMS = ["alphabet soup", "cream cheese", "butter", "orange juice"]
# Grid "items": the hardest taught wording (C) and the hardest unseen wording (F),
# before (N = 0) and after one demo per item (N = 1). Initial states where N = 0
# fails and every N >= 1 succeeds, from results.jsonl: C has one for all four items
# (7); F has none in common, so each item uses its own.
ITEMS_GRID_INITS = {"C": [7, 7, 7, 7], "F": [8, 0, 5, 6]}


def wording_line(wording: str, item: str = "{item}") -> str:
    """'taught: "…"' or 'unseen: "…"' (or 'original: "…"'), the sentence from data/wordings.json; {item} stays as written unless an item is given."""
    texts = json.loads((HERE / "data" / "wordings.json").read_text())
    group = next((g for g in ("taught", "unseen") if wording in texts[g]), None)
    sentence = (texts[group][wording] if group else texts["original"]).replace("{item}", item)
    return f"{group or 'original'}: “{sentence}”"


def run_name(n: int) -> str:
    return "n0" if n == 0 else f"n{n}-s1000"


def episode(n: int, item: str, wording: str, init: int) -> dict:
    lines = (EVAL / f"{run_name(n)}-test" / "results.jsonl").read_text().splitlines()
    row = next(r for r in map(json.loads, lines)
               if r["item"] == item and r["wording"] == wording and r["init_state"] == init)
    # results.jsonl holds the pod path (/workspace/outputs/week2/eval/...); map it to the local copy.
    row["local_video"] = EVAL / row["video"].split("/eval/", 1)[1]
    return row


def duration_s(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("layout", choices=["items", "trend", "forget", "pair"])
    parser.add_argument("--item", help="trend, pair: which item; forget: the trained item")
    parser.add_argument("--other", help="forget: an item left out of fine-tuning")
    parser.add_argument("--wording", help="trend, pair: original or A-H")
    parser.add_argument("--init", type=int, help="trend, pair: initial state 0-9")
    parser.add_argument("--ns", type=int, nargs="+", default=[0, 1],
                        help="trend, pair: the N to show (pair: two; trend: up to four, one row)")
    args = parser.parse_args()

    if args.layout == "items":
        tiles = [(episode(n, item, w, inits[col]), f"N={n}")
                 for w, inits in ITEMS_GRID_INITS.items() for n in (0, 1)
                 for col, item in enumerate(TRAINED_ITEMS)]
        columns, tile, name = len(TRAINED_ITEMS), 256, "taught_C_unseen_F_n0_vs_n1"
        # Under each pair of rows (N = 0 above N = 1), a white band names the wording;
        # a thin grey gap separates the two rows of a pair, and a dark line under the
        # first band separates the two pairs (grey blends into the grey wall of the scene).
        row_gaps = [4, BAND + SEPARATOR, 4]
        bands = [(2 * k + 1, wording_line(w)) for k, w in enumerate(ITEMS_GRID_INITS)]
    elif args.layout == "trend":
        # One scene across several models in one row of 256 px tiles, as wide as the
        # four-column items grid when four N are given.
        tiles = [(episode(n, args.item, args.wording, args.init), f"N={n}") for n in args.ns]
        columns, tile, row_gaps = len(args.ns), 256, []
        bands = [(0, wording_line(args.wording, args.item))]
        name = f"{args.item.replace(' ', '_')}_{args.wording}_init{args.init}_n{'_'.join(map(str, args.ns))}"
    elif args.layout == "forget":
        # Original wording, N = 0 and 1, for a trained item and for an item left out of
        # fine-tuning, side by side in one row: only the left-out item is forgotten.
        tiles = [(episode(n, item, "original", args.init), f"N={n}")
                 for item in (args.item, args.other) for n in (0, 1)]
        columns, tile, row_gaps = 4, 256, []
        bands = [(0, wording_line("original"))]
        halves = ["trained item", "untrained item"]
        name = f"forgetting_{args.item.replace(' ', '_')}_vs_{args.other.replace(' ', '_')}_init{args.init}_n0_vs_n1"
    else:
        tiles = [(episode(n, args.item, args.wording, args.init), f"N={n}") for n in args.ns]
        columns, tile, row_gaps = 2, 360, []  # original size, for a closer look
        bands = [(0, wording_line(args.wording, args.item))]
        name = f"{args.item.replace(' ', '_')}_{args.wording}_init{args.init}_n{args.ns[0]}_vs_n{args.ns[1]}"

    length = max(duration_s(row["local_video"]) for row, _ in tiles)
    box = f"fontfile={LABEL_FONT}:fontsize={LABEL_SIZE}:box=1:boxcolor=black@0.5:boxborderw=3:y_align=font"
    filters, layout = [], []
    for i, (row, label) in enumerate(tiles):
        chain = (f"[{i}:v]setpts=PTS-STARTPTS,fps={CONTROL_FPS},scale={tile}:{tile},"
                 f"tpad=stop_mode=clone:stop_duration={length:.2f}")
        # Top left: N in white, then the outcome in bold green or light red on a darker
        # box (plain red on the dark floor was hard to read).
        outcome, color = ("success", "lime") if row["success"] else ("fail", "0xff6060")
        outcome_x = 6 + round(len(label) * LABEL_SIZE * MONO_ADVANCE) + 12
        chain += f",drawtext=text='{label}':x=6:y=6:fontcolor=white:{box}"
        chain += (f",drawtext=text='{outcome}':x={outcome_x}:y=6:fontcolor={color}:fontfile={LABEL_FONT_BOLD}"
                  f":fontsize={LABEL_SIZE}:box=1:boxcolor=black@0.8:boxborderw=3:y_align=font")
        # Bottom center: the item.
        chain += f",drawtext=text='{row['item']}':x=(w-tw)/2:y={tile - 8 - round(LABEL_SIZE * 1.3)}:fontcolor=white:{box}"
        filters.append(f"{chain}[v{i}]")
        r = i // columns
        layout.append(f"{(i % columns) * tile}_{r * tile + sum(row_gaps[:r])}")
    inputs = "".join(f"[v{i}]" for i in range(len(tiles)))
    # White bands with the wording, centered: one under the last row (added by pad),
    # the others in the gap left for them between rows. With halves, the last band gets
    # a first line naming the left and the right half, and the wording goes below it.
    halves = locals().get("halves", [])
    rows = len(tiles) // columns
    notes = f"pad=iw:ih+{BAND * (2 if halves else 1)}:0:0:color=white"
    if halves:
        # Dark vertical line between the halves, drawn over the tile edges so the grid
        # stays as wide as the others.
        notes += (f",drawbox=x={columns * tile // 2 - SEPARATOR // 2}:y=0:w={SEPARATOR}"
                  f":h={rows * tile + sum(row_gaps)}:color=0x303030:t=fill")
    for j, text in enumerate(halves):
        notes += (f",drawtext=text='{text}':fontfile={BAND_FONT_BOLD}:fontsize={BAND_SIZE}:fontcolor=black"
                  f":x={j}*w/{len(halves)}+(w/{len(halves)}-tw)/2:y={rows * tile + sum(row_gaps)}+({BAND}-th)/2")
    for after, text in bands:
        y = (after + 1) * tile + sum(row_gaps[:after]) + (BAND if halves and after == rows - 1 else 0)
        if after < rows - 1:
            notes += f",drawbox=x=0:y={y}:w=iw:h={BAND}:color=white:t=fill"
            notes += f",drawbox=x=0:y={y + BAND}:w=iw:h={SEPARATOR}:color=0x303030:t=fill"
        safe = text.replace(":", r"\:")  # ":" separates drawtext options
        notes += (f",drawtext=text='{safe}':fontfile={BAND_FONT}:fontsize={BAND_SIZE}:fontcolor=black"
                  f":x=(w-tw)/2:y={y}+({BAND}-th)/2")
    filters.append(f"{inputs}xstack=inputs={len(tiles)}:layout={'|'.join(layout)}:fill=gray,{notes}[out]")

    grid_dir, asset_dir = HERE / "outputs" / "grids", HERE / "assets"
    grid_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(exist_ok=True)
    mp4, webp = grid_dir / f"{name}.mp4", asset_dir / f"{name}.webp"
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for row, _ in tiles:
        command += ["-i", str(row["local_video"])]
    command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-t", f"{length:.2f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)]
    subprocess.run(command, check=True)
    # Animated WebP for the docs: loops, plays inline on GitHub and DEV.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-c:v", "libwebp_anim",
                    "-loop", "0", "-q:v", "70", str(webp)], check=True)
    successes = sum(row["success"] for row, _ in tiles)
    print(f"{mp4}\n{webp}  ({successes}/{len(tiles)} success, {length:.1f} s, "
          f"{webp.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
