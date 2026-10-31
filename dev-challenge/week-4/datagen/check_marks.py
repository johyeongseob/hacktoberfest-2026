"""Check that the item boxes of datagen/mark.py sit on the items.

For each scene, opens one initial state, lets the objects settle, and draws
every item's box on both cameras, one row per item. The images are turned
180 degrees, as the recorder stores them and as the model sees them. Each
box is drawn four times, once for each way the projection could map onto
the raw renders, so one look shows which way is right:
    green: rows flipped (mark.py's default)   red: rows and columns flipped
    blue: no flip                             yellow: columns flipped
Writes OUT/scene<N>.png.

Usage (on the pod, from dev-challenge/week-4):
    python datagen/check_marks.py --scenes 3 9 --out /workspace/outputs/week4/check-marks
"""

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from mark import CAMERAS, draw_box, item_box
from PIL import Image, ImageDraw
from scripted_pick import ITEMS, object_name

SIZE = 256
# (flip_rows, flip_cols, color)
CANDIDATES = [
    (True, False, (0, 255, 0)),
    (True, True, (255, 0, 0)),
    (False, False, (0, 128, 255)),
    (False, True, (255, 255, 0)),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=int, nargs="+", default=[3, 9])
    parser.add_argument("--init-state", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    for scene in args.scenes:
        task = suite.get_task(scene)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=SIZE, camera_widths=SIZE)
        env.seed(0)
        env.reset()
        env.set_init_state(suite.get_task_init_states(scene)[args.init_state])
        for _ in range(10):  # let objects settle, as the recorder does
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        sim = env.env.sim
        items = [item for item in ITEMS if object_name(item) in env.env.object_states_dict]

        rows = []
        for item in items:
            tiles = []
            for key, camera in CAMERAS.items():
                image = obs[key]
                for flip_rows, flip_cols, color in CANDIDATES:
                    box = item_box(sim, object_name(item), camera, SIZE, SIZE, flip_rows, flip_cols)
                    image = draw_box(image, box, color=color, width=1)
                tiles.append(np.ascontiguousarray(image[::-1, ::-1]))  # as stored and as the model sees it
            row = Image.fromarray(np.concatenate(tiles, axis=1))
            ImageDraw.Draw(row).text((4, 4), f"scene {scene}: {item}", fill=(255, 255, 255))
            rows.append(np.asarray(row))
        path = args.out / f"scene{scene}.png"
        imageio.imwrite(path, np.concatenate(rows, axis=0))
        print(f"wrote {path} ({len(items)} items; left: agent camera, right: wrist camera)")
        env.close()


if __name__ == "__main__":
    main()
