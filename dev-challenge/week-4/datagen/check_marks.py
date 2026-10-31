"""Check that the item boxes of datagen/mark.py sit on the items.

For each scene, opens one initial state, lets the objects settle, and marks
each item in turn on both cameras, one row per item, as the recorder will:
one green box around the asked item. The images are turned 180 degrees, as
the recorder stores them and as the model sees them. Writes OUT/scene<N>.png.

Usage (on the pod, from dev-challenge/week-4):
    python datagen/check_marks.py --scenes 3 9 --out /workspace/outputs/week4/marks
"""

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from mark import CAMERAS, mark
from PIL import Image, ImageDraw
from scripted_pick import ITEMS, object_name

SIZE = 256


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
        items = [item for item in ITEMS if object_name(item) in env.env.object_states_dict]

        rows = []
        for item in items:
            marked = mark(env.env.sim, obs, object_name(item))
            # As stored and as the model sees them: turned 180 degrees.
            row = np.concatenate([np.ascontiguousarray(marked[key][::-1, ::-1]) for key in CAMERAS], axis=1)
            row = Image.fromarray(row)
            ImageDraw.Draw(row).text((4, 4), f"scene {scene}: {item}", fill=(255, 255, 255))
            rows.append(np.asarray(row))
        path = args.out / f"scene{scene}.png"
        imageio.imwrite(path, np.concatenate(rows, axis=0))
        print(f"wrote {path} ({len(items)} items; left: agent camera, right: wrist camera)")
        env.close()


if __name__ == "__main__":
    main()
