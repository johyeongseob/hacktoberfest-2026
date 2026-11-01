"""Check a frozen detector on the LIBERO items, without moving the robot.

For every LIBERO-Object scene at one initial state (default 0, the test
layout), ask the detector (--detector, datagen/detect.py) for each of the scene's six items by name on both cameras
and keep its top box. The simulator's exact box (datagen/mark.py, the box the
training data and --mark gt draw) is the answer:
- iou: overlap of the top box with the asked item's exact box
- hit: iou >= --hit-iou
- boxed: the item on the table whose exact box the top box overlaps most,
  so a miss shows which item the detector took for the asked one
- score: the detector's confidence in the top box (Florence-2 gives none);
  ms: one detection, on the GPU

The detectors are the ones --mark detector uses. Ten scenes x six items x
two cameras = 120 questions; the summary counts the hits over all 120.

Writes OUT/results.jsonl (one line per item and camera), prints a summary,
and saves one picture per camera (OUT/agentview.jpg, OUT/robot0_eye_in_hand.jpg):
every scene (rows) and item (columns), exact box green, detector box red, with
the item and the iou on top.

Example (from dev-challenge/week-4):
    python eval/check_detector.py --detector owlv2 --out /workspace/outputs/week4/eval/detector-check/owlv2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datagen"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from detect import DETECTORS, iou, make_detector  # noqa: E402
from make_chain_plan import SCENE_ITEMS  # noqa: E402
from mark import draw_box, item_box  # noqa: E402

SUITE = "libero_object"
# LeRobot's LIBERO observation keys -> MuJoCo cameras (as in eval/chain_rollout.py).
CAMERAS = {"image": "agentview", "image2": "robot0_eye_in_hand"}
RED = (255, 0, 0)


def object_name(item: str) -> str:
    return item.replace(" ", "_") + "_1"


def tile(raw: np.ndarray, exact, found, label: str) -> Image.Image:
    image = draw_box(draw_box(raw, exact), found, color=RED)[::-1, ::-1]
    picture = Image.fromarray(np.ascontiguousarray(image))
    ImageDraw.Draw(picture).text((3, 2), label, fill=(255, 255, 255))
    return picture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=DETECTORS, default="owlv2")
    parser.add_argument("--scenes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--query", default="{item}", help="Text asked for, with {item} as the item name")
    parser.add_argument("--hit-iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    detector = make_detector(args.detector, args.device)
    rows = []
    tiles = {}
    for scene in args.scenes:
        env_cfg = LiberoEnvConfig(task=SUITE, task_ids=[scene])
        vec_env = make_env(env_cfg, n_envs=1, use_async_envs=False)[SUITE][scene]
        env = vec_env.envs[0]
        libero_env = env.unwrapped
        libero_env.init_state_id = args.init_state
        observation, _ = env.reset(seed=1000 + args.init_state, options={NEW_ROLLOUT_OPTION: True})
        sim = libero_env._env.env.sim
        items = SCENE_ITEMS[scene]
        for key, camera in CAMERAS.items():
            raw = observation["pixels"][key]
            height, width = raw.shape[:2]
            exact = {item: item_box(sim, object_name(item), camera, height, width) for item in items}
            for item in items:
                found, score, ms = detector.box(raw, args.query.format(item=item))
                overlaps = {other: iou(found, box) for other, box in exact.items()}
                boxed = max(overlaps, key=overlaps.get) if max(overlaps.values()) > 0 else None
                row = {
                    "detector": args.detector, "init_state": args.init_state,
                    "scene": scene, "camera": camera, "item": item, "query": args.query.format(item=item),
                    "in_view": exact[item] is not None, "exact": exact[item], "found": found,
                    "iou": round(overlaps[item], 3), "hit": overlaps[item] >= args.hit_iou,
                    "boxed": boxed, "score": round(score, 3), "ms": round(ms, 1),
                }
                rows.append(row)
                tiles[(camera, scene, item)] = tile(raw, exact[item], found, f"{item} {row['iou']:.2f}")
        vec_env.close()
        print(f"scene {scene} done", flush=True)

    with open(args.out / "results.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    for camera in CAMERAS.values():
        seen = [r for r in rows if r["camera"] == camera and r["in_view"]]
        out_of_view = sum(1 for r in rows if r["camera"] == camera and not r["in_view"])
        if not seen:
            print(f"{camera}: no item in view")
            continue
        hits = sum(r["hit"] for r in seen)
        mean_iou = sum(r["iou"] for r in seen) / len(seen)
        print(f"{camera}: hit {hits}/{len(seen)} (iou >= {args.hit_iou}), mean iou {mean_iou:.2f}, "
              f"{out_of_view} items out of view")
        for r in seen:
            if not r["hit"]:
                print(f"  scene {r['scene']} {r['item']}: boxed {r['boxed']} (iou {r['iou']}, score {r['score']})")
    hits = sum(r["hit"] for r in rows)
    print(f"{args.detector}, initial state {args.init_state}: hit {hits}/{len(rows)} over both cameras")
    ms = [r["ms"] for r in rows]
    print(f"detection: {sum(ms) / len(ms):.1f} ms mean, {max(ms):.1f} ms max, on {args.device}")

    size = next(iter(tiles.values())).size
    for camera in CAMERAS.values():
        sheet = Image.new("RGB", (size[0] * 6, size[1] * len(args.scenes)))
        for r, scene in enumerate(args.scenes):
            for c, item in enumerate(SCENE_ITEMS[scene]):
                sheet.paste(tiles[(camera, scene, item)], (c * size[0], r * size[1]))
        sheet.save(args.out / f"{camera}.jpg", quality=85)
    print(f"wrote {args.out}: results.jsonl, " + ", ".join(f"{camera}.jpg" for camera in CAMERAS.values()))


if __name__ == "__main__":
    main()
