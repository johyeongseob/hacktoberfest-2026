"""Check OWLv2 as the item detector, without moving the robot.

For every LIBERO-Object scene at one initial state (default 0, the test
layout), ask OWLv2 for each of the scene's six items by name on both cameras
and keep its top box. The simulator's exact box (datagen/mark.py, the box the
training data and --mark gt draw) is the answer:
- iou: overlap of the top box with the asked item's exact box
- hit: iou >= --hit-iou
- boxed: the item on the table whose exact box the top box overlaps most,
  so a miss shows which item the detector took for the asked one
- score: OWLv2's confidence in the top box; ms: one detection, on the GPU

The detector sees the image upright (the raw render turned 180 degrees, as
the model sees it) and its box is turned back onto the raw render.

Writes OUT/results.jsonl (one line per item and camera), prints a summary,
and saves OUT/agentview.jpg: every scene (rows) and item (columns) on the
agent camera, exact box green, OWLv2 box red.

Example (from dev-challenge/week-4):
    python eval/check_detector.py --out /workspace/outputs/week4/eval/detector-check/owlv2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from PIL import Image, ImageDraw
from transformers import Owlv2ForObjectDetection, Owlv2Processor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datagen"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from make_chain_plan import SCENE_ITEMS  # noqa: E402
from mark import draw_box, item_box  # noqa: E402

SUITE = "libero_object"
MODEL = "google/owlv2-base-patch16-ensemble"
# LeRobot's LIBERO observation keys -> MuJoCo cameras (as in eval/chain_rollout.py).
CAMERAS = {"image": "agentview", "image2": "robot0_eye_in_hand"}
RED = (255, 0, 0)


def object_name(item: str) -> str:
    return item.replace(" ", "_") + "_1"


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    top, left = max(a[0], b[0]), max(a[1], b[1])
    bottom, right = min(a[2], b[2]), min(a[3], b[3])
    inter = max(bottom - top + 1, 0) * max(right - left + 1, 0)
    area = lambda box: (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    return inter / (area(a) + area(b) - inter)


class Owlv2:
    def __init__(self, device: str):
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(device).eval()

    @torch.inference_mode()
    def box(self, raw: np.ndarray, query: str):
        """Top box for query as (top, left, bottom, right) on the raw render, its score, and ms."""
        start = time.perf_counter()
        upright = np.ascontiguousarray(raw[::-1, ::-1])
        inputs = self.processor(text=[[query]], images=upright, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        scores = outputs.logits[0, :, 0].sigmoid()
        best = int(scores.argmax())
        # Boxes are (cx, cy, w, h), relative to the image padded to a square at the bottom and right.
        cx, cy, w, h = outputs.pred_boxes[0, best].tolist()
        if self.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        height, width = raw.shape[:2]
        side = max(height, width)
        top, bottom = (cy - h / 2) * side, (cy + h / 2) * side
        left, right = (cx - w / 2) * side, (cx + w / 2) * side
        top, bottom = max(top, 0), min(bottom, height - 1)
        left, right = max(left, 0), min(right, width - 1)
        # Upright -> raw: the 180-degree turn flips both axes.
        box = (height - 1 - bottom, width - 1 - right, height - 1 - top, width - 1 - left)
        return tuple(int(round(v)) for v in box), float(scores[best]), ms


def tile(raw: np.ndarray, exact, found, label: str) -> Image.Image:
    image = draw_box(draw_box(raw, exact), found, color=RED)[::-1, ::-1]
    picture = Image.fromarray(np.ascontiguousarray(image))
    ImageDraw.Draw(picture).text((3, 2), label, fill=(255, 255, 255))
    return picture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--query", default="{item}", help="Text asked for, with {item} as the item name")
    parser.add_argument("--hit-iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    detector = Owlv2(args.device)
    detector.box(np.zeros((256, 256, 3), np.uint8), "warm up")  # the first call is slow
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
                    "scene": scene, "camera": camera, "item": item, "query": args.query.format(item=item),
                    "in_view": exact[item] is not None, "exact": exact[item], "found": found,
                    "iou": round(overlaps[item], 3), "hit": overlaps[item] >= args.hit_iou,
                    "boxed": boxed, "score": round(score, 3), "ms": round(ms, 1),
                }
                rows.append(row)
                if camera == "agentview":
                    tiles[(scene, item)] = tile(raw, exact[item], found, f"{item} {row['iou']:.2f}")
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
    ms = [r["ms"] for r in rows]
    print(f"detection: {sum(ms) / len(ms):.1f} ms mean, {max(ms):.1f} ms max, on {args.device}")

    size = next(iter(tiles.values())).size
    sheet = Image.new("RGB", (size[0] * 6, size[1] * len(args.scenes)))
    for r, scene in enumerate(args.scenes):
        for c, item in enumerate(SCENE_ITEMS[scene]):
            sheet.paste(tiles[(scene, item)], (c * size[0], r * size[1]))
    sheet.save(args.out / "agentview.jpg", quality=85)
    print(f"wrote {args.out}/results.jsonl and agentview.jpg")


if __name__ == "__main__":
    main()
