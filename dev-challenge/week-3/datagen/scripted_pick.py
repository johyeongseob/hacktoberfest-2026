"""Scripted demo: put LIBERO-Object items in the basket without a learned policy.

The simulator knows where every object is, so a few lines of control can do
what SmolVLA could not (for example the butter in the cream cheese scene).
For each item, in order, the gripper moves above it, goes down, closes,
lifts, moves above the basket, and opens. Actions are the same 7-D deltas
the LIBERO demos use (OSC_POSE: x, y, z, rotation, gripper), so episodes
made this way can become training data.

Usage (from dev-challenge/week-3):
    python datagen/scripted_pick.py --task-id 1 --items butter
    python datagen/scripted_pick.py --task-id 1 --items butter "cream cheese" --init-state 10
    python datagen/scripted_pick.py --task-id 1 --each --episodes 5 --no-video   # every item alone

Appends one JSON line per episode to OUTPUT_DIR/results.jsonl: whether
LIBERO's goal predicate (In <item>_1 basket_1_contain_region) holds for each
item right after its own step and at the end. Videos (agent camera, 20 fps)
unless --no-video.
"""

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import transform_utils as T

BASKET = "basket_1_contain_region"
MAX_DELTA_M = 0.05  # OSC_POSE: an action of 1.0 moves the target 5 cm
MAX_DELTA_RAD = 0.5  # and turns it 0.5 rad
OPEN, CLOSE = -1.0, 1.0


def object_name(item: str) -> str:
    return item.replace(" ", "_") + "_1"


class Scripted:
    def __init__(self, env, frames: list):
        self.env, self.frames = env, frames
        self.obs = None
        self.steps = 0
        self.home_quat = None  # gripper orientation to hold (set after the first step)

    def step(self, delta_xyz, gripper):
        action = np.zeros(7)
        action[:3] = np.clip(np.asarray(delta_xyz) / MAX_DELTA_M, -1.0, 1.0)
        if self.home_quat is not None:
            # Hold the starting orientation; without this the gripper slowly
            # turns and misses the next item.
            error = T.quat2axisangle(T.quat_multiply(self.home_quat, T.quat_inverse(self.obs["robot0_eef_quat"])))
            action[3:6] = np.clip(error / MAX_DELTA_RAD, -1.0, 1.0)
        action[6] = gripper
        self.obs, _, _, _ = self.env.step(action)
        self.frames.append(self.obs["agentview_image"][::-1, ::-1])  # LIBERO renders upside down
        self.steps += 1

    def eef(self):
        return self.obs["robot0_eef_pos"]

    def move_to(self, target, gripper, tol=0.01, max_steps=120, gain=1.0):
        # Proportional move of the gripper to target (world frame).
        for _ in range(max_steps):
            error = np.asarray(target) - self.eef()
            if np.linalg.norm(error) < tol:
                return True
            self.step(gain * error, gripper)
        return False

    def hold(self, gripper, n):
        for _ in range(n):
            self.step(np.zeros(3), gripper)


def pick_and_place(robot: Scripted, item: str, travel_z: float) -> None:
    pos = lambda name: robot.obs[f"{name}_pos"].copy()
    target = pos(object_name(item))
    above = target + np.array([0, 0, 0.10])
    robot.move_to([robot.eef()[0], robot.eef()[1], travel_z], OPEN)
    robot.move_to([above[0], above[1], travel_z], OPEN)
    robot.move_to(above, OPEN, tol=0.005)
    target = pos(object_name(item))  # re-read: objects can shift slightly
    # Aim at the object center: 1 cm higher stopped some grasps above small boxes.
    robot.move_to(target, OPEN, tol=0.005)
    robot.hold(CLOSE, 15)
    robot.move_to([robot.eef()[0], robot.eef()[1], travel_z], CLOSE)
    basket = pos("basket_1")
    robot.move_to([basket[0], basket[1], travel_z], CLOSE, tol=0.01)
    robot.move_to([basket[0], basket[1], basket[2] + 0.12], CLOSE, tol=0.01)
    robot.hold(OPEN, 15)
    robot.move_to([basket[0], basket[1], travel_z], OPEN)


ITEMS = [
    "alphabet soup",
    "cream cheese",
    "salad dressing",
    "bbq sauce",
    "ketchup",
    "tomato sauce",
    "butter",
    "milk",
    "chocolate pudding",
    "orange juice",
]


def run_episode(env, init_state, items, video_path=None) -> dict:
    """Put items in the basket in order from one initial state; return what held."""
    env.reset()
    env.set_init_state(init_state)
    frames = []
    robot = Scripted(env, frames)
    robot.hold(OPEN, 10)  # let objects settle, as LeRobot does after a reset
    robot.home_quat = robot.obs["robot0_eef_quat"].copy()
    inside = lambda item: bool(env.env._eval_predicate(["in", object_name(item), BASKET]))
    done_at = {}
    for item in items:
        pick_and_place(robot, item, travel_z=robot.obs["basket_1_pos"][2] + 0.25)
        done_at[item] = robot.steps if inside(item) else None
    robot.hold(OPEN, 10)
    at_end = {item: inside(item) for item in items}
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=20, macro_block_size=8)
    return {
        "items": items,
        "success": all(at_end.values()),
        "in_basket_at_end": at_end,
        "in_basket_after_own_step": done_at,
        "steps": robot.steps,
        "video": str(video_path) if video_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-id", type=int, required=True, help="LIBERO-Object scene (task ID 0-9)")
    parser.add_argument("--items", nargs="+", default=None, help="Default: every item in the scene")
    parser.add_argument("--each", action="store_true", help="Every item in its own episode")
    parser.add_argument("--init-state", type=int, default=10, help="First initial state (10 and up: training)")
    parser.add_argument("--episodes", type=int, default=1, help="Initial states init-state, init-state+1, ...")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scripted"))
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    size = 64 if args.no_video else 256  # small renders when nothing is saved
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size)
    env.seed(0)
    env.reset()
    in_scene = [item for item in ITEMS if object_name(item) in env.env.object_states_dict]
    items = args.items or in_scene
    missing = [item for item in items if item not in in_scene]
    if missing:
        raise SystemExit(f"Not in scene {args.task_id}: {missing}; the scene has {in_scene}")
    groups = [[item] for item in items] if args.each else [items]
    init_states = suite.get_task_init_states(args.task_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "results.jsonl", "a") as f:
        for group in groups:
            tag = "-".join(item.replace(" ", "") for item in group)
            for k in range(args.init_state, args.init_state + args.episodes):
                video = None if args.no_video else args.output_dir / f"task{args.task_id}_{tag}_init{k}.mp4"
                result = {"task_id": args.task_id, "init_state": k, **run_episode(env, init_states[k], group, video)}
                f.write(json.dumps(result) + "\n")
                f.flush()
                print(json.dumps(result), flush=True)
    env.close()


if __name__ == "__main__":
    main()
