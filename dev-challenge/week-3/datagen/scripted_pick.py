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

Writes a 20 fps video (agent camera) and prints, per item, whether LIBERO's
goal predicate (In <item>_1 basket_1_contain_region) holds at the end.
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-id", type=int, required=True, help="LIBERO-Object scene (task ID 0-9)")
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--init-state", type=int, default=10, help="10 and up: training states")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scripted"))
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    env.set_init_state(suite.get_task_init_states(args.task_id)[args.init_state])
    domain = env.env

    frames = []
    robot = Scripted(env, frames)
    robot.hold(OPEN, 10)  # let objects settle, as LeRobot does after a reset
    robot.home_quat = robot.obs["robot0_eef_quat"].copy()
    for item in args.items:
        if f"{object_name(item)}_pos" not in robot.obs:
            raise SystemExit(f"{item} is not in scene {args.task_id}")
        pick_and_place(robot, item, travel_z=robot.obs["basket_1_pos"][2] + 0.25)
    robot.hold(OPEN, 10)

    inside = {item: bool(domain._eval_predicate(["in", object_name(item), BASKET])) for item in args.items}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(item.replace(" ", "") for item in args.items)
    video = args.output_dir / f"task{args.task_id}_{tag}_init{args.init_state}.mp4"
    imageio.mimsave(video, frames, fps=20, macro_block_size=8)
    print(f"steps: {robot.steps}  in basket: {inside}  success: {all(inside.values())}  video: {video}")
    env.close()


if __name__ == "__main__":
    main()
