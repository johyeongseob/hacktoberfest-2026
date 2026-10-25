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
ACTION_LIMIT = 0.9375  # the largest action in the LIBERO demos (lerobot/libero stats)
OPEN, CLOSE = -1.0, 1.0


def step_sentence(item: str) -> str:
    # The LIBERO-Object training sentence for one item.
    return f"pick up the {item} and place it in the basket"


def object_name(item: str) -> str:
    return item.replace(" ", "_") + "_1"


class Scripted:
    def __init__(self, env, frames: list, recorder=None):
        # recorder(obs, action, label) is called before every action once a
        # label is set, so a dataset sees the same observation the action used.
        self.env, self.frames, self.recorder = env, frames, recorder
        self.label = None
        self.obs = None
        self.steps = 0
        self.home_quat = None  # gripper orientation to hold (set after the first step)

    def step(self, delta_xyz, gripper):
        action = np.zeros(7)
        action[:3] = np.clip(np.asarray(delta_xyz) / MAX_DELTA_M, -ACTION_LIMIT, ACTION_LIMIT)
        if self.home_quat is not None:
            # Hold the starting orientation; without this the gripper slowly
            # turns and misses the next item.
            error = T.quat2axisangle(T.quat_multiply(self.home_quat, T.quat_inverse(self.obs["robot0_eef_quat"])))
            action[3:6] = np.clip(error / MAX_DELTA_RAD, -ACTION_LIMIT, ACTION_LIMIT)
        action[6] = gripper
        if self.recorder is not None and self.label is not None:
            self.recorder(self.obs, action.copy(), self.label)
        self.obs, _, _, _ = self.env.step(action)
        self.frames.append(self.obs["agentview_image"][::-1, ::-1])  # LIBERO renders upside down
        self.steps += 1

    def eef(self):
        return self.obs["robot0_eef_pos"]

    def move_to(self, target, gripper, tol=0.01, max_steps=120, gain=1.0, stall_steps=None):
        # Proportional move of the gripper to target (world frame).
        # stall_steps: also stop once the gripper has not come 1 mm closer in
        # that many steps. A target it cannot reach (the fingers on the table
        # below an item's center, an item already in the basket) otherwise
        # costs all max_steps with tiny actions, which the recordings showed
        # as 5-second pauses that no human demo has.
        best, since_best = np.inf, 0
        for _ in range(max_steps):
            error = np.asarray(target) - self.eef()
            distance = np.linalg.norm(error)
            if distance < tol:
                return True
            if distance < best - 0.001:
                best, since_best = distance, 0
            else:
                since_best += 1
                if stall_steps is not None and since_best >= stall_steps:
                    return False
            self.step(gain * error, gripper)
        return False

    def hold(self, gripper, n):
        for _ in range(n):
            self.step(np.zeros(3), gripper)


# How the gripper moves. "slow" made the first recordings (about 300 steps per
# item, twice the human LIBERO-Object demos' 146; tools/compare_demos.py).
# "fast" (the default since then) follows the human pace: a stronger pull
# toward each waypoint, looser tolerances on the way, one diagonal move
# instead of up-then-across, and shorter waits for the gripper.
# "steady" is "fast" without the pauses: a move that stops getting closer
# (stall, in steps) ends there instead of creeping on to its step limit.
PACES = {
    "slow": dict(gain=1.0, tol=0.01, fine_tol=0.005, close_wait=15, open_wait=15, diagonal=False, above=0.10,
                 stall=None),
    "fast": dict(gain=3.0, tol=0.02, fine_tol=0.006, close_wait=8, open_wait=6, diagonal=True, above=0.06,
                 stall=None),
    "steady": dict(gain=3.0, tol=0.02, fine_tol=0.006, close_wait=8, open_wait=6, diagonal=True, above=0.06,
                   stall=5),
}


def pick_and_place(
    robot: Scripted, item: str, travel_z: float, release_height: float = 0.12, center_object: bool = False,
    spot=(0.0, 0.0), pace: str = "fast",
) -> None:
    p = PACES[pace]
    pos = lambda name: robot.obs[f"{name}_pos"].copy()
    robot.label = step_sentence(item)
    target = pos(object_name(item))
    above = target + np.array([0, 0, p["above"]])
    if not p["diagonal"]:
        robot.move_to([robot.eef()[0], robot.eef()[1], travel_z], OPEN, stall_steps=p["stall"])
    robot.move_to([above[0], above[1], travel_z], OPEN, tol=p["tol"], gain=p["gain"], stall_steps=p["stall"])
    robot.move_to(above, OPEN, tol=p["fine_tol"], gain=p["gain"], stall_steps=p["stall"])
    target = pos(object_name(item))  # re-read: objects can shift slightly
    # Aim at the object center: 1 cm higher stopped some grasps above small boxes.
    robot.move_to(target, OPEN, tol=p["fine_tol"], gain=p["gain"], stall_steps=p["stall"])
    robot.hold(CLOSE, p["close_wait"])
    robot.move_to(
        [robot.eef()[0], robot.eef()[1], travel_z], CLOSE, tol=p["tol"], gain=p["gain"], stall_steps=p["stall"]
    )
    basket = pos("basket_1")
    basket[:2] += spot  # where in the basket this item goes
    target_z = basket[2] + release_height
    if center_object:
        # Tall bottles are held near the top and hang below and off-center:
        # bring the item, not the gripper, over the middle of the basket, and
        # lower until the item (not the gripper) is release_height above it.
        hang = robot.eef() - pos(object_name(item))
        basket[:2] += hang[:2]
        target_z += hang[2]
    robot.move_to([basket[0], basket[1], travel_z], CLOSE, tol=p["tol"], gain=p["gain"], stall_steps=p["stall"])
    robot.move_to([basket[0], basket[1], target_z], CLOSE, tol=0.01, gain=p["gain"], stall_steps=p["stall"])
    robot.hold(OPEN, p["open_wait"])
    robot.move_to([basket[0], basket[1], travel_z], OPEN, tol=p["tol"], gain=p["gain"], stall_steps=p["stall"])


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


# Tall bottles hang below the gripper and pushed the basket away at the
# usual heights. datagen/sweep_grasp.sh (2026-10-24, 30 episodes per item):
# ketchup 8 -> 25, salad dressing 11 -> 15 with these settings, while the
# other items did as well or better with the defaults.
TALL_ITEM_SETTINGS = {"center_object": True, "travel_height": 0.35, "release_height": 0.10}
TALL_ITEMS = {"ketchup", "salad dressing"}
# Items dropped at the basket's center landed on the one before and stayed
# above LIBERO's "in the basket" region; each item in a chain gets its own
# spot (the region is a 12 x 12 cm box around the basket center).
SPOTS = [(-0.03, -0.03), (0.03, 0.03), (-0.03, 0.03), (0.03, -0.03)]


def run_episode(
    env, init_state, items, video_path=None, recorder=None, yaw=0.0, release_height=0.12, center_object=False,
    travel_height=0.25, per_item=False, pace="fast",
) -> dict:
    """Put items in the basket in order from one initial state; return what held."""
    env.reset()
    env.set_init_state(init_state)
    frames = []
    robot = Scripted(env, frames, recorder)
    robot.hold(OPEN, 10)  # let objects settle, as LeRobot does after a reset
    # Hold the starting orientation, turned by yaw (radians) about the vertical.
    robot.home_quat = T.quat_multiply(T.axisangle2quat(np.array([0.0, 0.0, yaw])), robot.obs["robot0_eef_quat"])
    inside = lambda item: bool(env.env._eval_predicate(["in", object_name(item), BASKET]))
    done_at = {}
    for index, item in enumerate(items):
        # per_item: the tall bottles get TALL_ITEM_SETTINGS, everything else the arguments.
        settings = dict(center_object=center_object, travel_height=travel_height, release_height=release_height)
        if per_item and item in TALL_ITEMS:
            settings = TALL_ITEM_SETTINGS
        pick_and_place(
            robot, item, robot.obs["basket_1_pos"][2] + settings["travel_height"], settings["release_height"],
            settings["center_object"], spot=SPOTS[index % len(SPOTS)] if len(items) > 1 else (0.0, 0.0),
            pace=pace,
        )
        done_at[item] = robot.steps if inside(item) else None
    robot.label = None  # the closing wait is not part of any step
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
    parser.add_argument("--yaw", type=float, default=0.0, help="Gripper turn about the vertical, radians")
    parser.add_argument("--release-height", type=float, default=0.12, help="Gripper height over the basket when opening")
    parser.add_argument("--center-object", action="store_true", help="Place by the item's position, not the gripper's")
    parser.add_argument("--travel-height", type=float, default=0.25, help="Gripper height over the basket when moving")
    parser.add_argument("--pace", choices=list(PACES), default="fast")
    parser.add_argument("--per-item", action="store_true", help="Tall bottles get TALL_ITEM_SETTINGS (as recorded)")
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
                result = {
                    "task_id": args.task_id, "init_state": k, "yaw": args.yaw, "release_height": args.release_height,
                    "center_object": args.center_object, "travel_height": args.travel_height,
                    "pace": args.pace, "per_item": args.per_item,
                    **run_episode(
                        env, init_states[k], group, video, yaw=args.yaw, release_height=args.release_height,
                        center_object=args.center_object, travel_height=args.travel_height,
                        per_item=args.per_item, pace=args.pace,
                    ),
                }
                f.write(json.dumps(result) + "\n")
                f.flush()
                print(json.dumps(result), flush=True)
    env.close()


if __name__ == "__main__":
    main()
