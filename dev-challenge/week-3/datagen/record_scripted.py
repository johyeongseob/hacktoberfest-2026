"""Record scripted LIBERO-Object chains as a LeRobot dataset (lerobot/libero format).

Runs datagen/scripted_pick.py's controller and stores every step the way
lerobot/libero does, so the Week 2 training code can read it:
- observation.images.image / image2: agent and wrist cameras, 256x256,
  turned 180 degrees (LIBERO renders upside down; the dataset and LeRobot's
  evaluation both flip them)
- observation.state: end-effector position (3), axis-angle (3), gripper
  finger positions (2), as LeRobot's LIBERO processor builds it
- action: the 7-D OSC_POSE action that was executed
- task: the LIBERO-Object sentence of the item being picked in that step
Only episodes where every item ends in the basket are kept. Each chain is
tried from initial states --init-state, +1, ... until one succeeds or
--max-tries runs out (N = 1: the first success).

Usage (from dev-challenge/week-3):
    python datagen/record_scripted.py --task-id 1 --chains "butter|cream cheese" --root outputs/datasets/check
    python datagen/record_scripted.py --task-id 1 --random-chains 6 --lengths 2 3 4 --root outputs/datasets/scene1
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import transform_utils as T
from scripted_pick import ITEMS, object_name, run_episode

FPS = 10  # lerobot/libero's fps field; one frame per control step, as there
FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def to_frame(obs, action, label) -> dict:
    state = np.concatenate([obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
    return {
        "observation.images.image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "observation.images.image2": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "observation.state": state.astype(np.float32),
        "action": action.astype(np.float32),
        "task": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-id", type=int, required=True, help="LIBERO-Object scene (task ID 0-9)")
    parser.add_argument("--chains", nargs="*", default=[], help='Items in order, joined by "|"')
    parser.add_argument("--random-chains", type=int, default=0, help="Add this many random chains per length")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-state", type=int, default=10, help="First initial state (10 and up: training)")
    parser.add_argument("--max-tries", type=int, default=5)
    parser.add_argument("--root", type=Path, required=True, help="New dataset folder")
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    # horizon: four items take about 1,000 steps, robosuite's default limit.
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, horizon=3000)
    env.seed(0)
    env.reset()
    in_scene = [item for item in ITEMS if object_name(item) in env.env.object_states_dict]

    chains = [chain.split("|") for chain in args.chains]
    rng = random.Random(f"{args.seed}-{args.task_id}")
    for length in args.lengths if args.random_chains else []:
        for _ in range(args.random_chains):
            chains.append(rng.sample(in_scene, length))
    for chain in chains:
        missing = [item for item in chain if item not in in_scene]
        if missing:
            raise SystemExit(f"Not in scene {args.task_id}: {missing}; the scene has {in_scene}")

    dataset = LeRobotDataset.create(
        repo_id=f"local/{args.root.name}", fps=FPS, features=FEATURES, root=args.root, robot_type="panda"
    )
    init_states = suite.get_task_init_states(args.task_id)
    log = []
    for chain in chains:
        for k in range(args.init_state, args.init_state + args.max_tries):
            frames = []
            result = run_episode(env, init_states[k], chain, recorder=lambda o, a, l: frames.append(to_frame(o, a, l)))
            if result["success"]:
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                break
        entry = {"task_id": args.task_id, "items": chain, "init_state": k, "success": result["success"],
                 "tries": k - args.init_state + 1, "frames": len(frames) if result["success"] else 0}
        log.append(entry)
        print(json.dumps(entry), flush=True)
    dataset.finalize()
    env.close()
    (args.root / "chains.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
    print(f"saved {sum(e['success'] for e in log)}/{len(log)} chains to {args.root}")


if __name__ == "__main__":
    main()
