"""Run SmolVLA on one LIBERO-Object picnic item with an optional custom instruction.

This is our own observe -> infer -> act loop. It builds the environment, policy,
and processors the same way `lerobot-eval` does (LeRobot 0.6.1), so with the
original instruction it should reproduce the `lerobot-eval` episodes. The
difference is that the language instruction can be replaced.

Examples:

    python picnic_rollout.py --task-id 4
    python picnic_rollout.py --task-id 4 --instruction "grab the ketchup for the picnic basket"
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed

POLICY_PATH = "lerobot/smolvla_libero"
SUITE = "libero_object"
# The checkpoint was trained with the LIBERO cameras renamed (see its train_config.json).
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-id", type=int, required=True, help="LIBERO-Object task ID (0-9)")
    parser.add_argument("--instruction", default=None, help="Replace the original LIBERO instruction")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/picnic-rollout"))
    return parser.parse_args()


def add_batch_dim(value):
    # LeRobot processors expect batched observations, as produced by a vector env.
    if isinstance(value, dict):
        return {key: add_batch_dim(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None, ...]
    return value


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Same setup order as lerobot-eval: seed, environment, policy, processors.
    set_seed(args.seed)

    env_cfg = LiberoEnvConfig(task=SUITE, task_ids=[args.task_id])
    vec_env = make_env(env_cfg, n_envs=1, use_async_envs=False)[SUITE][args.task_id]
    env = vec_env.envs[0]  # one episode at a time, without vector autoreset
    libero_env = env.unwrapped  # LiberoEnv attributes sit behind LeRobot's wrapper

    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_cfg.pretrained_path = Path(POLICY_PATH)
    policy_cfg.device = args.device
    # The checkpoint ships 50, which replays a whole chunk blind (huggingface/lerobot#4614).
    policy_cfg.n_action_steps = args.n_action_steps
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=RENAME_MAP)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    original_instruction = libero_env.task_description
    instruction = args.instruction or original_instruction
    max_steps = libero_env._max_episode_steps
    # Keep videos from different instructions apart in one output directory.
    label = "original" if args.instruction is None else re.sub(r"[^a-z0-9]+", "-", instruction.lower()).strip("-")
    print(f"Item: {ITEMS[args.task_id]} (task {args.task_id})")
    print(f"Original instruction: {original_instruction}")
    print(f"Instruction used:     {instruction}")

    results = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        policy.reset()
        observation, _ = env.reset(seed=seed, options={NEW_ROLLOUT_OPTION: True})
        frames = [libero_env.render()]
        inference_times = []
        success = False
        step = 0
        episode_start = time.perf_counter()

        while step < max_steps:
            # Observe: camera images and robot state, plus the language instruction.
            batch = preprocess_observation(add_batch_dim(observation))
            batch["task"] = [instruction]
            batch = preprocessor(env_preprocessor(batch))

            # Infer: SmolVLA predicts a 50-action chunk when its queue is empty,
            # otherwise it returns the next queued action.
            runs_model = len(policy._queues[ACTION]) == 0
            sync(args.device)
            start = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(batch)
            sync(args.device)
            if runs_model:
                inference_times.append(time.perf_counter() - start)

            # Act: 7-D end-effector delta and gripper command.
            action = env_postprocessor({ACTION: postprocessor(action)})[ACTION]
            observation, _, terminated, _, info = env.step(action.to("cpu").numpy()[0])
            frames.append(libero_env.render())
            step += 1
            if terminated:
                success = bool(info["is_success"])
                break

        episode_s = time.perf_counter() - episode_start
        video_path = args.output_dir / f"task{args.task_id}_{label}_ep{episode}.mp4"
        # Write at the control rate so the video plays in simulated real time.
        write_video(video_path, frames, fps=env_cfg.fps)

        result = {
            "task_id": args.task_id,
            "item": ITEMS[args.task_id],
            "instruction": instruction,
            "original_instruction": original_instruction,
            "episode": episode,
            "seed": seed,
            "success": success,
            "steps": step,
            "max_steps": max_steps,
            "episode_s": round(episode_s, 2),
            "inference_calls": len(inference_times),
            "inference_total_s": round(sum(inference_times), 2),
            "inference_mean_s": round(sum(inference_times) / len(inference_times), 3),
            "device": args.device,
            "video": str(video_path),
        }
        results.append(result)
        print(json.dumps(result))

    with open(args.output_dir / "results.jsonl", "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    successes = sum(result["success"] for result in results)
    print(f"Success: {successes}/{len(results)}")
    vec_env.close()


if __name__ == "__main__":
    main()
