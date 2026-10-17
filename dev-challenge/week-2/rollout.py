"""Evaluate a SmolVLA checkpoint on one LIBERO-Object item with several wordings.

Based on the Week 1 loop (week-1/paraphrases/picnic_rollout.py). Changes:
- the checkpoint, device, first initial state, and wordings are arguments
  (validation starts at initial state 10, the test at 0; wordings are keys
  in data/wordings.json: original, A to H)
- one process runs several wordings, so the policy loads once
- the seed is set before every episode (initial state k uses seed 1000 + k),
  so an episode's result does not depend on what ran before it

Examples:
    python rollout.py --policy-path lerobot/smolvla_libero --task-id 6 --wordings original A B C D E F G H
    python rollout.py --policy-path outputs/n5/checkpoints/001000/pretrained_model --task-id 6 \
        --wordings A B C D --init-state 10 --episodes 5 --output-dir outputs/n5-val-001000
"""

import argparse
import json
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

SUITE = "libero_object"
BASE_SEED = 1000
# The checkpoints were trained with the LIBERO cameras renamed.
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
WORDINGS_FILE = Path(__file__).resolve().parent / "data" / "wordings.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--policy-path", required=True, help="Hub id, Hub id/subfolder, or local pretrained_model folder"
    )
    parser.add_argument("--task-id", type=int, required=True, help="LIBERO-Object task ID (0-9)")
    parser.add_argument("--wordings", nargs="+", default=["original"], help="Keys from data/wordings.json")
    parser.add_argument("--init-state", type=int, default=0, help="First initial state")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rollout"))
    parser.add_argument(
        "--results-file", type=Path, default=None, help="Default: OUTPUT_DIR/results.jsonl (overwritten)"
    )
    return parser.parse_args()


def load_wordings() -> dict[str, str]:
    data = json.loads(WORDINGS_FILE.read_text())
    return {"original": data["original"], **data["taught"], **data["unseen"]}


def resolve_policy_path(path: str) -> str:
    # "owner/repo/subfolder" on the Hub (one run of upload_chosen.py) -> local folder.
    parts = path.split("/")
    if Path(path).exists() or len(parts) <= 2:
        return path
    from huggingface_hub import snapshot_download

    repo_id, subfolder = "/".join(parts[:2]), "/".join(parts[2:])
    return str(Path(snapshot_download(repo_id, allow_patterns=[f"{subfolder}/*"])) / subfolder)


def add_batch_dim(value):
    # LeRobot processors expect batched observations, as produced by a vector env.
    if isinstance(value, dict):
        return {key: add_batch_dim(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None, ...]
    return value


def main() -> None:
    args = parse_args()
    templates = load_wordings()
    unknown = [key for key in args.wordings if key not in templates]
    if unknown:
        raise SystemExit(f"Unknown wordings {unknown}; choose from {list(templates)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    item = ITEMS[args.task_id]

    # Same setup order as lerobot-eval: seed, environment, policy, processors.
    # Each episode reseeds below as well.
    set_seed(BASE_SEED + args.init_state)

    env_cfg = LiberoEnvConfig(task=SUITE, task_ids=[args.task_id])
    vec_env = make_env(env_cfg, n_envs=1, use_async_envs=False)[SUITE][args.task_id]
    env = vec_env.envs[0]  # one episode at a time, without vector autoreset
    libero_env = env.unwrapped  # LiberoEnv attributes sit behind LeRobot's wrapper

    policy_path = resolve_policy_path(args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = args.device
    # Checkpoints ship 50, which replays a whole chunk blind (huggingface/lerobot#4614).
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
    if original_instruction != templates["original"].format(item=item):
        raise SystemExit(f"Task {args.task_id} instruction {original_instruction!r} does not match {item!r}")
    max_steps = libero_env._max_episode_steps
    print(f"Item: {item} (task {args.task_id}), policy: {args.policy_path}")

    results = []
    for key in args.wordings:
        instruction = templates[key].format(item=item)
        for episode in range(args.episodes):
            init_state = args.init_state + episode
            seed = BASE_SEED + init_state
            # SmolVLA samples flow-matching noise from the global torch RNG.
            set_seed(seed)
            policy.reset()
            # Pin the initial state; LiberoEnv otherwise counts resets per process.
            libero_env.init_state_id = init_state
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
                start = time.perf_counter()
                with torch.inference_mode():
                    action = policy.select_action(batch)
                if runs_model:
                    if args.device == "cuda":
                        torch.cuda.synchronize()
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
            video_path = None
            if not args.no_video:
                # e.g. videos/task6_butter/C/init3.mp4
                item_dir = f"task{args.task_id}_{item.replace(' ', '-')}"
                video_path = args.output_dir / "videos" / item_dir / key / f"init{init_state}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                # Write at the control rate so the video plays in simulated real time.
                write_video(video_path, frames, fps=env_cfg.fps)

            result = {
                "policy": args.policy_path,
                "task_id": args.task_id,
                "item": item,
                "wording": key,
                "instruction": instruction,
                "init_state": init_state,
                "seed": seed,
                "success": success,
                "steps": step,
                "max_steps": max_steps,
                "episode_s": round(episode_s, 2),
                "inference_calls": len(inference_times),
                "inference_total_s": round(sum(inference_times), 2),
                "inference_mean_s": round(sum(inference_times) / len(inference_times), 3),
                "device": args.device,
                "video": str(video_path) if video_path else None,
            }
            results.append(result)
            print(json.dumps(result), flush=True)

    # One results file per process; run_eval.py merges them, so parallel
    # processes never write to the same file.
    results_file = args.results_file or args.output_dir / "results.jsonl"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    for key in args.wordings:
        rows = [r for r in results if r["wording"] == key]
        print(f"{item} {key}: {sum(r['success'] for r in rows)}/{len(rows)}")
    vec_env.close()


if __name__ == "__main__":
    main()
