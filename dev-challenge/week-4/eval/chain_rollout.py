"""Ask SmolVLA for LIBERO-Object items in one scene, one after another.

The instruction is the LIBERO-Object sentence for one item at a time ("pick
up the cream cheese and place it in the basket"); when that item is in the
basket, the instruction switches to the next item. The scene is the
LIBERO-Object task given by --task-id, and every item must be in it.

- the episode does not stop when the scene's own target item is in the
  basket; it stops when all items are in the basket at once, or at
  --max-steps (default --steps-per-item per item)
- each step checks every item with LIBERO's own goal predicate
  (In <item>_1 basket_1_contain_region) and records the first step it held
- every item on the table is followed too: how high it was lifted and
  whether it ended in the basket, so a failure shows whether the robot
  went for another item or missed the one it was asked for
- inference time is recorded per model call

eval/run_cross_items.sh (one item, --each) and eval/run_chain_eval.sh (two
items from data/pair_plan.json) run all ten scenes.

Example (from dev-challenge/week-4):
    python eval/chain_rollout.py --task-id 1 --items butter "cream cheese" --episodes 1
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
BASKET = "basket_1_contain_region"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--task-id", type=int, required=True, help="LIBERO-Object scene (task ID 0-9)")
    parser.add_argument(
        "--items", nargs="+", default=None, help="Items in the order named"
    )
    parser.add_argument("--plan", type=Path, default=None, help="data/pair_plan.json: run this scene's chains")
    parser.add_argument("--split", default="test", choices=["train", "test", "test_seen"])
    parser.add_argument("--lengths", type=int, nargs="+", default=[2], help="Chain lengths to take from --plan")
    parser.add_argument("--each", action="store_true", help="Ask for every item on its own")
    parser.add_argument("--init-state", type=int, default=0, help="First initial state")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Default: --steps-per-item per item of each chain")
    parser.add_argument("--steps-per-item", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chain"))
    return parser.parse_args()


def step_sentence(item: str) -> str:
    # The LIBERO-Object training sentence for one item.
    return f"pick up the {item} and place it in the basket"


def object_name(item: str) -> str:
    return item.replace(" ", "_") + "_1"


def add_batch_dim(value):
    # LeRobot processors expect batched observations, as produced by a vector env.
    if isinstance(value, dict):
        return {key: add_batch_dim(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None, ...]
    return value


def run_group(args, items, env, libero_env, env_cfg, policy, preprocessor, postprocessor,
              env_preprocessor, env_postprocessor) -> list[dict]:
    """Run --episodes episodes asking for items, in order, and return one result per episode."""
    instruction = " / ".join(map(step_sentence, items))
    print(f"Instruction: {instruction!r}")
    tag = "steps_" + "-".join(item.replace(" ", "") for item in items)
    max_steps = args.max_steps or args.steps_per_item * len(items)
    results = []
    for episode in range(args.episodes):
        init_state = args.init_state + episode
        seed = BASE_SEED + init_state
        set_seed(seed)  # SmolVLA samples flow-matching noise from the global torch RNG
        policy.reset()
        libero_env.init_state_id = init_state
        observation, _ = env.reset(seed=seed, options={NEW_ROLLOUT_OPTION: True})
        domain = libero_env._env.env  # the LIBERO problem, which evaluates BDDL predicates
        # robosuite ends an episode at its horizon (1,000 steps by default);
        # four items may take longer.
        domain.horizon = max(domain.horizon, max_steps + 10)
        missing = [item for item in items if object_name(item) not in domain.object_states_dict]
        if missing:
            raise SystemExit(f"Not in scene {args.task_id}: {missing}")

        frames = [libero_env.render()]
        first_in = {item: None for item in items}
        on_table = [item for item in ITEMS if object_name(item) in domain.object_states_dict]
        height = lambda item: float(domain.object_states_dict[object_name(item)].get_geom_state()["pos"][2])
        start_height = {item: height(item) for item in on_table}
        lifted = {item: 0.0 for item in on_table}
        current = 0  # index of the item being asked for
        switches = []  # (step, item) whenever the instruction changes
        instruction_now = step_sentence(items[0])
        inference_times = []
        success = False
        step = 0
        episode_start = time.perf_counter()
        while step < max_steps:
            batch = preprocess_observation(add_batch_dim(observation))
            batch["task"] = [instruction_now]
            batch = preprocessor(env_preprocessor(batch))
            runs_model = len(policy._queues[ACTION]) == 0
            start = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(batch)
            if runs_model:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                inference_times.append(time.perf_counter() - start)

            action = env_postprocessor({ACTION: postprocessor(action)})[ACTION]
            # The scene's own goal ends LeRobot's episode ("terminated"); ignore it
            # and judge all named items instead.
            observation, _, _, _, _ = env.step(action.to("cpu").numpy()[0])
            frames.append(libero_env.render())
            step += 1
            inside = {item: bool(domain._eval_predicate(["in", object_name(item), BASKET])) for item in items}
            for item in on_table:
                lifted[item] = max(lifted[item], height(item) - start_height[item])
            for item, now in inside.items():
                if now and first_in[item] is None:
                    first_in[item] = step
            if all(inside.values()):
                success = True
                break
            if inside[items[current]] and current + 1 < len(items):
                # This item is done: ask for the next one, and drop the queued
                # actions planned for the old instruction.
                current += 1
                instruction_now = step_sentence(items[current])
                switches.append({"step": step, "item": items[current]})
                policy.reset()

        episode_s = time.perf_counter() - episode_start
        video_path = None
        if not args.no_video:
            video_path = args.output_dir / "videos" / f"task{args.task_id}_{tag}" / f"init{init_state}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            write_video(video_path, frames, fps=env_cfg.fps)  # control rate: real time

        result = {
            "policy": args.policy_path,
            "task_id": args.task_id,
            "items": items,
            "mode": "steps",
            "instruction": instruction,
            "switches": switches,
            "init_state": init_state,
            "seed": seed,
            "success": success,
            "items_in_basket_at_end": [item for item, now in inside.items() if now],
            "first_in_step": first_in,
            # Every item on the table: highest lift (m) and the ones in the basket at the end.
            "lifted_m": {item: round(rise, 3) for item, rise in lifted.items()},
            "any_in_basket_at_end": [
                item for item in on_table if domain._eval_predicate(["in", object_name(item), BASKET])
            ],
            "steps": step,
            "max_steps": max_steps,
            "episode_s": round(episode_s, 2),
            "inference_calls": len(inference_times),
            "inference_mean_s": round(sum(inference_times) / len(inference_times), 3),
            "device": args.device,
            "video": str(video_path) if video_path else None,
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    with open(args.output_dir / "results.jsonl", "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"{instruction}: {sum(r['success'] for r in results)}/{len(results)} all items in the basket")
    return results


def main() -> None:
    args = parse_args()
    # --each: every item is its own one-item episode, in one process.
    if args.plan is not None:
        scene_plan = json.loads(args.plan.read_text())["scenes"][str(args.task_id)][args.split]
        groups = [chain for chain in scene_plan if len(chain) in args.lengths]
    elif args.items is None:
        raise SystemExit("Give --items or --plan")
    else:
        groups = [[item] for item in args.items] if args.each else [args.items]
    for items in groups:
        unknown = [item for item in items if item not in ITEMS]
        if unknown or not items or len(set(items)) != len(items):
            raise SystemExit(f"Give different items from {ITEMS}; got {items}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(BASE_SEED + args.init_state)
    longest = args.max_steps or args.steps_per_item * max(len(items) for items in groups)
    env_cfg = LiberoEnvConfig(task=SUITE, task_ids=[args.task_id], episode_length=longest)
    vec_env = make_env(env_cfg, n_envs=1, use_async_envs=False)[SUITE][args.task_id]
    env = vec_env.envs[0]  # one episode at a time, without vector autoreset
    libero_env = env.unwrapped  # LiberoEnv attributes sit behind LeRobot's wrapper

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
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
    print(f"Scene: task {args.task_id} ({libero_env.task_description})")
    results = []
    for items in groups:
        results += run_group(args, items, env, libero_env, env_cfg, policy, preprocessor, postprocessor,
                             env_preprocessor, env_postprocessor)
    vec_env.close()


if __name__ == "__main__":
    main()
