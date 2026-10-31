"""Re-render one original LIBERO-Object human demo per item, with the item boxed.

lerobot/libero holds the human demos as videos, without simulator states, so
the item's box cannot be drawn on them. The original LIBERO files (one HDF5
per task, about 0.7 GB each) keep the simulator state of every step: this
script puts each state back into the simulator, renders both cameras, marks
the task's item (datagen/mark.py), and stores the steps the way
datagen/record_scripted.py stores scripted chains, with the demo's actions.
The training set then uses these ten instead of the copies from lerobot/libero.

Usage (on the pod, from dev-challenge/week-4):
    python datagen/record_originals.py --root /workspace/outputs/week4/originals-mark
"""

import argparse
import os
from pathlib import Path

import h5py
from huggingface_hub import hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from record_scripted import FEATURES, FPS, to_frame
from scripted_pick import item_of

REPO_ID = "yifengzhu-hf/LIBERO-datasets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="New dataset folder")
    parser.add_argument("--demo", type=int, default=0, help="Which demo of each task (demo_<N> in the HDF5)")
    parser.add_argument("--no-mark", action="store_true", help="Re-render without the box")
    args = parser.parse_args()

    dataset = LeRobotDataset.create(
        repo_id=f"local/{args.root.name}", fps=FPS, features=FEATURES, root=args.root, robot_type="panda"
    )
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    for task_id in range(10):
        task = suite.get_task(task_id)
        sentence = task.language  # "pick up the <item> and place it in the basket"
        path = hf_hub_download(REPO_ID, f"libero_object/{task.name}_demo.hdf5", repo_type="dataset")
        with h5py.File(path, "r") as f:
            demo = f["data"][f"demo_{args.demo}"]
            states, actions = demo["states"][()], demo["actions"][()]

        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        for state, action in zip(states, actions):
            obs = env.set_init_state(state)  # the simulator at this step of the demo
            dataset.add_frame(to_frame(obs, action, sentence, None if args.no_mark else env.env.sim))
        dataset.save_episode()
        env.close()
        print(f"task {task_id} ({item_of(sentence)}): demo_{args.demo}, {len(states)} steps", flush=True)
    dataset.finalize()
    print(f"saved 10 demos to {args.root}")


if __name__ == "__main__":
    main()
