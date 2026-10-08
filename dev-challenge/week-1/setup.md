# Local Setup (WSL, CPU)

Target: WSL Ubuntu 24.04, Python 3.12, CPU only.

LeRobot 0.6.x requires Python 3.12 or newer. LIBERO runs only on Linux.

## 1. System packages

MuJoCo needs an OpenGL backend for offscreen rendering. `ffmpeg` is used
for evaluation videos.

```bash
sudo apt update
sudo apt install -y libegl1 libgl1 libosmesa6 ffmpeg
```

## 2. Virtual environment

Week 1 uses its own environment so that LeRobot's pinned PyTorch does not
affect the Weekend Challenge environment.

```bash
python3.12 -m venv ~/.venvs/hf26-week1
source ~/.venvs/hf26-week1/bin/activate
python -m pip install --upgrade pip
```

## 3. PyTorch (CPU build)

Installing the CPU build first avoids downloading the large CUDA wheels.
The version range matches LeRobot 0.6.1.

```bash
pip install "torch>=2.7,<2.12" "torchvision>=0.22,<0.27" \
  --index-url https://download.pytorch.org/whl/cpu
```

## 4. LeRobot with SmolVLA and LIBERO

```bash
pip install "lerobot[smolvla,libero]==0.6.1"
```

## LIBERO config and rendering

LIBERO stores its asset and init-state paths in `~/.libero/config.yaml` by
default. Another LIBERO installation (for example, Physical AI Studio) may
already own that file with paths into its own virtual environment. Give this
environment its own config directory and set the rendering backend on
activation:

```bash
echo 'export LIBERO_CONFIG_PATH="$HOME/.libero-hf26-week1"' >> ~/.venvs/hf26-week1/bin/activate
echo 'export MUJOCO_GL=egl' >> ~/.venvs/hf26-week1/bin/activate
source ~/.venvs/hf26-week1/bin/activate
```

On the first run, LIBERO asks whether to use a custom dataset folder.
Answer `N`. Evaluation does not need the LIBERO datasets.

## Model storage

`lerobot-eval` downloads `lerobot/smolvla_libero` on the first run into the
default Hugging Face cache, not into the repository's `models/` folder:

```bash
~/.cache/huggingface/hub/models--lerobot--smolvla_libero/
```

The first run also downloads the SmolVLM2-500M backbone
(`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, about 2 GB). After the first
download, set `HF_HUB_OFFLINE=1` to run without internet.

## 5. Smoke test: ketchup task, one episode

`lerobot/smolvla_libero` was trained with the LIBERO cameras renamed to
`camera1` and `camera2` (see its `train_config.json`). The LIBERO environment
still emits `image` and `image2`, so evaluation needs the same `--rename_map`.
In LeRobot 0.6.1 the saved mapping is not applied automatically
([huggingface/lerobot#4578](https://github.com/huggingface/lerobot/pull/4578)).

The checkpoint also ships `n_action_steps=50`, which replays a whole action
chunk before looking again and costs about 20 points on LIBERO
([huggingface/lerobot#4614](https://github.com/huggingface/lerobot/issues/4614)).
Use `--policy.n_action_steps=10`.

EGL is set in the activate script. If rendering fails, run
`export MUJOCO_GL=osmesa` and retry.

[run_eval.py](run_eval.py) takes the same arguments as `lerobot-eval`. It also
times every SmolVLA action-chunk prediction, prints a summary, and writes
`inference_timing.json` to the output directory.

```bash
cd /mnt/c/Users/silve/Desktop/hacktoberfest-2026/dev-challenge/week-1
python run_eval.py \
  --policy.path=lerobot/smolvla_libero \
  --policy.device=cpu \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[4]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --output_dir=./outputs/smoke-ketchup
```

Record:

- Whether the model loads and actions are produced.
- Whether the episode succeeds.
- Wall-clock time for the episode.
- Any errors, including first-run LIBERO configuration prompts.

`outputs/` holds local results and should stay out of Git.
