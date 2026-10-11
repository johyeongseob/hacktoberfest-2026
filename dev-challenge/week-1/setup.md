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

`lerobot/smolvla_libero` expects the cameras renamed to `camera1` and
`camera2`, but LeRobot 0.6.1 does not apply the saved mapping automatically
([huggingface/lerobot#4578](https://github.com/huggingface/lerobot/pull/4578)),
so the command passes `--rename_map`. The checkpoint also ships
`n_action_steps=50`, which costs about 20 points on LIBERO
([huggingface/lerobot#4614](https://github.com/huggingface/lerobot/issues/4614)),
so the command passes `--policy.n_action_steps=10`.

The first run, from `dev-challenge/week-1` with internet access, downloaded
about 3.3 GB (SmolVLA checkpoint, SmolVLM2 backbone, LIBERO assets):

```bash
python baseline/benchmark_eval.py   --policy.path=lerobot/smolvla_libero   --policy.device=cpu   --policy.n_action_steps=10   --env.type=libero   --env.task=libero_object   --env.task_ids='[4]'   --eval.batch_size=1   --eval.n_episodes=1   --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'   --output_dir=./outputs/smoke-ketchup-timing
```

The episode succeeded (1/1, 176 steps); see
[baseline_results.md](baseline/baseline_results.md#smoke-test-october-9-2026).
The same command is in [run_smoke_test.sh](baseline/run_smoke_test.sh).
