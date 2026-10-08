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

## 5. Smoke test: ketchup task, one episode

Try EGL first. If rendering fails, switch to OSMesa.

```bash
export MUJOCO_GL=egl   # fallback: export MUJOCO_GL=osmesa
cd /mnt/c/Users/silve/Desktop/hacktoberfest-2026/dev-challenge/week-1
lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --policy.device=cpu \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[4]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir=./outputs/smoke-ketchup
```

Record:

- Whether the model loads and actions are produced.
- Whether the episode succeeds.
- Wall-clock time for the episode.
- Any errors, including first-run LIBERO configuration prompts.

`outputs/` holds local results and should stay out of Git.
