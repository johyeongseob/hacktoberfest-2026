#!/usr/bin/env bash
# Prepare a DigitalOcean NVIDIA GPU Droplet (AI/ML-ready Ubuntu 24.04 image)
# for the Week 1 SmolVLA LIBERO evaluation.
#
# Run from the cloned repository:
#   bash dev-challenge/week-1/setup_gpu_server.sh

set -euo pipefail

VENV="$HOME/.venvs/hf26-week1"
LIBERO_DIR="$HOME/.libero-hf26-week1"

echo "== GPU check"
nvidia-smi

echo "== System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3.12-venv libegl1 libgl1 libosmesa6 ffmpeg

echo "== Virtual environment"
if [ ! -d "$VENV" ]; then
  python3.12 -m venv "$VENV"
  # Same per-environment settings as the local WSL setup.
  echo "export LIBERO_CONFIG_PATH=\"$LIBERO_DIR\"" >> "$VENV/bin/activate"
  echo 'export MUJOCO_GL=egl' >> "$VENV/bin/activate"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip

echo "== LeRobot with SmolVLA and LIBERO (default CUDA build of PyTorch)"
pip install "lerobot[smolvla,libero]==0.6.1"
python -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('lerobot', lerobot.__version__)"

echo "== LIBERO config (answers the first-run dataset prompt with N)"
printf 'N\n' | python -c "import libero.libero" || true
cat "$LIBERO_DIR/config.yaml"

echo "== Model downloads"
hf download lerobot/smolvla_libero
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct

echo "== EGL check (NVIDIA EGL library should be listed)"
ls /usr/lib/x86_64-linux-gnu/ | grep -i -E 'libEGL_nvidia' || echo "NVIDIA EGL library not found. Use MUJOCO_GL=osmesa."

echo "Done. Run: source $VENV/bin/activate"
