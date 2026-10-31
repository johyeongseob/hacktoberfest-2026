# GPU Setup (RunPod)

- **Pod:** 1x A100 SXM 80 GB, PyTorch 2.8.0 (CUDA 12.8) template, volume
  disk at `/workspace`, SSH over exposed TCP enabled. Recording uses the
  CPUs (16 on this pod); tests and fine-tuning use the GPU.
- **Connect:** the "SSH over exposed TCP" command from the pod's Connect
  menu, for example `ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519`.

## Environment

```
apt update
apt install -y git python3-pip libegl1 libgl1 libosmesa6 ffmpeg tmux rsync
mkdir -p /workspace && cd /workspace
git clone https://github.com/johyeongseob/hacktoberfest-2026.git
pip install --break-system-packages uv
uv venv --python 3.12 /workspace/venvs/hf26-week3
source /workspace/venvs/hf26-week3/bin/activate
uv pip install "torch>=2.7,<2.12" "torchvision>=0.22,<0.27" --index-url https://download.pytorch.org/whl/cu128
uv pip install "lerobot[smolvla,libero]==0.6.1"
cat >> /workspace/venvs/hf26-week3/bin/activate <<'EOF'
export HF_HOME=/workspace/hf
export LIBERO_CONFIG_PATH=/workspace/.libero-hf26-week3
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
EOF
source /workspace/venvs/hf26-week3/bin/activate
```

The environment (named `hf26-week3`, shared with Week 3) lives on the
volume, but the `apt` packages live on the container disk, which starts
empty after a Stop and Start or a pod migration. Run the first two lines
again then; without `libegl1`, `import mujoco` fails with
`'NoneType' object has no attribute 'eglQueryString'`.

The thread limits matter: the pod reports the host's CPUs, and without them
each process starts hundreds of threads. With them, `nproc` prints 1.

Run long jobs inside `tmux` so they keep going if the SSH connection drops.

## Copy the results to the laptop

From WSL on the laptop, in `dev-challenge/week-4`: one file per test, the
scenes in order (videos stay on the pod).

```
E=/workspace/outputs/week4/eval/base; mkdir -p results/before-fine-tuning
for t in single:single seen:trained heldout:heldout; do
  ssh -p POD_PORT -i ~/.ssh/id_ed25519 root@POD_IP "cat $E/${t%%:*}/scene{0..9}/results.jsonl" > results/before-fine-tuning/${t##*:}.jsonl
done
```

Stop the pod in the RunPod console when the work is done.
