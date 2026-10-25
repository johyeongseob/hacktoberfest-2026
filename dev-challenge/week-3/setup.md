# GPU Setup (RunPod)

The pod is set up as in Week 2 ([week-2/setup.md](../week-2/setup.md),
steps 1 to 5), with a Week 3 environment name.

- **Pod:** 1x A100 SXM 80 GB, PyTorch 2.8.0 (CUDA 12.8) template, container
  disk 50 GB, volume disk at `/workspace`, SSH over exposed TCP enabled.
  Rollouts need a few GB of GPU memory each; the 80 GB is for fine-tuning
  later on the same pod.
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

The environment lives on the volume, but the `apt` packages live on the
container disk, which starts empty after a Stop and Start or a pod
migration. Run the first two lines again then; without `libegl1`,
`import mujoco` fails with `'NoneType' object has no attribute 'eglQueryString'`.

The thread limits matter: the pod reports the host's CPUs, and without them
each process starts hundreds of threads (Week 2 step 5).

## Smoke test: one item, a few steps

```
cd /workspace/hacktoberfest-2026/dev-challenge/week-3
python eval/chain_rollout.py --mode steps --task-id 1 --items "cream cheese" --max-steps 20 --device cuda --no-video --output-dir /workspace/outputs/week3/smoke
```

## Can the model pick other items in a scene?

All ten LIBERO-Object scenes at once, six items each, one episode per item.
Run it inside `tmux` so it keeps going if the SSH connection drops.

```
cd /workspace/hacktoberfest-2026/dev-challenge/week-3
DEVICE=cuda PARALLEL=10 VENV=/workspace/venvs/hf26-week3/bin/activate OUT=/workspace/outputs/week3/cross-items bash eval/run_cross_items.sh
```

## Copy the results to the laptop

From WSL on the laptop, in `dev-challenge/week-3`:

```
rsync -av -e "ssh -p POD_PORT -i ~/.ssh/id_ed25519" root@POD_IP:/workspace/outputs/week3/ outputs/
```

Terminate the pod in the RunPod console when the work is done.
