# GPU Setup (RunPod, A100)

Target: one RunPod GPU pod with **1x A100 SXM 80 GB (`NVIDIA A100-SXM4-80GB`)**,
Python 3.12, LeRobot 0.6.1 with
CUDA. The pod runs both training and the LIBERO evaluation. Everything that
must survive lives under `/workspace` (the pod volume), so the paths match
the scripts.

RunPod bills from prepaid credit while the pod runs. If the credit runs out,
the pod is stopped and the jobs end, so keep enough credit for the whole run
plus a margin. Before terminating the pod, copy the results (step 9) and
upload the chosen checkpoints (step 10).

Measured on 2026-10-18 (A100 SXM, $1.82/hour with disks): setup about 40
minutes, `run_all.sh` with all six runs 5 hours 19 minutes (each training
run about 48 minutes, its test done about 25 minutes later), about 6 hours
and $11 from creating the pod to `ALL_DONE.txt`.

DigitalOcean, a Hacktoberfest sponsor, was the first choice, but a new
account cannot create GPU Droplets until its limit is raised, either with a
$250 non-refundable prepay or through a support ticket that takes one to two
business days.

## 1. Create the pod

- Account: add credit (Billing) and the laptop's SSH public key
  (Settings, SSH Public Keys; `cat ~/.ssh/id_ed25519.pub` in WSL)
- GPU: **1x A100 SXM** (80 GB, 16 vCPU, $1.79/hour on 2026-10-18; $1.82/hour
  with the disks). Check the count is 1 and note the hourly price
- Template: PyTorch 2.8.0 (CUDA 12.8); the PyTorch inside is
  replaced in step 4
- Container disk: 50 GB; volume disk: 200 GB, mounted at `/workspace`
  (six runs keep up to about 80 GB of checkpoints)
- Enable SSH over exposed TCP (public IP), which `rsync` in step 9 needs
- Note the vCPU count shown for the pod; `run_eval.py --parallel` should
  stay below it

Connect with the "SSH over exposed TCP" command from the pod's Connect
menu; it looks like this (address and port differ per pod):

```
ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519
```

Check the GPU:

```
nvidia-smi
```

## 2. System packages

MuJoCo renders offscreen through EGL; on RunPod the NVIDIA EGL driver is
present, so rendering runs on the GPU. `ffmpeg` is needed for the video
decoder (`torchcodec`) and for evaluation videos, `rsync` for step 9.

```
apt update
apt install -y git python3-pip libegl1 libgl1 libosmesa6 ffmpeg tmux rsync
```

## 3. Repository and Python 3.12

LeRobot 0.6.x needs Python 3.12 or newer; `uv` creates the environment.
The template is Ubuntu 24.04, whose system pip refuses global installs
without `--break-system-packages`.

```
mkdir -p /workspace && cd /workspace
git clone https://github.com/johyeongseob/hacktoberfest-2026.git
pip install --break-system-packages uv
uv venv --python 3.12 /workspace/venvs/hf26-week2
source /workspace/venvs/hf26-week2/bin/activate
```

## 4. PyTorch (CUDA) and LeRobot

The RunPod PyTorch 2.8 template ships a CUDA 12.8 driver stack, so the
wheels come from the CUDA 12.8 index (newer default wheels may target a newer
CUDA than the driver supports). The version range matches LeRobot 0.6.1.

```
uv pip install "torch>=2.7,<2.12" "torchvision>=0.22,<0.27" --index-url https://download.pytorch.org/whl/cu128
uv pip install "lerobot[smolvla,libero]==0.6.1"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 5. Environment variables

Keep the Hugging Face cache, the LIBERO config, the rendering backend, and
two process settings in the virtual environment's activate script:

```
cat >> /workspace/venvs/hf26-week2/bin/activate <<'EOF'
export HF_HOME=/workspace/hf
export LIBERO_CONFIG_PATH=/workspace/.libero-hf26-week2
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
EOF
source /workspace/venvs/hf26-week2/bin/activate
```

- **Thread limits:** the pod reports 256 CPUs (`nproc` shows the host) but
  has 16 vCPU. Without the limits, each evaluation process started hundreds
  of threads, used 400 to 600% CPU, and starved training (stuck at step 0 for
  nine minutes). Rendering runs on the GPU, so one thread per process costs
  nothing.
- **`PYTHONUNBUFFERED`:** without it, the training log reaches the tmux
  window in late bursts because `run_train.sh` pipes it through `tee`.

If the smoke test below fails with an EGL error, try `export
MUJOCO_GL=osmesa` (software rendering, slower) and note it in the results.

Log in to the Hugging Face Hub (needed in step 10). Choose "Log in with
your browser", open https://hf.co/oauth/device on the laptop, enter the code
shown, and authorize; then check the account:

```
hf auth login
hf auth whoami
```

If the upload in step 10 is refused for lack of write access, log in again
with "Paste an access token" and a token with Write permission.

## 6. Smoke test: butter, original wording, one episode

The Week 2 rollout loop on the GPU. On the first run, LIBERO asks whether to
use a custom dataset folder; answer `N`. The first run also downloads the
checkpoint, the SmolVLM2 backbone, and the LIBERO assets (about 3.3 GB).

```
cd /workspace/hacktoberfest-2026/dev-challenge/week-2
python rollout.py --policy-path lerobot/smolvla_libero --task-id 6 --wordings original --episodes 1 --output-dir /workspace/outputs/week2/smoke
```

Expected: success in about 145 steps (145 on the laptop CPU; the GPU may
differ by a few steps), with an inference time per call far below the 1.15 s
of the Week 1 laptop run.

## 7. Training probe: memory and speed

The real training script for N = 1 with 200 steps. It checks the whole
pipeline (download, relabeling, training, checkpoint saving) and measures
peak GPU memory and seconds per step. The log should show `[relabel]` lines
with mixed wordings.

```
cd /workspace/hacktoberfest-2026/dev-challenge/week-2
STEPS=200 OUT_ROOT=/workspace/outputs/week2/probe bash run_train.sh 1
```

In a second SSH session, watch memory while it runs:

```
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 5
```

Record the peak memory and the step time. Measured: about 28 GB of GPU
memory, 0.46 s per step with the GPU to itself, plus about 1.5 minutes of
data loader start-up before the first step. The probe saves only the final
checkpoint at step 200 (below the 1000-step save interval).

## 7b. Upload check (private repo)

Checks write access before the long run, so a token problem does not show
up only after the experiment. It creates the private repository with a
one-line README, which the model card replaces later:

```
echo "Week 2 checkpoints (private, work in progress)." > /tmp/README.md
hf upload johyeongseob725/smolvla-libero-picnic-wordings /tmp/README.md README.md --private
```

If it fails with a permission error, log in again with "Paste an access
token" and a Write token (step 5), then repeat. Check on
https://huggingface.co/johyeongseob725/smolvla-libero-picnic-wordings that
the repository exists and shows as private.

## 8. Run the experiment

One command starts everything in a tmux session named `week2` and returns
in a few seconds; the jobs keep running after the SSH connection closes.
Training and evaluation share the A100; while eight evaluation processes run
next to it, a training step takes about 2 s instead of 0.46 s.

```
cd /workspace/hacktoberfest-2026/dev-challenge/week-2
bash run_all.sh
```

| tmux window | What runs |
| --- | --- |
| `n0` | Test set on `lerobot/smolvla_libero` (N = 0) |
| `train` | `run_train.sh` for N = 1, 5, 40, then the extras 10, 3, 20, one after another. Each run saves a checkpoint at 1000, 2000, and 3000 steps and logs the loss every 10 steps |
| `n1`, `n5`, ... | `watch_eval.py` per run: waits for the 1000, 2000, and 3000-step checkpoints, scores each on the validation set while training continues, then runs the test set on the best one (ties go to the earlier step). A watcher for a run that has not started yet just waits |
| `done` | Waits for every test summary, then writes `/workspace/outputs/week2/eval/ALL_DONE.txt` |

To run only the core runs: `RUNS="1 5 40" bash run_all.sh`.

Attach with `tmux attach -t week2` (from inside another tmux session:
`tmux switch-client -t week2`), go to a window with `Ctrl-b` and its number,
detach with `Ctrl-b d`. For checks, open a spare window with `Ctrl-b c`
rather than typing into a window that runs a job.

The validation curve and the chosen step land in
`/workspace/outputs/week2/eval/<run>-val.json`, the test results in
`/workspace/outputs/week2/eval/<run>-test/summary.txt`.

Watch the run for a while before leaving it: problems such as the CPU
starvation in step 5 show up early. In the Week 2 run, the checks below
were done about 30 minutes after `run_all.sh` started; then the laptop was
turned off, and the pod ran on its own to the end.

Before leaving the pod unattended, check:

- the `train` window keeps printing steps and loss past step 1000
- the first checkpoint exists:
  `ls /workspace/outputs/week2/train/n1-s1000/checkpoints/`
- the `n0` test finished (`/workspace/outputs/week2/eval/n0-test/summary.txt`)
- the `n1` window printed `step 1000: validation x/80` (about 40 minutes in)
- `top` shows no process far above 100% CPU (see the thread limits in step 5)
- there is room for the checkpoints (about 80 GB for all six runs):
  `df -h /workspace`
- the RunPod credit covers the remaining hours (Billing page)

If a training run fails, the runs chained after it do not start, and its
watcher keeps waiting; check the `train` window and
`/workspace/outputs/week2/train/*.log` on return.

## 9. Copy the results to the laptop

From WSL on the laptop, with the address and port of the "SSH over exposed
TCP" command, and `LOCAL_DIR` replaced by the laptop's
`dev-challenge/week-2` folder. Evaluation folders hold `results.jsonl`,
`summary.txt`, logs, and videos; training logs hold the step times.

```
mkdir -p LOCAL_DIR/outputs
rsync -av -e "ssh -p POD_PORT -i ~/.ssh/id_ed25519" root@POD_IP:/workspace/outputs/week2/eval LOCAL_DIR/outputs/
rsync -av -m -e "ssh -p POD_PORT -i ~/.ssh/id_ed25519" --include='*/' --include='*.log' --exclude='*' root@POD_IP:/workspace/outputs/week2/train LOCAL_DIR/outputs/
```

The evaluation folder was 454 MB with 2,940 test videos. `-m` skips the
empty checkpoint folders that the include rule would otherwise create.

## 10. Upload the chosen checkpoints (private)

Only the `pretrained_model` folder is needed for inference (about 0.9 GB);
`training_state` is not uploaded. One private repository holds the chosen
checkpoint of each run (the step in `<run>-val.json`, not
`checkpoints/last`), one folder per run:

```
cd /workspace/hacktoberfest-2026/dev-challenge/week-2
python upload_chosen.py --repo johyeongseob725/smolvla-libero-picnic-wordings --runs n1-s1000 n3-s1000 n5-s1000 n10-s1000 n20-s1000 n40-s1000
```

Leave out runs that did not finish.

Later, any of them runs on the laptop CPU, for example:

```
python rollout.py --policy-path johyeongseob725/smolvla-libero-picnic-wordings/n5-s1000 --task-id 6 --wordings C --episodes 1 --device cpu
```

## 11. Before terminating the pod

Terminating deletes the pod and its volume, so everything not copied in
steps 9 and 10 is gone. Check:

- the laptop has `outputs/eval/` (results, summaries, videos) and the
  training logs
- the private repository shows one folder per run
- no other checkpoint is worth keeping (for example an N = 1 checkpoint
  before and after overfitting); upload it the same way with `hf upload`
  if so

Then **terminate** the pod in the RunPod console. A running pod offers
only Stop in its menu: stop it first (the volume disk still bills while
stopped), then choose Terminate Pod from the same menu, and check that it no
longer appears under Pods.
