# GPU Setup (DigitalOcean GPU Droplet, H100)

Target: one DigitalOcean GPU Droplet with 1x H100, Python 3.12, LeRobot 0.6.1
with CUDA. The Droplet runs both training and the LIBERO evaluation.
Everything lives under `/workspace` so the paths match the scripts.

The Droplet is billed per second until it is **destroyed**; powering it off
does not stop billing. Before destroying it, copy the results (step 9) and
upload the chosen checkpoints (step 10).

## 1. Create the Droplet

- GPU: 1x H100 (80 GB)
- Image: the AI/ML-ready Ubuntu image, which ships the NVIDIA driver
- Authentication: SSH key
- Note the vCPU count shown on the plan; `run_eval.py --parallel` should
  stay below it

Connect as root with the address shown on the Droplet page:

```
ssh root@DROPLET_IP
```

Check the GPU:

```
nvidia-smi
```

## 2. System packages

MuJoCo renders offscreen through EGL. `ffmpeg` is needed for the video
decoder (`torchcodec`) and for evaluation videos.

```
apt update
apt install -y git python3-pip libegl1 libgl1 libosmesa6 ffmpeg tmux
```

## 3. Repository and Python 3.12

LeRobot 0.6.x needs Python 3.12 or newer; `uv` installs it without touching
the system Python. On Ubuntu 24.04, add `--break-system-packages` to the
`pip install uv` line.

```
mkdir -p /workspace && cd /workspace
git clone https://github.com/johyeongseob/hacktoberfest-2026.git
pip install uv
uv venv --python 3.12 /workspace/venvs/hf26-week2
source /workspace/venvs/hf26-week2/bin/activate
```

## 4. PyTorch (CUDA) and LeRobot

The default PyPI wheels of PyTorch include CUDA. The version range matches
LeRobot 0.6.1.

```
uv pip install "torch>=2.7,<2.12" "torchvision>=0.22,<0.27"
uv pip install "lerobot[smolvla,libero]==0.6.1"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 5. Environment variables

Keep the Hugging Face cache, the LIBERO config, and the rendering backend
in the virtual environment's activate script:

```
echo 'export HF_HOME=/workspace/hf' >> /workspace/venvs/hf26-week2/bin/activate
echo 'export LIBERO_CONFIG_PATH=/workspace/.libero-hf26-week2' >> /workspace/venvs/hf26-week2/bin/activate
echo 'export MUJOCO_GL=egl' >> /workspace/venvs/hf26-week2/bin/activate
source /workspace/venvs/hf26-week2/bin/activate
```

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
differ by a few steps), with an inference time per call far below the
laptop's 1.2 s.

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

Record the peak memory and the step time. The full run uses 3000 steps, so
one training run takes about 15 times the probe's training time. The probe
saves no checkpoint (200 is below the 500-step save interval) except the
final one at step 200.

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
Training and evaluation share the H100.

```
cd /workspace/hacktoberfest-2026/dev-challenge/week-2
bash run_all.sh
```

| tmux window | What runs |
| --- | --- |
| `n0` | Test set on `lerobot/smolvla_libero` (N = 0) |
| `train` | `run_train.sh` for N = 1, 5, 40, then the extras 10, 3, 20, one after another. Each run saves a checkpoint every 500 steps (six in all) and logs the loss every 10 steps |
| `n1`, `n5`, ... | `watch_eval.py` per run: waits for each checkpoint, scores it on the validation set while training continues, then runs the test set on the best checkpoint (ties go to the earlier step). A watcher for a run that has not started yet just waits |
| `done` | Waits for every test summary, then writes `/workspace/outputs/week2/eval/ALL_DONE.txt` |

To run only the core runs: `RUNS="1 5 40" bash run_all.sh`.

Attach with `tmux attach -t week2`, switch windows with `Ctrl-b n` and
`Ctrl-b p`, detach with `Ctrl-b d`.

The validation curve and the chosen step land in
`/workspace/outputs/week2/eval/<run>-val.json`, the test results in
`/workspace/outputs/week2/eval/<run>-test/summary.txt`.

Before leaving the Droplet unattended, check:

- the `train` window keeps printing steps and loss past step 500
- the first checkpoint exists:
  `ls /workspace/outputs/week2/train/n1-s1000/checkpoints/`
- the `n1` window printed `step 500: validation x/80`
- there is room for the checkpoints (about 150 GB for all six runs):
  `df -h /workspace`

If a training run fails, the runs chained after it do not start, and its
watcher keeps waiting; check the `train` window and
`/workspace/outputs/week2/train/*.log` on return.

## 9. Copy the results to the laptop

From WSL on the laptop. Evaluation folders hold `results.jsonl`,
`summary.txt`, logs, and videos; training logs hold the step times.

```
mkdir -p /mnt/c/Users/silve/Desktop/hacktoberfest-2026/dev-challenge/week-2/outputs
rsync -av root@DROPLET_IP:/workspace/outputs/week2/eval /mnt/c/Users/silve/Desktop/hacktoberfest-2026/dev-challenge/week-2/outputs/
rsync -av --include='*/' --include='*.log' --exclude='*' root@DROPLET_IP:/workspace/outputs/week2/train /mnt/c/Users/silve/Desktop/hacktoberfest-2026/dev-challenge/week-2/outputs/
```

## 10. Upload the chosen checkpoints (private)

Only the `pretrained_model` folder is needed for inference (about 0.9 GB);
`training_state` is not uploaded. One private repository holds the chosen
checkpoint of each run (the step in `<run>-val.json`, not
`checkpoints/last`), one folder per run:

```
cd /workspace/hacktoberfest-2026/dev-challenge/week-2
python upload_chosen.py --repo johyeongseob725/smolvla-libero-picnic-wordings --runs n1-s1000 n5-s1000 n40-s1000
```

If the extra runs finished, add `n10-s1000 n3-s1000 n20-s1000` to `--runs`.

Later, any of them runs on the laptop CPU, for example:

```
python rollout.py --policy-path johyeongseob725/smolvla-libero-picnic-wordings/n5-s1000 --task-id 6 --wordings C --episodes 1 --device cpu
```

## 11. Before destroying the Droplet

Destroying it deletes everything not copied in steps 9 and 10. Check:

- the laptop has `outputs/eval/` (results, summaries, videos) and the
  training logs
- the private repository shows one folder per run
- no other checkpoint is worth keeping (for example an N = 1 checkpoint
  before and after overfitting); upload it the same way with `hf upload`
  if so

Then destroy the Droplet from the DigitalOcean control panel.
