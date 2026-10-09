# DigitalOcean GPU Run

Goal: run the same ketchup episode as the local CPU smoke test on an NVIDIA
RTX 4000 Ada GPU Droplet and compare SmolVLA inference time.

Billing runs from Droplet creation until the Droplet is destroyed. Powering
off does not stop billing. Prepare everything locally first and destroy the
Droplet as soon as results are downloaded.

## 0. Before creating the Droplet (no cost)

- All Week 1 changes are committed and pushed, because the server clones the
  public GitHub repository.
- An SSH key is registered in the DigitalOcean account.
- The RTX 4000 Ada plan is available in a region, and its current price is
  checked on the [pricing page](https://www.digitalocean.com/pricing/gpu-droplets).

## 1. Create the Droplet (dashboard)

- Type: GPU Droplet, NVIDIA RTX 4000 Ada, 1 GPU
- Image: AI/ML-ready NVIDIA image (Ubuntu 24.04 with NVIDIA driver and CUDA)
- Authentication: SSH key
- Record the creation time for cost tracking.

## 2. Connect and set up

From local WSL:

```bash
ssh root@<DROPLET_IP>
```

On the server:

```bash
git clone https://github.com/johyeongseob/hacktoberfest-2026.git
cd hacktoberfest-2026
bash dev-challenge/week-1/setup_gpu_server.sh 2>&1 | tee ~/setup.log
source ~/.venvs/hf26-week1/bin/activate
```

The script checks the GPU, installs system packages and LeRobot 0.6.1,
creates the LIBERO config, downloads the models, and checks for the NVIDIA
EGL library. If that library is missing, run `export MUJOCO_GL=osmesa`.

## 3. Ketchup episode on GPU

Same settings and seed as the local CPU run, with `--policy.device=cuda`:

```bash
cd ~/hacktoberfest-2026/dev-challenge/week-1
nvidia-smi > outputs-gpu-info.txt 2>&1
python run_eval.py \
  --policy.path=lerobot/smolvla_libero \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[4]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --output_dir=./outputs/gpu-smoke-ketchup 2>&1 | tee gpu-smoke-ketchup.log
```

The first run also downloads the LIBERO assets (about 0.4 GB).

## 4. Download results (from local WSL)

```bash
cd /mnt/c/Users/silve/Desktop/hacktoberfest-2026/dev-challenge/week-1
scp -r root@<DROPLET_IP>:~/hacktoberfest-2026/dev-challenge/week-1/outputs/gpu-smoke-ketchup ./outputs/
scp root@<DROPLET_IP>:~/hacktoberfest-2026/dev-challenge/week-1/{outputs-gpu-info.txt,gpu-smoke-ketchup.log} ./outputs/gpu-smoke-ketchup/
```

## 5. Destroy the Droplet (dashboard)

Destroy it, then confirm that no GPU Droplet remains in the Droplets list.
Record the destroy time.
