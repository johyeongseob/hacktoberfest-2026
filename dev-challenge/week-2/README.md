# Teaching the Picnic Robot My Words

Hacktoberfest 2026 DEV Challenge, Week 2 (theme: Side Hustle).

Status: planning.

## Idea

In Week 1, `lerobot/smolvla_libero` packed picnic items every time with its
training sentence (40/40) but only a third of the time with four other ways
of saying the same request (A to D, 52/160). Week 2 teaches the robot those
four wordings with a few demos per item and asks two questions:

1. How many demos per item does it take before the taught wordings work?
2. Does learning four wordings help with wordings it has never seen?

Side Hustle angle: the curve is a price list. "Adapting an open VLA to the
way your users talk takes N demos per item and $X of GPU time." That service
only exists because the weights are open: a closed VLA cannot be fine-tuned
on a customer's own data, inspected, or run on the customer's own hardware.

## Experiment

- **Start:** `lerobot/smolvla_libero`, the Week 1 checkpoint.
- **Training data:** LIBERO-Object demos from `lerobot/libero` for the four
  Week 1 items only (alphabet soup, cream cheese, butter, orange juice), the
  first N demos per item (each item has 44 or 45 demos). Each N is a
  separate fine-tune from `smolvla_libero`, and the subsets are nested
  (1 within 5 within 40). Each demo is trained with five instructions: the
  original and wordings A to D. Training on all ten items is left for
  Week 3 if needed.
  - **Core:** N = 1, 5, 40 (start, rise, end of the curve).
  - **If time allows, in this order:** N = 10; N = 3 and 20; two more
    training seeds for N = 1 and 5 (error bars).
- **Training length:** every N trains for the same 3000 steps, the whole
  model as in the `smolvla_libero` recipe, with the loss logged every 10
  steps. A checkpoint every 500 steps (six per run) is scored on validation
  initial states 10 to 14 (wordings A to D, 80 episodes) while training
  continues, and the best one per N goes to the test. The test initial
  states 0 to 9 are never used for this choice. The validation scores give
  a success-versus-training-step curve per N that shows overfitting, next
  to the loss curve.
- **Compared with the original `smolvla_libero` training** (from its
  `train_config.json`), 3000 steps is short but enough, because the
  starting point already does the task and the data is much smaller:

| | `smolvla_libero` training | This fine-tune |
| --- | --- | --- |
| Start | `smolvla_base` (never saw LIBERO) | `smolvla_libero` (already does the task) |
| To learn | Arm motion for 40 tasks | Four new wordings for four items |
| Data | 40 tasks, about 273,000 frames | 4 items, about 600 to 24,000 frames (N = 1 to 40) |
| Steps x batch | 25,000 x 32 | 3,000 x 32 |
| Passes over each frame | About 2.9 | About 4 (N = 40) to 160 (N = 1) |
| Learning rate | 1e-4, warmup 1,000 | 2.5e-5, warmup 200 |

- **Test wordings, 4 items (alphabet soup, cream cheese, butter, orange
  juice) x 10 episodes each, the Week 1 protocol:**

| Set | Wordings | Role |
| --- | --- | --- |
| Taught | A to D (Week 1) | Did the taught wordings get fixed? |
| Unseen | E to H in [data/wordings.json](data/wordings.json), fixed before any training under written rules (no verb or sentence structure shared with A to D); test only, never used for training or checkpoint choice | Does it generalize to new wordings? |
| Original | Training sentence | Did the robot forget what it already did? |
| Other items | Training sentence for the six untrained items (60 episodes) | Did training on four items break the other six? |

- **N = 0:** Week 1 results for A to D (52/160), the original (40/40), and
  the other six items (49/60), re-checked on the GPU machine; E to H
  measured before any training.
- **Result figure:** success rate against demos per item (0, 1, 5, 40, plus
  any extra points), one line each for taught, unseen, original, and other
  items.
- **Result videos:** the same scene and wording across N (for example "pack
  the butter for our picnic" at N = 0, 1, 5, 40), built with the Week 1
  `tools/make_grid.py`.
- **Cost:** GPU hours and dollars per run, and inference timing, go in the
  results doc from the start.

## Plan

| Step | Work | Done when |
| --- | --- | --- |
| M0 | Rent a GPU, install LeRobot 0.6.1 with CUDA, run a smoke test there (EGL rendering) | Butter, original wording: success in about 145 steps, as on the laptop |
| M1 | Fix wordings E to H; build the N-demo subsets and five-instruction relabeling; pin the `lerobot/libero` revision | Done: [data/wordings.json](data/wordings.json), [data/subsets.json](data/subsets.json), [train_relabel.py](train_relabel.py) |
| M2 | Measure N = 0 on the GPU (the full test set) | Baseline row of the results table |
| M3 | Fine-tune N = 1, 5, 40; pick each checkpoint on the validation states | Chosen checkpoints uploaded to a private Hugging Face repo, training cost logged |
| M4 | Test the chosen checkpoints; draw the figure and the video grid | Results doc with the table, figure, videos, cost, and inference timing |
| M5 | Publish the best checkpoint on the Hugging Face Hub with a model card | Public model repo that runs with a documented command |
| Extra | N = 10, N = 3 and 20, extra seeds for N = 1 and 5 | Points added to the same table and figure |
| M6 | DEV post | Published with `devchallenge`, `hf26challenge` |

## Risks

- **Deadline:** October 18, 11:59 PM PDT (October 19, 15:59 KST). If time
  runs short, stop after the core N = 1, 5, 40. Training and evaluation run
  on the GPU machine at the same time; the laptop CPU is too slow for the
  roughly 4,400 evaluation episodes.
- **CPU vs GPU:** Week 1 ran on CPU; GPU arithmetic can flip a few
  episodes. N = 0 is re-measured on the GPU for that reason.
- **Small N is noisy and can overfit:** with 1 to 5 demos per item, the
  choice of demos and the training seed can move the result, and the same
  frames are repeated many times. Validation-based checkpoint choice and
  the extra seeds address this.

## Stack

| Layer | Choice |
| --- | --- |
| VLA model | `lerobot/smolvla_libero`, fine-tuned per N |
| Training data | `lerobot/libero`, LIBERO-Object episodes (MP4 video) |
| Runner | LeRobot 0.6.1. Training: [run_train.sh](run_train.sh) runs [train_relabel.py](train_relabel.py), `lerobot-train` with the five-wording relabeling. Evaluation: [run_eval.py](run_eval.py) runs [rollout.py](rollout.py) (the Week 1 loop with the checkpoint, initial state, and wordings as arguments) in parallel; [watch_eval.py](watch_eval.py) validates each checkpoint during training and tests the best. [run_all.sh](run_all.sh) starts all of it in one tmux session |
| Benchmark | LIBERO-Object, same items and protocol as Week 1 |
| Hardware | One cloud GPU, DigitalOcean H100 or RunPod A100 (to be decided); setup in [setup.md](setup.md) |
| Demo subsets | [data/subsets.json](data/subsets.json), built by [data/make_subsets.py](data/make_subsets.py) from `lerobot/libero` revision `a1aaacb` |
