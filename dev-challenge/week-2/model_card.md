---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
base_model: lerobot/smolvla_libero
datasets:
- lerobot/libero
tags:
- lerobot
- smolvla
- robotics
- libero
- vision-language-action
---

# SmolVLA LIBERO, picnic wordings

`lerobot/smolvla_libero` fine-tuned so that four picnic items in
LIBERO-Object can be requested in new words. Out of the box, the model packs
an item every time with its training sentence ("pick up the butter and place
it in the basket") but only about a third of the time with other ways of
saying it ("pack the butter for our picnic"). These checkpoints were
fine-tuned on N demos per item, each labeled with the original sentence and
four new wordings, to measure how many demos it takes.

One demo per item (four demos in total, about $1.50 of rented A100 time)
lifts the taught wordings from 31% to 80% and four wordings never seen in
training from 29% to 78%. The experiment, all results, and the code are in
[johyeongseob/hacktoberfest-2026, Week 2](https://github.com/johyeongseob/hacktoberfest-2026/tree/main/dev-challenge/week-2).

Made for the Hacktoberfest 2026 DEV Challenge, Week 2 (theme: Side Hustle).

## Checkpoints

One folder per N (demos per item). Each was fine-tuned separately from
`lerobot/smolvla_libero`; the folder holds the checkpoint chosen on
validation states (the step in the table). In a folder name such as
`n5-s1000`, `n5` is N = 5 and `s1000` is the training seed 1000, not a step.

| Folder | Demos per item (total) | Chosen step | Taught A-D | Unseen E-H | Original | Other 6 items |
| --- | --- | --- | --- | --- | --- | --- |
| (base) `lerobot/smolvla_libero` | 0 | - | 31% | 29% | 100% | 87% |
| `n1-s1000` | 1 (4) | 3000 | 80% | 78% | 95% | 60% |
| `n3-s1000` | 3 (12) | 1000 | 86% | 74% | 90% | 70% |
| `n5-s1000` | 5 (20) | 2000 | 84% | 74% | 88% | 70% |
| `n10-s1000` | 10 (40) | 3000 | 86% | 77% | 92% | 72% |
| `n20-s1000` | 20 (80) | 1000 | 92% | 79% | 98% | 68% |
| `n40-s1000` | 40 (160) | 3000 | 92% | 78% | 98% | 68% |

Test set: LIBERO-Object initial states 0 to 9, seed 1000 + initial state.
Taught, unseen, and original are over the four trained items (160, 160, and
40 episodes); "Other 6 items" is the six untrained items with the original
sentence (60 episodes). 95% intervals and per-item and per-wording results
are in [results/tables.md](https://github.com/johyeongseob/hacktoberfest-2026/blob/main/dev-challenge/week-2/results/tables.md).

Which one to use: `n20-s1000` or `n40-s1000` for the best success on the
taught wordings; `n1-s1000` to see what four demos already do. All of them
are worse than the base model on the six items they were not trained on.

## Wordings

The four trained items are alphabet soup, cream cheese, butter, and orange
juice; `{item}` stands for the item name.

| | Wording | Used in training |
| --- | --- | --- |
| original | pick up the {item} and place it in the basket | yes |
| A | grab the {item} and put it in the basket | yes |
| B | {item} in the basket | yes |
| C | pack the {item} for our picnic | yes |
| D | could you put the {item} in the picnic basket? | yes |
| E | toss the {item} into the basket | no (test only) |
| F | the basket still needs the {item} | no (test only) |
| G | can you get the {item} into the basket for me? | no (test only) |
| H | move the {item} over to the basket | no (test only) |

E to H were written and fixed before any training.

## How to run

The checkpoints are ordinary LeRobot 0.6.1 SmolVLA checkpoints. The Week 2
rollout script runs one in LIBERO-Object with any wording, on CPU or GPU:

```
git clone https://github.com/johyeongseob/hacktoberfest-2026.git
cd hacktoberfest-2026/dev-challenge/week-2
python rollout.py --policy-path johyeongseob725/smolvla-libero-picnic-wordings/n1-s1000 --task-id 6 --wordings C --episodes 1 --device cpu
```

`--task-id 6` is butter and `--wordings C` is "pack the butter for our
picnic". Installation (LeRobot 0.6.1 with the `smolvla` and `libero`
extras) is in the repository's setup guide. To load a checkpoint yourself:

```python
from huggingface_hub import snapshot_download
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

folder = "n1-s1000"
path = snapshot_download("johyeongseob725/smolvla-libero-picnic-wordings", allow_patterns=[f"{folder}/*"])
policy = SmolVLAPolicy.from_pretrained(f"{path}/{folder}")
```

## Training

- **Start:** `lerobot/smolvla_libero` (whole model trained, as in its own
  recipe).
- **Data:** `lerobot/libero` (revision `a1aaacb`), LIBERO-Object, the first
  N demos of each of the four items. Each training sample gets one of five
  instructions at random: the original or wordings A to D.
- **Schedule:** 3000 steps, batch 32, learning rate 2.5e-5 with 200 warmup
  steps and cosine decay, seed 1000.
- **Checkpoint choice:** steps 1000, 2000, and 3000 were scored on
  validation initial states 10 to 14 with wordings A to D (80 episodes); the
  best one was kept (ties go to the earlier step). Test states were never
  used for this choice.
- **Hardware and cost:** one rented A100 SXM 80 GB (RunPod, $1.82/hour),
  about 28 GB of GPU memory, about 48 minutes per run while sharing the GPU
  with evaluation (about $1.50). The whole experiment, evaluation included,
  took about six hours and $11.

## Limitations

- **Forgetting:** the fine-tune only saw four items, so the other six
  LIBERO-Object items got worse (87% to 60-72%), whatever N.
- **More demos do not help new wordings:** unseen wordings stay at 74% to
  79% from N = 1 to 40; for orange juice they drop from 36/40 to 23/40.
- **One training seed per N:** small-N results may change with other demos
  or seeds.
- **Simulation only:** LIBERO-Object in MuJoCo; not tested on a real robot.
- **Narrow task:** four items, one scene layout family, pick and place into
  a basket.

## Credits

Base model [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero)
([SmolVLA](https://arxiv.org/abs/2506.01844), Hugging Face) and the
[`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) dataset,
both Apache-2.0; benchmark [LIBERO](https://libero-project.github.io/).
