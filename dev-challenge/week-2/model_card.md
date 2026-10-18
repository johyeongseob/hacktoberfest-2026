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

`lerobot/smolvla_libero` fine-tuned on N demos per item so that four
LIBERO-Object picnic items can be requested in new words. One demo per item
(four in total, about $1.50 of rented A100 time) lifts taught wordings from
31% to 80% and never-seen wordings from 29% to 78%.
Code and full results: [GitHub, Week 2](https://github.com/johyeongseob/hacktoberfest-2026/tree/main/dev-challenge/week-2).

## Results

<p align="center">
  <img src="https://raw.githubusercontent.com/johyeongseob/hacktoberfest-2026/main/dev-challenge/week-2/results/success_vs_n.png" alt="Test success against demos per item for taught, unseen, original, and other items">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/johyeongseob/hacktoberfest-2026/main/dev-challenge/week-2/results/wording_by_n.png" alt="Successes out of 40 per wording and N">
</p>

| Folder | Demos per item | Chosen step | Taught A-D | Unseen E-H | Original | Other 6 items |
| --- | --- | --- | --- | --- | --- | --- |
| base `lerobot/smolvla_libero` | 0 | - | 31% | 29% | 100% | 87% |
| `n1-s1000` | 1 | 3000 | 80% | 78% | 95% | 60% |
| `n3-s1000` | 3 | 1000 | 86% | 74% | 90% | 70% |
| `n5-s1000` | 5 | 2000 | 84% | 74% | 88% | 70% |
| `n10-s1000` | 10 | 3000 | 86% | 77% | 92% | 72% |
| `n20-s1000` | 20 | 1000 | 92% | 79% | 98% | 68% |
| `n40-s1000` | 40 | 3000 | 92% | 78% | 98% | 68% |

LIBERO-Object test states 0 to 9; taught, unseen, and original over the
four trained items (160, 160, 40 episodes), other items with the original
sentence (60 episodes). `s1000` is the training seed, not a step. Per-item
results and 95% intervals: [tables.md](https://github.com/johyeongseob/hacktoberfest-2026/blob/main/dev-challenge/week-2/results/tables.md).

| | Wording (`{item}`: alphabet soup, cream cheese, butter, orange juice) |
| --- | --- |
| original ● | pick up the {item} and place it in the basket |
| A-D ● | grab the {item} and put it in the basket / {item} in the basket / pack the {item} for our picnic / could you put the {item} in the picnic basket? |
| E-H ○ | toss the {item} into the basket / the basket still needs the {item} / can you get the {item} into the basket for me? / move the {item} over to the basket |

● in the fine-tuning data, ○ test only (fixed before training).

## Use

```
git clone https://github.com/johyeongseob/hacktoberfest-2026.git
cd hacktoberfest-2026/dev-challenge/week-2
python rollout.py --policy-path johyeongseob725/smolvla-libero-picnic-wordings/n1-s1000 --task-id 6 --wordings C --episodes 1 --device cpu
```

Butter (`--task-id 6`) with "pack the butter for our picnic" (`--wordings C`),
LeRobot 0.6.1 with the `smolvla` and `libero` extras.

## Training

From `lerobot/smolvla_libero`, whole model; `lerobot/libero` (revision
`a1aaacb`), first N demos of each item, each sample labeled at random with
the original sentence or A-D; 3000 steps, batch 32, lr 2.5e-5 (200 warmup,
cosine), seed 1000; checkpoint chosen at step 1000, 2000, or 3000 on
validation states 10 to 14 (A-D only). One A100 SXM 80 GB, about 28 GB of
memory, about 48 minutes per run.

## Limitations

- The six untrained items get worse (87% to 60-72%), whatever N.
- More demos do not help unseen wordings (74-79% for every N).
- One training seed per N; simulation only.

## Credits

[`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero)
([SmolVLA](https://arxiv.org/abs/2506.01844)) and
[`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero), both
Apache-2.0; benchmark [LIBERO](https://libero-project.github.io/).
