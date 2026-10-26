# Was it the normalization stats? (No.)

A small side experiment from Week 3: one suspect for a broken fine-tune, tested
and ruled out.

## The goal of Week 3

Week 3 asks the picnic robot (`lerobot/smolvla_libero` in LIBERO-Object) to
pick the item I name, not just the item it already knows. Each LIBERO-Object
scene was built around one item, and the published model had learned to pick
that item no matter what I asked for:

- asked for the scene's own item: **10/10**
- asked for any of the 5 other items on the table: **1/50**

So I fine-tuned it on scripted demos where the robot also picks the other
items in each scene, hoping it would learn to listen to the item name. Full story:
[My Picnic Robot Memorized the Table, Not My Words](https://dev.to/johyeongseob/my-picnic-robot-memorized-the-table-not-my-words-1dd5).

## The problem

After only 1,000 training steps, the robot lost the one thing it could do:
the scene's own item fell from 10/10 to **0/10**.

## The suspect

While reading LeRobot's training code, I found that `lerobot-train` throws
away the checkpoint's normalization statistics when fine-tuning (the mean and
std used to scale state and action) and uses the new dataset's instead.

My scripted demos moved very differently from the original human demos. If
the statistics changed, the same network output would turn into a different
action, which could easily explain a sudden collapse.

## The test

An ablation: change one thing, keep everything else the same, and see if the
result changes. I trained on the same data twice:

| Run | Statistics |
|---|---|
| `dataset-stats` | LeRobot's default: the fine-tune dataset's own statistics |
| `pretrained-stats` | The original `smolvla_libero` statistics, copied in with [`use_base_stats.py`](../../train/use_base_stats.py) |

Same starting model, same data (347 episodes: scripted single-item picks
plus one original demo per item, built by
[`prepare_data.py`](../../train/prepare_data.py)), same settings (lr 2.5e-5, batch 32, seed 1000). The two training configs differ
only in the run name and output folder ([train-config/](train-config/)).

Each checkpoint was tested the same way as above: in all 10 scenes, ask for
the scene's own item (10 tries) and for each of the 5 other items (50 tries),
from the same starting layout (initial state 46), 300 steps max per try.

## The result

Both runs start from the published model (own item 10/10, other items 1/50).

| Training steps | Own item, `dataset-stats` | Own item, `pretrained-stats` | Other items, `dataset-stats` | Other items, `pretrained-stats` |
|---|---|---|---|---|
| 1,000 | 0/10 | 3/10 | 2/50 | 2/50 |
| 2,000 | 3/10 | 2/10 | 5/50 | 3/50 |
| 3,000 | 2/10 | 2/10 | 4/50 | 2/50 |

Keeping the original statistics did not stop the drop. Both runs fell to
about 2/10 on the scene's own item and were at the same 2/10 by 3,000 steps.
**The statistics were not the cause.**

**Limits:** one run per setting, one starting layout, and only 10 tries for
the own item. Small gaps like 0/10 vs 3/10 can happen by chance.

## What happened next

With the statistics ruled out, I kept looking. The scripted demos turned out
to have hidden pauses, and fixing them reduced the "picked up nothing"
failures. The own-item drop itself stayed, and I kept the original statistics
in later runs anyway. Details are in the post linked above.

## Files

- [`episodes.csv`](episodes.csv): every try as one row (run, steps, scene,
  asked item, success). Easiest place to look.
- `published/`, `dataset-stats-*/`, `pretrained-stats-*/`: raw evaluation
  output, one JSON line per try (`sceneK.jsonl` = LIBERO-Object task K).
- `train-config/`: the training config of each run.
- `summarize.py`: rebuilds the table and `episodes.csv` from the raw output
  (`python summarize.py`).
