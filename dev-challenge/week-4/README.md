# Exorcising the Training Table

Hacktoberfest 2026 DEV Challenge, Week 4 (theme: Ghost in the Machine).
Deadline: 2026-10-31 11:59 PM PDT (2026-11-01 15:59 KST).

Status: step 1 (reproduce): fine-tuning on the 310-episode set.

## Idea

Week 3 found a ghost in `lerobot/smolvla_libero`: asked for any item on a
LIBERO-Object table, it picks the item that table was trained on. Scripted
demos of every item broke that habit but did not teach the model to read the
item name. Week 4 asks whether marking the asked item on the camera image
(a box from a detector) does what more data did not.

1. **Reproduce** with a new training set and a test that holds out table
   layouts and item orders.
2. **Add one detection module**: the same data with the asked item marked,
   the same test.

Every command below runs from `dev-challenge/week-4` on the pod, set up as
in [setup.md](setup.md), with the environment active.

| Folder | What |
| --- | --- |
| [`data`](data) | The plan: which chains are trained and which are tested |
| [`datagen`](datagen) | Scripted demos: the controller and the recorder |
| [`train`](train) | Training-set builder and normalization statistics |
| [`eval`](eval) | The 180-episode test |
| [`results`](results) | Test results without videos |

## Training set: 310 episodes

| Part | Episodes |
| --- | --- |
| Original LIBERO-Object demos, one per scene | 10 |
| Single picks: every item of every scene once (10 scenes x 6 items) | 60 |
| Two-item chains: 24 of the 30 ordered pairs per scene | 240 |

### 1. The plan

```
python data/make_chain_plan.py
```

Per scene, with its six items A to F in `SCENE_ITEMS` order: `train` holds
the six single items and the 24 pairs outside the cycle A->B, B->C, ...,
F->A; `test` holds that cycle (held-out orders); `test_seen` holds the
reverse cycle B->A, ..., A->F (trained orders, tested as a control).
The result is [`data/pair_plan.json`](data/pair_plan.json).

### 2. Record the 300 scripted chains

```
bash datagen/run_record_all.sh
```

[`datagen/scripted_pick.py`](datagen/scripted_pick.py) reads the object
positions from the simulator, moves the gripper over the asked item, grasps
it, and places it in the basket, at about the human LIBERO-Object pace.
[`datagen/record_scripted.py`](datagen/record_scripted.py) stores every step
as LeRobot data (both cameras, arm state, action, the sentence of the item
being picked). Chains start from training initial states 10 to 49 (0 to 9
are for the test); only chains with every item in the basket are kept, and
a failed chain is retried from the next state, up to 40 states.

All 300 recorded, in about 35 minutes on 16 CPUs (14 recorders in
parallel, 3 per scene), into `/workspace/outputs/week4/pairs30`.

Tall bottles (ketchup, salad dressing) are carried 0.30 m over the basket.
At 0.35 m the elbow straightens to its joint limit, and 13 chains of this
plan (a tall bottle first in scenes 3, 8 and 9) failed from every state.

### 3. Merge with the original demos, keep smolvla_libero's normalization

```
python train/prepare_data.py --chains /workspace/outputs/week4/pairs30/scripted-chains \
  --out /workspace/outputs/week4/train-data/pairs30
python train/use_base_stats.py /workspace/outputs/week4/train-data/pairs30
```

Result: 310 episodes, 10 instructions (one per item).

## Test: 180 episodes

Every episode starts from initial state 0, which no training episode uses,
with up to 300 steps per item.

| Test | Episodes | Command |
| --- | --- | --- |
| Every item in every scene | 60 | `bash eval/run_cross_items.sh` |
| Two items, trained orders | 60 | `SPLIT=test_seen bash eval/run_chain_eval.sh` |
| Two items, held-out orders | 60 | `SPLIT=test bash eval/run_chain_eval.sh` |

The first item of every two-item episode is one more single pick, so the
test has 180 single-pick records in all.

Before fine-tuning (`smolvla_libero` as published):

```
E=/workspace/outputs/week4/eval/base
OUT=$E/single bash eval/run_cross_items.sh
SPLIT=test_seen VIDEO=1 OUT=$E/seen bash eval/run_chain_eval.sh
SPLIT=test VIDEO=1 OUT=$E/heldout bash eval/run_chain_eval.sh
```

## Results

### Before fine-tuning (smolvla_libero as published)

[`results/smolvla-libero`](results/smolvla-libero): `single`, `seen`
(trained orders) and `heldout` (held-out orders), one `results.jsonl` per
scene.

| Test | Result |
| --- | --- |
| Scene's own item | 10/10 |
| Any other item | 0/50: 27 lifted another item (23 of them the scene's own item), 23 lifted nothing |
| Two items, trained orders | 0/60; first item in 10, 17 of 120 items in the basket |
| Two items, held-out orders | 0/60; first item in 10, 13 of 120 items in the basket |

The Week 3 shortcut holds on a layout no run has trained on (Week 3,
state 46: 10/10, 1/50, 24 of 26 wrong lifts). For this model the trained
and held-out orders are the same test, since it never saw any of them.
