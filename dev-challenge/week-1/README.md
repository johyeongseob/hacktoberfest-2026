# Picnic Packing with SmolVLA

Hacktoberfest 2026 DEV Challenge, Week 1 (theme: Touch Grass).

Status: M1 and M2 done. M3 next.

## Idea

Before heading out for a picnic, a person tells a robot which items to pack
from the kitchen. A vision-language-action (VLA) model, SmolVLA, reads the
camera images, the robot state, and the instruction, and outputs continuous
arm actions that pick up each item and place it in a basket. The person then
takes the picnic outside and records how it went.

The robot runs in simulation because there is no physical robot.

## Stack

| Layer | Choice |
| --- | --- |
| VLA model | `lerobot/smolvla_libero` (SmolVLA fine-tuned from `lerobot/smolvla_base` on `lerobot/libero`) |
| Runner | LeRobot 0.6.1 (`lerobot-eval`), wrapped by [benchmark_eval.py](benchmark_eval.py) for inference timing |
| Benchmark | LIBERO-Object (`libero_object`) |
| Simulation | robosuite on MuJoCo |

## Picnic items (LIBERO-Object task IDs)

All ten tasks are "pick up the X and place it in the basket".

| ID | Item | ID | Item |
| --- | --- | --- | --- |
| 0 | alphabet soup | 5 | tomato sauce |
| 1 | cream cheese | 6 | butter |
| 2 | salad dressing | 7 | milk |
| 3 | bbq sauce | 8 | chocolate pudding |
| 4 | ketchup | 9 | orange juice |

## Known limits

- The LIBERO-Object scene places items on a floor, not a kitchen counter.
  The project frames it as a kitchen staging area.
- One item per episode. Packing several items in one episode is outside
  the training distribution.
- Producing actions is not the same as completing the task. Results record
  both separately.
- No training is planned for Week 1. `lerobot/smolvla_libero` is used for
  inference as published.
- The checkpoint was fine-tuned on all ten LIBERO-Object tasks. Evaluation
  measures whether it repeats trained tasks from new initial states, not
  whether it solves unseen tasks. Paraphrased instructions are the unseen
  part.
- CPU inference is slower than real time. See below.

### CPU inference is slower than real time

The robot is controlled at 20 Hz, so each action covers 0.05 s. SmolVLA
predicts a chunk of 50 actions per call, and we execute the first 10 before
looking again (`n_action_steps=10`). One call therefore has to plan 0.5 s of
motion.

On this laptop CPU, one call takes 1.15 s when the machine is otherwise idle
(M1) and 1.81 s median, 2.17 s mean during the long M2 run.

```text
Simulation (time pauses while the model thinks):
  [think 1.15 s][move 0.5 s][think 1.15 s][move 0.5 s] ...
  The simulator waits, so success rates are unaffected.

Real robot (time keeps going):
  [think 1.15 s][move 0.5 s][think 1.15 s][move 0.5 s] ...
   arm stopped              arm stopped
```

On a real arm, the robot would either stop for about 1.15 s after every
0.5 s of motion, moving only about 30% of the time (0.5 / 1.65 s), or keep
going while the next plan is computed from an image that is already more
than a second old. Either way, the motion would be stop-and-go or late to
react, and the 10-step plan would need to be about 2.3 times faster to keep
up.

Ways to close the gap, none tested here:

- Run inference on a GPU or an optimized runtime such as OpenVINO.
- Compute the next chunk while the current one executes (asynchronous
  inference, described in the SmolVLA paper).
- Execute more of each chunk per call. This trades accuracy for speed:
  running all 50 steps costs about 20 points on LIBERO
  ([huggingface/lerobot#4614](https://github.com/huggingface/lerobot/issues/4614)).

In simulation the delay does not change the results, so all success rates
in this project are valid regardless of speed.

## Results

### M1: Local smoke test (October 9, 2026)

Setup: WSL Ubuntu 24.04 on an Intel Core Ultra X7 358H laptop, CPU only,
`--policy.n_action_steps=10`, seed 1000. Command and options are in
[setup.md](setup.md).

| Task | Episodes | Success | Env steps | Episode time |
| --- | --- | --- | --- | --- |
| Ketchup (ID 4) | 1 | 1/1 | 176 of max 280 | 37.7 s |

The saved video was checked by eye: the arm places the ketchup in the basket.
One episode with a fixed seed confirms the pipeline works. It is not a
success rate.

Inference timing from `inference_timing.json`:

| Metric | Value |
| --- | --- |
| SmolVLA calls (one per 10-action chunk) | 18 |
| Time per call, mean (p90) | 1.15 s (1.25 s) |
| Total inference time | 20.6 s |
| Simulation, rendering, and other time in the episode | about 17 s |
| Whole run including model loading | 52.8 s |

On this CPU, one call takes 1.15 s to plan 10 actions, which cover 0.5 s at
the 20 Hz control rate. Inference is about 2.3 times slower than real time,
which is fine in simulation but would not drive a real robot smoothly.

The first run also downloaded about 3.3 GB (SmolVLM2 backbone, SmolVLA
checkpoint, LIBERO assets). Later runs work offline.

### M2: Picnic item success rates (October 9, 2026)

Same setup as M1. All ten LIBERO-Object tasks, 10 episodes each (initial
states 0–9, seeds 1000–1009), original instructions, one `lerobot-eval` run
via [benchmark_eval.py](benchmark_eval.py) from 15:43 to 17:08 local time.

| ID | Item | Success |
| --- | --- | --- |
| 0 | alphabet soup | 10/10 |
| 1 | cream cheese | 10/10 |
| 2 | salad dressing | 9/10 |
| 3 | bbq sauce | 6/10 |
| 4 | ketchup | 8/10 |
| 5 | tomato sauce | 8/10 |
| 6 | butter | 10/10 |
| 7 | milk | 9/10 |
| 8 | chocolate pudding | 9/10 |
| 9 | orange juice | 10/10 |
| | **Total** | **89/100** |

With 10 episodes per item, only large gaps are meaningful. BBQ sauce (6/10)
looks weaker than the 10/10 items, but 8/10 and 9/10 cannot be ranked
against each other.

For reference, [huggingface/lerobot#4614](https://github.com/huggingface/lerobot/issues/4614)
reports 42/50 (84%) on `libero_object` for this checkpoint with
`n_action_steps=10`.

Inference timing from `inference_timing.json`:

| Metric | Value |
| --- | --- |
| Env steps / SmolVLA calls | 15,220 / 1,563 |
| Time per call, mean / median / p90 | 2.17 s / 1.81 s / 2.94 s |
| Slowest call | 59.5 s |
| Total inference time | 3,392 s (67% of the run) |
| Whole run | 5,084 s (84.7 min), 50.7 s per episode |

Calls were slower than in M1 (1.15 s). The laptop was also running other
desktop apps during this long run, and memory use was high (about 27 of
31 GB), so the M1 number is the better estimate of an unloaded CPU. The
59.5 s call is an outlier, likely a system stall rather than the model.

## M3 instructions

Every LIBERO-Object task in the training data uses one template with only the
item name changed:

```text
pick up the {item} and place it in the basket
```

M3 asks whether a person can hand off a picnic item in their own words. If
only the exact template works, the person stays at the screen rephrasing
instead of heading out.

Items: cream cheese (task 1) and butter (task 6). Both succeeded 10/10 with
the original instruction in M2, so a drop with a paraphrase points to the
wording, not to the grasp.

| Style | What it tests | Cream cheese (task 1) | Butter (task 6) |
| --- | --- | --- | --- |
| Original | Baseline from M2 | pick up the cream cheese and place it in the basket | pick up the butter and place it in the basket |
| A. Different verbs | Same structure, new verbs | grab the cream cheese and put it in the basket | grab the butter and put it in the basket |
| B. Short | Minimal wording | cream cheese in the basket | butter in the basket |
| C. Picnic context | Goal stated as a purpose | pack the cream cheese for our picnic | pack the butter for our picnic |
| D. Polite question | Everyday request | could you put the cream cheese in the picnic basket? | could you put the butter in the picnic basket? |

Each paraphrase runs for 10 episodes with the same seeds and initial states
as M2 (2 items x 4 paraphrases x 10 episodes = 80 episodes). The original
instruction is also rerun through [picnic_rollout.py](picnic_rollout.py) to
check that our loop matches the M2 success rate. Individual episodes can
differ from M2 because `lerobot-eval` ran all tasks in one process, so the
random state at the start of each task is not the same. Commands are in
[run_m3.sh](run_m3.sh).

The paraphrases were drafted with AI assistance and reviewed by the author.

## Plan

Submission deadline: October 12, 2026, 06:59 UTC.

| Milestone | Target | Work | Done when |
| --- | --- | --- | --- |
| M0. Plan | Oct 9 | Choose the topic, model, checkpoint, and benchmark. Write this README and [setup.md](setup.md). | Done |
| M1. Local smoke test | Oct 9 | Install in WSL, then run the ketchup task (ID 4) for one episode on CPU. | Done. See [Results](#m1-local-smoke-test-october-9-2026). |
| M2. Picnic item success rates | Oct 9–10 | Run LIBERO-Object tasks 0–9 with N episodes each. | Done: 89/100. See [Results](#m2-picnic-item-success-rates-october-9-2026). Failure videos still to review. |
| M3. Paraphrased instructions | Oct 10 | Run the [M3 instructions](#m3-instructions): 2 items x 4 paraphrases x 10 episodes, plus the original instruction through our own loop. | A table compares original and paraphrased success rates. |
| M4. Real picnic | Oct 10–11 | Pack a real picnic with the same items, go outside, and record it. | Photos and notes from the outdoor use exist. |
| M5. Clean up | Oct 11 | Update results in this README. Review and redact the development session. | The repository shows how to reproduce the results and what they were. |
| M6. DEV post | Oct 11–12 | Write and publish the post with the `#hf26challenge` tag. | The post is published before the deadline, aiming for the morning of Oct 12 (KST). |

If time runs short, cut scope in this order:

1. Reduce M3 to one item and two phrasings.
2. Reduce the number of episodes per task in M2.
