# Picnic Packing with SmolVLA

Hacktoberfest 2026 DEV Challenge, Week 1 (theme: Touch Grass).

Status: planning. Nothing has been run yet.

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
| Runner | LeRobot (`lerobot-eval`) |
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

## Plan

Submission deadline: October 12, 2026, 06:59 UTC.

| Milestone | Target | Work | Done when |
| --- | --- | --- | --- |
| M0. Plan | Oct 9 | Choose the topic, model, checkpoint, and benchmark. Write this README and [setup.md](setup.md). | Done |
| M1. Local smoke test | Oct 9 | Install in WSL, then run the ketchup task (ID 4) for one episode on CPU. | The model loads, actions are produced, the episode result and video are saved, and the run time is recorded. |
| M2. Picnic item success rates | Oct 9–10 | Run LIBERO-Object tasks 0–9 with N episodes each. Use a DigitalOcean GPU Droplet only if CPU runs are too slow, and delete it afterward. | A per-item success rate table and representative success and failure videos exist. |
| M3. Paraphrased instructions | Oct 10 | Replace the original instruction with user-style paraphrases for 2–3 items and 3–4 phrasings. | A table compares original and paraphrased success rates. |
| M4. Real picnic | Oct 10–11 | Pack a real picnic with the same items, go outside, and record it. | Photos and notes from the outdoor use exist. |
| M5. Clean up | Oct 11 | Update results in this README. Review and redact the development session. | The repository shows how to reproduce the results and what they were. |
| M6. DEV post | Oct 11–12 | Write and publish the post with the `#hf26challenge` tag. | The post is published before the deadline, aiming for the morning of Oct 12 (KST). |

If time runs short, cut scope in this order:

1. Reduce M3 to one item and two phrasings.
2. Reduce the number of episodes per task in M2.
