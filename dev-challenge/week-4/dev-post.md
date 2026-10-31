<!--
DEV draft source for the Week 4 post. Edit here, then sync to DEV.
Title: My Picnic Robot Is Haunted by the Table It Trained On
Tags: devchallenge, hf26challenge, robotics, machinelearning
Series: Hacktoberfest 2026: Exploring Physical AI
-->

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 4: Ghost in the Machine](https://dev.to/challenges/hacktoberfest-week4-2026-10-26)*

## What I Built

Last week I found a ghost in my picnic robot. Ask the open robot model [SmolVLA](https://huggingface.co/lerobot/smolvla_libero) for any item on a LIBERO-Object table, and it reaches for the one item that table was trained on, whatever the sentence says. In [Week 3](https://dev.to/johyeongseob/my-picnic-robot-memorized-the-table-not-my-words-1dd5) more demos broke that habit but did not teach the robot to read the item name.

This week I tried to starve it out with a clean experiment: a new training set where every item of every table is picked equally often, and a test on table layouts and item orders the robot never trained on. If the data stops feeding the ghost, does it leave, and does the robot start reading the item name instead?

## Demo

![A scripted training demo: ketchup, then salad dressing](https://raw.githubusercontent.com/johyeongseob/hacktoberfest-2026/main/dev-challenge/week-4/assets/scripted_demo_ketchup_then_dressing.webp)

One of the 300 scripted training demos: "pick up the ketchup …", then, once it is in, "pick up the salad dressing …". A script writes these from the simulator's object positions, with no teleoperation. This is what the robot learned from; what it did with it is below.

## The ghost, measured again

Every LIBERO-Object table holds six groceries, but the published model learned one task per table. I asked for each of the six items on each of the ten tables, from a layout no training run has used (initial state 0):

| `smolvla_libero`, before fine-tuning | Result |
| --- | --- |
| The table's own item | 10/10 |
| Any other item on the same table | **0/50** |
| Wrong lifts that were the table's own item | **23 of 27** |

Same ghost as Week 3 (10/10, 1/50, 24 of 26 there), on a layout it had never seen.

## Starving the ghost

The training set gives the table nothing to memorize:

| Part | Episodes |
| --- | --- |
| Single picks: every item of every table once (10 tables × 6 items) | 60 |
| Two-item chains: 24 of the 30 ordered pairs per table | 240 |
| Original LIBERO human demos, one per table | 10 |

The six pairs left out of each table (A→B, B→C, …, F→A) are the held-out test; their reverses (B→A, …) are a trained-order control. All scripted demos run at the same "steady" pace, from training layouts 10 to 49 only.

![Every item in every scene, before and after fine-tuning](https://raw.githubusercontent.com/johyeongseob/hacktoberfest-2026/main/dev-challenge/week-4/assets/cross_items_before_after.png)

*Left: before fine-tuning, successes sit only on the outlined diagonal, the item each table was trained on. Right: after fine-tuning on the 310 demos, they leave the diagonal but stay rare and scattered. Rows are tables, columns the item asked for, one episode per cell.*

| Test (180 episodes, unseen layout) | Before fine-tuning | Fine-tuned |
| --- | --- | --- |
| The table's own item | 10/10 | 2/10 |
| Any other item | 0/50 | 3/50 |
| Wrong lifts that were the table's own item | 23 of 27 (85%) | 10 of 28 (36%) |
| Two items, trained orders | 0/60 | 1/60 |
| Two items, held-out orders | 0/60 | 1/60 |

**The ghost left.** A wrong pick is no longer the table's item 85% of the time; at 36% it is close to picking among the five other items at random (20%).

**Nothing moved in.** The asked item is found 3 times in 50, and the robot also lost the one thing it did well: its own item fell from 10/10 to 2/10. Trained and held-out orders score the same, so no order was memorized; picking the right item is the bottleneck, and two in a row is luck.

This repeats Week 3's finding with a cleaner experiment: uniform demos, one pace, and a test that holds out layouts and orders. More balanced data removes the shortcut; it does not create the skill the shortcut stood in for.

## Notes from the Bench: a ghost in my own script

Recording the 300 demos, 13 chains failed from every one of the 40 training layouts, all of them asking for a tall bottle first. The arm reached the bottle, carried it, and then grabbed air at the next item.

Logging the joints found it: carrying a tall bottle 0.35 m above the basket straightened the elbow to its joint limit (joint 4 at -4°). With one joint pinned, the controller gave up orientation to keep position, the gripper turned about 17.5°, and the arm could not pull back toward its base. A phantom twist, invisible in the success logs. Carrying tall bottles at 0.30 m fixed all 13, and the tall-bottle demos went from about 300 steps to about 145, the same as the human ones. The Week 3 data had the same quiet slowdown in it.

## How I Built It

| Part | Choice |
| --- | --- |
| Model | [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero), 450M parameters, LeRobot 0.6.1 |
| Simulation | LIBERO-Object on robosuite and MuJoCo |
| Demos | Scripted pick-and-place from simulator object poses, one pace, failed attempts discarded and retried from the next layout |
| Fine-tuning | Whole model (vision encoder frozen), batch 32, lr 2.5e-5, 9,000 steps, about 75 minutes on one A100 |
| Test | 180 episodes from an unseen layout: every item in every table (60), two items in trained (60) and held-out (60) orders; failure kinds from the lift height of every object |
| Inference | 0.47 s per model call (10 actions) with ten tests sharing one GPU |

**Limits:** one training run per setting, one test layout, simulation only, and the test is small: 3/50 against 0/50 is a direction, not a precise rate.

## Code

{% embed https://github.com/johyeongseob/hacktoberfest-2026 %}

Everything for this week is in [`dev-challenge/week-4`](https://github.com/johyeongseob/hacktoberfest-2026/tree/main/dev-challenge/week-4): the plan, the scripted demo recorder, the training-set builder, the test, and the results as one line per episode.

## Why Does Open Innovation Matter?

| Open piece | What it made possible |
| --- | --- |
| Open simulator (LIBERO, robosuite, MuJoCo) | It knows where every object is, so a script writes demos no one recorded, and I could log the arm's joints to find the phantom twist |
| Open data | The original LIBERO demos are there to mix in, one per table |
| Open weights and training code | I could fine-tune the model and test it on any item, table and order I chose |
| Open discussion | Others had measured the same shortcut (linked in Week 3), so I knew what kind of ghost I was looking at |

A closed model would have given me a success rate. Open pieces let me find the ghost, starve it, and see what was left.

## What's Next

Balanced data removed the shortcut but did not teach the robot to read the item name. Next I will try helping it find the asked item on the camera image, and I will keep updating this post after the deadline.

---

**Disclosure:** I set the goal, ran every recording, training and test command, and reviewed every result. An AI coding agent (Claude Code) wrote the scripts with me, analyzed the results, made the charts, and drafted this post.
