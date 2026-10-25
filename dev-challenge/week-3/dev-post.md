<!--
DEV draft source for the Week 3 post. Edit here, then sync to DEV
(article 4915045, unpublished). Title: My Picnic Robot Memorized the Table, Not My Words
Tags: devchallenge, hf26challenge, robotics, machinelearning
-->

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 3: Slow and Steady](https://dev.to/challenges/hacktoberfest-week3-2026-10-19)*

## What I Built

In [Week 2](https://dev.to/johyeongseob/for-the-price-of-a-coffee-my-picnic-robot-learned-how-i-talk-11n9), fine-tuning worked: one demo per item, relabeled with the ways people actually ask, taught my open robot model ([SmolVLA](https://huggingface.co/lerobot/smolvla_libero)) new phrasings (31% to 80%) for the price of a coffee. It still packed one item at a time.

So this week I pushed the same idea toward the laziest picnic request of all: say one sentence, "grab the cream cheese and the salad dressing", and walk away while the robot puts both in the basket, in that order. If fine-tuning could teach it how I talk, could it teach it a two-item task?

I built the pieces around the model to make that a single request:

| Piece | What it does |
| --- | --- |
| Request splitter | "grab the cream cheese and the salad dressing" becomes two steps, in the order named |
| Step checker | The simulator reports when an item is in the basket, and the instruction moves on to the next item |
| SmolVLA | Does one step at a time from two camera images and the arm state |
| Scripted demo generator | Writes training demos from the simulator's object positions, no teleoperation |

It did not go to plan, and the reasons turned out to be the interesting part.

## Demo

![SmolVLA puts the cream cheese and then the salad dressing in the basket](https://raw.githubusercontent.com/johyeongseob/hacktoberfest-2026/main/dev-challenge/week-3/assets/two_items_creamcheese_then_dressing.webp)

"Pick up the cream cheese ..." then, once it is in, "pick up the salad dressing ...". Neither item is the one this scene was trained on (bbq sauce). The cream cheese lands at step 95, the dressing at step 371 (20 steps per second).

**The honest number: this is 1 success in 80 two-item episodes.** The rest of this post is why.

## Finding 1: the robot memorized the table, not the sentence

Every LIBERO-Object scene has six groceries, but the model only ever learned one task per scene: "pick up {that scene's item} and place it in the basket". It scores 96% on that benchmark. So I asked for each of the six items in each of the ten scenes.

| Published SmolVLA | Success |
| --- | --- |
| The scene's own item | 10/10 |
| Any other item on the same table | **1/50** |

When it failed on another item, it lifted the scene's own item instead in **24 of 26** episodes. It was not reading the item name; it was recognizing the table. I am not the first to see this: [VLA or IL?](https://dev.to/fn8211/vla-or-il-a-controlled-dataset-for-testing-whether-finetuning-turns-your-vla-into-a-fancy-4m6i) by @fn8211 shows the same with π0.5 and summarizes LIBERO-PRO and LIBERO-Plus, which found that fine-tuned models keep the same trajectory when the instruction changes. This is the same failure in a 450M model, measured on every item of every scene.

A single sentence for two items ("put both the cream cheese and the butter in the basket") failed 0/3: the cream cheese, that scene's item, went in every time; the butter never did.

This also explains Week 2. There, every new phrasing still named the scene's own item, so the table alone gave the right answer. Two items in one scene is the first time the item name has to matter.

## Finding 2: breaking the shortcut is not the same as reading

LIBERO-Object has no demos of "pick the other item", so I generated them. A short script reads the object positions from the simulator, moves the gripper over the asked item, grasps it, and places it in the basket. The training set had single picks of all six items in every scene (8 per item), chains of two to four items, and a few original demos; no teleoperation.

![Every item in every scene before and after fine-tuning](https://raw.githubusercontent.com/johyeongseob/hacktoberfest-2026/main/dev-challenge/week-3/assets/cross_items_before_after.png)

| | Published | Fine-tuned on the scripted demos |
| --- | --- | --- |
| Scene's own item | 10/10 | 3 to 5 of 10 (3k, 6k, 9k steps) |
| Other items | 1/50 | 3 to 5 of 50 |
| Wrong lifts that were the scene's own item | 24 of 26 | 12 of 40 (3k), 16 of 41 (6k) |

The fine-tune broke the habit: wrong picks mostly stopped going to the scene's own item. But they did not go to the asked item either. I checked whether a wrong pick at least shared the asked item's kind (box, carton, can, bottle): 6 of 40 and 10 of 41, against 7.6 expected by chance. When it is wrong, the item name plays no role.

**Removing a shortcut did not create the skill it was standing in for.** The model stopped recognizing the table, and nothing took its place.

## Two items, in order

| Model (same 80 episodes: 40 trained item orders, 2 table layouts) | Both items, in order | First item |
| --- | --- | --- |
| Fine-tuned on the first scripted chains (6k) | 3/80 | |
| Fine-tuned on the scripted demos above (9k) | 1/80 | 15/80 |

With picking the right item at about 10%, two in a row is about 1%. The step checker and splitter work; the bottleneck is grounding one item at a time.

## The lazy payoff

| | Effort |
| --- | --- |
| What I wanted | One sentence, then walk away |
| What works today | One sentence, then watch: the robot gets the order right about 1 time in 80 |
| What the pipeline saves already | No teleoperation: 657 training demos written by a script in under an hour on the rented machine's CPUs |

The data side is lazy in the good way. The robot side is not there yet, and I would rather show that than a cherry-picked clip.

## How I Built It

| Part | Choice |
| --- | --- |
| Model | [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero), 450M parameters, LeRobot 0.6.1 |
| Simulation | LIBERO-Object on robosuite and MuJoCo |
| Demos | Scripted pick-and-place from simulator object poses, human pace, failed attempts discarded |
| Fine-tuning | Whole model (vision encoder frozen), batch 32, lr 2.5e-5, 9,000 steps, about 75 minutes |
| Evaluation | Every item in every scene (60 episodes), two-item chains (80 episodes), failure kinds from the lift height of every object |
| Hardware | 1x A100 80 GB on RunPod ($1.82/h) |

The request splitter is the smallest part:

```python
def split_request(request: str) -> list[str]:
    text = request.lower()
    names = {item: item for item in ITEMS} | SHORT_NAMES
    found = []  # (start, end, item)
    # Longest names first, so "cream cheese" is taken before "cheese".
    for name in sorted(names, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(name)}\b", text):
            if not any(match.start() < end and start < match.end() for start, end, _ in found):
                found.append((match.start(), match.end(), names[name]))
    items = []
    for _, _, item in sorted(found):
        if item not in items:
            items.append(item)
    return items
```

`split_request("grab the cream cheese and the salad dressing")` returns `["cream cheese", "salad dressing"]`; each item becomes "pick up the {item} and place it in the basket", the sentence the model was trained on.

**Limits:** all numbers are from one training layout per scene (state 46) or two (46 and 47); simulation only; one training run per setting.

## Code

{% embed https://github.com/johyeongseob/hacktoberfest-2026 %}

| Where | What |
| --- | --- |
| [`week-3/datagen`](https://github.com/johyeongseob/hacktoberfest-2026/tree/main/dev-challenge/week-3/datagen) | Scripted demo generator and recorder |
| [`week-3/eval`](https://github.com/johyeongseob/hacktoberfest-2026/tree/main/dev-challenge/week-3/eval) | Step-by-step rollout, request splitter, every-item-in-every-scene test |
| [`week-3/train`](https://github.com/johyeongseob/hacktoberfest-2026/tree/main/dev-challenge/week-3/train) | Training-set builder, normalization fix, training script |

## Why Does Open Innovation Matter?

| Open piece | What it made possible |
| --- | --- |
| Open weights | I could fine-tune the model and inspect every checkpoint |
| Open simulator and benchmark | The simulator knows where every object is, so a script could write demos no one recorded |
| Open data | The original LIBERO demos are there to mix back in against forgetting |
| Open training code | Reading LeRobot's training loop let me check what fine-tuning changes, down to the normalization statistics |
| Open discussion | Others had already measured the same shortcut, so I knew what I was looking at |

A closed model would have given me a success rate. Open pieces let me find out why the robot fails.

## What's Next

Grounding is the missing skill, and a detector already has it. Next I want to mark the asked item on the camera image (first with the simulator's own boxes, then with an open-vocabulary detector) and see whether SmolVLA learns to follow the mark.

## Notes from the Bench

Things I tried along the way that did not make the main story.

**1. My scripted demos had hidden pauses.** The script moved the gripper in proportion to the remaining distance, and some targets were out of reach: the fingers stopped against a tall bottle or the basket floor, and the script kept nudging until its step limit. The average pace still matched the human demos (about 146 steps per item), so success rates and episode lengths looked fine.

![Frame-to-frame motion in a scripted demo before and after the fix](https://raw.githubusercontent.com/johyeongseob/hacktoberfest-2026/main/dev-challenge/week-3/assets/demo_stillness.png)

A "steady" version ends a move once the gripper stops getting closer. After retraining on it, "picked up nothing" dropped from 34 to 18 of 60 at 3,000 steps (Fisher's exact test, p ≈ 0.006; the published model: 22). The steady set also had more single picks (about 450 vs 168), so this is the likely cause, not an isolated one. @ankk98 saw the same in teleop data: [idle footage teaches a policy to linger](https://dev.to/ankk98/teleoperation-data-quality-for-imitation-learning-what-actually-breaks-the-model-1abc). Slow and steady, it turns out, has to be steady.

**2. Normalization statistics were not the culprit.** LeRobot replaces a checkpoint's state and action statistics with the new dataset's when fine-tuning. Keeping the original statistics barely mattered here (own item 2/10 either way at 3,000 steps), but I keep them anyway.

**3. Co-training with the original demos.** *(result pending)*

---

**Disclosure:** I set the goal, ran every training and evaluation command, and reviewed every result. An AI coding agent (Claude Code) wrote the scripts with me, analyzed the results, made the charts, and drafted this post.
