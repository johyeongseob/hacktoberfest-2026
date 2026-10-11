# Paraphrased instructions

<p align="center">
  <b>Alphabet soup</b><br>
  <img src="../assets/paraphrases/paraphrases_alphabet_soup.webp" alt="Alphabet soup: original instruction and paraphrases A to D">
</p>

<p align="center">
  <b>Cream cheese</b><br>
  <img src="../assets/paraphrases/paraphrases_cream_cheese.webp" alt="Cream cheese: original instruction and paraphrases A to D">
</p>

<p align="center">
  <b>Butter</b><br>
  <img src="../assets/paraphrases/paraphrases_butter.webp" alt="Butter: original instruction and paraphrases A to D">
</p>

<p align="center">
  <b>Orange juice</b><br>
  <img src="../assets/paraphrases/paraphrases_orange_juice.webp" alt="Orange juice: original instruction and paraphrases A to D">
</p>

One row per item, left to right: the original training sentence and
paraphrases A to D. Each tile shows the first successful episode out of the
ten episodes for that instruction, or episode 0 when all 10 failed.

## Question

Can a person ask the robot to pack a picnic item in their own words? Every
LIBERO-Object task in the training data uses one template with only the item
name changed:

```text
pick up the {item} and place it in the basket
```

If only the exact template works, the person stays at the screen rephrasing
instead of heading out. This experiment rewords the instruction in four ways
for four picnic items, 10 episodes each: the original template succeeds
**40/40 (100%)**, the paraphrases far less often:

| Original | A. grab ... put | B. {item} in the basket | C. pack ... for our picnic | D. could you put ...? |
| --- | --- | --- | --- | --- |
| **40/40 (100%)** | 22/40 (55%) | 2/40 (5%) | 2/40 (5%) | 26/40 (65%) |

Across all four paraphrases: 52/160 (33%). Everything ran on a laptop CPU
(Intel Core Ultra X7 358H, WSL Ubuntu 24.04) without a GPU.

## Instruction variants

### Parts of the original instruction

```text
pick up    the {item}   and   place it   in the basket
[action 1] [object]           [action 2] [destination]
```

### What each paraphrase changes

The object (item name) is kept in every paraphrase.

| Style | Instruction | Action 1 | Action 2 | Destination | Change |
| --- | --- | --- | --- | --- | --- |
| Original | pick up the {item} and place it in the basket | pick up | place | in the basket | none |
| A | grab the {item} and put it in the basket | **grab** | **put** | in the basket | both verbs swapped, structure kept |
| B | {item} in the basket | **removed** | **removed** | in the basket | no verb at all |
| C | pack the {item} for our picnic | **pack** | **removed** | **for our picnic** (a purpose) | one new verb; destination replaced by a purpose |
| D | could you put the {item} in the picnic basket? | **removed** | put | in the **picnic** basket | question form ("could you"), single verb, "picnic" added |

Each paraphrase changes more than one part at once. C swaps the verb and
replaces the destination with a purpose, D combines a question form, a dropped
first verb, and an added word, and A swaps both verbs. The results can show
which wordings fail, but not which single change causes the failure.

**Note:** The paraphrases were drafted with AI assistance and reviewed by the author.

### Items

Alphabet soup (task 0), cream cheese (task 1), butter (task 6), and orange
juice (task 9), the four items that succeeded 10/10 with the original
instruction in the [baseline](../baseline/README.md). A drop with a
paraphrase therefore points to the wording, not to the grasp.

## Results (October 9–10, 2026)

Same model settings as the baseline, run through
[picnic_rollout.py](picnic_rollout.py): 4 items x 5 instructions x 10
episodes = 200 episodes, with the same initial states and seeds for every
instruction. The original instruction is rerun through the same loop as the
control. The commands for all four items are in
[run_paraphrases.sh](run_paraphrases.sh).

Success out of 10 episodes per item and instruction:

| Instruction | 0: alphabet soup | 1: cream cheese | 6: butter | 9: orange juice | All |
| --- | --- | --- | --- | --- | --- |
| Original: pick up the {item} and place it in the basket | 10/10 | 10/10 | 10/10 | 10/10 | **40/40** |
| A. grab the {item} and put it in the basket | 6/10 | 3/10 | 10/10 | 3/10 | 22/40 |
| B. {item} in the basket | 2/10 | 0/10 | 0/10 | 0/10 | **2/40** |
| C. pack the {item} for our picnic | 0/10 | 0/10 | 2/10 | 0/10 | **2/40** |
| D. could you put the {item} in the picnic basket? | 7/10 | 3/10 | 10/10 | 6/10 | 26/40 |
| **All paraphrases** | 15/40 | 6/40 | 22/40 | 9/40 | **52/160** |

Every failure ran to the 280-step limit.

What the numbers show:

- The original template works for all four items (40/40, matching the
  baseline). Any rewording lowers the success rate: 52/160 (33%) across all
  paraphrases.
- B (no verb) and C (a new verb, no basket) almost never work: 2/40 each.
- A and D keep the "put ... in the basket" part and land in between (22/40
  and 26/40), but with large item differences. Butter is 10/10 on both;
  cream cheese is 3/10 on both. The same wording can work for one item and
  fail for another.
- With 10 episodes per cell, only large gaps are meaningful. The 2/40
  results and the butter vs cream cheese gap are large; differences of one
  or two episodes are not.

## Why do paraphrases fail?

The causes below are hypotheses; the runs above cannot tell which of them is
responsible.

| Cause | Likelihood | Evidence |
| --- | --- | --- |
| One fixed sentence per task in the training data | High | Every item fails once the wording leaves the template; the model may treat the sentence as a task ID rather than as meaning |
| The language model was fine-tuned on the same 40 sentences | High | The checkpoint config has `train_expert_only: False`, so the SmolVLM2 language layers were trained too and may have lost general English |
| Small model size | Medium | The language part is about 252M parameters and uses 16 of SmolVLM2's 32 layers; a larger backbone may map "grab" or "pack" closer to "pick up" |
| Unseen words and shifted word positions | Medium | A and D keep "put ... in the basket" and succeed about half the time; B moves the item name to the start, and C, D add words such as "pack", "could you", and "picnic" that never appear in training |

Running the same instructions on a larger LIBERO checkpoint, such as
`lerobot/pi05_libero_finetuned` (about 3B parameters), would test the model
size hypothesis.

## Limitation: CPU inference is not real time

Each SmolVLA call plans 0.5 s of motion but took 2.07 s on average here
(4,448 calls). The simulator waits, so success rates are unaffected, but a
real arm would stop and go, as shown in the
[baseline smoke test](../baseline/README.md#smoke-test-october-9-2026).
Real-time control would need faster inference (for example, a GPU) and
asynchronous inference; neither was tested.
