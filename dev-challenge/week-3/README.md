# Packing the Whole Picnic From One Lazy Sentence

Hacktoberfest 2026 DEV Challenge, Week 3 (theme: Slow and Steady: build
something with open-weight models or open-source AI that rewards laziness).
Deadline: 2026-10-25 11:59 PM PDT (2026-10-26 15:59 KST).

Status: planning. Working title.

## Idea

Weeks 1 and 2 asked `lerobot/smolvla_libero` to pack one picnic item at a
time. Week 3 asks for several items in one sentence ("grab the butter and
the ketchup") and lets the system do the rest: an open-weight LLM splits
the request into single-item steps (and says which items are not in the
scene), SmolVLA carries out each step in the same LIBERO-Object scene, and
a check after each step retries what failed.

The public LIBERO data has only single-item demos for LIBERO-Object, so the
state "another item is already in the basket" never appears in training.
The harder question is whether the robot can learn that state from its own
successful runs (filtered behavior cloning), and whether it then handles
item orders it was never trained on.

## Related work

LIBERO-10 already has two-item basket tasks with full two-item demos, and
`smolvla_libero` was trained on them:

- put both the alphabet soup and the cream cheese box in the basket
- put both the cream cheese box and the butter in the basket
- put both the alphabet soup and the tomato sauce in the basket

These run first, as the reference for what full two-item demos give.

## Found so far

- **One sentence for two items fails.** "put both the cream cheese and the
  butter in the basket" in the cream cheese scene: 0/3; the cream cheese
  goes in every time, the butter never.
- **Step-by-step instructions fail too.** Switching to "pick up the butter
  ..." after the cream cheese is in: 0/3. Butter first: 0/3, the butter
  never goes in.
- **The model packs the scene, not the sentence.** Asked for each of the
  six items in each of the ten LIBERO-Object scenes (training state 10,
  one episode each): the scene's own item 8/10, any other item 1/50; the
  one success did not repeat (0/10 on states 11 to 20).
- **A scripted demo can do it.** Using the object positions the simulator
  knows, a few lines of control put one, two, and three items in the
  basket in the cream cheese scene ([datagen/scripted_pick.py](datagen/scripted_pick.py)).

So the robot cannot collect its own successes yet (2%). Week 3 starts from
scripted demos instead, then lets the fine-tuned robot collect its own.

## Plan

| Step | Work | Where | Status |
| --- | --- | --- | --- |
| M0 | LIBERO-10 two-item tasks with `smolvla_libero` (reference) | Laptop | Tried, one episode each |
| M1 | Chains in one LIBERO-Object scene: one sentence and step by step; every item in every scene | Laptop, GPU | Done (above) |
| B1 | Scripted pick of every item in every scene on training states; fix what fails | GPU pod | |
| B2 | Save scripted episodes in the `lerobot/libero` format; replay one to check | Laptop, then pod | |
| B3 | Generate chains of two to four items in mixed orders, step labels; hold some orders out | GPU pod | |
| B4 | Fine-tune `smolvla_libero` on them plus one original demo per item (N = 1) | GPU pod | |
| B5 | Test states 0 to 9: other items alone, chains of two to four (trained and held-out orders), original items | GPU pod | |
| C | Fine-tuned robot collects its own successes; train again | GPU pod | If time allows |
| M5 | Results, videos, DEV post (a first version by 10/25 15:59 KST) | | |
