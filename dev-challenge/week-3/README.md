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

## Plan

| Step | Work | Date |
| --- | --- | --- |
| M0 | Run the three LIBERO-10 two-item tasks with `smolvla_libero` (test states 0 to 9) | 10/22 |
| M1 | Chain items in one LIBERO-Object scene without a reset; list which picnic items share a scene (up to four?); measure success of step 2, 3, and 4 | 10/22-23 |
| M2 | LLM planner (Ollama): sentence to steps, items not in the scene reported; check and retry after each step | 10/23 |
| M3 | Test chains of two to four items: direct sentence vs planner vs planner + retry | 10/23-24 |
| M4 | On a rented GPU: collect successful later steps (2 to 4) on training states, fine-tune, test chains of two to four items in trained and held-out orders | 10/24 |
| M5 | Results, videos, DEV post | 10/25 by 15:59 |

M0 to M3 run on the laptop and are the result that ships even if M4 does
not finish.

### Fine-tuning for long chains

- **Start states for later steps:** step k starts with k-1 items already in
  the basket. Besides real chained runs, these states can be made in the
  simulator by placing the earlier items in the basket and letting physics
  settle, so data for steps 3 and 4 does not depend on the robot first
  succeeding k-1 times in a row.
- **Data:** only steps the simulator marks as successful, each labeled with
  its own step instruction (plus Week 2 style rewordings), item orders
  balanced, and some original single-item demos mixed in against
  forgetting.
- **Splits:** test states 0 to 9 are never used for collection; some item
  orders are held out for testing only.
- **Rounds:** if time allows, collect again with the fine-tuned model and
  train once more.
