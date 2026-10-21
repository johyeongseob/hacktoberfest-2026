# Packing the Whole Picnic From One Lazy Sentence

Hacktoberfest 2026 DEV Challenge, Week 3 (theme: Slow and Steady: build
something with open-weight models or open-source AI that rewards laziness).
Deadline: 2026-10-25 15:59 KST.

Status: planning. Working title.

## Idea

Weeks 1 and 2 asked `lerobot/smolvla_libero` to pack one picnic item at a
time. Week 3 asks for the whole picnic in one sentence ("pack the picnic")
and lets the system do the rest: an open-weight LLM splits the request into
single-item steps, SmolVLA carries out each step in the same LIBERO-Object
scene, and a check after each step retries what failed.

The public LIBERO data has only single-item demos for LIBERO-Object, so the
state "another item is already in the basket" never appears in training.
The harder question is whether the robot can learn that state from its own
successful runs (filtered behavior cloning), and whether it then handles
item orders it was never trained on.

## Questions

1. How often does a chain of two (then three) items succeed with the
   published model, with and without planning and retries?
2. After fine-tuning on the robot's own successful steps, does chained
   success improve for the trained orders, and for held-out orders?
3. How close does this get to LIBERO-10's two-item basket tasks, which were
   trained with full two-item demos?

## Notes

- [Related work](related-work.md)
