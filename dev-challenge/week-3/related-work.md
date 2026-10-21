# Related work

Survey made on 2026-10-20 before starting Week 3. Numbers are as reported by
each paper, not reproduced here.

## Long-horizon manipulation is a known problem

Chaining single-step skills fails mostly at the hand-off between steps:
errors add up, and each step starts from a state the skill never saw.

| Work | What it does | Relation to Week 3 |
| --- | --- | --- |
| [LIBERO](https://libero-project.github.io/) (Liu et al., NeurIPS 2023) | Benchmark; LIBERO-10 (LIBERO-Long) has two-step tasks with about 50 human demos each | Source of the scenes. Its two-item basket tasks are our reference point (below) |
| [Long-VLA](https://arxiv.org/abs/2508.19958) (Fan et al., CoRL 2025) | End-to-end VLA for long tasks; masks inputs by phase (moving vs interacting); L-CALVIN benchmark | Same failure mode (skill chaining), solved inside the model instead of with a planner |
| [LoHoVLA](https://arxiv.org/abs/2506.00411) (2025) | One VLM outputs both the next sub-task in words and the actions; re-plans after repeated failures | Planner and executor in one model; ours keeps them separate |
| [SeqVLA](https://arxiv.org/abs/2509.14138) (2025) | Detects when each sub-task is complete, since wrong detection cascades | Our step check plays this role, with the simulator's success signal |
| [LiLo-VLA](https://arxiv.org/abs/2602.21531) (Yang et al., 2026) | Splits reaching from object-centric interaction; LIBERO-Long++ and Ultra-Long suites; reports 69% vs much lower for pi0.5 and OpenVLA-OFT | Shows strong VLAs still drop on chained LIBERO tasks |
| [BATON](https://arxiv.org/abs/2608.16889) (Xu et al., 2026) | LLM coding agent drives a frozen VLA; explores each sub-task separately and keeps transition-aware memory with a verifier | Closest system design to ours (agent + frozen VLA + verifier), on RoboMemArena |
| [BEHAVIOR Challenge solution](https://arxiv.org/abs/2512.10071) (2025) | Competition report with openpi | Notes that many skill-chaining fixes conflict with end-to-end VLA training |

## LLM planner over robot skills

| Work | What it does | Relation to Week 3 |
| --- | --- | --- |
| SayCan (Ahn et al., 2022) | LLM proposes the next skill, weighted by how feasible each skill is | The original planner + skills idea |
| [Inner Monologue](https://arxiv.org/abs/2207.05608) (Huang et al., CoRL 2022) | Feeds success detection and scene descriptions back into the LLM prompt; retries and re-plans; no LLM fine-tuning | Our verify + retry loop is a small version of this |

## Learning new chains from the robot's own experience

| Work | What it does | Relation to Week 3 |
| --- | --- | --- |
| [BOSS](https://arxiv.org/abs/2310.10021) (Zhang et al., CoRL 2023) | Starts from primitive skills; an LLM suggests which skills to chain; the agent practices chains and learns them, then runs unseen long tasks zero-shot | Closest learning setup: LLM-guided chaining + self-collected data. Ours uses a modern VLA, LIBERO, and held-out orders |
| [Autonomous Improvement of Instruction Following Skills](https://auto-improvement.github.io/) (2024) | A VLM chooses skills to practise and filters the robot's own data for successes | Same filtered-data idea; real robot, not LIBERO |
| [PLD: Self-Improving VLA via Residual RL](https://arxiv.org/abs/2511.00091) (Xiao et al., ICLR 2026) | Frozen VLA + small residual RL actors take over on failure; the collected data is distilled back with SFT; about 99% on LIBERO | Self-improvement on LIBERO, but with RL; ours keeps only plain successful rollouts |

## Making demos instead of collecting them

| Work | What it does | Relation to Week 3 |
| --- | --- | --- |
| [MimicGen](https://arxiv.org/abs/2310.17596) (Mandlekar et al., CoRL 2023) | Splits human demos into object-centric segments and replays them, transformed, in new scenes; about 50K demos from about 200 | Fallback if the robot's own success rate is too low to collect data |
| [SkillMimicGen](https://arxiv.org/abs/2410.18907) (Garrett et al., CoRL 2024) | Skill segments joined by motion planning; 24K demos from 60 | Same fallback, closer to chaining |

## Reference point inside our own data

`lerobot/libero` (40 tasks) already has three "put both ... in the basket"
tasks from LIBERO-10, two of them with picnic items:

- put both the alphabet soup and the cream cheese box in the basket
- put both the cream cheese box and the butter in the basket
- put both the alphabet soup and the tomato sauce in the basket

Both `lerobot/smolvla_libero` and `RLWRLD/RLDX-1-FT-LIBERO` were trained on
these. They are a different scene (a living-room table) with a fixed task,
so they serve as an upper reference: what full two-item demos give.

## Where Week 3 sits

Chaining skills, an LLM planner, verify and retry, and learning from
self-collected successes have all been studied. We did not find a public
study that, with a small open VLA (SmolVLA, 450M) on LIBERO-Object and a
budget of a few dollars:

- chains single-item skills that were never demonstrated together,
- fine-tunes on the robot's own successful steps only, and
- reports success separately for trained and held-out item orders.

This needs a broader search before it is written as a claim; for the post,
"a known hard problem, tried with a small open model on a coffee budget"
is the safe framing.
