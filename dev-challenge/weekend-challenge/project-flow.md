# Shared Workbench Reset Assistant: Project Flow

## Goal

Help one coworker restore a shared workbench after use. Compare a reference photo
with the current layout and suggest a short cleanup checklist. Confirm the
coworker's actual difficulty and preferred layout before claiming the problem
is solved.

## Workflow

1. **Confirm the need:** Ask the coworker what makes cleanup difficult.
2. **Prepare images:** Start with C1 `frame000.png` as the reference and
   `frame004.png` as the current layout. Restore frame004 toward frame000.
3. **Run local AI:** Use SmolVLM2 on the CPU with `python compare_layout.py`.
4. **Review the checklist:** Check object recognition, movement directions,
   orientation, and uncertainty against the images and `expected-result.json`.
   Never provide expected annotations to the model.
5. **Complete the demo:** Show the images and checklist in a simple usable workflow.
6. **Get feedback:** Let the coworker try it, if feasible, and record actual feedback.
7. **Submit on DEV:** Explain the problem, demo, open AI benefits, limitations,
   dataset attribution, and AI assistance. Optionally link a sanitized DevRelay session.

## Open AI and Project Scope

The open-weight model powers image comparison locally. Verify offline operation
before claiming it, and explain why local processing and editable behavior help
this coworker. This is a VLM prototype that suggests actions for human review;
robot control and VLA integration are future work.

TTU samples are demo data, not company workbench photos. Expected annotations are
qualitative review aids, not official dataset labels. See `README.md` for setup
and dataset attribution.

## Current Status

- **Ready:** WSL environment, installed packages, downloaded model, four demo
  sequences, expected annotations, and comparison script.
- **Verified:** C1 inference runs on the CPU (30.20 seconds for the first response),
  but the initial response repeats contradictory statements and is unusable.
- **Pending:** A useful checklist, usable handoff, coworker
  feedback, offline verification, and DEV submission.
- **Next:** Run `python compare_layout.py` in the Weekend Challenge directory
  with the revised prompt and review the C1 result saved under `results/`.

