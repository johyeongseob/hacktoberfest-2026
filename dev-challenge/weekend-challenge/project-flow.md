# Shared Workbench Reset Assistant: Project Flow

## Goal

Help one coworker restore a shared workbench after use. Compare a reference photo
with the current layout and suggest a short cleanup checklist. The user reports
that the coworker has trouble remembering objects' original locations. Agree on
the target layout with them; a coworker trial has not yet been completed.

## Workflow

1. **Define the reference:** Use a photo to make original locations easy to check.
2. **Prepare images:** Start with C1 `frame000.png` as the reference and
   `frame004.png` as the current layout. Restore frame004 toward frame000.
3. **Run local AI:** The browser app uses SmolVLM2 2.2B Q8 on the CPU through llama.cpp.
4. **Review observations:** Correct AI positions and orientations against the photos.
5. **Prepare a checklist:** Generate from reviewed observations, then edit and
   approve the final instructions. Never supply expected annotations to inference.
6. **Get feedback:** Let the coworker try it, if feasible, and record actual feedback.
7. **Submit on DEV:** Explain the problem, demo, open AI benefits, limitations,
   dataset attribution, and AI assistance. Optionally link a sanitized DevRelay session.

## Open AI and Project Scope

The open-weight model powers image descriptions and checklist drafting locally. Verify offline operation
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
  The revised run finished without repetition in 22.63 seconds; its checklist
  still contains an incorrect can position and incomplete restoration guidance.
- **Pending:** Coworker feedback, offline verification, and DEV submission.
- **Implemented:** A browser demo with photo selection, editable observations,
  review approvals, checklist drafting, and a reviewed JSON download.
- **Completed:** One full C1 browser workflow in WSL, including reviewed JSON
  export. The local model made observation errors and repeated observations
  instead of generating actions. The coding assistant helped correct the text;
  the user reviewed and approved the final checklist against the photos.
- **Next:** Preserve demo evidence and write the DEV submission with clear
  attribution of model outputs, assisted corrections, and remaining limitations.

