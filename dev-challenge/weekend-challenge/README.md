# Shared Workbench Reset Assistant

A Weekend Challenge prototype for a coworker who has trouble remembering where
objects originally belonged on a shared workbench. Keep the reference and current
photos side by side, review local AI observations, and draft a cleanup checklist
from the corrected observations. A coworker trial has not yet been completed.

See [Project Flow](project-flow.md) for the end-to-end workflow, deliverables,
and current progress.

## Run the Browser Demo

The demo uses the downloaded SmolVLM2 2.2B Q8 model through a local llama.cpp
server. It does not use PyTorch for inference. Run both services in WSL and keep
both terminals open. No hosted inference API key is required.

Download the browser demo's Q8 model and vision projector from the repository root:

```bash
source ~/.venvs/hacktoberfest-2026/bin/activate
cd /path/to/hacktoberfest-2026
hf download ggml-org/SmolVLM2-2.2B-Instruct-GGUF \
  SmolVLM2-2.2B-Instruct-Q8_0.gguf \
  mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf \
  --local-dir models/SmolVLM2-2.2B-Instruct-GGUF
```

Skip this download if both files are already present. The 500M setup below is
for the earlier PyTorch experiments, not the browser demo.
Replace `/path/to/hacktoberfest-2026` with your repository path in WSL.

First, build the server using the existing llama.cpp checkout and install the UI:

```bash
source ~/.venvs/hacktoberfest-2026/bin/activate
cmake --build ~/tools/llama.cpp/build --config Release -j 8 --target llama-server
python -m pip install 'streamlit>=1.40,<2'
python -m pip check
```

Terminal 1: start the model server and wait until it reports that it is listening:

```bash
cd /path/to/hacktoberfest-2026/dev-challenge/weekend-challenge
bash start_model_server.sh
```

Terminal 2: start the browser interface:

```bash
source ~/.venvs/hacktoberfest-2026/bin/activate
cd /path/to/hacktoberfest-2026/dev-challenge/weekend-challenge
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 \
  --browser.gatherUsageStats false
```

Open `http://localhost:8501` in your Windows browser. Choose the C1 demo or upload
two photos, click **Analyze both photos**, correct the observations against the
images, and confirm them. Then draft, edit, and confirm the final checklist. The
app resets approvals and checklist state when the photos or observations change.
Download the reviewed result if you want to preserve the experiment.

Uploads are normalized in memory and sent to the fixed localhost model endpoint.
The app does not write uploaded photos to the repository. Downloads include AI
text and user edits without embedded photos or machine paths; inspect free text
for private information before sharing. The services bind to loopback addresses.

### What Is Ready and What Still Needs Validation

The C1 browser workflow was completed in WSL: local image analysis, observation
editing and approval, checklist drafting, final editing and approval, and JSON
export. The user's screenshots document this run:
[Photo comparison](screenshots/reference-and-current.png),
[reviewed observations](screenshots/reviewed-observations.png), and
[reviewed checklist](screenshots/reviewed-checklist.png).
The reviewed export is kept locally as
`results/C1-template_00000-traj_00000-reviewed-result.json`.
The entire `results/` directory is excluded from Git; it is not part of the
public repository. Earlier committed results may remain in Git history.

The model's observations included incorrect cup orientation, can color, and
remote-control face orientation. Its checklist repeated mixed observations
instead of giving restoration instructions. The corrected observations and final
checklist were supplied with assistance from the coding assistant and confirmed
by the user against the photos. They are not successful autonomous outputs from
the local model. The export preserves both raw AI responses and reviewed text.

User review is part of the workflow; this is not autonomous cleanup or a
robot-control system. The app never reads expected-result annotations to generate
observations or checklists. Other samples and personal-photo uploads have not
been validated through the complete browser workflow.

Before submitting, upload the screenshots to DEV, add a demo video or deployed
link as requested by the submission template, and explain these limitations in
the DEV post. If possible, ask the
coworker to try it and record actual feedback. Test with internet disconnected
before claiming offline execution. DEV signup is separate from project submission.

Open innovation matters here because the open-weight model performs image
analysis locally, the prompt and review workflow are editable, and the reference
photo can remain on the user's laptop. The demo has no hosted inference API fee;
setup, hardware, and electricity still have costs. These benefits do not establish
better accuracy than a closed model or a validated offline deployment.

References: [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[Q8 model files](https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/tree/main).

## Local Model Setup

Run these commands in WSL Ubuntu 24.04 with the project virtual environment active.
From this challenge directory, install compatible CPU PyTorch and torchvision
wheels together before the remaining packages:

```bash
python -m pip install --upgrade pip
python -m pip install 'torch>=2.6,<3' torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

This initial setup uses CPU inference with individual image inputs. CUDA,
FlashAttention, and video decoding packages are not required for this workflow.
However, the SmolVLM processor imports torchvision even when only still images
are supplied. Let pip resolve a compatible torch/torchvision pair from the CPU
wheel index; do not select their versions independently.
Dependency ranges are an initial setup configuration, not a verified lockfile.
Record the resolved versions after the first successful inference run.

Download the model into the shared `models/` directory at the repository root.
The following command assumes you are in this challenge directory in WSL:

```bash
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --local-dir ../../models/SmolVLM2-500M-Video-Instruct \
  --exclude 'onnx/*'
```

The repository-root `models/` directory is excluded from Git and shared across
the month's projects. The inference script locates it relative to its own file.
Downloading the
files alone does not validate model loading or inference.

References: [Model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).

## First Comparison

From this challenge directory in WSL, run:

```bash
python compare_layout.py
```

The script loads the downloaded model locally on the CPU in float32 and compares
the C1 sample's reference (`frame000.png`) and current (`frame004.png`) images.
It does not read `expected-result.json`. This first run uses two still images;
the three intermediate frames are not inputs yet. No robot actions are executed.

The response, prompt, image paths, package versions, and timing are saved to
`results/C1-template_00000-traj_00000-comparison.json`. Repeating the same sample
overwrites that result. Review the response against the images and expected
annotations, especially movement directions and cup handle orientation. Model
output is an unverified suggestion, not ground truth. Successful installation
does not establish accuracy; loading and inference still need to be tested.

Use `--sample D10-template_00000-traj_00000` to select another demo, or
`--max-new-tokens 384` if a response reaches the default token limit.

## Demo Samples

### Q8 Position and Orientation Diagnostic

With the Q8 model and projector downloaded and `llama-mtmd-cli` built in
`~/tools/llama.cpp/build/bin/`, run from this challenge directory:

```bash
bash describe_layout_q8.sh
```

The script asks for positions and visible orientations independently for C1's
reference and current images. Each image uses a new process without the other
image's answer, object names, or expected annotations. Review both descriptions
before using them to construct any restoration checklist. Coarse position labels
may miss small movements. Raw logs are saved in a unique directory under `/tmp/`,
outside Git, because runtime output may contain machine paths. Preserve reviewed
findings in `results/` after inspecting the output.

Override `LLAMA_MTMD_CLI` if the executable is installed elsewhere. A C1 run
completed, but the reference description misreported the cup handle and remote
orientation; the current description omitted orientation details. This diagnostic
does not establish reliable spatial understanding.

To test whether a user-provided object inventory helps the two-image comparison:

```bash
python compare_layout.py --mode named-compare \
  --objects "red cup" "beverage can" "remote control"
```

This experiment adds object names only, without positions, orientations, or
expected actions. It uses the same two images and generation settings as compare
mode and saves to `results/C1-template_00000-traj_00000-named-comparison.json`.
Evaluate it as assisted recognition: the object names are supplied by the user,
not discovered independently by the model. Expected-result files are never read.

To compare all five ordered images while retaining the endpoint restoration task:

```bash
python compare_layout.py --mode sequence --max-new-tokens 256
```

This mode supplies frame000 through frame004 in order and asks to restore the
last layout to the first. It adds no object names or expected actions. Results
are saved separately to `results/C1-template_00000-traj_00000-sequence.json`.
This is a multi-image experiment, not video with verified timing; its sequence
labels require a slightly different prompt from the two-image run.

To diagnose scene understanding separately from comparison, describe the two
endpoint images independently:

```bash
python compare_layout.py --mode describe --max-new-tokens 128
```

The model is loaded once, but each image receives a separate prompt without the
other image or its description. Results are saved to
`results/C1-template_00000-traj_00000-descriptions.json`, leaving the comparison
result intact. Expected annotations are not provided in either mode.

### Initial Inference Result

The first C1 run loaded successfully and generated 256 tokens in 30.20 seconds
on the CPU. It repeated contradictory statements about the cup and reached the
token limit, so it did not produce a usable restoration checklist. The original
output is preserved in `results/C1-template_00000-traj_00000-baseline.json`.

The next experiment uses a shorter prompt with at most three checklist items,
a repetition penalty of 1.1, and a repeated 10-token sequence restriction.
These settings are experiments, not an accuracy guarantee. Review the next
output against the images before using it. The three-item limit is for the
initial C1 experiment and does not cover all objects in larger samples.

The selected images come from the [Tabletop Tidying Up (TTU) Dataset](https://github.com/rllab-snu/TTU-Dataset):

Four five-frame sequences are included under `data/` (twenty images):

| Demo directory | Objects | Original trajectory |
| --- | --- | --- |
| `O5-template_00000-traj_00000` | Cup, beverage can, marker, glue, stapler | `dataset_sample/O5/template_00000/traj_00000` |
| `O5-template_00001-traj_00000` | Different cup, beverage can, marker, glue, stapler | `dataset_sample/O5/template_00001/traj_00000` |
| `C1-template_00000-traj_00000` | Cup, beverage can, remote control | `dataset_sample/C1/template_00000/traj_00000` |
| `D10-template_00000-traj_00000` | Plate, fork, knife, fruit model, bowl, drinking glass | `dataset_sample/D10/template_00000/traj_00000` |

### Ordered Frame Sequences

Each of the four demo directories contains five images directly in the directory:

```text
frame000.png
frame001.png
frame002.png
frame003.png
frame004.png
sequence.json
expected-result.json
```

Each image comes from the corresponding original frame directory's `rgb_top.png`.
`sequence.json` records the ordered inputs and original source paths.
`frame000.png` is the reference layout; `frame004.png` is the final/current layout.
There are no separate numbered frame directories or duplicate endpoint images.

Frame timing has not been verified. Treat these as ordered layout states, not
video with a known frame rate. Each `expected-result.json` evaluates restoration
from frame004 to frame000 only, not intermediate transitions.

The reference and current roles are assigned for this prototype. They are not
verified official tidy/untidy labels. These samples are not photographs of our
company workbench.

Demo samples in `data/` are intended to be committed so the demo can
be reproduced without downloading the dataset. The dataset license and citation are preserved in `data/LICENSE.txt` and `data/CITATION.cff`. The demo uses the four frame sequences listed above;
exploratory inspection copies have been removed.

## Dataset Attribution and License

**Dataset:** Tabletop Tidying Up (TTU) Dataset  
**Authors:** Hogun Kee, Wooseok Oh, and Songhwai Oh  
**Year:** 2024  
**Source:** https://github.com/rllab-snu/TTU-Dataset/  
**Upstream stated license:** MIT

The upstream README states that the dataset is distributed under the MIT
License. Its `LICENSE.txt` contains the notice reproduced in
[LICENSE.txt](data/LICENSE.txt). The copyright holder in that file is
Othneil Drew (2018); this is preserved verbatim and should not be interpreted
as the dataset author attribution. A separate data-specific license was not
verified. Confirm upstream data terms before redistributing sample images.

Retain the applicable copyright and permission notice when distributing copies
covered by the MIT License. This attribution does not assign a license to our
own project code.

Suggested citation from the upstream README:

```bibtex
@misc{Kee_TTU_Dataset_Dataset_2024,
    author = {Kee, Hogun and Oh, Wooseok and Oh, Songhwai},
    month = sep,
    title = {{TTU Dataset: Dataset for Tabletop Tidying Up Problem}},
    url = {https://github.com/rllab-snu/TTU-Dataset/},
    year = {2024}
}
```

