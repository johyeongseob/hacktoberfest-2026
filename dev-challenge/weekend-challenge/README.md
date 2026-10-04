# Shared Workbench Reset Assistant

A Weekend Challenge prototype for helping a coworker restore a shared workbench
after use. The planned local VLM workflow compares a reference layout with a
current layout and suggests a short cleanup checklist.

See [Project Flow](project-flow.md) for the end-to-end workflow, deliverables,
and current progress.

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

