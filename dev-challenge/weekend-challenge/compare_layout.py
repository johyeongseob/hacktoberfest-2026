"""Compare two local tabletop images with SmolVLM2 on the CPU."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PROMPT = (
    "How should the objects in Image 2 be moved to match Image 1? "
    "Give at most three short checklist items, one per object. "
    "Use directions as seen in the pictures. "
    "If you cannot tell how an object moved, say uncertain."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="C1-template_00000-traj_00000")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.threads < 1:
        parser.error("Token and thread counts must be positive.")
    sample = ROOT / "data" / args.sample
    if sample.resolve().parent != (ROOT / "data").resolve():
        parser.error("Choose a sample directory directly under data.")
    images = [sample / "frame000.png", sample / "frame004.png"]
    model_path = REPO_ROOT / "models" / "SmolVLM2-500M-Video-Instruct"
    for path in [model_path / "config.json", *images]:
        if not path.is_file():
            parser.error(f"Missing required file: {path.relative_to(REPO_ROOT)}")

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(args.threads)
    print("Loading the local model on CPU (float32)...", flush=True)
    started = perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).to("cpu").eval()
    load_seconds = perf_counter() - started

    # Expected-result annotations are never loaded or passed to the model.
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Image 1: reference layout."},
        {"type": "image"},
        {"type": "text", "text": "Image 2: current layout."},
        {"type": "image"},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    rgb_images = []
    for path in images:
        with Image.open(path) as image:
            rgb_images.append(image.convert("RGB"))
    inputs = processor(text=text, images=rgb_images, return_tensors="pt")

    print("Comparing frame000.png and frame004.png...", flush=True)
    started = perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
            repetition_penalty=1.1, no_repeat_ngram_size=10,
        )
    generation_seconds = perf_counter() - started
    answer_ids = generated[0, inputs["input_ids"].shape[1]:]
    answer = processor.decode(answer_ids, skip_special_tokens=True).strip()
    result = {
        "model": MODEL_ID,
        "sample": args.sample,
        "input_images": [path.relative_to(ROOT).as_posix() for path in images],
        "prompt": PROMPT,
        "response": answer,
        "device": "cpu",
        "dtype": "float32",
        "threads": args.threads,
        "max_new_tokens": args.max_new_tokens,
        "generation_settings": {
            "do_sample": False,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 10,
        },
        "generated_tokens": len(answer_ids),
        "reached_token_limit": len(answer_ids) >= args.max_new_tokens,
        "load_seconds": round(load_seconds, 2),
        "generation_seconds": round(generation_seconds, 2),
        "package_versions": {
            name: version(name) for name in ["torch", "torchvision", "transformers", "Pillow"]
        },
        "review_status": "pending human review",
    }
    output = ROOT / "results" / f"{args.sample}-comparison.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n{answer}\n")
    print(f"Generation time: {generation_seconds:.2f} seconds")
    print(f"Saved: {output.relative_to(ROOT)}")
    if result["reached_token_limit"]:
        print("The response may be truncated. Review it before sharing.")


if __name__ == "__main__":
    main()

