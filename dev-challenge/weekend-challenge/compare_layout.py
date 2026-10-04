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
    parser.add_argument(
        "--mode", choices=["compare", "describe", "sequence", "named-compare"],
        default="compare",
    )
    parser.add_argument(
        "--objects", nargs="+",
        help="User-provided object names for named-compare; no positions or actions.",
    )
    parser.add_argument("--sample", default="C1-template_00000-traj_00000")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.threads < 1:
        parser.error("Token and thread counts must be positive.")
    if args.mode == "named-compare" and not args.objects:
        parser.error("named-compare requires --objects with the visible object names.")
    if args.objects and args.mode != "named-compare":
        parser.error("--objects is only supported in named-compare mode.")
    sample = ROOT / "data" / args.sample
    if sample.resolve().parent != (ROOT / "data").resolve():
        parser.error("Choose a sample directory directly under data.")
    images = [sample / "frame000.png", sample / "frame004.png"]
    if args.mode == "sequence":
        images = [sample / f"frame{index:03d}.png" for index in range(5)]
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

    if args.mode == "describe":
        describe_images(processor, model, images, args, load_seconds)
        return

    # Expected-result annotations are never loaded or passed to the model.
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Image 1: reference layout."},
        {"type": "image"},
        {"type": "text", "text": "Image 2: current layout."},
        {"type": "image"},
        {"type": "text", "text": PROMPT},
    ]}]
    if args.mode == "named-compare":
        # Only the user-supplied inventory is added; reference answers stay private.
        messages[0]["content"][-1]["text"] = (
            "The objects in both images are: " + ", ".join(args.objects) + ". " + PROMPT
        )
    if args.mode == "sequence":
        content = []
        for index, path in enumerate(images, start=1):
            content.extend([
                {"type": "text", "text": f"Image {index}: {path.name}, in sequence order."},
                {"type": "image"},
            ])
        # Preserve the checklist request, updating the current image's number.
        content.append({"type": "text", "text": (
            "These five pictures are ordered layout states. "
            "Image 1 (frame000.png) is the reference. "
            "Image 5 (frame004.png) is the current layout. "
            "The three middle pictures show intermediate states. "
            + PROMPT.replace("Image 2", "Image 5")
        )})
        messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    rgb_images = []
    for path in images:
        with Image.open(path) as image:
            rgb_images.append(image.convert("RGB"))
    inputs = processor(text=text, images=rgb_images, return_tensors="pt")

    print(f"Comparing {len(images)} images from frame000.png to frame004.png...", flush=True)
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
        "mode": args.mode,
        "provided_object_names": args.objects or [],
        "input_images": [path.relative_to(ROOT).as_posix() for path in images],
        "prompt": messages[0]["content"][-1]["text"],
        "text_instructions": [
            item["text"] for item in messages[0]["content"] if item["type"] == "text"
        ],
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
    suffix = {"sequence": "sequence", "named-compare": "named-comparison"}.get(
        args.mode, "comparison"
    )
    output = ROOT / "results" / f"{args.sample}-{suffix}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n{answer}\n")
    print(f"Generation time: {generation_seconds:.2f} seconds")
    print(f"Saved: {output.relative_to(ROOT)}")
    if result["reached_token_limit"]:
        print("The response may be truncated. Review it before sharing.")


def describe_images(processor, model, images, args, load_seconds):
    """Describe each image independently without reference answers or history."""
    import torch
    from PIL import Image

    prompt = (
        "Describe the objects visible on this tabletop and where each is located "
        "in the picture. Use short sentences. Describe only this image."
    )
    descriptions = []
    for path in images:
        print(f"Describing {path.name} independently...", flush=True)
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        with Image.open(path) as image:
            inputs = processor(
                text=text, images=[image.convert("RGB")], return_tensors="pt"
            )
        started = perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                repetition_penalty=1.1, no_repeat_ngram_size=10,
            )
        seconds = perf_counter() - started
        answer_ids = generated[0, inputs["input_ids"].shape[1]:]
        answer = processor.decode(answer_ids, skip_special_tokens=True).strip()
        descriptions.append({
            "image": path.relative_to(ROOT).as_posix(),
            "response": answer,
            "generation_seconds": round(seconds, 2),
            "generated_tokens": len(answer_ids),
            "reached_token_limit": len(answer_ids) >= args.max_new_tokens,
        })
        print(f"\n{answer}\nGeneration time: {seconds:.2f} seconds\n", flush=True)
    result = {
        "model": MODEL_ID,
        "sample": args.sample,
        "mode": "independent single-image descriptions",
        "prompt": prompt,
        "device": "cpu",
        "dtype": "float32",
        "threads": args.threads,
        "load_seconds": round(load_seconds, 2),
        "generation_settings": {
            "do_sample": False, "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": 1.1, "no_repeat_ngram_size": 10,
        },
        "descriptions": descriptions,
        "package_versions": {
            name: version(name) for name in ["torch", "torchvision", "transformers", "Pillow"]
        },
        "review_status": "pending human review",
    }
    output = ROOT / "results" / f"{args.sample}-descriptions.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output.relative_to(ROOT)}")
    if any(item["reached_token_limit"] for item in descriptions):
        print("A description may be truncated. Review it before sharing.")


if __name__ == "__main__":
    main()

