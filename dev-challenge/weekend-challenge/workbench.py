"""Local inference and image preparation for the workbench demo."""

import base64
from io import BytesIO
import json
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "smolvlm2-workbench"
OBSERVATION_PROMPT = (
    "Describe each tabletop object in this image. For each object, give its name, "
    "position in the picture, and visible orientation. For a cup, describe where "
    "its handle points. For an elongated object, describe its long-axis direction. "
    "If a detail is unclear, say uncertain. Use one short item per object. "
    "Describe only this image; do not suggest movements."
)


def normalize_image(data):
    """Return a size-limited RGB PNG without the uploaded image's metadata."""
    from PIL import Image, ImageOps

    with Image.open(BytesIO(data)) as original:
        original.load()
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((1024, 1024))
        clean_image = Image.new("RGB", image.size)
        clean_image.paste(image)
        output = BytesIO()
        clean_image.save(output, format="PNG")
        return output.getvalue()


def generate(content, max_tokens=192):
    """Send one independent request to the fixed localhost inference server."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "repeat_penalty": 1.1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = perf_counter()
    try:
        # Local requests should not be routed through a configured network proxy.
        with build_opener(ProxyHandler({})).open(request, timeout=300) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(
            f"The local model server returned HTTP {exc.code}. Check its terminal."
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Cannot reach the local model server, or the request timed out. "
            "Start it in the other WSL terminal and wait until it is ready."
        ) from exc
    try:
        choice = result["choices"][0]
        answer = choice["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Empty response")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("The model server returned no usable text.") from exc
    return {
        "text": answer.strip(),
        "seconds": round(perf_counter() - started, 2),
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage", {}),
    }


def describe(image):
    encoded = base64.b64encode(image).decode("ascii")
    return generate([
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
        {"type": "text", "text": OBSERVATION_PROMPT},
    ])


def make_checklist(reference, current):
    prompt = (
        "Write a short checklist for a person restoring a shared tabletop. "
        "Use only the following reviewed observations. The reference is the target; "
        "the current layout is the starting point. Do not reverse them. "
        "Include an object's movement or rotation only if supported by both descriptions. "
        "If locations are too vague to determine a direction, say to check the reference "
        "photo instead of guessing. Do not invent distances or extra objects. "
        "Use up/down/left/right relative to the pictures. Use at most six numbered items.\n\n"
        f"REFERENCE OBSERVATIONS:\n{reference}\n\nCURRENT OBSERVATIONS:\n{current}"
    )
    return generate([{"type": "text", "text": prompt}], max_tokens=256)
