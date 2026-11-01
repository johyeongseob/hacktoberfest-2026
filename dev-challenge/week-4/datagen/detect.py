"""Find the asked item on a LIBERO camera image with OWLv2, by its name.

The learned counterpart of mark.item_box: it gets only the image and the item
name, and returns a box in the same form, (top, left, bottom, right) on the
raw render, so mark.draw_box draws it the same way. The detector sees the
image upright (the raw render turned 180 degrees, as the model sees it), and
its box is turned back onto the raw render. It always returns its top box,
however low the score.

Used by eval/chain_rollout.py (--mark detector) and eval/check_detector.py.
"""

import time

import numpy as np
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL = "google/owlv2-base-patch16-ensemble"


def iou(a, b) -> float:
    """Overlap of two (top, left, bottom, right) boxes; 0 if either is None."""
    if a is None or b is None:
        return 0.0
    top, left = max(a[0], b[0]), max(a[1], b[1])
    bottom, right = min(a[2], b[2]), min(a[3], b[3])
    inter = max(bottom - top + 1, 0) * max(right - left + 1, 0)
    area = lambda box: (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    return inter / (area(a) + area(b) - inter)


class Owlv2:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(device).eval()
        self.box(np.zeros((256, 256, 3), np.uint8), "warm up")  # the first call is slow

    @torch.inference_mode()
    def box(self, raw: np.ndarray, query: str):
        """Top box for query as (top, left, bottom, right) on the raw render, its score, and ms."""
        start = time.perf_counter()
        upright = np.ascontiguousarray(raw[::-1, ::-1])
        inputs = self.processor(text=[[query]], images=upright, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        scores = outputs.logits[0, :, 0].sigmoid()
        best = int(scores.argmax())
        # Boxes are (cx, cy, w, h), relative to the image padded to a square at the bottom and right.
        cx, cy, w, h = outputs.pred_boxes[0, best].tolist()
        if self.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        height, width = raw.shape[:2]
        side = max(height, width)
        top, bottom = (cy - h / 2) * side, (cy + h / 2) * side
        left, right = (cx - w / 2) * side, (cx + w / 2) * side
        top, bottom = max(top, 0), min(bottom, height - 1)
        left, right = max(left, 0), min(right, width - 1)
        # Upright -> raw: the 180-degree turn flips both axes.
        box = (height - 1 - bottom, width - 1 - right, height - 1 - top, width - 1 - left)
        return tuple(int(round(v)) for v in box), float(scores[best]), ms
