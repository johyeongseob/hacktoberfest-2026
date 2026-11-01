"""Find the asked item on a LIBERO camera image by its name, with a frozen detector.

The learned counterpart of mark.item_box: a detector gets only the image and
the item name, and returns a box in the same form, (top, left, bottom, right)
on the raw render, so mark.draw_box draws it the same way. Every detector
sees the image upright (the raw render turned 180 degrees, as the model sees
it), its box is turned back onto the raw render, and it always returns its top
box, however low the score.

Detectors (make_detector(name)), all open vocabulary:
- owlv2: google/owlv2-base-patch16-ensemble
- owlvit: google/owlvit-base-patch32 (the OWL-ViT family MOO used)
- gdino: IDEA-Research/grounding-dino-tiny
- yoloworld: YOLO-World v2 small (ultralytics, AGPL-3.0; installed separately)
- florence2: Florence-2 base, open-vocabulary detection task (no score)

Used by eval/chain_rollout.py (--mark detector) and eval/check_detector.py.
"""

import time

import numpy as np
import torch

DETECTORS = ["owlv2", "owlvit", "gdino", "yoloworld", "florence2"]


def iou(a, b) -> float:
    """Overlap of two (top, left, bottom, right) boxes; 0 if either is None."""
    if a is None or b is None:
        return 0.0
    top, left = max(a[0], b[0]), max(a[1], b[1])
    bottom, right = min(a[2], b[2]), min(a[3], b[3])
    inter = max(bottom - top + 1, 0) * max(right - left + 1, 0)
    area = lambda box: (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    return inter / (area(a) + area(b) - inter)


def to_raw(left, top, right, bottom, height, width):
    """An upright (left, top, right, bottom) box in pixels -> (top, left, bottom, right) on the raw render."""
    top, bottom = max(top, 0), min(bottom, height - 1)
    left, right = max(left, 0), min(right, width - 1)
    # Upright -> raw: the 180-degree turn flips both axes.
    box = (height - 1 - bottom, width - 1 - right, height - 1 - top, width - 1 - left)
    return tuple(int(round(v)) for v in box)


class Detector:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def box(self, raw: np.ndarray, query: str):
        """Top box for query as (top, left, bottom, right) on the raw render, its score, and ms."""
        start = time.perf_counter()
        upright = np.ascontiguousarray(raw[::-1, ::-1])
        left, top, right, bottom, score = self.detect(upright, query)
        if self.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        return to_raw(left, top, right, bottom, *raw.shape[:2]), score, ms

    def warm_up(self):
        self.box(np.zeros((256, 256, 3), np.uint8), "warm up")  # the first call is slow


class Owl(Detector):
    """OWLv2 and OWL-ViT: a score per image patch for the text, and a box per patch."""

    def __init__(self, name: str, device: str = "cuda"):
        super().__init__(device)
        if name == "owlv2":
            from transformers import Owlv2ForObjectDetection as Model, Owlv2Processor as Processor
            repo = "google/owlv2-base-patch16-ensemble"
        else:
            from transformers import OwlViTForObjectDetection as Model, OwlViTProcessor as Processor
            repo = "google/owlvit-base-patch32"
        self.processor = Processor.from_pretrained(repo)
        self.model = Model.from_pretrained(repo).to(device).eval()
        self.warm_up()

    @torch.inference_mode()
    def detect(self, image: np.ndarray, query: str):
        inputs = self.processor(text=[[query]], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        scores = outputs.logits[0, :, 0].sigmoid()
        best = int(scores.argmax())
        # (cx, cy, w, h) relative to the input square (OWLv2 pads at the bottom
        # and right; OWL-ViT resizes). The LIBERO renders are square either way.
        cx, cy, w, h = outputs.pred_boxes[0, best].tolist()
        side = max(image.shape[:2])
        return (cx - w / 2) * side, (cy - h / 2) * side, (cx + w / 2) * side, (cy + h / 2) * side, float(scores[best])


class GroundingDino(Detector):
    """Grounding DINO: 900 box queries, each scored against every token of the text."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        repo = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(repo).to(device).eval()
        self.warm_up()

    @torch.inference_mode()
    def detect(self, image: np.ndarray, query: str):
        # The model expects lower case phrases, each ending with a period.
        inputs = self.processor(images=image, text=query.lower() + ".", return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        scores = outputs.logits[0].sigmoid().max(dim=-1).values  # best text token per box
        best = int(scores.argmax())
        cx, cy, w, h = outputs.pred_boxes[0, best].tolist()  # relative to the image
        height, width = image.shape[:2]
        return (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height, float(scores[best])


class YoloWorld(Detector):
    """YOLO-World: a YOLO detector whose class names are text embeddings."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        from ultralytics import YOLOWorld
        self.model = YOLOWorld("yolov8s-worldv2.pt")
        self.query = None
        self.warm_up()

    def detect(self, image: np.ndarray, query: str):
        if query != self.query:
            self.model.set_classes([query])
            self.query = query
        # ultralytics reads numpy images as BGR.
        result = self.model.predict(np.ascontiguousarray(image[..., ::-1]), conf=1e-4, device=self.device, verbose=False)[0]
        if len(result.boxes) == 0:
            return 0, 0, 0, 0, 0.0
        best = int(result.boxes.conf.argmax())
        left, top, right, bottom = result.boxes.xyxy[best].tolist()
        return left, top, right, bottom, float(result.boxes.conf[best])


class Florence2(Detector):
    """Florence-2: writes the boxes for the phrase as text; there is no score."""

    TASK = "<OPEN_VOCABULARY_DETECTION>"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        repo = "florence-community/Florence-2-base"
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = Florence2ForConditionalGeneration.from_pretrained(repo).to(device).eval()
        self.warm_up()

    @torch.inference_mode()
    def detect(self, image: np.ndarray, query: str):
        inputs = self.processor(text=self.TASK + query, images=image, return_tensors="pt").to(self.device)
        ids = self.model.generate(**inputs, max_new_tokens=64, num_beams=3, do_sample=False)
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        height, width = image.shape[:2]
        found = self.processor.post_process_generation(text, task=self.TASK, image_size=(width, height))[self.TASK]
        boxes = found.get("bboxes", [])
        if not boxes:
            return 0, 0, 0, 0, 0.0
        left, top, right, bottom = boxes[0]  # the first box written
        return left, top, right, bottom, 1.0


def make_detector(name: str, device: str = "cuda") -> Detector:
    if name in ("owlv2", "owlvit"):
        return Owl(name, device)
    if name == "gdino":
        return GroundingDino(device)
    if name == "yoloworld":
        return YoloWorld(device)
    if name == "florence2":
        return Florence2(device)
    raise ValueError(f"Unknown detector {name}; choose from {DETECTORS}")
