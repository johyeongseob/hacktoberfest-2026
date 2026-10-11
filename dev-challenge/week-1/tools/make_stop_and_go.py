"""Replay a simulated episode as a real robot would experience CPU inference.

In simulation the clock stops while SmolVLA thinks. On a real robot it does
not: the arm would hold still for each inference call, then move for
n_action_steps control steps. This script inserts a frozen pause before every
chunk, labeled with a wall-clock timer, so the video shows that stop-and-go
motion.

    python tools/make_stop_and_go.py
    python tools/make_stop_and_go.py --think-s 1.15 --n-action-steps 10
"""

import argparse
from pathlib import Path

import av
import numpy as np
from lerobot.utils.io_utils import write_video
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FPS = 20  # control rate; one video frame per control step


def read_frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def label(frame: np.ndarray, lines: list[tuple[str, str]], size: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, size)
    y = 6
    for text, color in lines:
        left, top, right, bottom = draw.textbbox((6, y), text, font=font)
        draw.rectangle((left - 3, top - 3, right + 3, bottom + 3), fill=(0, 0, 0))
        draw.text((6, y), text, font=font, fill=color)
        y = bottom + 8
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=Path("outputs/rollout-check/task4_original_ep0.mp4"),
                        help="20 fps episode video from picnic_rollout.py")
    parser.add_argument("--think-s", type=float, default=1.15, help="seconds per inference call (idle CPU)")
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/stop-and-go/ketchup_stop_and_go.mp4"))
    args = parser.parse_args()

    # frames[0] is the initial observation; frames[k] follows control step k.
    frames = read_frames(args.video)
    steps = len(frames) - 1
    size = max(12, frames[0].shape[0] // 22)
    pause = round(args.think_s * FPS)
    # Totals shown next to each running clock.
    total_calls = -(-steps // args.n_action_steps)
    sim_total = steps / FPS
    real_total = (steps + total_calls * pause) / FPS
    out, wall = [], 0.0
    calls = 0
    for step in range(steps):
        if step % args.n_action_steps == 0:
            # The model sees frames[step] and thinks; a real arm holds still.
            calls += 1
            for _ in range(pause):
                out.append(label(frames[step], [
                    (f"real {wall:4.1f}/{real_total:.1f} s  sim {step / FPS:3.1f}/{sim_total:.1f} s", "white"),
                    (f"thinking (call {calls}, {args.think_s:.2f} s)", "red"),
                ], size))
                wall += 1 / FPS
        # Each chunk executes n_action_steps actions (the last one may be shorter).
        chunk_start = step - step % args.n_action_steps
        move_s = min(args.n_action_steps, steps - chunk_start) / FPS
        out.append(label(frames[step + 1], [
            (f"real {wall:4.1f}/{real_total:.1f} s  sim {(step + 1) / FPS:3.1f}/{sim_total:.1f} s", "white"),
            (f"moving (call {calls}, {move_s:.2f} s)", "lime"),
        ], size))
        wall += 1 / FPS

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video(args.output, out, fps=FPS)
    print(f"{args.output}: {steps} steps, {calls} calls, sim {steps / FPS:.1f} s, real {wall:.1f} s")

    # Side by side: simulation time on the left (holds its last frame once the
    # episode ends), real-robot time on the right, both on one clock.
    sim = []
    for i in range(len(out)):
        k = min(i, steps)
        status = ("moving", "lime") if i < steps else ("done", "white")
        sim.append(label(frames[k], [(f"simulation {k / FPS:3.1f}/{sim_total:.1f} s", "white"), status], size))
    pair = [np.concatenate([left, right], axis=1) for left, right in zip(sim, out)]
    side = args.output.with_name(args.output.stem + "_side_by_side.mp4")
    write_video(side, pair, fps=FPS)
    print(f"{side}: left simulation time, right real-robot time")


if __name__ == "__main__":
    main()
