"""Re-time lerobot-eval videos to real time.

lerobot-eval saves LIBERO videos at 80 fps, but the robot is controlled at
20 Hz (one video frame per step), so the clips play four times too fast.
This script writes a copy of each video at 20 fps with a clock, so one
second of video is one second of robot motion. Time spent computing actions
is not shown: the robot is assumed to keep up with its 20 Hz control rate.

Usage (from dev-challenge/week-3):
    python tools/realtime_video.py outputs/libero10-smoke

Writes outputs/realtime/<run>/<task>_<episode>.mp4; the original videos are
left untouched.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CONTROL_HZ = 20  # LIBERO control rate; one video frame per step


def label(frame: np.ndarray, clock_s: float, font) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), f"{clock_s:5.1f} s", fill="white", font=font, stroke_width=2, stroke_fill="black")
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="lerobot-eval output folder")
    parser.add_argument("--out-root", type=Path, default=Path("outputs/realtime"))
    args = parser.parse_args()

    info = json.loads((args.run_dir / "eval_info.json").read_text())
    font = ImageFont.load_default(size=20)
    out_dir = args.out_root / args.run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in info["per_task"]:
        for episode, video in enumerate(task["metrics"]["video_paths"]):
            # Paths in eval_info.json are relative to where the run started.
            path = args.run_dir / "videos" / Path(video).parent.name / Path(video).name
            frames = iio.imread(path, plugin="pyav")
            out = out_dir / f"{task['task_group']}_{task['task_id']}_ep{episode}.mp4"
            writer = imageio.get_writer(out, fps=CONTROL_HZ, codec="libx264", quality=7, macro_block_size=8)
            for i, frame in enumerate(frames):
                writer.append_data(label(frame, i / CONTROL_HZ, font))
            writer.close()
            print(f"{out}: {len(frames)} steps, {len(frames) / CONTROL_HZ:.1f} s")


if __name__ == "__main__":
    main()
