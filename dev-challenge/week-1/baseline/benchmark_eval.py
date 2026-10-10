"""Run the official LIBERO benchmark with `lerobot-eval` and report SmolVLA inference time.

Usage: pass the same arguments as `lerobot-eval`, for example

    python baseline/benchmark_eval.py --policy.path=lerobot/smolvla_libero --env.type=libero ...

The script times every SmolVLA forward pass that predicts an action chunk,
prints a summary, and writes `inference_timing.json` to `--output_dir`.
"""

import json
import statistics
import sys
import time
from pathlib import Path

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts import lerobot_eval

# Inference durations and env step count, filled in while lerobot-eval runs.
chunk_times_s: list[float] = []
select_action_calls = 0

# Original methods, called by the wrappers below.
_original_get_action_chunk = SmolVLAPolicy._get_action_chunk
_original_select_action = SmolVLAPolicy.select_action


# Runs one SmolVLA forward pass (50-action chunk) and records its duration.
def _timed_get_action_chunk(self, *args, **kwargs):
    start = time.perf_counter()
    actions = _original_get_action_chunk(self, *args, **kwargs)
    chunk_times_s.append(time.perf_counter() - start)
    return actions


# Called once per env step; counts steps.
def _counted_select_action(self, *args, **kwargs):
    global select_action_calls
    select_action_calls += 1
    return _original_select_action(self, *args, **kwargs)


# Swap in the wrappers before lerobot-eval builds the policy.
SmolVLAPolicy._get_action_chunk = _timed_get_action_chunk
SmolVLAPolicy.select_action = _counted_select_action


# Returns --output_dir from the lerobot-eval arguments.
def _output_dir() -> Path | None:
    for arg in sys.argv[1:]:
        if arg.startswith("--output_dir="):
            return Path(arg.split("=", 1)[1])
    return None


# Percentile of a list (0.5 = median, 0.9 = p90).
def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


# Builds the timing summary saved to inference_timing.json.
def _summary(wall_s: float) -> dict:
    inference_s = sum(chunk_times_s)
    summary = {
        "wall_s": round(wall_s, 2),
        "inference_total_s": round(inference_s, 2),
        "non_inference_s": round(wall_s - inference_s, 2),
        "env_steps": select_action_calls,
        "inference_calls": len(chunk_times_s),
    }
    if chunk_times_s:
        summary.update(
            {
                "inference_mean_s": round(statistics.mean(chunk_times_s), 3),
                "inference_p50_s": round(_percentile(chunk_times_s, 0.5), 3),
                "inference_p90_s": round(_percentile(chunk_times_s, 0.9), 3),
                "inference_max_s": round(max(chunk_times_s), 3),
                "inference_first_s": round(chunk_times_s[0], 3),
            }
        )
    return summary


# Runs lerobot-eval unchanged, then prints and saves the timing summary.
def main() -> None:
    start = time.perf_counter()
    try:
        lerobot_eval.main()
    finally:
        summary = _summary(time.perf_counter() - start)
        print("\n=== Inference timing ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        output_dir = _output_dir()
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "inference_timing.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
