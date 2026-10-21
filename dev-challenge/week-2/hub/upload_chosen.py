"""Upload the chosen checkpoint of each run to a private Hugging Face repo.

For each run, reads the chosen step from EVAL_ROOT/<run>-val.json (written by
watch_eval.py) and uploads that checkpoint's `pretrained_model` folder to
<repo>/<run>/. `training_state` is not uploaded. The repo is created as
private if it does not exist; this script never makes it public.

Needs a write token (`hf auth login`).

Example:
    python upload_chosen.py --repo HF_USER/smolvla-libero-picnic-wordings --runs n1-s1000 n5-s1000 n40-s1000
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="e.g. HF_USER/smolvla-libero-picnic-wordings")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--train-root", type=Path, default=Path("/workspace/outputs/week2/train"))
    parser.add_argument("--eval-root", type=Path, default=Path("/workspace/outputs/week2/eval"))
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(args.repo).private:
        raise SystemExit(f"{args.repo} is public; this script only uploads to private repos")

    for run in args.runs:
        chosen = json.loads((args.eval_root / f"{run}-val.json").read_text())["chosen_step"]
        folder = args.train_root / run / "checkpoints" / f"{chosen:06d}" / "pretrained_model"
        print(f"{run}: uploading step {chosen} from {folder}", flush=True)
        api.upload_folder(
            repo_id=args.repo,
            folder_path=folder,
            path_in_repo=run,
            commit_message=f"{run}: checkpoint at step {chosen} (best on validation)",
        )
    print(f"Done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
