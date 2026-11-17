"""Create/update the public ZeroGPU Space from the explicit safe allowlist."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import validate_repo_id

from src.hf_deployment import build_space_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face Space id, for example username/vietnamese-asr-stt19.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_repo_id(args.repo_id)
    api = HfApi()
    account = api.whoami()
    username = str(account.get("name", ""))
    if not username or not args.repo_id.startswith(f"{username}/"):
        raise ValueError("The Space must be created under the authenticated account.")

    manifest = build_space_manifest(PROJECT_ROOT)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="zero-a10g",
        private=False,
        exist_ok=True,
    )
    operations = [
        CommitOperationAdd(
            path_in_repo=item.path_in_repo,
            path_or_fileobj=item.local_path,
        )
        for item in manifest
    ]
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="space",
        operations=operations,
        commit_message="Deploy Vietnamese ASR ZeroGPU demo",
    )
    print(f"Published: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
