from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import configure_utf8_stdio
from src.config import load_config
from src.train import run_fine_tuning
from src.youtube_learning import build_youtube_continual_manifests


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Trộn mẫu YouTube đồng thuận với dữ liệu thật và huấn luyện "
            "checkpoint PhoWhisper ứng viên riêng."
        )
    )
    parser.add_argument(
        "--config",
        default="config.phowhisper_youtube_candidate.yaml",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Chỉ tạo manifest trộn, chưa huấn luyện.",
    )
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    learning = config["youtube_learning"]
    summary = build_youtube_continual_manifests(
        base_processed_dir=learning["base_processed_data"],
        youtube_manifest=learning["manifest"],
        output_dir=config["paths"]["processed_data"],
        minimum_youtube_samples=int(learning.get("minimum_samples", 20)),
        replay_per_youtube_sample=int(learning.get("replay_per_youtube_sample", 4)),
        seed=int(config["training"].get("seed", 42)),
    )
    if args.prepare_only:
        print(json.dumps({"status": "prepared", **summary}, ensure_ascii=False, indent=2))
        return

    result = run_fine_tuning(
        config,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(
        json.dumps(
            {"manifest": summary, "training": result},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
