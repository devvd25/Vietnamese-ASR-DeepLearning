from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptation import prepare_supervised_adaptation_manifests
from src.cli import configure_utf8_stdio


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Tạo manifest supervised adaptation ST23D có validation độc lập."
    )
    parser.add_argument("--audio", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=64)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_supervised_adaptation_manifests(
        audio_path=args.audio,
        transcript=args.transcript,
        validation_manifest=args.validation_manifest,
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        validation_limit=args.validation_limit,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
