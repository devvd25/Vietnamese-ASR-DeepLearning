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
from src.evaluate import run_model_evaluation


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Đánh giá Wav2Vec2-Vietnamese hoặc Whisper fine-tuned bằng WER/CER."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--model",
        choices=["wav2vec2_vietnamese", "whisper_fine_tuned"],
        default="wav2vec2_vietnamese",
    )
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-beams", type=int)
    args = parser.parse_args()
    result = run_model_evaluation(
        load_config(args.config),
        model_choice=args.model,
        split=args.split,
        limit=args.limit,
        num_beams=args.num_beams,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
