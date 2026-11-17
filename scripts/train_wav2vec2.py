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
from src.train_ctc import run_wav2vec2_fine_tuning


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Fine-tune Wav2Vec2-CTC trên manifest đã chuẩn bị."
    )
    parser.add_argument("--config", default="config.wav2vec2_fpt.yaml")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    result = run_wav2vec2_fine_tuning(
        load_config(args.config),
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
