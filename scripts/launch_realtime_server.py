from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import configure_utf8_stdio
from src.realtime_server import create_realtime_app


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Chạy server local cho Chrome Extension phụ đề YouTube realtime."
    )
    parser.add_argument(
        "--config",
        default="config.phowhisper_vietspeech_10h.yaml",
        help="Cấu hình chứa hai model ASR local.",
    )
    parser.add_argument(
        "--host",
        choices=["127.0.0.1", "localhost"],
        default="127.0.0.1",
        help="Chỉ cho phép loopback; mặc định 127.0.0.1.",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("--port phải nằm trong khoảng 1..65535")
    app = create_realtime_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
