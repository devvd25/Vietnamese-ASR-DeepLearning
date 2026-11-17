from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gradio_app import launch_demo
from src.cli import configure_utf8_stdio


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Khởi chạy demo Gradio ASR tiếng Việt.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch_demo(
        args.config,
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
