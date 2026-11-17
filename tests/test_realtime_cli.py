from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


def test_realtime_server_cli_exposes_loopback_options() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/launch_realtime_server.py", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1"},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--port" in result.stdout
    assert "127.0.0.1" in result.stdout
