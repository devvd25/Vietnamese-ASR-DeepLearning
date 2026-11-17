from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prepare_cli_help_survives_legacy_windows_encoding() -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1258"
    result = subprocess.run(
        [sys.executable, "scripts/prepare_data.py", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert "Chuẩn bị manifest" in result.stdout.decode("utf-8")
