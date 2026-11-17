from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prepare_vietspeech_subset_cli_documents_bounded_streaming() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/prepare_vietspeech_subset.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "VietSpeech" in result.stdout
    assert "--target-hours" in result.stdout
    assert "--base-processed-dir" in result.stdout
