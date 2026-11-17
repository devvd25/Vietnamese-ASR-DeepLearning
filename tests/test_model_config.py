from __future__ import annotations

import pytest

from src.model_utils import validate_model_config


def test_validate_model_config_accepts_multilingual_whisper() -> None:
    validated = validate_model_config(
        {"name": "openai/whisper-tiny", "language": "Vietnamese", "task": "transcribe"}
    )
    assert validated["name"] == "openai/whisper-tiny"


def test_validate_model_config_rejects_english_only_checkpoint() -> None:
    with pytest.raises(ValueError, match="đa ngôn ngữ"):
        validate_model_config(
            {"name": "openai/whisper-tiny.en", "language": "Vietnamese", "task": "transcribe"}
        )


def test_validate_model_config_rejects_translation_task() -> None:
    with pytest.raises(ValueError, match="transcribe"):
        validate_model_config(
            {"name": "openai/whisper-tiny", "language": "Vietnamese", "task": "translate"}
        )
