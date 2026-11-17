from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config


def test_vietspeech_demo_config_declares_wav2vec2_model() -> None:
    project_root = Path(__file__).resolve().parents[1]

    config = load_config(project_root / "config.phowhisper_vietspeech_10h.yaml")

    assert (
        config["wav2vec2_model"]["name"]
        == "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    )
    assert config["wav2vec2_model"]["decoder"]["enabled"] is True


def test_load_config_resolves_project_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  name: openai/whisper-tiny\n  language: Vietnamese\n  task: transcribe\n"
        "data:\n  root: data/raw/common-voice-27-vi\n  max_audio_seconds: 30\n"
        "training:\n  epochs: 1\n  learning_rate: 0.00005\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["data"]["root"] == str((tmp_path / "data/raw/common-voice-27-vi").resolve())


def test_load_config_resolves_secondary_dataset_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  name: vinai/PhoWhisper-base\n  language: Vietnamese\n  task: transcribe\n"
        "data:\n  root: data/raw/vivos\n"
        "  common_voice_root: data/raw/common-voice-vi-21/vi\n"
        "  max_audio_seconds: 30\n"
        "training:\n  epochs: 1\n  learning_rate: 0.000002\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["data"]["common_voice_root"] == str(
        (tmp_path / "data/raw/common-voice-vi-21/vi").resolve()
    )


def test_load_config_resolves_fpt_dataset_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  name: vinai/PhoWhisper-base\n  language: Vietnamese\n  task: transcribe\n"
        "data:\n  root: data/raw/vivos\n"
        "  fpt_root: data/raw/fpt-open-speech\n"
        "  max_audio_seconds: 30\n"
        "training:\n  epochs: 1\n  learning_rate: 0.000002\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["data"]["fpt_root"] == str(
        (tmp_path / "data/raw/fpt-open-speech").resolve()
    )


def test_load_config_rejects_non_positive_epochs(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "model:\n  name: openai/whisper-tiny\n  language: Vietnamese\n  task: transcribe\n"
        "data:\n  root: data\n  max_audio_seconds: 30\n"
        "training:\n  epochs: 0\n  learning_rate: 0.00005\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="epochs"):
        load_config(config_path)
