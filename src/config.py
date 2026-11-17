"""Nạp và kiểm tra cấu hình YAML tái lập được."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .model_utils import validate_model_config


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Cấu hình YAML phải là một mapping.")
    config["model"] = validate_model_config(config.get("model", {}))
    data = config.get("data")
    training = config.get("training")
    if not isinstance(data, dict) or not isinstance(training, dict):
        raise TypeError("Cấu hình phải có các nhóm data và training.")
    if float(data.get("max_audio_seconds", 0)) <= 0:
        raise ValueError("data.max_audio_seconds phải dương.")
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("training.epochs phải dương.")
    if float(training.get("learning_rate", 0)) <= 0:
        raise ValueError("training.learning_rate phải dương.")
    base_dir = path.parent
    for key in ("root", "common_voice_root", "fpt_root", "kaggle_speech_root"):
        if key in data:
            root = Path(str(data[key]))
            data[key] = str(
                (base_dir / root).resolve() if not root.is_absolute() else root.resolve()
            )
    paths = config.setdefault("paths", {})
    for key, value in list(paths.items()):
        candidate = Path(str(value))
        paths[key] = str(
            (base_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        )
    config["_config_path"] = str(path)
    return config
