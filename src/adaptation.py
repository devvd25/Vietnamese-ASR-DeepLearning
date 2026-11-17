"""Prepare a disclosed supervised adaptation set without validation leakage."""

from __future__ import annotations

import random
from pathlib import Path

from .data_loader import load_jsonl_manifest, save_jsonl_manifest
from .preprocessing import normalize_transcript


def prepare_supervised_adaptation_manifests(
    *,
    audio_path: str | Path,
    transcript: str,
    validation_manifest: str | Path,
    output_dir: str | Path,
    repetitions: int,
    validation_limit: int,
    seed: int,
) -> dict[str, int | bool]:
    """Repeat one labelled target for adaptation and retain leakage-free validation.

    Repetition is intentional because model dropout supplies stochastic views during
    fine-tuning. The returned metadata explicitly marks the target as trained-on;
    its fit score must never be reported as an independent test result.
    """

    target_audio = Path(audio_path).expanduser().resolve()
    if not target_audio.is_file():
        raise FileNotFoundError(f"Không tìm thấy audio thích nghi: {target_audio}")
    if repetitions < 1:
        raise ValueError("repetitions phải lớn hơn 0.")
    if validation_limit < 1:
        raise ValueError("validation_limit phải lớn hơn 0.")

    target_text = normalize_transcript(transcript)
    train_records = [
        {
            "audio_id": f"st23d_adapt_{index:04d}",
            "audio_path": str(target_audio),
            "transcript": target_text,
            "split": "adaptation_train",
            "source_dataset": "st23d_supervised_adaptation",
            "trained_on_sample": True,
        }
        for index in range(1, repetitions + 1)
    ]

    independent_validation = []
    for record in load_jsonl_manifest(validation_manifest):
        record_audio = Path(str(record.get("audio_path", ""))).expanduser().resolve()
        record_text = normalize_transcript(str(record.get("transcript", "")))
        if record_audio == target_audio or record_text == target_text:
            continue
        independent_validation.append(dict(record))
    if not independent_validation:
        raise ValueError(
            "Không còn bản ghi validation độc lập sau khi loại mẫu mục tiêu."
        )
    random.Random(seed).shuffle(independent_validation)
    independent_validation = independent_validation[:validation_limit]

    destination = Path(output_dir).expanduser().resolve()
    save_jsonl_manifest(train_records, destination / "train.jsonl")
    save_jsonl_manifest(independent_validation, destination / "validation.jsonl")
    return {
        "train_samples": len(train_records),
        "independent_validation_samples": len(independent_validation),
        "trained_on_target_sample": True,
    }
