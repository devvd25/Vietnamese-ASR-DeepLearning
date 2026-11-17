from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.adaptation import prepare_supervised_adaptation_manifests


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_prepare_adaptation_repeats_target_and_keeps_validation_independent(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "st23d.wav"
    audio_path.write_bytes(b"RIFF-test")
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        validation_path,
        [
            {
                "audio_id": "leaked_path",
                "audio_path": str(audio_path),
                "transcript": "nội dung khác",
            },
            {
                "audio_id": "leaked_text",
                "audio_path": str(tmp_path / "other.wav"),
                "transcript": "XIN CHÀO ST23D",
            },
            {
                "audio_id": "independent_1",
                "audio_path": str(tmp_path / "one.wav"),
                "transcript": "một câu độc lập",
            },
            {
                "audio_id": "independent_2",
                "audio_path": str(tmp_path / "two.wav"),
                "transcript": "một câu khác",
            },
        ],
    )

    summary = prepare_supervised_adaptation_manifests(
        audio_path=audio_path,
        transcript="  Xin chào   ST23D  ",
        validation_manifest=validation_path,
        output_dir=tmp_path / "prepared",
        repetitions=3,
        validation_limit=2,
        seed=7,
    )

    train = [
        json.loads(line)
        for line in (tmp_path / "prepared" / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    validation = [
        json.loads(line)
        for line in (tmp_path / "prepared" / "validation.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["audio_id"] for record in train] == [
        "st23d_adapt_0001",
        "st23d_adapt_0002",
        "st23d_adapt_0003",
    ]
    assert {record["audio_path"] for record in train} == {str(audio_path.resolve())}
    assert {record["transcript"] for record in train} == {"xin chào st23d"}
    assert {record["split"] for record in train} == {"adaptation_train"}
    assert {record["trained_on_sample"] for record in train} == {True}
    assert {record["audio_id"] for record in validation} == {
        "independent_1",
        "independent_2",
    }
    assert summary == {
        "train_samples": 3,
        "independent_validation_samples": 2,
        "trained_on_target_sample": True,
    }


def test_prepare_adaptation_rejects_validation_with_only_leaked_target(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "st23d.wav"
    audio_path.write_bytes(b"RIFF-test")
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        validation_path,
        [
            {
                "audio_id": "same_text",
                "audio_path": str(tmp_path / "other.wav"),
                "transcript": "xin chào st23d",
            }
        ],
    )

    with pytest.raises(ValueError, match="validation độc lập"):
        prepare_supervised_adaptation_manifests(
            audio_path=audio_path,
            transcript="xin chào st23d",
            validation_manifest=validation_path,
            output_dir=tmp_path / "prepared",
            repetitions=2,
            validation_limit=1,
            seed=42,
        )
