from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.prepare_data import _add_duration, _load_splits


def _write_vivos(root: Path, split: str, speaker_ids: list[str]) -> None:
    split_dir = root / split
    lines = []
    for index, speaker_id in enumerate(speaker_ids):
        audio_id = f"{speaker_id}_R{index:03d}"
        wave_dir = split_dir / "waves" / speaker_id
        wave_dir.mkdir(parents=True, exist_ok=True)
        (wave_dir / f"{audio_id}.wav").write_bytes(b"placeholder")
        lines.append(f"{audio_id} XIN CHÀO VIỆT NAM")
    (split_dir / "prompts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_common_voice(root: Path, speakers: int = 6) -> None:
    clips = root / "clips"
    clips.mkdir(parents=True)
    manifest = root / "validated.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["client_id", "path", "sentence"],
            delimiter="\t",
        )
        writer.writeheader()
        for index in range(speakers):
            filename = f"cv-{index}.mp3"
            (clips / filename).write_bytes(b"placeholder")
            writer.writerow(
                {
                    "client_id": f"cv-speaker-{index}",
                    "path": filename,
                    "sentence": "Đây là câu tiếng Việt",
                }
            )


def _write_fpt(root: Path) -> None:
    audio_dir = root / "mp3"
    audio_dir.mkdir(parents=True)
    filename = "FPTOpenSpeechData_Set001_V0.1_000001.mp3"
    (audio_dir / filename).write_bytes(b"placeholder")
    (root / "transcriptAll.txt").write_text(
        f"{filename}|Đây là dữ liệu FPT|0.00000-1.00000\n",
        encoding="utf-8",
    )


def _write_kaggle_speech(root: Path, records: int = 4) -> None:
    train_dir = root / "train"
    audio_dir = train_dir / "audio"
    audio_dir.mkdir(parents=True)
    with (train_dir / "train.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "audio", "text"])
        writer.writeheader()
        for index in range(records):
            audio_id = f"train-{index:05d}"
            filename = f"{audio_id}.flac"
            (audio_dir / filename).write_bytes(b"placeholder")
            writer.writerow(
                {
                    "id": audio_id,
                    "audio": f"audio/{filename}",
                    "text": f"Đây là câu số {index}",
                }
            )


def test_load_mixed_corpus_keeps_vivos_test_external(tmp_path: Path) -> None:
    vivos_root = tmp_path / "vivos"
    common_voice_root = tmp_path / "common-voice"
    _write_vivos(vivos_root, "train", ["VIVOS01", "VIVOS02", "VIVOS03", "VIVOS04"])
    _write_vivos(vivos_root, "test", ["VIVOSTEST"])
    _write_common_voice(common_voice_root)
    config = {
        "data": {
            "dataset_name": "vivos_common_voice",
            "root": str(vivos_root),
            "common_voice_root": str(common_voice_root),
            "validation_ratio": 0.25,
            "common_voice_validation_ratio": 0.2,
            "common_voice_test_ratio": 0.2,
        },
        "training": {"seed": 42},
    }

    splits = _load_splits(config)

    assert set(splits) == {"train", "validation", "test", "common_voice_test"}
    assert {row["source_dataset"] for row in splits["train"]} == {
        "vivos",
        "common_voice_vi_21",
    }
    assert {row["source_dataset"] for row in splits["validation"]} == {
        "vivos",
        "common_voice_vi_21",
    }
    assert {row["source_dataset"] for row in splits["test"]} == {"vivos"}
    assert {row["source_dataset"] for row in splits["common_voice_test"]} == {
        "common_voice_vi_21"
    }
    assert all(":" in row["audio_id"] for rows in splits.values() for row in rows)
    train_speakers = {row["speaker_id"] for row in splits["train"]}
    validation_speakers = {row["speaker_id"] for row in splits["validation"]}
    assert train_speakers.isdisjoint(validation_speakers)


def test_load_three_corpora_adds_fpt_only_to_training(tmp_path: Path) -> None:
    vivos_root = tmp_path / "vivos"
    common_voice_root = tmp_path / "common-voice"
    fpt_root = tmp_path / "fpt"
    _write_vivos(vivos_root, "train", ["VIVOS01", "VIVOS02", "VIVOS03", "VIVOS04"])
    _write_vivos(vivos_root, "test", ["VIVOSTEST"])
    _write_common_voice(common_voice_root)
    _write_fpt(fpt_root)
    config = {
        "data": {
            "dataset_name": "vivos_common_voice_fpt",
            "root": str(vivos_root),
            "common_voice_root": str(common_voice_root),
            "fpt_root": str(fpt_root),
            "validation_ratio": 0.25,
            "common_voice_validation_ratio": 0.2,
            "common_voice_test_ratio": 0.2,
        },
        "training": {"seed": 42},
    }

    splits = _load_splits(config)

    assert {row["source_dataset"] for row in splits["train"]} == {
        "vivos",
        "common_voice_vi_21",
        "fpt_open_speech",
    }
    assert {row["source_dataset"] for row in splits["validation"]} == {
        "vivos",
        "common_voice_vi_21",
    }
    assert {row["source_dataset"] for row in splits["test"]} == {"vivos"}
    assert {row["source_dataset"] for row in splits["common_voice_test"]} == {
        "common_voice_vi_21"
    }


def test_load_kaggle_speech_keeps_new_test_split_out_of_training(
    tmp_path: Path,
) -> None:
    vivos_root = tmp_path / "vivos"
    common_voice_root = tmp_path / "common-voice"
    kaggle_speech_root = tmp_path / "kaggle-speech"
    _write_vivos(vivos_root, "train", ["VIVOS01", "VIVOS02", "VIVOS03", "VIVOS04"])
    _write_vivos(vivos_root, "test", ["VIVOSTEST"])
    _write_common_voice(common_voice_root)
    _write_kaggle_speech(kaggle_speech_root)
    config = {
        "data": {
            "dataset_name": "vivos_common_voice_kaggle_speech",
            "root": str(vivos_root),
            "common_voice_root": str(common_voice_root),
            "kaggle_speech_root": str(kaggle_speech_root),
            "validation_ratio": 0.25,
            "common_voice_validation_ratio": 0.2,
            "common_voice_test_ratio": 0.2,
            "kaggle_speech_test_ratio": 0.25,
        },
        "training": {"seed": 42},
    }

    splits = _load_splits(config)

    assert set(splits) == {
        "train",
        "validation",
        "test",
        "common_voice_test",
        "kaggle_speech_test",
    }
    kaggle_train = {
        row["audio_id"]
        for row in splits["train"]
        if row["source_dataset"] == "kaggle_vietnamese_speech_recognition_data"
    }
    kaggle_test = {row["audio_id"] for row in splits["kaggle_speech_test"]}
    assert kaggle_train
    assert kaggle_test
    assert kaggle_train.isdisjoint(kaggle_test)
    assert {row["source_dataset"] for row in splits["kaggle_speech_test"]} == {
        "kaggle_vietnamese_speech_recognition_data"
    }


def test_add_duration_rejects_implausibly_long_transcript(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.zeros(16_000, dtype=np.float32), 16_000)
    records = [
        {
            "audio_id": "valid",
            "audio_path": str(audio_path),
            "transcript": "xin chào",
        },
        {
            "audio_id": "malformed",
            "audio_path": str(audio_path),
            "transcript": "a" * 501,
        },
    ]

    valid, rejected = _add_duration(
        records,
        max_seconds=30,
        max_transcript_characters=500,
    )

    assert [row["audio_id"] for row in valid] == ["valid"]
    assert rejected[0]["audio_id"] == "malformed"
    assert "transcript_characters=501" in rejected[0]["reason"]
