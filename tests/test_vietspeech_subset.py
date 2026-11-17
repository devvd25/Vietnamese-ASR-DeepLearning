from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf

import src.vietspeech as vietspeech
from src.data_loader import load_jsonl_manifest, save_jsonl_manifest
from src.vietspeech import materialize_vietspeech_subset


def _wav_bytes(seconds: float, sample_rate: int = 16_000) -> bytes:
    buffer = BytesIO()
    sf.write(
        buffer,
        np.zeros(round(seconds * sample_rate), dtype=np.float32),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


def _row(seconds: float, transcript: str, index: int) -> dict:
    return {
        "audio": {
            "bytes": _wav_bytes(seconds),
            "path": f"sample-{index}.wav",
        },
        "transcription": transcript,
    }


def test_materialize_vietspeech_subset_filters_leaks_and_invalid_audio(
    tmp_path: Path,
) -> None:
    rows = [
        _row(2.0, "Xin ch\u00e0o \u0111\u1ed9c l\u1eadp", 1),
        _row(0.5, "qu\u00e1 ng\u1eafn", 2),
        _row(2.0, "C\u00e2u h\u1ee3p l\u1ec7 th\u1ee9 nh\u1ea5t", 3),
        _row(2.0, "C\u00e2u h\u1ee3p l\u1ec7 th\u1ee9 nh\u1ea5t", 4),
        _row(30.0, "qu\u00e1 d\u00e0i", 5),
        _row(2.0, "C\u00e2u h\u1ee3p l\u1ec7 th\u1ee9 hai", 6),
    ]

    records, summary = materialize_vietspeech_subset(
        rows,
        output_dir=tmp_path / "subset",
        target_duration_seconds=3.5,
        min_audio_seconds=1.0,
        max_audio_seconds=28.0,
        excluded_transcripts={"xin ch\u00e0o \u0111\u1ed9c l\u1eadp"},
    )

    assert [record["transcript"] for record in records] == [
        "c\u00e2u h\u1ee3p l\u1ec7 th\u1ee9 nh\u1ea5t",
        "c\u00e2u h\u1ee3p l\u1ec7 th\u1ee9 hai",
    ]
    assert all(Path(record["audio_path"]).is_file() for record in records)
    assert all(record["source_dataset"] == "NhutP/VietSpeech" for record in records)
    assert summary["accepted_records"] == 2
    assert summary["accepted_duration_seconds"] == 4.0
    assert summary["rejected_by_reason"] == {
        "independent_test_overlap": 1,
        "duration_out_of_range": 2,
        "duplicate_transcript": 1,
    }


def test_build_training_manifests_adds_vietspeech_to_train_only(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "base"
    base_records = {
        "train": [{"audio_id": "base-train", "transcript": "train"}],
        "validation": [{"audio_id": "base-validation", "transcript": "validation"}],
        "test": [{"audio_id": "base-test", "transcript": "test"}],
        "common_voice_test": [{"audio_id": "base-cv", "transcript": "cv"}],
    }
    for split, records in base_records.items():
        save_jsonl_manifest(records, base_dir / f"{split}.jsonl")
    subset = [
        {
            "audio_id": "vietspeech:new",
            "audio_path": "new.wav",
            "transcript": "c\u00e2u m\u1edbi",
            "source_dataset": "NhutP/VietSpeech",
        }
    ]

    summary = vietspeech.build_vietspeech_training_manifests(
        base_processed_dir=base_dir,
        vietspeech_records=subset,
        output_dir=tmp_path / "combined",
    )

    assert [
        row["audio_id"]
        for row in load_jsonl_manifest(tmp_path / "combined" / "train.jsonl")
    ] == ["base-train", "vietspeech:new"]
    for split in ("validation", "test", "common_voice_test"):
        assert load_jsonl_manifest(
            tmp_path / "combined" / f"{split}.jsonl"
        ) == base_records[split]
    assert summary == {
        "train": 2,
        "validation": 1,
        "test": 1,
        "common_voice_test": 1,
        "vietspeech_added_to_train": 1,
    }
