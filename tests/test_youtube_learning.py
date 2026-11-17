from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.realtime_caption import CapturedLearningWindow
from src.youtube_learning import (
    YouTubeLearningStore,
    build_youtube_continual_manifests,
    choose_consensus_label,
)


def test_consensus_uses_wav2vec2_teacher_only_when_both_models_agree() -> None:
    decision = choose_consensus_label(
        {
            "whisper_fine_tuned": {
                "transcript": "xin chào các bạn hôm nay",
                "confidence": 0.81,
            },
            "wav2vec2_vietnamese": {
                "transcript": "xin chào các bạn hôm nay",
                "confidence": 0.88,
            },
        }
    )

    assert decision is not None
    assert decision["transcript"] == "xin chào các bạn hôm nay"
    assert decision["label_source"] == "wav2vec2_teacher_consensus"
    assert decision["agreement"] == 1.0


def test_consensus_rejects_disagreement_and_low_confidence() -> None:
    disagreement = choose_consensus_label(
        {
            "whisper_fine_tuned": {
                "transcript": "xin chào các bạn hôm nay",
                "confidence": 0.9,
            },
            "wav2vec2_vietnamese": {
                "transcript": "thời tiết ngoài trời mưa lớn",
                "confidence": 0.9,
            },
        }
    )
    low_confidence = choose_consensus_label(
        {
            "whisper_fine_tuned": {
                "transcript": "xin chào các bạn hôm nay",
                "confidence": 0.2,
            },
            "wav2vec2_vietnamese": {
                "transcript": "xin chào các bạn hôm nay",
                "confidence": 0.9,
            },
        }
    )

    assert disagreement is None
    assert low_confidence is None


class _TeacherTranscriber:
    def transcribe_array(self, audio, *, sample_rate: int, **kwargs):
        return {
            "transcript": "xin chào các bạn hôm nay",
            "confidence": 0.91,
            "inference_seconds": 0.01,
            "model_source": "teacher",
            "device": "cpu",
        }


def test_learning_store_writes_deduplicated_audio_and_training_manifest(
    tmp_path: Path,
) -> None:
    store = YouTubeLearningStore(tmp_path)
    window = CapturedLearningWindow(
        sequence=4,
        audio=np.full(16_000, 0.1, dtype=np.float32),
        sample_rate=16_000,
        start_seconds=3.0,
        end_seconds=4.0,
        model_choice="whisper_fine_tuned",
        transcript="xin chào các bạn hôm nay",
        confidence=0.87,
    )

    first = store.evaluate_and_store(
        window,
        secondary_model="wav2vec2_vietnamese",
        secondary_transcriber=_TeacherTranscriber(),
    )
    second = store.evaluate_and_store(
        window,
        secondary_model="wav2vec2_vietnamese",
        secondary_transcriber=_TeacherTranscriber(),
    )

    assert first["accepted"] is True
    assert first["total_samples"] == 1
    assert second["accepted"] is False
    assert second["reason"] == "duplicate"
    records = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["transcript"] == "xin chào các bạn hôm nay"
    assert records[0]["source_dataset"] == "youtube_consensus_learning"
    audio, sample_rate = sf.read(records[0]["audio_path"], dtype="float32")
    assert sample_rate == 16_000
    assert audio.shape == (16_000,)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_continual_manifest_limits_pseudo_labels_to_twenty_percent(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    youtube = tmp_path / "youtube" / "train.jsonl"
    output = tmp_path / "continual"
    _write_jsonl(
        base / "train.jsonl",
        [
            {"audio_id": f"human-{index}", "audio_path": f"human-{index}.wav", "transcript": "dữ liệu thật"}
            for index in range(20)
        ],
    )
    _write_jsonl(base / "validation.jsonl", [{"audio_id": "validation"}])
    _write_jsonl(base / "test.jsonl", [{"audio_id": "test"}])
    _write_jsonl(
        youtube,
        [
            {"audio_id": f"youtube-{index}", "audio_path": f"youtube-{index}.wav", "transcript": "mẫu youtube"}
            for index in range(3)
        ],
    )

    summary = build_youtube_continual_manifests(
        base_processed_dir=base,
        youtube_manifest=youtube,
        output_dir=output,
        minimum_youtube_samples=3,
        replay_per_youtube_sample=4,
        seed=7,
    )

    train_rows = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary == {
        "youtube_samples": 3,
        "human_replay_samples": 12,
        "total_train_samples": 15,
        "pseudo_label_fraction": 0.2,
    }
    assert sum(row["audio_id"].startswith("youtube-") for row in train_rows) == 3
    assert (output / "validation.jsonl").read_text(encoding="utf-8") == (
        base / "validation.jsonl"
    ).read_text(encoding="utf-8")
    assert (output / "test.jsonl").read_text(encoding="utf-8") == (
        base / "test.jsonl"
    ).read_text(encoding="utf-8")
