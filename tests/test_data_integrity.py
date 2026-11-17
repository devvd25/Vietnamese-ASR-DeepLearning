from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.data_loader import (
    dataset_statistics,
    load_common_voice_manifest,
    load_fpt_open_speech_manifest,
    load_jsonl_manifest,
    load_kaggle_vietnamese_speech_manifest,
    load_vivos_split,
    resolve_manifest_columns,
    save_jsonl_manifest,
    split_records,
)


def test_load_kaggle_vietnamese_speech_manifest_maps_audio_and_text(
    tmp_path: Path,
) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_path = audio_dir / "train-00000.flac"
    audio_path.write_bytes(b"audio")
    manifest = tmp_path / "train.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "audio", "text"])
        writer.writeheader()
        writer.writerow(
            {
                "id": "train-00000",
                "audio": "audio/train-00000.flac",
                "text": "Lớp chúng tôi có ba mươi hai học sinh.",
            }
        )

    records = load_kaggle_vietnamese_speech_manifest(manifest)

    assert records == [
        {
            "audio_id": "train-00000",
            "audio_path": str(audio_path.resolve()),
            "transcript": "lớp chúng tôi có ba mươi hai học sinh.",
            "source_dataset": "kaggle_vietnamese_speech_recognition_data",
        }
    ]


def test_load_fpt_manifest_maps_pipe_rows_to_canonical_audio(tmp_path: Path) -> None:
    audio_dir = tmp_path / "mp3"
    audio_dir.mkdir()
    canonical = audio_dir / "FPTOpenSpeechData_Set001_V0.1_000001.mp3"
    canonical.write_bytes(b"audio")
    (audio_dir / "FPTOpenSpeechData_Set001_V0.1_000001.mp3 1").write_bytes(
        b"duplicate-upload"
    )
    manifest = tmp_path / "transcriptAll.txt"
    manifest.write_text(
        "FPTOpenSpeechData_Set001_V0.1_000001.mp3|Cách để đi|0.00000-1.27298\n",
        encoding="utf-8",
    )

    records = load_fpt_open_speech_manifest(manifest, audio_dir=audio_dir)

    assert records == [
        {
            "audio_id": "FPTOpenSpeechData_Set001_V0.1_000001",
            "audio_path": str(canonical.resolve()),
            "transcript": "cách để đi",
            "source_dataset": "fpt_open_speech",
        }
    ]


def test_load_fpt_manifest_rejects_malformed_timestamp(tmp_path: Path) -> None:
    audio_dir = tmp_path / "mp3"
    audio_dir.mkdir()
    filename = "FPTOpenSpeechData_Set001_V0.1_000001.mp3"
    (audio_dir / filename).write_bytes(b"audio")
    manifest = tmp_path / "transcriptAll.txt"
    manifest.write_text(f"{filename}|cách để đi|not-a-range\n", encoding="utf-8")

    with pytest.raises(ValueError, match="timestamp"):
        load_fpt_open_speech_manifest(manifest, audio_dir=audio_dir)


def test_load_fpt_manifest_accepts_multiple_time_ranges_and_escaped_newline(
    tmp_path: Path,
) -> None:
    audio_dir = tmp_path / "mp3"
    audio_dir.mkdir()
    filename = "FPTOpenSpeechData_Set001_V0.1_000036.mp3"
    (audio_dir / filename).write_bytes(b"audio")
    manifest = tmp_path / "transcriptAll.txt"
    manifest.write_text(
        f"{filename}|địa chỉ email \\r\\n @ gmx|0.00000-10.6710 10.6710-19.8263\n",
        encoding="utf-8",
    )

    records = load_fpt_open_speech_manifest(manifest, audio_dir=audio_dir)

    assert records[0]["transcript"] == "địa chỉ email @ gmx"


def test_load_fpt_manifest_accepts_scientific_notation_timestamp(tmp_path: Path) -> None:
    audio_dir = tmp_path / "mp3"
    audio_dir.mkdir()
    filename = "FPTOpenSpeechData_Set001_V0.1_004524.mp3"
    (audio_dir / filename).write_bytes(b"audio")
    manifest = tmp_path / "transcriptAll.txt"
    manifest.write_text(
        f"{filename}|công nghiệp|6.93889e-17-7.70694\n",
        encoding="utf-8",
    )

    records = load_fpt_open_speech_manifest(manifest, audio_dir=audio_dir)

    assert records[0]["transcript"] == "công nghiệp"


def test_resolve_manifest_columns_supports_common_voice_names() -> None:
    mapping = resolve_manifest_columns(["client_id", "path", "sentence"])
    assert mapping.audio == "path"
    assert mapping.transcript == "sentence"
    assert mapping.speaker == "client_id"


def test_load_manifest_maps_audio_and_transcript(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "sample.mp3").write_bytes(b"not-decoded-in-this-test")
    manifest = tmp_path / "validated.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client_id", "path", "sentence"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"client_id": "speaker-1", "path": "sample.mp3", "sentence": "Xin chào"})

    records = load_common_voice_manifest(manifest, clips_dir=clips)

    assert len(records) == 1
    assert records[0]["audio_id"] == "sample"
    assert records[0]["audio_path"] == str((clips / "sample.mp3").resolve())
    assert records[0]["transcript"] == "xin chào"


def test_load_manifest_reports_missing_audio(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text("path\tsentence\nmissing.mp3\tXin chào\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing.mp3"):
        load_common_voice_manifest(manifest, clips_dir=tmp_path / "clips")


def test_load_tsv_does_not_merge_rows_after_unmatched_quote(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "first.mp3").write_bytes(b"placeholder")
    (clips / "second.mp3").write_bytes(b"placeholder")
    manifest = tmp_path / "validated.tsv"
    manifest.write_text(
        "client_id\tpath\tsentence\n"
        'speaker-1\tfirst.mp3\t"Ăn nữa đi con\n'
        "speaker-1\tsecond.mp3\tXin chào Việt Nam\n",
        encoding="utf-8",
    )

    records = load_common_voice_manifest(manifest, clips_dir=clips)

    assert len(records) == 2
    assert records[1]["audio_id"] == "second"
    assert records[1]["transcript"] == "xin chào việt nam"


def test_speaker_split_prevents_speaker_leakage() -> None:
    records = [
        {"audio_id": f"a{i}", "speaker_id": f"s{i // 2}", "transcript": "xin chào"}
        for i in range(20)
    ]

    splits = split_records(records, seed=42, group_key="speaker_id")

    speaker_sets = [
        {record["speaker_id"] for record in splits[name]}
        for name in ("train", "validation", "test")
    ]
    assert speaker_sets[0].isdisjoint(speaker_sets[1])
    assert speaker_sets[0].isdisjoint(speaker_sets[2])
    assert speaker_sets[1].isdisjoint(speaker_sets[2])
    assert sum(len(part) for part in splits.values()) == len(records)


def test_speaker_split_allows_official_test_to_stay_external() -> None:
    records = [
        {"audio_id": f"a{i}", "speaker_id": f"s{i}", "transcript": "xin chào"}
        for i in range(10)
    ]
    splits = split_records(records, ratios=(0.9, 0.1, 0.0), seed=1)
    assert len(splits["test"]) == 0
    assert len(splits["train"]) == 9
    assert len(splits["validation"]) == 1


def test_speaker_split_assigns_dominant_speaker_to_largest_split() -> None:
    sizes = [60, 15, 10, 5, 5, 5]
    records = []
    audio_index = 0
    for speaker_index, size in enumerate(sizes):
        for _ in range(size):
            records.append(
                {
                    "audio_id": f"a{audio_index}",
                    "speaker_id": f"s{speaker_index}",
                    "transcript": "xin chào",
                }
            )
            audio_index += 1

    splits = split_records(records, ratios=(0.8, 0.1, 0.1), seed=42)

    assert "s0" in {record["speaker_id"] for record in splits["train"]}
    assert splits["validation"]
    assert splits["test"]


def test_dataset_statistics_reports_empty_and_duplicate_transcripts() -> None:
    stats = dataset_statistics(
        [
            {"transcript": "xin chào", "duration_seconds": 1.0},
            {"transcript": "xin chào", "duration_seconds": 3.0},
            {"transcript": "", "duration_seconds": 2.0},
        ]
    )

    assert stats["num_records"] == 3
    assert stats["total_duration_seconds"] == 6.0
    assert stats["mean_duration_seconds"] == 2.0
    assert stats["empty_transcripts"] == 1
    assert stats["duplicate_transcripts"] == 1


def test_load_vivos_split_maps_prompts_to_nested_wave_files(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    wave_dir = split_dir / "waves" / "VIVOSSPK01"
    wave_dir.mkdir(parents=True)
    (wave_dir / "VIVOSSPK01_R001.wav").write_bytes(b"wav-placeholder")
    (split_dir / "prompts.txt").write_text(
        "VIVOSSPK01_R001 XIN CHÀO VIỆT NAM\n", encoding="utf-8"
    )

    records = load_vivos_split(tmp_path, "train")

    assert records == [
        {
            "audio_id": "VIVOSSPK01_R001",
            "audio_path": str((wave_dir / "VIVOSSPK01_R001.wav").resolve()),
            "transcript": "xin chào việt nam",
            "speaker_id": "VIVOSSPK01",
            "split": "train",
        }
    ]


def test_load_vivos_split_uses_parent_directory_for_non_r_audio_ids(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    wave_dir = split_dir / "waves" / "VIVOSSPK13"
    wave_dir.mkdir(parents=True)
    (wave_dir / "VIVOSSPK13_001.wav").write_bytes(b"wav-placeholder")
    (split_dir / "prompts.txt").write_text(
        "VIVOSSPK13_001 XIN CHÀO VIỆT NAM\n", encoding="utf-8"
    )

    records = load_vivos_split(tmp_path, "train")

    assert records[0]["speaker_id"] == "VIVOSSPK13"


def test_jsonl_manifest_round_trip_preserves_unicode(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    expected = [{"audio_id": "a1", "transcript": "tiếng Việt", "duration_seconds": 1.25}]
    save_jsonl_manifest(expected, path)
    assert load_jsonl_manifest(path) == expected
