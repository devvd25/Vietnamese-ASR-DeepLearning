"""Utilities for building a bounded local subset of VietSpeech."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import soundfile as sf

from .data_loader import load_jsonl_manifest, save_jsonl_manifest
from .preprocessing import normalize_transcript


def materialize_vietspeech_subset(
    rows: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    target_duration_seconds: float,
    min_audio_seconds: float,
    max_audio_seconds: float,
    excluded_transcripts: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_duration_seconds <= 0:
        raise ValueError("target_duration_seconds must be positive")
    if min_audio_seconds <= 0 or max_audio_seconds < min_audio_seconds:
        raise ValueError("invalid audio duration bounds")

    destination = Path(output_dir).expanduser().resolve()
    audio_dir = destination / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    excluded = {
        normalize_transcript(value)
        for value in (excluded_transcripts or set())
        if normalize_transcript(value)
    }
    accepted_transcripts: set[str] = set()
    rejected: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    accepted_duration = 0.0

    for row in rows:
        transcript = normalize_transcript(str(row.get("transcription", "")))
        if not transcript:
            rejected["invalid_transcript"] += 1
            continue
        if transcript in excluded:
            rejected["independent_test_overlap"] += 1
            continue
        if transcript in accepted_transcripts:
            rejected["duplicate_transcript"] += 1
            continue

        audio = row.get("audio")
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            rejected["missing_audio_bytes"] += 1
            continue
        try:
            info = sf.info(BytesIO(audio_bytes))
            duration = float(info.frames / info.samplerate)
        except Exception:  # noqa: BLE001 - corrupt source rows are counted and skipped.
            rejected["invalid_audio"] += 1
            continue
        if duration < min_audio_seconds or duration > max_audio_seconds:
            rejected["duration_out_of_range"] += 1
            continue

        original_path = str(audio.get("path", ""))
        suffix = Path(original_path).suffix.casefold()
        if not suffix or len(suffix) > 10 or not suffix[1:].isalnum():
            suffix = ".wav"
        digest = hashlib.sha256(
            audio_bytes + b"\0" + transcript.encode("utf-8")
        ).hexdigest()
        audio_path = audio_dir / f"vietspeech-{digest[:20]}{suffix}"
        if not audio_path.exists():
            audio_path.write_bytes(audio_bytes)
        records.append(
            {
                "audio_id": f"vietspeech:{digest[:20]}",
                "audio_path": str(audio_path),
                "transcript": transcript,
                "source_dataset": "NhutP/VietSpeech",
                "duration_seconds": duration,
                "sample_rate": int(info.samplerate),
            }
        )
        accepted_transcripts.add(transcript)
        accepted_duration += duration
        if accepted_duration >= target_duration_seconds:
            break

    return records, {
        "accepted_records": len(records),
        "accepted_duration_seconds": accepted_duration,
        "rejected_by_reason": dict(rejected),
    }


def build_vietspeech_training_manifests(
    *,
    base_processed_dir: str | Path,
    vietspeech_records: Iterable[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, int]:
    """Add VietSpeech to train while preserving every independent split."""
    base = Path(base_processed_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    train_manifest = base / "train.jsonl"
    if not train_manifest.is_file():
        raise FileNotFoundError(f"Missing base training manifest: {train_manifest}")
    additions = list(vietspeech_records)
    if not additions:
        raise ValueError("VietSpeech subset is empty")

    manifests = sorted(base.glob("*.jsonl"))
    summary: dict[str, int] = {}
    for manifest in manifests:
        split = manifest.stem
        records = load_jsonl_manifest(manifest)
        if split == "train":
            existing_ids = {str(record.get("audio_id", "")) for record in records}
            duplicate_ids = {
                str(record.get("audio_id", ""))
                for record in additions
                if str(record.get("audio_id", "")) in existing_ids
            }
            if duplicate_ids:
                raise ValueError(
                    f"Duplicate audio_id across base and VietSpeech: {sorted(duplicate_ids)}"
                )
            records.extend(additions)
        save_jsonl_manifest(records, destination / manifest.name)
        summary[split] = len(records)
    summary["vietspeech_added_to_train"] = len(additions)
    return summary
