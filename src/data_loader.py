"""Nạp manifest ASR, tự nhận diện tên cột giữa các phiên bản dataset."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .preprocessing import normalize_transcript


@dataclass(frozen=True)
class ManifestColumns:
    audio: str
    transcript: str
    speaker: str | None = None


def _first_match(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if candidate in columns), None)


def resolve_manifest_columns(columns: Iterable[str]) -> ManifestColumns:
    available = {str(column).strip() for column in columns}
    audio = _first_match(available, ("audio_path", "path", "file", "filename", "audio"))
    transcript = _first_match(available, ("transcript", "sentence", "text", "normalized_text"))
    speaker = _first_match(available, ("client_id", "speaker_id", "speaker", "user_id"))
    if audio is None or transcript is None:
        raise ValueError(
            "Manifest phải có cột audio/path và transcript/sentence; "
            f"cột hiện có: {sorted(available)}"
        )
    return ManifestColumns(audio=audio, transcript=transcript, speaker=speaker)


def load_common_voice_manifest(
    manifest_path: str | Path,
    *,
    clips_dir: str | Path | None = None,
    strict_audio_paths: bool = True,
) -> list[dict[str, str]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest: {path}")
    clips = Path(clips_dir).expanduser().resolve() if clips_dir else path.parent / "clips"
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(
            handle,
            delimiter=delimiter,
            # Common Voice TSV có thể chứa dấu nháy đơn lẻ trong sentence. TSV
            # không dùng dấu nháy làm cú pháp bao trường, nên không được để csv
            # ghép các dòng kế tiếp thành một transcript khổng lồ.
            quoting=csv.QUOTE_NONE if delimiter == "\t" else csv.QUOTE_MINIMAL,
        )
        if reader.fieldnames is None:
            raise ValueError(f"Manifest không có header: {path}")
        mapping = resolve_manifest_columns(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            relative_audio = str(row.get(mapping.audio, "")).strip()
            raw_text = str(row.get(mapping.transcript, ""))
            if not relative_audio:
                raise ValueError(f"Dòng {row_number} thiếu đường dẫn audio.")
            audio_path = Path(relative_audio)
            if not audio_path.is_absolute():
                audio_path = clips / audio_path
            audio_path = audio_path.resolve()
            if strict_audio_paths and not audio_path.is_file():
                raise FileNotFoundError(
                    f"Dòng {row_number}: không tìm thấy audio {relative_audio} tại {audio_path}"
                )
            transcript = normalize_transcript(raw_text)
            record = {
                "audio_id": audio_path.stem,
                "audio_path": str(audio_path),
                "transcript": transcript,
            }
            if mapping.speaker:
                record["speaker_id"] = str(row.get(mapping.speaker, "")).strip()
            records.append(record)
    if not records:
        raise ValueError(f"Manifest không có bản ghi hợp lệ: {path}")
    return records


def load_kaggle_vietnamese_speech_manifest(
    manifest_path: str | Path,
    *,
    strict_audio_paths: bool = True,
) -> list[dict[str, str]]:
    """Load the Kaggle ``id,audio,text`` manifest with paths relative to the CSV."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest Kaggle speech: {path}")
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "audio", "text"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Manifest Kaggle speech phải có các cột id, audio, text."
            )
        for row_number, row in enumerate(reader, start=2):
            audio_id = str(row.get("id", "")).strip()
            relative_audio = str(row.get("audio", "")).strip()
            if not audio_id or not relative_audio:
                raise ValueError(
                    f"Dòng Kaggle speech {row_number} thiếu id hoặc audio."
                )
            if audio_id in seen_ids:
                raise ValueError(f"Trùng audio_id Kaggle speech: {audio_id}")
            seen_ids.add(audio_id)
            audio_path = Path(relative_audio)
            if not audio_path.is_absolute():
                audio_path = path.parent / audio_path
            audio_path = audio_path.resolve()
            if strict_audio_paths and not audio_path.is_file():
                raise FileNotFoundError(
                    f"Dòng Kaggle speech {row_number}: không tìm thấy audio {relative_audio}"
                )
            records.append(
                {
                    "audio_id": audio_id,
                    "audio_path": str(audio_path),
                    "transcript": normalize_transcript(str(row.get("text", ""))),
                    "source_dataset": "kaggle_vietnamese_speech_recognition_data",
                }
            )
    if not records:
        raise ValueError(f"Manifest Kaggle speech không có bản ghi hợp lệ: {path}")
    return records


def load_fpt_open_speech_manifest(
    manifest_path: str | Path,
    *,
    audio_dir: str | Path | None = None,
) -> list[dict[str, str]]:
    """Load the official FPT pipe manifest and ignore Kaggle's suffixed duplicates."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest FPT: {path}")
    audio_root = (
        Path(audio_dir).expanduser().resolve() if audio_dir else path.parent / "mp3"
    )
    timestamp_number = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    time_range = rf"{timestamp_number}-{timestamp_number}"
    timestamp_pattern = re.compile(rf"^{time_range}(?:\s+{time_range})*$")
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                filename_and_text, timestamp = stripped.rsplit("|", 1)
                filename, transcript = filename_and_text.split("|", 1)
            except ValueError as exc:
                raise ValueError(f"Dòng FPT {line_number} không đủ ba trường.") from exc
            filename = filename.strip()
            if Path(filename).name != filename or not filename.casefold().endswith(".mp3"):
                raise ValueError(f"Dòng FPT {line_number} có tên audio không hợp lệ: {filename}")
            if not timestamp_pattern.fullmatch(timestamp.strip()):
                raise ValueError(f"Dòng FPT {line_number} có timestamp không hợp lệ.")
            audio_path = (audio_root / filename).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Dòng FPT {line_number}: không tìm thấy audio {filename}"
                )
            audio_id = Path(filename).stem
            if audio_id in seen_ids:
                raise ValueError(f"Trùng audio_id FPT: {audio_id}")
            seen_ids.add(audio_id)
            records.append(
                {
                    "audio_id": audio_id,
                    "audio_path": str(audio_path),
                    "transcript": normalize_transcript(
                        re.sub(r"\\[rn]", " ", transcript)
                    ),
                    "source_dataset": "fpt_open_speech",
                }
            )
    if not records:
        raise ValueError(f"Manifest FPT không có bản ghi hợp lệ: {path}")
    return records


def split_records(
    records: Iterable[dict[str, str]],
    *,
    seed: int = 42,
    group_key: str = "speaker_id",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[dict[str, str]]]:
    records = list(records)
    if len(ratios) != 3 or any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios phải gồm ba tỷ lệ không âm có tổng bằng 1.")
    if len(records) < 3:
        raise ValueError("Cần ít nhất 3 bản ghi để chia train/validation/test.")
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        audio_id = str(record.get("audio_id", index))
        if audio_id in seen_ids:
            raise ValueError(f"Trùng audio_id: {audio_id}")
        seen_ids.add(audio_id)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, record in enumerate(records):
        group = str(record.get(group_key) or f"__record_{index}")
        grouped[group].append(record)
    groups = list(grouped.items())
    if len(groups) < 3:
        raise ValueError("Cần ít nhất 3 nhóm độc lập để tránh leakage giữa các split.")
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    names = ("train", "validation", "test")
    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    targets = [len(records) * ratio for ratio in ratios]
    totals = [0, 0, 0]
    selected: dict[str, list[tuple[str, list[dict[str, str]]]]] = {
        name: [] for name in names
    }
    for group_index, group in enumerate(groups):
        empty = [index for index in active if not selected[names[index]]]
        groups_remaining = len(groups) - group_index
        candidates = empty if groups_remaining == len(empty) else active
        selected_index = max(
            candidates,
            key=lambda index: (
                (targets[index] - totals[index]) / targets[index],
                ratios[index],
                -index,
            ),
        )
        selected[names[selected_index]].append(group)
        totals[selected_index] += len(group[1])
    return {
        split: [record for _, group_records in parts for record in group_records]
        for split, parts in selected.items()
    }


def dataset_statistics(*args, **kwargs):
    records = list(args[0]) if args else list(kwargs["records"])
    durations = [float(record.get("duration_seconds", 0.0) or 0.0) for record in records]
    transcripts = [str(record.get("transcript", "")).strip() for record in records]
    counts = Counter(text for text in transcripts if text)
    total = sum(durations)
    lengths = [len(text) for text in transcripts]
    return {
        "num_records": len(records),
        "total_duration_seconds": total,
        "total_duration_hours": total / 3600,
        "mean_duration_seconds": total / len(records) if records else 0.0,
        "min_duration_seconds": min(durations, default=0.0),
        "max_duration_seconds": max(durations, default=0.0),
        "empty_transcripts": sum(not text for text in transcripts),
        "duplicate_transcripts": sum(count - 1 for count in counts.values() if count > 1),
        "mean_transcript_characters": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def load_vivos_split(*args, **kwargs):
    root = Path(args[0] if args else kwargs.pop("root")).expanduser().resolve()
    split = str(args[1] if len(args) > 1 else kwargs.pop("split")).casefold()
    if split not in {"train", "test", "validation", "dev"}:
        raise ValueError("VIVOS split phải là train, test, validation hoặc dev.")
    split_dir = root / split
    prompts_path = split_dir / "prompts.txt"
    waves_dir = split_dir / "waves"
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy prompts VIVOS: {prompts_path}")
    if not waves_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục waves VIVOS: {waves_dir}")
    audio_by_id = {path.stem: path.resolve() for path in waves_dir.rglob("*.wav")}
    records: list[dict[str, str]] = []
    with prompts_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Dòng prompts {line_number} không có transcript.")
            audio_id, transcript = parts
            audio_path = audio_by_id.get(audio_id)
            if audio_path is None:
                raise FileNotFoundError(
                    f"Không tìm thấy wave cho {audio_id} (dòng {line_number})."
                )
            # VIVOS dùng nhiều mẫu tên file (_R001, _T001 hoặc _001); thư mục cha
            # mới là định danh người nói ổn định cho mọi mẫu.
            speaker_id = audio_path.parent.name
            records.append(
                {
                    "audio_id": audio_id,
                    "audio_path": str(audio_path),
                    "transcript": normalize_transcript(transcript),
                    "speaker_id": speaker_id,
                    "split": split,
                }
            )
    if not records:
        raise ValueError(f"VIVOS split {split} không có bản ghi.")
    return records


def save_jsonl_manifest(records: Iterable[dict], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_jsonl_manifest(manifest_path: str | Path) -> list[dict]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest JSONL: {path}")
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Dòng {line_number} không phải JSON object.")
            records.append(value)
    if not records:
        raise ValueError(f"Manifest JSONL rỗng: {path}")
    return records
