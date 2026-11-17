"""Controlled pseudo-labelling for local YouTube continual learning."""

from __future__ import annotations

import hashlib
import json
import random
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .preprocessing import normalize_transcript
from .realtime_caption import CapturedLearningWindow


def _word_edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_word in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_word in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_word != right_word),
                )
            )
        previous = current
    return previous[-1]


def _word_agreement(left: str, right: str) -> float:
    left_words = normalize_transcript(left, reject_empty=False).split()
    right_words = normalize_transcript(right, reject_empty=False).split()
    denominator = max(len(left_words), len(right_words), 1)
    return max(0.0, 1.0 - _word_edit_distance(left_words, right_words) / denominator)


def choose_consensus_label(
    outputs: Mapping[str, Mapping[str, Any]],
    *,
    minimum_confidence: float = 0.45,
    minimum_agreement: float = 0.85,
    minimum_words: int = 3,
) -> dict[str, Any] | None:
    """Accept only high-confidence agreement and use Wav2Vec2 as teacher."""

    required = {"whisper_fine_tuned", "wav2vec2_vietnamese"}
    if set(outputs) != required:
        return None
    whisper = outputs["whisper_fine_tuned"]
    teacher = outputs["wav2vec2_vietnamese"]
    whisper_text = normalize_transcript(
        str(whisper.get("transcript", "")), reject_empty=False
    )
    teacher_text = normalize_transcript(
        str(teacher.get("transcript", "")), reject_empty=False
    )
    if min(len(whisper_text.split()), len(teacher_text.split())) < minimum_words:
        return None
    confidences: list[float] = []
    for output in (whisper, teacher):
        confidence = output.get("confidence")
        if confidence is None:
            return None
        numeric = float(confidence)
        if not np.isfinite(numeric) or numeric < minimum_confidence:
            return None
        confidences.append(numeric)
    agreement = _word_agreement(whisper_text, teacher_text)
    if agreement < minimum_agreement:
        return None
    return {
        "transcript": teacher_text,
        "label_source": "wav2vec2_teacher_consensus",
        "agreement": agreement,
        "whisper_confidence": confidences[0],
        "wav2vec2_confidence": confidences[1],
    }


class YouTubeLearningStore:
    """Persist accepted consensus windows as a local Whisper training manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.audio_dir = self.root / "audio"
        self.manifest_path = self.root / "train.jsonl"
        self.audit_path = self.root / "audit.jsonl"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._known_ids: set[str] = set()
        if self.manifest_path.is_file():
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._known_ids.add(str(json.loads(line).get("audio_id", "")))

    @property
    def total_samples(self) -> int:
        return len(self._known_ids)

    @staticmethod
    def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            handle.flush()

    def evaluate_and_store(
        self,
        window: CapturedLearningWindow,
        *,
        secondary_model: str,
        secondary_transcriber: Any,
        secondary_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        secondary = secondary_transcriber.transcribe_array(
            window.audio,
            sample_rate=window.sample_rate,
            max_audio_seconds=window.end_seconds - window.start_seconds,
            **dict(secondary_options or {}),
        )
        outputs = {
            window.model_choice: {
                "transcript": window.transcript,
                "confidence": window.confidence,
            },
            secondary_model: secondary,
        }
        decision = choose_consensus_label(outputs)
        if decision is None:
            with self._lock:
                self._append_jsonl(
                    self.audit_path,
                    {
                        "sequence": window.sequence,
                        "accepted": False,
                        "reason": "models_disagree_or_low_confidence",
                    },
                )
            return {
                "accepted": False,
                "reason": "models_disagree_or_low_confidence",
                "total_samples": self.total_samples,
            }

        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(window.audio, dtype="<f4").tobytes())
        digest.update(decision["transcript"].encode("utf-8"))
        audio_id = f"youtube-{digest.hexdigest()[:20]}"
        with self._lock:
            if audio_id in self._known_ids:
                return {
                    "accepted": False,
                    "reason": "duplicate",
                    "total_samples": self.total_samples,
                }
            audio_path = (self.audio_dir / f"{audio_id}.wav").resolve()
            sf.write(audio_path, window.audio, window.sample_rate, subtype="PCM_16")
            record = {
                "audio_id": audio_id,
                "audio_path": str(audio_path),
                "transcript": decision["transcript"],
                "source_dataset": "youtube_consensus_learning",
                "label_source": decision["label_source"],
                "agreement": decision["agreement"],
                "whisper_confidence": decision["whisper_confidence"],
                "wav2vec2_confidence": decision["wav2vec2_confidence"],
                "duration_seconds": window.end_seconds - window.start_seconds,
            }
            self._append_jsonl(self.manifest_path, record)
            self._append_jsonl(
                self.audit_path,
                {"sequence": window.sequence, "accepted": True, **record},
            )
            self._known_ids.add(audio_id)
            return {
                "accepted": True,
                "reason": "consensus",
                "total_samples": self.total_samples,
                "audio_id": audio_id,
                "agreement": decision["agreement"],
            }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def build_youtube_continual_manifests(
    *,
    base_processed_dir: str | Path,
    youtube_manifest: str | Path,
    output_dir: str | Path,
    minimum_youtube_samples: int = 20,
    replay_per_youtube_sample: int = 4,
    seed: int = 42,
) -> dict[str, int | float]:
    """Mix at most 20% pseudo labels with deterministic human-labelled replay."""

    base = Path(base_processed_dir).expanduser().resolve()
    youtube_path = Path(youtube_manifest).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    youtube_records = _read_jsonl(youtube_path)
    if len(youtube_records) < int(minimum_youtube_samples):
        raise ValueError(
            f"Cần ít nhất {minimum_youtube_samples} mẫu YouTube đồng thuận; "
            f"hiện có {len(youtube_records)}."
        )
    human_records = _read_jsonl(base / "train.jsonl")
    replay_count = min(
        len(human_records), len(youtube_records) * int(replay_per_youtube_sample)
    )
    replay = random.Random(seed).sample(human_records, replay_count)
    combined = replay + youtube_records
    random.Random(seed).shuffle(combined)
    _write_jsonl(output / "train.jsonl", combined)
    for split in ("validation", "test"):
        source = base / f"{split}.jsonl"
        if source.is_file():
            _write_jsonl(output / f"{split}.jsonl", _read_jsonl(source))
    fraction = len(youtube_records) / len(combined)
    return {
        "youtube_samples": len(youtube_records),
        "human_replay_samples": replay_count,
        "total_train_samples": len(combined),
        "pseudo_label_fraction": fraction,
    }
