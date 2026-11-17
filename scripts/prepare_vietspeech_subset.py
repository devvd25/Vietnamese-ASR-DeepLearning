from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import configure_utf8_stdio
from src.data_loader import load_jsonl_manifest, save_jsonl_manifest
from src.vietspeech import (
    build_vietspeech_training_manifests,
    materialize_vietspeech_subset,
)


def _excluded_transcripts(base_processed_dir: Path) -> set[str]:
    excluded: set[str] = set()
    for manifest in sorted(base_processed_dir.glob("*.jsonl")):
        for record in load_jsonl_manifest(manifest):
            transcript = str(record.get("transcript", "")).strip()
            if transcript:
                excluded.add(transcript)
    return excluded


def _bounded_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_scanned_records: int,
    progress_interval: int = 2_000,
) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(rows, start=1):
        if index > max_scanned_records:
            break
        if index == 1 or index % progress_interval == 0:
            print(f"VietSpeech rows scanned: {index:,}", flush=True)
        yield row


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Stream a bounded VietSpeech subset, remove transcript overlap, "
            "and add it only to the local training manifest."
        )
    )
    parser.add_argument("--repo-id", default="NhutP/VietSpeech")
    parser.add_argument("--target-hours", type=float, default=10.0)
    parser.add_argument("--min-audio-seconds", type=float, default=1.0)
    parser.add_argument("--max-audio-seconds", type=float, default=28.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=2_000)
    parser.add_argument("--max-scanned-records", type=int, default=100_000)
    parser.add_argument(
        "--base-processed-dir",
        default="data/processed/vivos_cv21",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/vietspeech_subset_10h",
    )
    parser.add_argument(
        "--combined-processed-dir",
        default="data/processed/vivos_cv21_vietspeech_10h",
    )
    parser.add_argument(
        "--summary-path",
        default="outputs/vietspeech_subset/selection_summary.json",
    )
    args = parser.parse_args()

    if args.target_hours <= 0:
        parser.error("--target-hours must be positive")
    if args.shuffle_buffer <= 0 or args.max_scanned_records <= 0:
        parser.error("shuffle and scan limits must be positive")

    from datasets import Audio, load_dataset

    base_dir = Path(args.base_processed_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    combined_dir = Path(args.combined_processed_dir).expanduser().resolve()
    target_seconds = float(args.target_hours) * 3_600
    excluded = _excluded_transcripts(base_dir)
    print(f"Excluded existing transcripts: {len(excluded):,}", flush=True)

    stream = load_dataset(args.repo_id, split="train", streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    records, selection = materialize_vietspeech_subset(
        _bounded_rows(
            stream,
            max_scanned_records=args.max_scanned_records,
        ),
        output_dir=output_dir,
        target_duration_seconds=target_seconds,
        min_audio_seconds=float(args.min_audio_seconds),
        max_audio_seconds=float(args.max_audio_seconds),
        excluded_transcripts=excluded,
    )
    if selection["accepted_duration_seconds"] < target_seconds:
        raise RuntimeError(
            "VietSpeech stream ended before the requested duration was collected: "
            f"{selection['accepted_duration_seconds'] / 3600:.3f}h"
        )

    save_jsonl_manifest(records, output_dir / "subset.jsonl")
    manifests = build_vietspeech_training_manifests(
        base_processed_dir=base_dir,
        vietspeech_records=records,
        output_dir=combined_dir,
    )
    summary = {
        "dataset": args.repo_id,
        "license": "apache-2.0",
        "streaming": True,
        "target_hours": float(args.target_hours),
        "selection": selection,
        "manifests": manifests,
        "base_processed_dir": str(base_dir),
        "combined_processed_dir": str(combined_dir),
    }
    summary_path = Path(args.summary_path).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
