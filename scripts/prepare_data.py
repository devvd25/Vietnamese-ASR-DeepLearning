from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

from src.cli import configure_utf8_stdio
from src.config import load_config
from src.data_loader import (
    dataset_statistics,
    load_common_voice_manifest,
    load_fpt_open_speech_manifest,
    load_kaggle_vietnamese_speech_manifest,
    load_vivos_split,
    save_jsonl_manifest,
    split_records,
)


def _add_duration(
    records: list[dict],
    max_seconds: float,
    *,
    max_transcript_characters: int = 500,
) -> tuple[list[dict], list[dict]]:
    valid, rejected = [], []
    for record in records:
        try:
            transcript_characters = len(str(record.get("transcript", "")))
            if transcript_characters > max_transcript_characters:
                raise ValueError(f"transcript_characters={transcript_characters}")
            info = sf.info(record["audio_path"])
            duration = float(info.frames / info.samplerate)
            if duration <= 0 or duration > max_seconds:
                raise ValueError(f"duration={duration:.3f}s")
            valid.append({**record, "duration_seconds": duration, "sample_rate": info.samplerate})
        except Exception as exc:  # noqa: BLE001 - record corruptions are logged and skipped.
            rejected.append({"audio_id": record.get("audio_id"), "reason": str(exc)})
    return valid, rejected


def _load_splits(config: dict) -> dict[str, list[dict]]:
    data = config["data"]
    root = Path(data["root"])
    dataset_name = data["dataset_name"].casefold()
    if dataset_name in {
        "vivos_common_voice",
        "vivos_common_voice_fpt",
        "vivos_common_voice_kaggle_speech",
    }:
        common_voice_root = Path(data["common_voice_root"])
        seed = int(config["training"]["seed"])
        vivos_train = load_vivos_split(root, "train")
        vivos_parts = split_records(
            vivos_train,
            ratios=(1.0 - float(data["validation_ratio"]), float(data["validation_ratio"]), 0.0),
            seed=seed,
        )
        common_voice = load_common_voice_manifest(
            common_voice_root / "validated.tsv",
            clips_dir=common_voice_root / "clips",
        )
        common_voice_validation_ratio = float(data["common_voice_validation_ratio"])
        common_voice_test_ratio = float(data["common_voice_test_ratio"])
        common_voice_parts = split_records(
            common_voice,
            ratios=(
                1.0 - common_voice_validation_ratio - common_voice_test_ratio,
                common_voice_validation_ratio,
                common_voice_test_ratio,
            ),
            seed=seed,
        )

        def tagged(records: list[dict], namespace: str, source: str) -> list[dict]:
            tagged_records = []
            for record in records:
                value = dict(record)
                value["audio_id"] = f"{namespace}:{record['audio_id']}"
                if record.get("speaker_id"):
                    value["speaker_id"] = f"{namespace}:{record['speaker_id']}"
                value["source_dataset"] = source
                tagged_records.append(value)
            return tagged_records

        splits = {
            "train": tagged(vivos_parts["train"], "vivos", "vivos")
            + tagged(common_voice_parts["train"], "cv21", "common_voice_vi_21"),
            "validation": tagged(vivos_parts["validation"], "vivos", "vivos")
            + tagged(
                common_voice_parts["validation"], "cv21", "common_voice_vi_21"
            ),
            "test": tagged(load_vivos_split(root, "test"), "vivos", "vivos"),
            "common_voice_test": tagged(
                common_voice_parts["test"], "cv21", "common_voice_vi_21"
            ),
        }
        if dataset_name == "vivos_common_voice_fpt":
            fpt_root = Path(data["fpt_root"])
            fpt_records = load_fpt_open_speech_manifest(
                fpt_root / "transcriptAll.txt",
                audio_dir=fpt_root / "mp3",
            )
            splits["train"].extend(tagged(fpt_records, "fpt", "fpt_open_speech"))
        if dataset_name == "vivos_common_voice_kaggle_speech":
            kaggle_speech_root = Path(data["kaggle_speech_root"])
            kaggle_records = load_kaggle_vietnamese_speech_manifest(
                kaggle_speech_root / "train" / "train.csv"
            )
            kaggle_test_ratio = float(data["kaggle_speech_test_ratio"])
            kaggle_parts = split_records(
                kaggle_records,
                ratios=(1.0 - kaggle_test_ratio, 0.0, kaggle_test_ratio),
                seed=seed,
            )
            splits["train"].extend(
                tagged(
                    kaggle_parts["train"],
                    "kaggle_speech",
                    "kaggle_vietnamese_speech_recognition_data",
                )
            )
            splits["kaggle_speech_test"] = tagged(
                kaggle_parts["test"],
                "kaggle_speech",
                "kaggle_vietnamese_speech_recognition_data",
            )
        return splits
    if dataset_name == "vivos":
        source_train = load_vivos_split(root, "train")
        official_test = load_vivos_split(root, "test")
        train_val = split_records(
            source_train,
            ratios=(1.0 - float(data["validation_ratio"]), float(data["validation_ratio"]), 0.0),
            seed=int(config["training"]["seed"]),
        )
        return {
            "train": train_val["train"],
            "validation": train_val["validation"],
            "test": official_test,
        }
    manifest_names = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}
    return {
        split: load_common_voice_manifest(root / filename, clips_dir=root / "clips")
        for split, filename in manifest_names.items()
    }


def _plot_distributions(splits: dict[str, list[dict]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for field, label, filename in (
        ("duration_seconds", "Thời lượng (giây)", "audio_duration_distribution.png"),
        ("transcript_length", "Độ dài transcript (ký tự)", "transcript_length_distribution.png"),
    ):
        plt.figure(figsize=(9, 5))
        for split, records in splits.items():
            values = [
                len(item["transcript"]) if field == "transcript_length" else item[field]
                for item in records
            ]
            plt.hist(values, bins=30, alpha=0.45, label=split)
        plt.xlabel(label)
        plt.ylabel("Số bản ghi")
        plt.title(f"Phân phối {label.lower()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=160)
        plt.close()


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Chuẩn bị manifest ASR và thống kê dataset thật.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    splits = _load_splits(config)
    processed_dir = Path(config["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rejected_all: list[dict] = []
    summary: dict[str, dict] = {}
    for split, records in splits.items():
        valid, rejected = _add_duration(
            records,
            float(config["data"]["max_audio_seconds"]),
            max_transcript_characters=int(
                config["data"].get("max_transcript_characters", 500)
            ),
        )
        for row in rejected:
            row["split"] = split
        rejected_all.extend(rejected)
        splits[split] = valid
        save_jsonl_manifest(valid, processed_dir / f"{split}.jsonl")
        summary[split] = dataset_statistics(valid)
        summary[split]["records_before_filter"] = len(records)
        summary[split]["records_after_filter"] = len(valid)
        summary[split]["rejected_records"] = len(rejected)
    with (metrics_dir / "dataset_statistics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if rejected_all:
        with (processed_dir / "rejected_records.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["split", "audio_id", "reason"])
            writer.writeheader()
            writer.writerows(rejected_all)
    _plot_distributions(splits, Path(config["paths"]["figures_dir"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
