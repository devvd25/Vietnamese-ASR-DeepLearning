"""Chấm lại mọi file prediction đã lưu bằng chuẩn hóa chấm điểm mới (bỏ dấu câu).

Không chạy lại model: chỉ đọc cột ``reference``/``prediction`` trong
``outputs/**/predictions/*.csv`` và so sánh cách chấm cũ (giữ dấu câu) với cách mới.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import configure_utf8_stdio
from src.metrics import compute_corpus_metrics


def _metrics_json_for(prediction_path: Path) -> Path | None:
    """``x/predictions/<stem>_predictions.csv`` ↔ ``x/tables/<stem>_metrics.json``."""
    stem = prediction_path.name.removesuffix("_predictions.csv")
    candidate = prediction_path.parent.parent / "tables" / f"{stem}_metrics.json"
    return candidate if candidate.is_file() else None


def rescore_file(path: Path, *, root: Path, resamples: int) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"reference", "prediction"}.issubset(rows[0]):
        raise ValueError(f"{path} thiếu cột reference/prediction.")
    references = [row["reference"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    legacy = compute_corpus_metrics(references, predictions, strip_punctuation=False)
    scored = compute_corpus_metrics(
        references, predictions, bootstrap_resamples=resamples
    )
    metrics_path = _metrics_json_for(path)
    recorded = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path else {}
    )
    digest = hashlib.sha256(
        "\n".join(f"{r}\t{p}" for r, p in zip(references, predictions)).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "predictions_csv": path.relative_to(root).as_posix(),
        "metrics_json": metrics_path.relative_to(root).as_posix() if metrics_path else None,
        "model_choice": recorded.get("model_choice"),
        "model_source": recorded.get("model_source"),
        "split": recorded.get("split"),
        "decoder": recorded.get("decoder"),
        "num_samples": scored["num_samples"],
        "references_with_punctuation": sum(
            any(not char.isalnum() and not char.isspace() for char in reference)
            for reference in references
        ),
        "content_sha256_16": digest,
        "legacy": {key: legacy[key] for key in ("wer", "cer", "ser")},
        "recorded_wer": recorded.get("wer"),
        "scored": {
            key: scored[key]
            for key in ("wer", "cer", "ser", "wer_ci95", "word_errors", "reference_words")
        },
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# Chấm lại predictions với chuẩn hóa bỏ dấu câu",
        "",
        "Sinh tự động bởi `scripts/rescore_predictions.py`. *Cũ* = NFC + chữ thường (giữ dấu câu);",
        "*Mới* = thêm bỏ dấu câu (trừ `%`). SER = tỷ lệ câu có ít nhất một lỗi từ.",
        "",
        "| File | Split | n | Ref có dấu câu | WER cũ | WER mới | CI 95% | CER mới | SER mới |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for entry in entries:
        low, high = entry["scored"]["wer_ci95"]
        lines.append(
            f"| `{entry['predictions_csv']}` | {entry['split'] or '?'} | {entry['num_samples']} "
            f"| {entry['references_with_punctuation']} | {_pct(entry['legacy']['wer'])} "
            f"| {_pct(entry['scored']['wer'])} | {_pct(low)}–{_pct(high)} "
            f"| {_pct(entry['scored']['cer'])} | {_pct(entry['scored']['ser'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--destination", default="outputs/rescored")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    outputs = Path(args.outputs)
    destination = Path(args.destination)
    paths = sorted(
        path
        for path in outputs.glob("**/predictions/*_predictions.csv")
        if destination not in path.parents
    )
    entries = [
        rescore_file(path, root=outputs.parent if outputs.parent != Path("") else Path("."),
                     resamples=args.resamples)
        for path in paths
    ]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "rescored_predictions.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "RESCORE_SUMMARY.md").write_text(
        render_markdown(entries), encoding="utf-8"
    )
    for entry in entries:
        print(
            f"{entry['predictions_csv']}: split={entry['split']} n={entry['num_samples']} "
            f"WER {_pct(entry['legacy']['wer'])} -> {_pct(entry['scored']['wer'])}"
        )
    print(f"Đã chấm lại {len(entries)} file → {destination}")


if __name__ == "__main__":
    main()
