"""Đánh giá định lượng và phân tích lỗi cho kết quả ASR thật."""

from __future__ import annotations

import csv
import json
import math
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .metrics import (
    calculate_cer,
    calculate_wer,
    compute_corpus_metrics,
    normalize_for_scoring,
)


def describe_decoder(transcriber, num_beams: int) -> str:
    """Tên decoder thực tế để artifact đánh giá không ghi nhầm greedy/LM."""
    from .inference import Wav2Vec2Transcriber

    if isinstance(transcriber, Wav2Vec2Transcriber):
        return str(getattr(transcriber, "decoder_name", "ctc_greedy"))
    return f"whisper_beam_{num_beams}"


def summarize_confidence(values: Sequence[float | None]) -> float | None:
    """Tính trung bình proxy hữu hạn; ``None`` nếu không có giá trị hợp lệ."""
    available = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(available) / len(available) if available else None


def summarize_by_duration(samples: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Tính WER/CER corpus cho ba nhóm thời lượng cố định."""
    groups: dict[str, list[dict[str, Any]]] = {"<3s": [], "3-6s": [], ">=6s": []}
    for sample in samples:
        duration = float(sample["duration_seconds"])
        label = "<3s" if duration < 3.0 else "3-6s" if duration < 6.0 else ">=6s"
        groups[label].append(sample)
    summary: dict[str, dict[str, Any]] = {}
    for label, rows in groups.items():
        if not rows:
            summary[label] = {"num_samples": 0, "wer": None, "cer": None}
            continue
        summary[label] = compute_corpus_metrics(
            [str(row["reference"]) for row in rows],
            [str(row["prediction"]) for row in rows],
        )
    return summary


def _strip_vietnamese_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _word_error_counts(reference: str, prediction: str) -> tuple[int, int, int]:
    """Trả về số thay thế, xóa và chèn theo căn chỉnh Levenshtein tối ưu."""
    ref_words = reference.split()
    pred_words = prediction.split()
    rows = len(ref_words) + 1
    columns = len(pred_words) + 1
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)
    ]
    for i in range(1, rows):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, columns):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, rows):
        for j in range(1, columns):
            if ref_words[i - 1] == pred_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            diagonal = dp[i - 1][j - 1]
            above = dp[i - 1][j]
            left = dp[i][j - 1]
            candidates = (
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (above[0] + 1, above[1], above[2] + 1, above[3]),
                (left[0] + 1, left[1], left[2], left[3] + 1),
            )
            dp[i][j] = min(candidates, key=lambda item: item[0])
    _, substitutions, deletions, insertions = dp[-1][-1]
    return substitutions, deletions, insertions


def _error_note(
    reference: str,
    prediction: str,
    *,
    substitutions: int,
    deletions: int,
    insertions: int,
) -> str:
    if reference == prediction:
        return "Khớp hoàn toàn"
    if _strip_vietnamese_diacritics(reference) == _strip_vietnamese_diacritics(prediction):
        return "Sai dấu tiếng Việt"
    if deletions and not substitutions and not insertions:
        return "Thiếu từ"
    if insertions and not substitutions and not deletions:
        return "Thừa từ"
    if substitutions and not deletions and not insertions:
        return "Thay thế từ"
    return (
        "Lỗi hỗn hợp "
        f"(thay thế={substitutions}, thiếu={deletions}, thừa={insertions})"
    )


def evaluate_predictions(
    audio_ids: Sequence[str],
    references: Sequence[str],
    predictions: Sequence[str],
) -> dict[str, Any]:
    if not (len(audio_ids) == len(references) == len(predictions)):
        raise ValueError("audio_ids, references và predictions phải cùng độ dài.")
    metrics = compute_corpus_metrics(references, predictions, bootstrap_resamples=2000)
    scored_pairs = [
        (normalize_for_scoring(reference), normalize_for_scoring(prediction))
        for reference, prediction in zip(references, predictions)
    ]
    exact_match_count = sum(reference == prediction for reference, prediction in scored_pairs)
    metrics["exact_match_count"] = exact_match_count
    metrics["exact_match_rate"] = exact_match_count / len(references) if references else 0.0
    samples = []
    for audio_id, reference, prediction, (scored_ref, scored_pred) in zip(
        audio_ids, references, predictions, scored_pairs
    ):
        substitutions, deletions, insertions = _word_error_counts(scored_ref, scored_pred)
        samples.append(
            {
                "audio_id": audio_id,
                "reference": reference,
                "prediction": prediction,
                "wer": calculate_wer(reference, prediction),
                "cer": calculate_cer(reference, prediction),
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
                "error_note": _error_note(
                    scored_ref,
                    scored_pred,
                    substitutions=substitutions,
                    deletions=deletions,
                    insertions=insertions,
                ),
            }
        )
    return {"metrics": metrics, "samples": samples}


def run_model_evaluation(
    config: dict[str, Any],
    *,
    model_choice: str,
    split: str = "test",
    limit: int | None = None,
    num_beams: int | None = None,
) -> dict[str, Any]:
    """Chạy inference local tuần tự và lưu prediction/reference thật."""
    from .data_loader import load_jsonl_manifest
    from .inference import Wav2Vec2Transcriber, load_transcriber

    manifest = Path(config["paths"]["processed_data"]) / f"{split}.jsonl"
    records = load_jsonl_manifest(manifest)
    requested_limit = limit or config["data"].get(f"max_{split}_samples")
    if requested_limit:
        records = records[: int(requested_limit)]
    beams = int(num_beams or config.get("evaluation", {}).get("generation_num_beams", 1))
    generation = config.get("evaluation", {})
    transcriber = load_transcriber(model_choice, config)
    audio_ids: list[str] = []
    references: list[str] = []
    predictions: list[str] = []
    timings: list[float] = []
    durations: list[float] = []
    confidences: list[float | None] = []
    confidence_methods: list[str] = []
    for record in records:
        if isinstance(transcriber, Wav2Vec2Transcriber):
            output = transcriber.transcribe_file(
                record["audio_path"],
                max_audio_seconds=float(config["data"]["max_audio_seconds"]),
            )
        else:
            output = transcriber.transcribe_file(
                record["audio_path"],
                num_beams=beams,
                max_audio_seconds=float(config["data"]["max_audio_seconds"]),
                max_length=int(generation.get("max_length", 225)),
                no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 3)),
                repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
            )
        audio_ids.append(str(record["audio_id"]))
        references.append(str(record["transcript"]))
        predictions.append(str(output["transcript"]))
        timings.append(float(output["inference_seconds"]))
        durations.append(float(record.get("duration_seconds", output["duration_seconds"])))
        confidence = output.get("confidence")
        confidences.append(None if confidence is None else float(confidence))
        confidence_methods.append(str(output.get("confidence_method", "unavailable")))
    result = evaluate_predictions(audio_ids, references, predictions)
    result["metrics"].update(
        {
            "model_choice": model_choice,
            "model_source": transcriber.model_source,
            "split": split,
            "decoder": describe_decoder(transcriber, beams),
            "num_beams": (
                None if isinstance(transcriber, Wav2Vec2Transcriber) else beams
            ),
            "mean_inference_seconds": sum(timings) / len(timings),
            "total_inference_seconds": sum(timings),
            "mean_confidence_proxy": summarize_confidence(confidences),
            "confidence_method": (
                confidence_methods[0]
                if len(set(confidence_methods)) == 1
                else "mixed"
            ),
        }
    )
    for sample, timing, duration, confidence, method in zip(
        result["samples"], timings, durations, confidences, confidence_methods
    ):
        sample["inference_seconds"] = timing
        sample["duration_seconds"] = duration
        sample["confidence_proxy"] = confidence
        sample["confidence_method"] = method
    result["metrics"]["duration_bins"] = summarize_by_duration(result["samples"])
    save_evaluation_artifacts(config, model_choice, result)
    return result


def save_evaluation_artifacts(
    config: dict[str, Any], model_choice: str, result: dict[str, Any]
) -> None:
    metrics_dir = Path(config["paths"]["metrics_dir"])
    predictions_dir = Path(config["paths"]["predictions_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    stem = model_choice.casefold().replace("-", "_")
    with (metrics_dir / f"{stem}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result["metrics"], handle, ensure_ascii=False, indent=2)
    samples = result["samples"]
    if samples:
        with (predictions_dir / f"{stem}_predictions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
    plot_available_results(config)


def plot_available_results(config: dict[str, Any]) -> None:
    """Chỉ vẽ từ metric/log thật đang tồn tại; thiếu file thì bỏ qua."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_dir = Path(config["paths"]["metrics_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparison: list[tuple[str, dict[str, Any]]] = []
    for choice in ("wav2vec2_vietnamese", "whisper_fine_tuned"):
        path = metrics_dir / f"{choice}_metrics.json"
        if path.is_file():
            comparison.append((choice, json.loads(path.read_text(encoding="utf-8"))))
    if comparison:
        labels = [label for label, _ in comparison]
        positions = list(range(len(labels)))
        width = 0.34
        plt.figure(figsize=(8, 5))
        plt.bar([value - width / 2 for value in positions], [row["wer"] for _, row in comparison], width, label="WER")
        plt.bar([value + width / 2 for value in positions], [row["cer"] for _, row in comparison], width, label="CER")
        plt.xticks(positions, labels)
        plt.ylabel("Error rate")
        sample_counts = {int(row["num_samples"]) for _, row in comparison}
        scope = (
            f"cùng test set VIVOS (n={sample_counts.pop()})"
            if len(sample_counts) == 1
            else "các kết quả hiện có"
        )
        plt.title(f"So sánh WER/CER trên {scope}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "wer_cer_comparison.png", dpi=160)
        plt.close()
    log_path = reports_dir / "trainer_log_history.json"
    if log_path.is_file():
        logs = json.loads(log_path.read_text(encoding="utf-8"))
        train_points = [(row["step"], row["loss"]) for row in logs if "loss" in row and "step" in row]
        eval_points = [(row["step"], row["eval_loss"]) for row in logs if "eval_loss" in row and "step" in row]
        if train_points or eval_points:
            plt.figure(figsize=(8, 5))
            if train_points:
                plt.plot(*zip(*train_points), marker="o", label="train loss")
            if eval_points:
                plt.plot(*zip(*eval_points), marker="s", label="validation loss")
            plt.xlabel("Optimizer step")
            plt.ylabel("Loss")
            plt.title("Training/validation loss từ Trainer log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "training_validation_loss.png", dpi=160)
            plt.close()
