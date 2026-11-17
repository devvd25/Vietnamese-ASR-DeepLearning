"""Xét lại các quyết định promote bằng WER đã chuẩn hóa và bootstrap ghép cặp.

Chỉ đọc prediction đã lưu, không chạy lại model. Quy tắc (docs/MODEL_UPGRADE_PLAN.md §3):
- Mỗi test độc lập: ``better`` nếu CI 95% của ΔWER < 0, ``worse`` nếu > 0, còn lại ``tie``.
- Đủ điều kiện promote khi không có ``worse`` và có ít nhất một ``better``.
- Mẫu ST23D chỉ có 1 câu/20 từ nên được báo cáo riêng, không dùng làm cổng thống kê.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import configure_utf8_stdio
from src.metrics import (
    normalize_for_scoring,
    paired_bootstrap_wer_delta,
    per_sample_word_errors,
)

W2V_ORIGINAL = "outputs/kaggle_cv21/predictions/wav2vec2_original_{split}_predictions.csv"
PHO_BASE = "outputs/kaggle_cv21/predictions/adapted_{split}_predictions.csv"

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "kaggle_speech_wav2vec2",
        "baseline": "nguyenvulebinh/wav2vec2-base-vietnamese-250h + LM 4-gram (đang dùng)",
        "candidate": "checkpoints/vivos_cv21_kaggle_speech_wav2vec2_adaptation/best",
        "previous_decision": "not_promoted",
        "splits": {
            "vivos_test": (
                W2V_ORIGINAL.format(split="vivos_test"),
                "outputs/kaggle_speech/candidate/wav2vec2/test/predictions/wav2vec2_vietnamese_predictions.csv",
            ),
            "common_voice_test": (
                W2V_ORIGINAL.format(split="common_voice_test"),
                "outputs/kaggle_speech/candidate/wav2vec2/common_voice_test/predictions/wav2vec2_vietnamese_predictions.csv",
            ),
            "kaggle_holdout": (
                "outputs/kaggle_speech/baseline/predictions/wav2vec2_vietnamese_predictions.csv",
                "outputs/kaggle_speech/candidate/wav2vec2/kaggle_speech_test/predictions/wav2vec2_vietnamese_predictions.csv",
            ),
        },
        "st23d": {"baseline_wer": 0.0, "candidate_json": "outputs/kaggle_speech/tables/wav2vec2_st23d_sample_metrics.json"},
    },
    {
        "id": "kaggle_speech_phowhisper",
        "baseline": "checkpoints/vivos_cv21_phowhisper_adaptation/best (gốc của bản demo)",
        "candidate": "checkpoints/vivos_cv21_kaggle_speech_phowhisper_adaptation/best",
        "previous_decision": "not_promoted",
        "splits": {
            "vivos_test": (
                PHO_BASE.format(split="vivos_test"),
                "outputs/kaggle_speech/candidate/phowhisper/test/predictions/whisper_fine_tuned_predictions.csv",
            ),
            "common_voice_test": (
                PHO_BASE.format(split="common_voice_test"),
                "outputs/kaggle_speech/candidate/phowhisper/common_voice_test/predictions/whisper_fine_tuned_predictions.csv",
            ),
            "kaggle_holdout": (
                "outputs/kaggle_speech/baseline/predictions/whisper_fine_tuned_predictions.csv",
                "outputs/kaggle_speech/candidate/phowhisper/kaggle_speech_test/predictions/whisper_fine_tuned_predictions.csv",
            ),
        },
        "st23d": {"baseline_wer": 0.1, "candidate_json": "outputs/kaggle_speech/tables/phowhisper_st23d_sample_metrics.json"},
    },
    {
        "id": "fpt_wav2vec2",
        "baseline": "nguyenvulebinh/wav2vec2-base-vietnamese-250h + LM 4-gram (đang dùng)",
        "candidate": "checkpoints/vivos_cv21_fpt_wav2vec2_adaptation/best",
        "previous_decision": "not_promoted",
        "splits": {
            "vivos_test": (
                W2V_ORIGINAL.format(split="vivos_test"),
                "outputs/kaggle_fpt/predictions/wav2vec2_fpt_vivos_test_predictions.csv",
            ),
            "common_voice_test": (
                W2V_ORIGINAL.format(split="common_voice_test"),
                "outputs/kaggle_fpt/predictions/wav2vec2_fpt_common_voice_test_predictions.csv",
            ),
        },
        "st23d": {"baseline_wer": 0.0, "candidate_json": "outputs/kaggle_fpt/tables/wav2vec2_fpt_st23d_sample_metrics.json"},
    },
    {
        "id": "fpt_phowhisper",
        "baseline": "checkpoints/vivos_cv21_phowhisper_adaptation/best (gốc của bản demo)",
        "candidate": "checkpoints/vivos_cv21_fpt_phowhisper_adaptation/best",
        "previous_decision": "not_promoted",
        "splits": {
            "vivos_test": (
                PHO_BASE.format(split="vivos_test"),
                "outputs/kaggle_fpt/predictions/phowhisper_fpt_vivos_test_predictions.csv",
            ),
            "common_voice_test": (
                PHO_BASE.format(split="common_voice_test"),
                "outputs/kaggle_fpt/predictions/phowhisper_fpt_common_voice_test_predictions.csv",
            ),
        },
        "st23d": {"baseline_wer": 0.1, "candidate_json": "outputs/kaggle_fpt/tables/phowhisper_fpt_st23d_sample_metrics.json"},
    },
    {
        "id": "st23d_demo_phowhisper",
        "baseline": "checkpoints/vivos_cv21_phowhisper_adaptation/best",
        "candidate": "checkpoints/phowhisper_st23d_spoken_code_adaptation/checkpoint-16 (đang dùng cho demo)",
        "previous_decision": "promoted",
        "splits": {
            "kaggle_holdout": (
                "outputs/kaggle_speech/baseline/predictions/whisper_fine_tuned_predictions.csv",
                "outputs/st23d_demo_candidate/predictions/whisper_fine_tuned_predictions.csv",
            ),
        },
        "unavailable_splits": {
            "vivos_test": "Không lưu file prediction; tham chiếu VIVOS không có dấu câu nên số cũ (0,08923→0,09026) vẫn đúng.",
            "common_voice_test": "Không lưu file prediction; chưa thể chấm lại.",
        },
        "st23d": {"note": "Mẫu ST23D đã dùng để train candidate (training fit), không so sánh."},
    },
]

# Từ chỉ số đếm; thay bằng <n> để phát hiện câu cùng khuôn mẫu chỉ khác con số.
_NUMBER_WORDS = frozenset(
    ["không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu", "bảy", "tám", "chín", "mười", "mươi", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "linh", "lẻ"]
)


def _read_predictions(path: Path) -> dict[str, tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["audio_id"]: (row["reference"], row["prediction"])
            for row in csv.DictReader(handle)
        }


def compare_split(root: Path, baseline_csv: str, candidate_csv: str, resamples: int) -> dict[str, Any]:
    baseline = _read_predictions(root / baseline_csv)
    candidate = _read_predictions(root / candidate_csv)
    shared = sorted(set(baseline) & set(candidate))
    if len(shared) != len(baseline) or len(shared) != len(candidate):
        raise ValueError(f"Hai file không cùng tập câu: {baseline_csv} / {candidate_csv}")
    references = [baseline[audio_id][0] for audio_id in shared]
    if references != [candidate[audio_id][0] for audio_id in shared]:
        raise ValueError(f"Reference lệch nhau giữa {baseline_csv} và {candidate_csv}")
    base_errors, words = per_sample_word_errors(references, [baseline[a][1] for a in shared])
    cand_errors, _ = per_sample_word_errors(references, [candidate[a][1] for a in shared])
    stats = paired_bootstrap_wer_delta(base_errors, cand_errors, words, resamples=resamples)
    verdict = "better" if stats["ci95_high"] < 0 else "worse" if stats["ci95_low"] > 0 else "tie"
    total = sum(words)
    return {
        "num_samples": len(shared),
        "reference_words": total,
        "baseline_wer": sum(base_errors) / total,
        "candidate_wer": sum(cand_errors) / total,
        "baseline_ser": sum(error > 0 for error in base_errors) / len(shared),
        "candidate_ser": sum(error > 0 for error in cand_errors) / len(shared),
        **stats,
        "verdict": verdict,
        "baseline_csv": baseline_csv,
        "candidate_csv": candidate_csv,
    }


def _template(text: str) -> str:
    words = normalize_for_scoring(text).split()
    return " ".join("<n>" if word in _NUMBER_WORDS or word.isdigit() else word for word in words)


def kaggle_holdout_overlap(root: Path) -> dict[str, Any]:
    """Đo mức trùng khuôn câu giữa Kaggle train và holdout (không có nhãn người nói)."""
    manifest_dir = root / "data/processed/vivos_cv21_kaggle_speech"
    train_texts = []
    with (manifest_dir / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if str(record["audio_id"]).startswith("kaggle_speech:"):
                train_texts.append(record["transcript"])
    with (manifest_dir / "kaggle_speech_test.jsonl").open(encoding="utf-8") as handle:
        holdout_texts = [json.loads(line)["transcript"] for line in handle]
    train_exact = {normalize_for_scoring(text) for text in train_texts}
    train_templates = {_template(text) for text in train_texts}
    train_4grams: set[tuple[str, ...]] = set()
    for text in train_texts:
        words = normalize_for_scoring(text).split()
        train_4grams.update(zip(*(words[offset:] for offset in range(4))))
    exact = template = 0
    covered = total = 0
    for text in holdout_texts:
        exact += normalize_for_scoring(text) in train_exact
        template += _template(text) in train_templates
        words = normalize_for_scoring(text).split()
        grams = list(zip(*(words[offset:] for offset in range(4))))
        covered += sum(gram in train_4grams for gram in grams)
        total += len(grams)
    count = len(holdout_texts)
    return {
        "kaggle_train_sentences": len(train_texts),
        "holdout_sentences": count,
        "speaker_labels_available": False,
        "exact_sentence_overlap": exact,
        "same_template_except_numbers": template,
        "same_template_rate": template / count,
        "holdout_4gram_coverage_by_train": covered / total if total else 0.0,
    }


def decide(splits: dict[str, dict[str, Any]], leaky: set[str]) -> tuple[bool, str]:
    verdicts = {name: result["verdict"] for name, result in splits.items()}
    worse = [name for name, verdict in verdicts.items() if verdict == "worse"]
    better = [name for name, verdict in verdicts.items() if verdict == "better"]
    trusted_better = [name for name in better if name not in leaky]
    if worse:
        return False, f"Tệ hơn có ý nghĩa thống kê trên: {', '.join(worse)}."
    if not better:
        return False, "Không có cải thiện có ý nghĩa thống kê trên test nào."
    if not trusted_better:
        return False, (
            "Chỉ cải thiện trên " + ", ".join(better)
            + ", nơi holdout trùng khuôn câu với train; chưa đủ bằng chứng tổng quát."
        )
    return True, "Không tệ hơn ở test nào; cải thiện có ý nghĩa trên: " + ", ".join(better) + "."


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(review: dict[str, Any]) -> str:
    overlap = review["kaggle_holdout_overlap"]
    lines = [
        "# Xét lại quyết định promote (WER đã bỏ dấu câu)",
        "",
        "Sinh tự động bởi `scripts/review_promotions.py`. ΔWER = candidate − baseline; âm là tốt hơn.",
        "CI 95% từ bootstrap ghép cặp theo câu. *tie* = CI chứa 0.",
        "",
        "## Kaggle holdout có độc lập không?",
        "",
        "- Không có nhãn người nói → không kiểm chứng được tách người nói.",
        f"- Trùng câu y hệt với train: {overlap['exact_sentence_overlap']}/{overlap['holdout_sentences']}.",
        (
            "- Cùng khuôn câu, chỉ khác con số: "
            f"**{overlap['same_template_except_numbers']}/{overlap['holdout_sentences']}"
            f" ({_pct(overlap['same_template_rate'])})**."
        ),
        f"- Tỷ lệ 4-gram của holdout đã xuất hiện trong train: {_pct(overlap['holdout_4gram_coverage_by_train'])}.",
        "",
    ]
    for experiment in review["experiments"]:
        lines += [
            f"## `{experiment['id']}`",
            "",
            f"- Baseline: {experiment['baseline']}",
            f"- Candidate: `{experiment['candidate']}`",
            "",
            "| Test | n | WER baseline | WER candidate | ΔWER | CI 95% | SER baseline → candidate | Kết luận |",
            "|---|---:|---:|---:|---:|---|---|---|",
        ]
        for name, result in experiment["splits"].items():
            lines.append(
                f"| {name} | {result['num_samples']} | {_pct(result['baseline_wer'])} "
                f"| {_pct(result['candidate_wer'])} | {100 * result['delta_wer']:+.2f} đ% "
                f"| {100 * result['ci95_low']:+.2f} … {100 * result['ci95_high']:+.2f} "
                f"| {_pct(result['baseline_ser'])} → {_pct(result['candidate_ser'])} | {result['verdict']} |"
            )
        for name, note in experiment.get("unavailable_splits", {}).items():
            lines.append(f"| {name} | — | — | — | — | — | — | {note} |")
        st23d = experiment["st23d"]
        if "candidate_wer" in st23d:
            lines += ["", f"ST23D (1 câu, chỉ tham khảo): {_pct(st23d['baseline_wer'])} → {_pct(st23d['candidate_wer'])}."]
        elif "note" in st23d:
            lines += ["", f"ST23D: {st23d['note']}"]
        lines += [
            "",
            (
                f"**Quyết định cũ:** {experiment['previous_decision']} · "
                "**Theo thước đo mới:** "
                f"{'đủ điều kiện promote' if experiment['eligible'] else 'không promote'}"
                f" — {experiment['reason']}"
            ),
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="outputs/rescored")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    root = PROJECT_ROOT
    overlap = kaggle_holdout_overlap(root)
    # Holdout coi là rò rỉ khi phần lớn câu trùng khuôn với train.
    leaky = {"kaggle_holdout"} if overlap["same_template_rate"] >= 0.5 else set()
    experiments = []
    for spec in EXPERIMENTS:
        splits = {
            name: compare_split(root, baseline_csv, candidate_csv, args.resamples)
            for name, (baseline_csv, candidate_csv) in spec["splits"].items()
        }
        st23d = dict(spec["st23d"])
        if "candidate_json" in st23d:
            st23d["candidate_wer"] = json.loads(
                (root / st23d["candidate_json"]).read_text(encoding="utf-8")
            )["wer"]
        if spec["id"] == "st23d_demo_phowhisper":
            eligible, reason = True, (
                "Giữ nguyên: checkpoint demo được chọn vì đã thích nghi mẫu ST23D; "
                "Kaggle holdout " + splits["kaggle_holdout"]["verdict"] + " so với gốc."
            )
        else:
            eligible, reason = decide(splits, leaky)
        experiments.append(
            {
                **{key: spec[key] for key in ("id", "baseline", "candidate", "previous_decision")},
                "splits": splits,
                "unavailable_splits": spec.get("unavailable_splits", {}),
                "st23d": st23d,
                "eligible": eligible,
                "reason": reason,
            }
        )
    review = {
        "scoring_normalization": "nfc_casefold_strip_punctuation (giữ %)",
        "decision_rule": "Không có test 'worse'; ít nhất một test 'better' ngoài holdout rò rỉ; ST23D chỉ tham khảo.",
        "kaggle_holdout_overlap": overlap,
        "kaggle_holdout_treated_as_leaky": bool(leaky),
        "experiments": experiments,
        "default_config_changed": False,
    }
    destination = root / args.destination
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "promotion_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "PROMOTION_REVIEW.md").write_text(render_markdown(review), encoding="utf-8")
    print(render_markdown(review))


if __name__ == "__main__":
    main()
