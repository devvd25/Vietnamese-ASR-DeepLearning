from __future__ import annotations

import pytest

from src.evaluate import (
    describe_decoder,
    evaluate_predictions,
    summarize_by_duration,
    summarize_confidence,
)
from src.inference import Wav2Vec2Transcriber


def test_evaluate_predictions_includes_per_sample_errors() -> None:
    result = evaluate_predictions(
        audio_ids=["a", "b"],
        references=["xin chào", "cảm ơn bạn"],
        predictions=["xin chào", "cảm ơn"],
    )

    assert result["metrics"]["num_samples"] == 2
    assert result["metrics"]["wer"] == 0.2
    assert result["metrics"]["exact_match_count"] == 1
    assert result["metrics"]["exact_match_rate"] == 0.5
    assert result["samples"][0]["wer"] == 0.0
    assert result["samples"][1]["error_note"] == "Thiếu từ"
    assert result["samples"][1]["substitutions"] == 0
    assert result["samples"][1]["deletions"] == 1
    assert result["samples"][1]["insertions"] == 0


def test_summarize_confidence_uses_only_available_finite_values() -> None:
    assert summarize_confidence([0.8, None, float("nan"), 0.6]) == pytest.approx(0.7)


def test_evaluate_predictions_identifies_diacritic_only_errors() -> None:
    result = evaluate_predictions(
        audio_ids=["tone"],
        references=["hòa bình"],
        predictions=["hoa binh"],
    )

    assert result["samples"][0]["error_note"] == "Sai dấu tiếng Việt"
    assert result["samples"][0]["substitutions"] == 2


def test_summarize_by_duration_reports_corpus_metrics_per_bin() -> None:
    samples = [
        {"duration_seconds": 2.5, "reference": "xin chào", "prediction": "xin chào"},
        {"duration_seconds": 4.0, "reference": "cảm ơn bạn", "prediction": "cảm ơn"},
        {"duration_seconds": 7.0, "reference": "hẹn gặp lại", "prediction": "hẹn lại"},
    ]

    summary = summarize_by_duration(samples)

    assert summary["<3s"]["num_samples"] == 1
    assert summary["3-6s"]["num_samples"] == 1
    assert summary[">=6s"]["num_samples"] == 1


def test_describe_decoder_reports_configured_wav2vec2_language_model() -> None:
    transcriber = object.__new__(Wav2Vec2Transcriber)
    transcriber.decoder_name = "ctc_beam_search_4gram"

    assert describe_decoder(transcriber, num_beams=1) == "ctc_beam_search_4gram"
