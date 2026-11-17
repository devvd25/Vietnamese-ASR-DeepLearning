from __future__ import annotations

import pytest

from src.metrics import (
    bootstrap_wer_interval,
    calculate_cer,
    calculate_wer,
    compute_corpus_metrics,
    normalize_for_scoring,
    paired_bootstrap_wer_delta,
    per_sample_word_errors,
)


def test_wer_counts_one_word_substitution() -> None:
    assert calculate_wer("xin chào bạn", "xin chào tôi") == pytest.approx(1 / 3)


def test_cer_counts_one_character_substitution() -> None:
    assert calculate_cer("xin chao", "xin chau") == pytest.approx(1 / 8)


def test_corpus_metrics_aggregate_edit_counts_not_sample_means() -> None:
    metrics = compute_corpus_metrics(
        references=["một hai", "ba"],
        predictions=["một ba", "ba bốn"],
    )

    assert metrics["word_errors"] == 2
    assert metrics["reference_words"] == 3
    assert metrics["wer"] == pytest.approx(2 / 3)
    assert 0 <= metrics["cer"] <= 1


def test_metrics_reject_empty_reference_corpus() -> None:
    with pytest.raises(ValueError, match="tham chiếu"):
        compute_corpus_metrics([""], ["xin chào"])


def test_scoring_ignores_punctuation_but_keeps_percent() -> None:
    assert normalize_for_scoring("Giảm  MƯỜI phần trăm, còn lại… 10%!") == (
        "giảm mười phần trăm còn lại 10%"
    )
    assert calculate_wer("bốn trăm, sáu nghìn đồng.", "bốn trăm sáu nghìn đồng") == 0
    assert calculate_cer("không?", "không") == 0


def test_legacy_scoring_still_counts_punctuation_for_comparison() -> None:
    references = ["hai trăm, ba lít."]
    predictions = ["hai trăm ba lít"]

    legacy = compute_corpus_metrics(references, predictions, strip_punctuation=False)
    scored = compute_corpus_metrics(references, predictions)

    assert legacy["wer"] == pytest.approx(2 / 4)
    assert scored["wer"] == 0
    assert scored["scoring_normalization"] == "nfc_casefold_strip_punctuation"


def test_corpus_metrics_report_sentence_error_rate_and_interval() -> None:
    metrics = compute_corpus_metrics(
        ["một hai", "ba", "bốn năm"],
        ["một hai", "ba bốn", "bốn năm"],
        bootstrap_resamples=200,
    )

    assert metrics["sentence_errors"] == 1
    assert metrics["ser"] == pytest.approx(1 / 3)
    low, high = metrics["wer_ci95"]
    assert 0 <= low <= metrics["wer"] <= high


def test_bootstrap_interval_is_deterministic_and_degenerate_without_errors() -> None:
    assert bootstrap_wer_interval([0, 0, 0], [3, 2, 4], resamples=50) == (0.0, 0.0)
    assert bootstrap_wer_interval([1, 0, 2], [3, 2, 4], resamples=50) == (
        bootstrap_wer_interval([1, 0, 2], [3, 2, 4], resamples=50)
    )


def test_paired_bootstrap_detects_consistent_improvement() -> None:
    errors_base = [2] * 40
    errors_cand = [0] * 40
    words = [5] * 40

    result = paired_bootstrap_wer_delta(errors_base, errors_cand, words, resamples=200)

    assert result["delta_wer"] == pytest.approx(-0.4)
    assert result["ci95_high"] < 0


def test_paired_bootstrap_interval_contains_zero_for_identical_systems() -> None:
    errors = [1, 0, 2, 0, 1]
    result = paired_bootstrap_wer_delta(errors, errors, [4, 3, 5, 2, 4], resamples=100)

    assert result == {"delta_wer": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}


def test_per_sample_word_errors_use_scoring_normalization() -> None:
    errors, words = per_sample_word_errors(["xin chào.", "ba bốn"], ["xin chào", "ba"])

    assert errors == [0, 1]
    assert words == [2, 2]
