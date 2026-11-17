"""WER/CER thuần Python để metric minh bạch và không phụ thuộc dịch vụ ngoài."""

from __future__ import annotations

import random
import unicodedata
from collections.abc import Sequence
from typing import Any

# "%" mang nghĩa (mười % ≠ mười), nên không bị coi là dấu câu khi chấm điểm.
_KEPT_PUNCTUATION = frozenset("%")


def normalize_for_scoring(text: str, *, strip_punctuation: bool = True) -> str:
    """Chuẩn hóa chỉ dùng để chấm WER/CER: NFC, chữ thường, bỏ dấu câu.

    Dấu câu không phải là thứ model nghe được; giữ chúng khiến ``trăm,`` khác
    ``trăm`` và một câu nghe đúng hoàn toàn vẫn bị tính lỗi. Transcript hiển thị
    và nhãn train vẫn dùng ``normalize_transcript`` như cũ.
    """
    text = unicodedata.normalize("NFC", str(text)).casefold()
    if strip_punctuation:
        text = "".join(
            " "
            if unicodedata.category(char).startswith("P") and char not in _KEPT_PUNCTUATION
            else char
            for char in text
        )
    return " ".join(text.split())


def _normalize(text: str) -> str:
    return normalize_for_scoring(text)


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Khoảng cách Levenshtein với bộ nhớ O(min(n, m))."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def calculate_wer(reference: str, prediction: str) -> float:
    ref_words = _normalize(reference).split()
    if not ref_words:
        raise ValueError("Transcript tham chiếu rỗng, không thể tính WER.")
    return edit_distance(ref_words, _normalize(prediction).split()) / len(ref_words)


def calculate_cer(reference: str, prediction: str) -> float:
    ref_chars = list(_normalize(reference))
    if not ref_chars:
        raise ValueError("Transcript tham chiếu rỗng, không thể tính CER.")
    return edit_distance(ref_chars, list(_normalize(prediction))) / len(ref_chars)


def bootstrap_wer_interval(
    word_errors: Sequence[int],
    reference_words: Sequence[int],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Khoảng tin cậy WER corpus bằng bootstrap theo câu (percentile)."""
    if len(word_errors) != len(reference_words) or not word_errors:
        raise ValueError("Cần danh sách lỗi/số từ cùng độ dài và không rỗng.")
    if resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("resamples phải dương và confidence nằm trong (0, 1).")
    rng = random.Random(seed)
    count = len(word_errors)
    rates = []
    for _ in range(resamples):
        picks = [rng.randrange(count) for _ in range(count)]
        words = sum(reference_words[index] for index in picks)
        rates.append(sum(word_errors[index] for index in picks) / words if words else 0.0)
    rates.sort()
    tail = (1 - confidence) / 2
    low = rates[int(tail * (resamples - 1))]
    high = rates[round((1 - tail) * (resamples - 1))]
    return low, high


def per_sample_word_errors(
    references: Sequence[str], predictions: Sequence[str]
) -> tuple[list[int], list[int]]:
    """Số lỗi từ và số từ tham chiếu của từng câu, theo chuẩn hóa chấm điểm."""
    if len(references) != len(predictions):
        raise ValueError("Số reference và prediction phải bằng nhau.")
    errors: list[int] = []
    words: list[int] = []
    for reference, prediction in zip(references, predictions):
        ref_words = normalize_for_scoring(reference).split()
        errors.append(edit_distance(ref_words, normalize_for_scoring(prediction).split()))
        words.append(len(ref_words))
    return errors, words


def paired_bootstrap_wer_delta(
    baseline_errors: Sequence[int],
    candidate_errors: Sequence[int],
    reference_words: Sequence[int],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """WER(candidate) − WER(baseline) trên cùng các câu, kèm CI bootstrap ghép cặp.

    Âm là candidate tốt hơn. Nếu CI chứa 0 thì khác biệt chưa có ý nghĩa thống kê.
    """
    count = len(reference_words)
    if not count or len(baseline_errors) != count or len(candidate_errors) != count:
        raise ValueError("Ba danh sách phải cùng độ dài và không rỗng.")
    total_words = sum(reference_words)
    delta = (sum(candidate_errors) - sum(baseline_errors)) / total_words
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        picks = [rng.randrange(count) for _ in range(count)]
        words = sum(reference_words[index] for index in picks)
        diff = sum(candidate_errors[index] - baseline_errors[index] for index in picks)
        deltas.append(diff / words if words else 0.0)
    deltas.sort()
    tail = (1 - confidence) / 2
    return {
        "delta_wer": delta,
        "ci95_low": deltas[int(tail * (resamples - 1))],
        "ci95_high": deltas[round((1 - tail) * (resamples - 1))],
    }


def compute_corpus_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    *,
    strip_punctuation: bool = True,
    bootstrap_resamples: int = 0,
) -> dict[str, Any]:
    """Tính micro-average bằng tổng edit / tổng đơn vị tham chiếu.

    ``strip_punctuation=False`` tái tạo cách chấm cũ (giữ dấu câu) để so sánh.
    ``bootstrap_resamples > 0`` bổ sung khoảng tin cậy 95% cho WER.
    """
    if len(references) != len(predictions):
        raise ValueError("Số reference và prediction phải bằng nhau.")
    if not references:
        raise ValueError("Tập transcript tham chiếu rỗng.")
    word_errors = char_errors = reference_words = reference_chars = 0
    sentence_errors = scored_samples = 0
    per_sample_errors: list[int] = []
    per_sample_words: list[int] = []
    for reference, prediction in zip(references, predictions):
        ref_text = normalize_for_scoring(reference, strip_punctuation=strip_punctuation)
        pred_text = normalize_for_scoring(prediction, strip_punctuation=strip_punctuation)
        ref_word_list = ref_text.split()
        if not ref_word_list:
            continue
        sample_word_errors = edit_distance(ref_word_list, pred_text.split())
        word_errors += sample_word_errors
        char_errors += edit_distance(list(ref_text), list(pred_text))
        reference_words += len(ref_word_list)
        reference_chars += len(ref_text)
        scored_samples += 1
        sentence_errors += sample_word_errors > 0
        per_sample_errors.append(sample_word_errors)
        per_sample_words.append(len(ref_word_list))
    if reference_words == 0 or reference_chars == 0:
        raise ValueError("Tập transcript tham chiếu không chứa nội dung hợp lệ.")
    metrics: dict[str, Any] = {
        "wer": word_errors / reference_words,
        "cer": char_errors / reference_chars,
        "ser": sentence_errors / scored_samples,
        "word_errors": word_errors,
        "char_errors": char_errors,
        "sentence_errors": sentence_errors,
        "reference_words": reference_words,
        "reference_characters": reference_chars,
        "num_samples": len(references),
        "scoring_normalization": (
            "nfc_casefold_strip_punctuation" if strip_punctuation else "nfc_casefold"
        ),
    }
    if bootstrap_resamples > 0:
        low, high = bootstrap_wer_interval(
            per_sample_errors, per_sample_words, resamples=bootstrap_resamples
        )
        metrics["wer_ci95"] = [low, high]
    return metrics
