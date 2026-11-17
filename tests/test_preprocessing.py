from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from src import preprocessing
from src.preprocessing import load_audio, normalize_transcript, preprocess_audio


def _write_pcm_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_normalize_transcript_preserves_vietnamese_and_normalizes_spacing() -> None:
    decomposed = "  TO\u0302I   ye\u0302u   Tiếng   Việt!  "
    assert normalize_transcript(decomposed) == "tôi yêu tiếng việt!"


def test_normalize_transcript_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="rỗng"):
        normalize_transcript(" \n\t ")


def test_normalize_domain_codes_converts_spoken_class_code_after_context() -> None:
    text = "xin chào lớp este hai bade mình là đặng công vũ"

    normalized, replacements = preprocessing.normalize_domain_codes(
        text, ["ST23D"]
    )

    assert normalized == "xin chào lớp st23d mình là đặng công vũ"
    assert replacements == [("este hai bade", "st23d")]


def test_normalize_domain_codes_accepts_common_asr_homophones() -> None:
    text = "xin chào lớp ế tê hay ba đê"

    normalized, replacements = preprocessing.normalize_domain_codes(
        text, ["ST23D"]
    )

    assert normalized == "xin chào lớp st23d"
    assert replacements == [("ế tê hay ba đê", "st23d")]


@pytest.mark.parametrize(
    "spoken",
    ["tê hay bờ đê", "ế tê hay bơ đê"],
)
def test_normalize_domain_codes_accepts_phowhisper_class_code_variants(
    spoken: str,
) -> None:
    text = f"xin chào lớp {spoken} mình là đặng công vũ"

    normalized, replacements = preprocessing.normalize_domain_codes(text, ["ST23D"])

    assert normalized == "xin chào lớp st23d mình là đặng công vũ"
    assert replacements == [(spoken, "st23d")]


def test_normalize_domain_codes_does_not_replace_without_code_context() -> None:
    text = "tôi nghe este hai bade trong một câu khác"

    normalized, replacements = preprocessing.normalize_domain_codes(
        text, ["ST23D"]
    )

    assert normalized == text
    assert replacements == []


def test_normalize_domain_phrases_replaces_only_an_exact_token_sequence() -> None:
    normalized, replacements = preprocessing.normalize_domain_phrases(
        "xin chờ tất cả các bạn nha",
        {"xin chờ tất cả": "xin chào tất cả"},
    )

    assert normalized == "xin chào tất cả các bạn nha"
    assert replacements == [("xin chờ tất cả", "xin chào tất cả")]

    unchanged, unchanged_replacements = preprocessing.normalize_domain_phrases(
        "xin chờ đợi tất cả các bạn",
        {"xin chờ tất cả": "xin chào tất cả"},
    )
    assert unchanged == "xin chờ đợi tất cả các bạn"
    assert unchanged_replacements == []


def test_load_audio_reads_a_valid_wav(tmp_path: Path) -> None:
    path = tmp_path / "valid.wav"
    signal = np.sin(2 * np.pi * 440 * np.arange(800) / 8000).astype(np.float32)
    _write_pcm_wav(path, signal, 8000)

    audio, sample_rate = load_audio(path)

    assert sample_rate == 8000
    assert audio.ndim == 1
    assert len(audio) == 800
    assert np.isfinite(audio).all()


def test_load_audio_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "missing.wav")


def test_preprocess_audio_converts_stereo_and_resamples() -> None:
    stereo = np.stack(
        [np.ones(800, dtype=np.float32), np.zeros(800, dtype=np.float32)], axis=1
    )

    processed = preprocess_audio(stereo, sample_rate=8000, target_sample_rate=16000)

    assert processed.ndim == 1
    assert len(processed) == 1600
    assert np.allclose(processed.mean(), 0.5, atol=0.02)
