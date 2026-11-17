"""Tiền xử lý audio và transcript dùng chung cho train, evaluate và demo."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}

_SPOKEN_CODE_ALIASES = {
    "a": ("a",),
    "b": ("b", "be"),
    "c": ("c", "ce", "xe"),
    "d": ("d", "de"),
    "e": ("e",),
    "f": ("f", "ep"),
    "g": ("g", "ge", "go"),
    "h": ("h", "hat"),
    "i": ("i",),
    "j": ("j", "gi"),
    "k": ("k", "ca"),
    "l": ("l", "el"),
    "m": ("m", "em"),
    "n": ("n", "en"),
    "o": ("o",),
    "p": ("p", "pe"),
    "q": ("q", "quy"),
    "r": ("r", "er"),
    # ASR đôi khi nuốt âm /s/ đầu trong cách đọc mã "ST...". Alias rỗng chỉ
    # được xét sau ngữ cảnh lớp/mã và chỉ cho mã đã khai báo trước.
    "s": ("s", "es", "et", "ay", "e", ""),
    "t": ("t", "te"),
    "u": ("u",),
    "v": ("v", "ve"),
    "w": ("w", "vekep"),
    "x": ("x", "ich", "xit"),
    "y": ("y", "ydai"),
    "0": ("0", "khong"),
    "1": ("1", "mot"),
    "2": ("2", "hai", "hay"),
    "3": ("3", "ba", "bo"),
    "4": ("4", "bon", "tu"),
    "5": ("5", "nam"),
    "6": ("6", "sau"),
    "7": ("7", "bay"),
    "8": ("8", "tam"),
    "9": ("9", "chin"),
}
_CODE_CONTEXTS = {"lop", "ma"}


def normalize_transcript(text: str, *, reject_empty: bool = True) -> str:
    """Chuẩn hóa Unicode NFC, chữ thường và khoảng trắng nhưng giữ dấu/câu."""
    if not isinstance(text, str):
        raise TypeError("Transcript phải là chuỗi.")
    normalized = unicodedata.normalize("NFC", " ".join(text.split())).casefold()
    if reject_empty and not normalized:
        raise ValueError("Transcript rỗng sau khi chuẩn hóa.")
    return normalized


def _ascii_compact(text: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD", text.casefold().replace("đ", "d")
    )
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character) and character.isalnum()
    )


def _matches_spoken_code(observed: str, code: str) -> bool:
    compact_code = _ascii_compact(code)
    if not compact_code or not observed:
        return False
    reachable = {0}
    for character in compact_code:
        aliases = _SPOKEN_CODE_ALIASES.get(character, (character,))
        next_reachable = {
            position + len(alias)
            for position in reachable
            for alias in aliases
            if observed.startswith(alias, position)
        }
        if not next_reachable:
            return False
        reachable = next_reachable
    return len(observed) in reachable


def normalize_domain_codes(
    text: str, domain_codes: list[str] | tuple[str, ...]
) -> tuple[str, list[tuple[str, str]]]:
    """Chuẩn hóa cách đọc mã chữ-số sau từ chỉ ngữ cảnh như ``lớp``/``mã``.

    Chỉ các mã được khai báo trước mới được thay thế; transcript tham chiếu không
    bao giờ được dùng để sửa kết quả nhận dạng.
    """
    words = str(text).split()
    replacements: list[tuple[str, str]] = []
    codes = [str(code).casefold() for code in domain_codes if _ascii_compact(str(code))]
    index = 0
    while index < len(words) - 1:
        if _ascii_compact(words[index]) not in _CODE_CONTEXTS:
            index += 1
            continue
        replaced = False
        start = index + 1
        max_end = min(len(words), start + 8)
        for code in codes:
            for end in range(start + 1, max_end + 1):
                source = " ".join(words[start:end])
                if _matches_spoken_code(_ascii_compact(source), code):
                    if source.casefold() != code:
                        words[start:end] = [code]
                        replacements.append((source, code))
                    index = start + 1
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            index += 1
    return " ".join(words), replacements


def normalize_domain_phrases(
    text: str, phrase_corrections: Mapping[str, str]
) -> tuple[str, list[tuple[str, str]]]:
    """Apply explicit whole-token phrase corrections and disclose every change.

    This intentionally performs no fuzzy matching: a rule only fires when all
    normalized source tokens occur consecutively in the decoded transcript.
    """

    words = normalize_transcript(str(text), reject_empty=False).split()
    rules: list[tuple[list[str], list[str], str, str]] = []
    for raw_source, raw_replacement in phrase_corrections.items():
        source = normalize_transcript(str(raw_source), reject_empty=False)
        replacement = normalize_transcript(str(raw_replacement), reject_empty=False)
        if not source or not replacement:
            raise ValueError("Quy tắc chuẩn hóa cụm từ không được rỗng.")
        rules.append((source.split(), replacement.split(), source, replacement))
    rules.sort(key=lambda rule: len(rule[0]), reverse=True)

    replacements: list[tuple[str, str]] = []
    index = 0
    while index < len(words):
        matched = False
        for source_words, replacement_words, source, replacement in rules:
            end = index + len(source_words)
            if words[index:end] != source_words:
                continue
            words[index:end] = replacement_words
            replacements.append((source, replacement))
            index += len(replacement_words)
            matched = True
            break
        if not matched:
            index += 1
    return " ".join(words), replacements


def load_audio(audio_path: str | Path) -> tuple[np.ndarray, int]:
    """Đọc audio cục bộ thành float32; báo lỗi rõ ràng với file không hợp lệ."""
    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp âm thanh: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Tệp âm thanh rỗng: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(f"Không hỗ trợ định dạng âm thanh: {path.suffix}")
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError(f"Không thể đọc tệp âm thanh {path.name}: {exc}") from exc
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or sample_rate <= 0:
        raise ValueError(f"Tệp âm thanh không có dữ liệu hợp lệ: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Tệp âm thanh chứa giá trị NaN/Inf: {path}")
    return audio, int(sample_rate)


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_sample_rate: int = 16_000,
    max_seconds: float | None = None,
) -> np.ndarray:
    """Chuyển mono và resample; không tự cắt audio dài quá giới hạn."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        raise ValueError("Audio rỗng.")
    if sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rate phải là số dương.")
    if samples.ndim == 2:
        channel_axis = 1 if samples.shape[1] <= 8 else 0
        samples = samples.mean(axis=channel_axis)
    elif samples.ndim != 1:
        raise ValueError("Audio phải có dạng mono hoặc stereo.")
    if not np.isfinite(samples).all():
        raise ValueError("Audio chứa NaN hoặc Inf.")
    if max_seconds is not None and len(samples) / sample_rate > max_seconds:
        raise ValueError(
            f"Audio dài {len(samples) / sample_rate:.2f}s, vượt giới hạn {max_seconds:.2f}s."
        )
    if sample_rate != target_sample_rate:
        divisor = math.gcd(sample_rate, target_sample_rate)
        samples = resample_poly(
            samples,
            target_sample_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
    return np.ascontiguousarray(samples, dtype=np.float32)


def prepare_asr_example(
    example: Mapping[str, Any],
    *,
    target_sample_rate: int = 16_000,
    max_seconds: float | None = 30.0,
) -> dict[str, Any]:
    """Chuẩn hóa một record có audio path/dict và transcript linh hoạt."""
    text = example.get("transcript", example.get("sentence", example.get("text", "")))
    transcript = normalize_transcript(str(text))
    audio_value = example.get("audio", example.get("audio_path", example.get("path")))
    if isinstance(audio_value, Mapping) and "array" in audio_value:
        audio = np.asarray(audio_value["array"], dtype=np.float32)
        sample_rate = int(audio_value["sampling_rate"])
    elif audio_value:
        audio, sample_rate = load_audio(Path(str(audio_value)))
    else:
        raise ValueError("Record thiếu đường dẫn hoặc mảng audio.")
    processed = preprocess_audio(
        audio,
        sample_rate,
        target_sample_rate=target_sample_rate,
        max_seconds=max_seconds,
    )
    return {
        **dict(example),
        "audio": {"array": processed, "sampling_rate": target_sample_rate},
        "transcript": transcript,
        "duration_seconds": len(processed) / target_sample_rate,
    }
