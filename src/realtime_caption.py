"""Core buffering and inference primitives for live browser captions."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


SUPPORTED_REALTIME_MODELS = {
    "wav2vec2_vietnamese",
    "whisper_fine_tuned",
}
_EXTENSION_ORIGIN = re.compile(r"^chrome-extension://[a-p]{32}$")


@dataclass(frozen=True)
class CapturedLearningWindow:
    """A speech window kept in memory until the learning pipeline consumes it."""

    sequence: int
    audio: np.ndarray
    sample_rate: int
    start_seconds: float
    end_seconds: float
    model_choice: str
    transcript: str
    confidence: float | None


def is_allowed_extension_origin(origin: str | None) -> bool:
    """Only unpacked/installed Chromium extensions may open the caption socket."""

    return bool(origin and _EXTENSION_ORIGIN.fullmatch(origin))


def _new_words(previous: str, current: str) -> str:
    """Remove an exact multi-word overlap between consecutive ASR windows."""

    previous_words = previous.split()
    current_words = current.split()
    max_overlap = min(len(previous_words), len(current_words))
    for overlap in range(max_overlap, 1, -1):
        if previous_words[-overlap:] == current_words[:overlap]:
            return " ".join(current_words[overlap:])
    return current


class RealtimeCaptionSession:
    """Accumulate mono float32 PCM and transcribe fixed-duration windows."""

    def __init__(
        self,
        *,
        model_choice: str,
        sample_rate: int,
        chunk_seconds: float,
        overlap_seconds: float = 0.0,
        transcriber_loader: Callable[[str], Any],
        transcribe_options: Mapping[str, Any] | None = None,
        silence_rms_threshold: float = 0.003,
        capture_learning: bool = False,
    ) -> None:
        if model_choice not in SUPPORTED_REALTIME_MODELS:
            raise ValueError("model must be wav2vec2_vietnamese or whisper_fine_tuned")
        if not 8_000 <= int(sample_rate) <= 192_000:
            raise ValueError("sample rate must be between 8000 and 192000 Hz")
        if not 1.0 <= float(chunk_seconds) <= 30.0:
            raise ValueError("chunk seconds must be between 1 and 30")
        if not 0.0 <= float(overlap_seconds) < float(chunk_seconds):
            raise ValueError("overlap seconds must be non-negative and shorter than chunk")
        if not math.isfinite(float(silence_rms_threshold)) or silence_rms_threshold < 0:
            raise ValueError("silence RMS threshold must be finite and non-negative")

        self.model_choice = model_choice
        self.sample_rate = int(sample_rate)
        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.chunk_samples = max(1, int(round(self.sample_rate * self.chunk_seconds)))
        self.overlap_samples = int(round(self.sample_rate * self.overlap_seconds))
        self.step_samples = self.chunk_samples - self.overlap_samples
        self.transcriber_loader = transcriber_loader
        self.transcribe_options = dict(transcribe_options or {})
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.capture_learning = bool(capture_learning)
        self._buffer = np.empty(0, dtype=np.float32)
        self._transcriber = None
        self._sequence = 0
        self._window_start_sample = 0
        self._previous_window_text = ""
        self._pending_learning_window: CapturedLearningWindow | None = None

    def take_learning_window(self) -> CapturedLearningWindow | None:
        """Return the last speech window once, without exposing it on the socket."""

        captured = self._pending_learning_window
        self._pending_learning_window = None
        return captured

    def ingest_pcm(self, packet: bytes) -> dict[str, Any] | None:
        """Consume little-endian float32 PCM and emit one caption/silence event."""

        if not packet or len(packet) % np.dtype("<f4").itemsize:
            raise ValueError("PCM packet must contain complete float32 samples")
        if len(packet) > 4 * 1024 * 1024:
            raise ValueError("PCM packet exceeds the 4 MB safety limit")
        incoming = np.frombuffer(packet, dtype="<f4").astype(np.float32, copy=True)
        if not np.isfinite(incoming).all():
            raise ValueError("PCM packet contains NaN or Inf")
        self._buffer = np.concatenate((self._buffer, incoming))
        if self._buffer.size < self.chunk_samples:
            return None

        window = np.ascontiguousarray(self._buffer[: self.chunk_samples])
        self._buffer = self._buffer[self.step_samples :].copy()
        self._sequence += 1
        start_seconds = self._window_start_sample / self.sample_rate
        end_seconds = start_seconds + self.chunk_seconds
        self._window_start_sample += self.step_samples
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
        if rms < self.silence_rms_threshold:
            self._previous_window_text = ""
            return {
                "type": "silence",
                "sequence": self._sequence,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
            }

        if self._transcriber is None:
            self._transcriber = self.transcriber_loader(self.model_choice)
        try:
            output = self._transcriber.transcribe_array(
                window,
                sample_rate=self.sample_rate,
                max_audio_seconds=self.chunk_seconds,
                **self.transcribe_options,
            )
        except RuntimeError as exc:
            # An empty CTC/seq2seq decode is normal for music, pauses and other
            # non-speech windows.  Keep the live session alive; genuine model
            # failures must still propagate to the WebSocket error handler.
            if "không sinh được transcript" not in str(exc).casefold():
                raise
            self._previous_window_text = ""
            return {
                "type": "silence",
                "sequence": self._sequence,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
            }
        window_text = str(output["transcript"]).strip()
        if self.capture_learning:
            raw_confidence = output.get("confidence")
            self._pending_learning_window = CapturedLearningWindow(
                sequence=self._sequence,
                audio=window.copy(),
                sample_rate=self.sample_rate,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                model_choice=self.model_choice,
                transcript=window_text,
                confidence=(
                    None if raw_confidence is None else float(raw_confidence)
                ),
            )
        text = _new_words(self._previous_window_text, window_text)
        self._previous_window_text = window_text
        event = {
            "type": "caption",
            "sequence": self._sequence,
            "text": text,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "inference_seconds": float(output["inference_seconds"]),
            "confidence": (
                None
                if output.get("confidence") is None
                else float(output["confidence"])
            ),
            "device": str(output.get("device", "unknown")),
            "model_source": str(output.get("model_source", "unknown")),
        }
        if text != window_text:
            event["window_text"] = window_text
        return event
