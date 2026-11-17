from __future__ import annotations

import numpy as np
import pytest

from src.realtime_caption import (
    RealtimeCaptionSession,
    is_allowed_extension_origin,
)


class _FakeTranscriber:
    model_source = "local-test-model"
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int]] = []

    def transcribe_array(self, audio, *, sample_rate: int, **kwargs):
        samples = np.asarray(audio, dtype=np.float32)
        self.calls.append((samples, sample_rate))
        return {
            "transcript": "xin chào việt nam",
            "inference_seconds": 0.12,
            "confidence": 0.8,
            "device": "cpu",
            "model_source": self.model_source,
        }


class _EmptyThenCaptionTranscriber(_FakeTranscriber):
    def transcribe_array(self, audio, *, sample_rate: int, **kwargs):
        if not self.calls:
            samples = np.asarray(audio, dtype=np.float32)
            self.calls.append((samples, sample_rate))
            raise RuntimeError("Mô hình Wav2Vec2 không sinh được transcript.")
        return super().transcribe_array(audio, sample_rate=sample_rate, **kwargs)


class _SequentialTranscriber(_FakeTranscriber):
    def __init__(self, transcripts: list[str]) -> None:
        super().__init__()
        self.transcripts = iter(transcripts)

    def transcribe_array(self, audio, *, sample_rate: int, **kwargs):
        output = super().transcribe_array(audio, sample_rate=sample_rate, **kwargs)
        output["transcript"] = next(self.transcripts)
        return output


def test_realtime_session_emits_caption_only_after_one_complete_window() -> None:
    transcriber = _FakeTranscriber()
    session = RealtimeCaptionSession(
        model_choice="wav2vec2_vietnamese",
        sample_rate=16_000,
        chunk_seconds=1.0,
        transcriber_loader=lambda _: transcriber,
    )
    first_half = np.full(8_000, 0.1, dtype="<f4").tobytes()
    second_half = np.full(8_000, 0.1, dtype="<f4").tobytes()

    assert session.ingest_pcm(first_half) is None
    caption = session.ingest_pcm(second_half)

    assert caption == {
        "type": "caption",
        "sequence": 1,
        "text": "xin chào việt nam",
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "inference_seconds": 0.12,
        "confidence": 0.8,
        "device": "cpu",
        "model_source": "local-test-model",
    }


def test_realtime_session_exposes_each_speech_window_once_for_controlled_learning() -> None:
    transcriber = _FakeTranscriber()
    session = RealtimeCaptionSession(
        model_choice="whisper_fine_tuned",
        sample_rate=16_000,
        chunk_seconds=1,
        transcriber_loader=lambda _: transcriber,
        capture_learning=True,
    )

    session.ingest_pcm(np.full(16_000, 0.1, dtype="<f4").tobytes())
    captured = session.take_learning_window()

    assert captured is not None
    assert captured.sequence == 1
    assert captured.model_choice == "whisper_fine_tuned"
    assert captured.transcript == "xin chào việt nam"
    assert captured.confidence == 0.8
    assert captured.audio.shape == (16_000,)
    assert session.take_learning_window() is None
    assert len(transcriber.calls) == 1
    samples, sample_rate = transcriber.calls[0]
    assert sample_rate == 16_000
    assert samples.shape == (16_000,)


def test_realtime_session_skips_silent_window_without_loading_model() -> None:
    loaded = False

    def loader(_):
        nonlocal loaded
        loaded = True
        return _FakeTranscriber()

    session = RealtimeCaptionSession(
        model_choice="whisper_fine_tuned",
        sample_rate=16_000,
        chunk_seconds=1.0,
        transcriber_loader=loader,
    )

    event = session.ingest_pcm(np.zeros(16_000, dtype="<f4").tobytes())

    assert event == {
        "type": "silence",
        "sequence": 1,
        "start_seconds": 0.0,
        "end_seconds": 1.0,
    }
    assert loaded is False


def test_realtime_session_treats_empty_model_output_as_no_speech_and_continues() -> None:
    transcriber = _EmptyThenCaptionTranscriber()
    session = RealtimeCaptionSession(
        model_choice="wav2vec2_vietnamese",
        sample_rate=16_000,
        chunk_seconds=1.0,
        transcriber_loader=lambda _: transcriber,
    )
    speech = np.full(16_000, 0.1, dtype="<f4").tobytes()

    no_speech = session.ingest_pcm(speech)
    caption = session.ingest_pcm(speech)

    assert no_speech == {
        "type": "silence",
        "sequence": 1,
        "start_seconds": 0.0,
        "end_seconds": 1.0,
    }
    assert caption is not None
    assert caption["type"] == "caption"
    assert caption["sequence"] == 2


def test_realtime_session_overlaps_windows_and_emits_only_new_words() -> None:
    transcriber = _SequentialTranscriber(
        ["xin chào các bạn", "các bạn hôm nay"]
    )
    session = RealtimeCaptionSession(
        model_choice="wav2vec2_vietnamese",
        sample_rate=8_000,
        chunk_seconds=1.0,
        overlap_seconds=0.5,
        transcriber_loader=lambda _: transcriber,
    )
    first_window = np.full(8_000, 0.1, dtype="<f4").tobytes()
    next_half_second = np.full(4_000, 0.2, dtype="<f4").tobytes()

    first = session.ingest_pcm(first_window)
    second = session.ingest_pcm(next_half_second)

    assert first is not None
    assert first["text"] == "xin chào các bạn"
    assert first["start_seconds"] == 0.0
    assert first["end_seconds"] == 1.0
    assert second is not None
    assert second["text"] == "hôm nay"
    assert second["window_text"] == "các bạn hôm nay"
    assert second["start_seconds"] == 0.5
    assert second["end_seconds"] == 1.5
    assert len(transcriber.calls) == 2
    assert np.allclose(transcriber.calls[1][0][:4_000], 0.1)
    assert np.allclose(transcriber.calls[1][0][4_000:], 0.2)


@pytest.mark.parametrize(
    ("model_choice", "sample_rate", "chunk_seconds", "message"),
    [
        ("unknown", 16_000, 4.0, "model"),
        ("wav2vec2_vietnamese", 0, 4.0, "sample rate"),
        ("wav2vec2_vietnamese", 16_000, 0.5, "chunk"),
        ("wav2vec2_vietnamese", 16_000, 31.0, "chunk"),
    ],
)
def test_realtime_session_rejects_unsafe_configuration(
    model_choice: str,
    sample_rate: int,
    chunk_seconds: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RealtimeCaptionSession(
            model_choice=model_choice,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            transcriber_loader=lambda _: _FakeTranscriber(),
        )


def test_realtime_session_rejects_malformed_pcm_packet() -> None:
    session = RealtimeCaptionSession(
        model_choice="wav2vec2_vietnamese",
        sample_rate=16_000,
        chunk_seconds=1.0,
        transcriber_loader=lambda _: _FakeTranscriber(),
    )

    with pytest.raises(ValueError, match="PCM"):
        session.ingest_pcm(b"abc")


def test_only_chrome_extension_origins_are_allowed() -> None:
    assert is_allowed_extension_origin(
        "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
    )
    assert not is_allowed_extension_origin("https://evil.example")
    assert not is_allowed_extension_origin("null")
    assert not is_allowed_extension_origin(None)
