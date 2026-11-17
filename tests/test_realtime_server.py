from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.realtime_server import create_realtime_app


class _FakeTranscriber:
    model_source = "fake/realtime-model"
    device = "cpu"

    def transcribe_array(self, audio, *, sample_rate: int, **kwargs):
        return {
            "transcript": "phụ đề trực tiếp",
            "inference_seconds": 0.05,
            "confidence": 0.75,
            "device": self.device,
            "model_source": self.model_source,
        }


class _DecoderSensitiveTranscriber(_FakeTranscriber):
    def __init__(self) -> None:
        self.decoder = object()
        self.decoder_name = "ctc_beam_search_4gram"

    def transcribe_array(self, audio, *, sample_rate: int, **kwargs):
        output = super().transcribe_array(audio, sample_rate=sample_rate, **kwargs)
        output["transcript"] = (
            "greedy không từ điển"
            if self.decoder is None
            else "beam có từ điển"
        )
        return output


@pytest.fixture
def realtime_client() -> TestClient:
    root = Path(__file__).resolve().parents[1]
    app = create_realtime_app(
        root / "config.phowhisper_vietspeech_10h.yaml",
        transcriber_loader=lambda _: _FakeTranscriber(),
    )
    return TestClient(app)


def test_health_endpoint_reports_local_realtime_capabilities(
    realtime_client: TestClient,
) -> None:
    response = realtime_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "vietnamese-asr-live-captions",
        "models": ["wav2vec2_vietnamese", "whisper_fine_tuned"],
        "default_model": "wav2vec2_vietnamese",
        "default_chunk_seconds": 4.0,
        "default_overlap_seconds": 1.0,
        "privacy": "Audio chỉ xử lý trên máy local.",
    }


def test_learning_status_reports_local_candidate_count(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    app = create_realtime_app(
        root / "config.phowhisper_vietspeech_10h.yaml",
        transcriber_loader=lambda _: _FakeTranscriber(),
        learning_root=tmp_path / "youtube-learning",
    )

    response = TestClient(app).get("/learning/status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "collecting",
        "accepted_samples": 0,
        "minimum_samples_before_training": 20,
        "active_checkpoint_unchanged": True,
    }


def test_websocket_stream_emits_ready_then_caption(
    realtime_client: TestClient,
) -> None:
    pcm = np.full(16_000, 0.1, dtype="<f4").tobytes()
    with realtime_client.websocket_connect(
        "/ws/captions?model=wav2vec2_vietnamese&sample_rate=16000&chunk_seconds=1",
        headers={"origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"},
    ) as socket:
        assert socket.receive_json() == {
            "type": "ready",
            "model": "wav2vec2_vietnamese",
            "sample_rate": 16_000,
            "chunk_seconds": 1.0,
            "overlap_seconds": 0.0,
        }
        socket.send_bytes(pcm)
        caption = socket.receive_json()

    assert caption["type"] == "caption"
    assert caption["text"] == "phụ đề trực tiếp"
    assert caption["start_seconds"] == 0.0
    assert caption["end_seconds"] == 1.0


def test_websocket_reports_configured_window_overlap(
    realtime_client: TestClient,
) -> None:
    with realtime_client.websocket_connect(
        "/ws/captions?model=wav2vec2_vietnamese&sample_rate=16000&chunk_seconds=3&overlap_seconds=1",
        headers={"origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"},
    ) as socket:
        ready = socket.receive_json()

    assert ready["type"] == "ready"
    assert ready["chunk_seconds"] == 3.0
    assert ready["overlap_seconds"] == 1.0


def test_websocket_controlled_learning_saves_only_consensus_windows(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    app = create_realtime_app(
        root / "config.phowhisper_vietspeech_10h.yaml",
        transcriber_loader=lambda _: _FakeTranscriber(),
        learning_root=tmp_path / "youtube-learning",
    )
    pcm = np.full(16_000, 0.1, dtype="<f4").tobytes()

    with TestClient(app).websocket_connect(
        "/ws/captions?model=whisper_fine_tuned&sample_rate=16000&chunk_seconds=1&learning=true",
        headers={"origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"},
    ) as socket:
        ready = socket.receive_json()
        socket.send_bytes(pcm)
        caption = socket.receive_json()
        learning = socket.receive_json()

    assert ready["learning_enabled"] is True
    assert caption["type"] == "caption"
    assert learning == {
        "type": "learning",
        "accepted": True,
        "reason": "consensus",
        "total_samples": 1,
        "agreement": 1.0,
    }
    assert (tmp_path / "youtube-learning" / "train.jsonl").is_file()


def test_realtime_wav2vec2_keeps_configured_language_model_decoder() -> None:
    root = Path(__file__).resolve().parents[1]
    transcriber = _DecoderSensitiveTranscriber()
    app = create_realtime_app(
        root / "config.phowhisper_vietspeech_10h.yaml",
        transcriber_loader=lambda _: transcriber,
    )
    pcm = np.full(16_000, 0.1, dtype="<f4").tobytes()

    with TestClient(app).websocket_connect(
        "/ws/captions?model=wav2vec2_vietnamese&sample_rate=16000&chunk_seconds=1",
        headers={"origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"},
    ) as socket:
        socket.receive_json()
        socket.send_bytes(pcm)
        caption = socket.receive_json()

    assert caption["text"] == "beam có từ điển"
    assert transcriber.decoder_name == "ctc_beam_search_4gram"


def test_websocket_rejects_non_extension_origin(realtime_client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as error:
        with realtime_client.websocket_connect(
            "/ws/captions?model=wav2vec2_vietnamese&sample_rate=16000&chunk_seconds=4",
            headers={"origin": "https://attacker.example"},
        ):
            pass

    assert error.value.code == 1008


def test_websocket_allows_only_one_live_caption_session(
    realtime_client: TestClient,
) -> None:
    path = "/ws/captions?model=wav2vec2_vietnamese&sample_rate=16000&chunk_seconds=4"
    headers = {"origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"}

    with realtime_client.websocket_connect(path, headers=headers) as first:
        assert first.receive_json()["type"] == "ready"
        with pytest.raises(WebSocketDisconnect) as error:
            with realtime_client.websocket_connect(path, headers=headers):
                pass

    assert error.value.code == 1013
