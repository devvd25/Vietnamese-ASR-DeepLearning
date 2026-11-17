"""Loopback-only FastAPI service used by the Chrome live-caption extension."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import load_config
from .inference import load_transcriber
from .realtime_caption import (
    SUPPORTED_REALTIME_MODELS,
    RealtimeCaptionSession,
    is_allowed_extension_origin,
)
from .youtube_learning import YouTubeLearningStore

DEFAULT_MODEL = "wav2vec2_vietnamese"
DEFAULT_CHUNK_SECONDS = 4.0
DEFAULT_OVERLAP_SECONDS = 1.0


class _TranscriberCache:
    def __init__(self, loader: Callable[[str], Any]) -> None:
        self.loader = loader
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, model_choice: str) -> Any:
        with self._lock:
            if model_choice not in self._models:
                self._models[model_choice] = self.loader(model_choice)
            return self._models[model_choice]


def create_realtime_app(
    config_path: str | Path = "config.phowhisper_vietspeech_10h.yaml",
    *,
    transcriber_loader: Callable[[str], Any] | None = None,
    learning_root: str | Path | None = None,
) -> FastAPI:
    """Build the local API without loading either neural model eagerly."""

    config = load_config(config_path)
    loader = transcriber_loader or (lambda choice: load_transcriber(choice, config))
    cache = _TranscriberCache(loader)
    configured_learning_root = config.get("paths", {}).get(
        "youtube_learning_data", "data/processed/youtube_learning"
    )
    learning_store = YouTubeLearningStore(
        learning_root if learning_root is not None else configured_learning_root
    )
    inference_lock = asyncio.Lock()
    connection_gate = threading.Lock()
    app = FastAPI(
        title="Vietnamese ASR Live Captions",
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"chrome-extension://[a-p]{32}",
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ready",
            "service": "vietnamese-asr-live-captions",
            "models": sorted(SUPPORTED_REALTIME_MODELS),
            "default_model": DEFAULT_MODEL,
            "default_chunk_seconds": DEFAULT_CHUNK_SECONDS,
            "default_overlap_seconds": DEFAULT_OVERLAP_SECONDS,
            "privacy": "Audio chỉ xử lý trên máy local.",
        }

    @app.get("/learning/status")
    async def learning_status() -> dict[str, Any]:
        return {
            "status": "collecting",
            "accepted_samples": learning_store.total_samples,
            "minimum_samples_before_training": int(
                config.get("youtube_learning", {}).get("minimum_samples", 20)
            ),
            "active_checkpoint_unchanged": True,
        }

    @app.websocket("/ws/captions")
    async def captions(websocket: WebSocket) -> None:
        if not is_allowed_extension_origin(websocket.headers.get("origin")):
            await websocket.close(code=1008, reason="Extension origin required")
            return
        if not connection_gate.acquire(blocking=False):
            await websocket.close(code=1013, reason="Another caption session is active")
            return
        try:
            try:
                model_choice = websocket.query_params.get("model", DEFAULT_MODEL)
                sample_rate = int(websocket.query_params.get("sample_rate", "48000"))
                chunk_seconds = float(
                    websocket.query_params.get(
                        "chunk_seconds", str(DEFAULT_CHUNK_SECONDS)
                    )
                )
                overlap_value = websocket.query_params.get("overlap_seconds")
                overlap_seconds = (
                    float(overlap_value)
                    if overlap_value is not None
                    else (DEFAULT_OVERLAP_SECONDS if chunk_seconds > 1.0 else 0.0)
                )
                learning_value = websocket.query_params.get("learning", "false").casefold()
                if learning_value not in {"true", "false", "1", "0"}:
                    raise ValueError("learning must be true or false")
                learning_enabled = learning_value in {"true", "1"}
                generation = config.get("evaluation", {})
                transcribe_options = (
                    {
                        "num_beams": int(generation.get("generation_num_beams", 1)),
                        "max_length": int(generation.get("max_length", 225)),
                        "no_repeat_ngram_size": int(
                            generation.get("no_repeat_ngram_size", 3)
                        ),
                        "repetition_penalty": float(
                            generation.get("repetition_penalty", 1.05)
                        ),
                    }
                    if model_choice == "whisper_fine_tuned"
                    else {}
                )
                session = RealtimeCaptionSession(
                    model_choice=model_choice,
                    sample_rate=sample_rate,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                    transcriber_loader=cache.get,
                    transcribe_options=transcribe_options,
                    capture_learning=learning_enabled,
                )
            except (TypeError, ValueError) as exc:
                await websocket.close(code=1008, reason=str(exc)[:120])
                return

            await websocket.accept()
            ready_event = {
                    "type": "ready",
                    "model": model_choice,
                    "sample_rate": sample_rate,
                    "chunk_seconds": chunk_seconds,
                    "overlap_seconds": overlap_seconds,
                }
            if learning_enabled:
                ready_event["learning_enabled"] = True
            await websocket.send_json(ready_event)
            try:
                while True:
                    packet = await websocket.receive_bytes()
                    try:
                        async with inference_lock:
                            event = await asyncio.to_thread(session.ingest_pcm, packet)
                        if event is not None:
                            await websocket.send_json(event)
                        captured = session.take_learning_window()
                        if learning_enabled and captured is not None:
                            secondary_model = (
                                "wav2vec2_vietnamese"
                                if model_choice == "whisper_fine_tuned"
                                else "whisper_fine_tuned"
                            )
                            secondary_options = (
                                {
                                    "num_beams": int(
                                        generation.get("generation_num_beams", 1)
                                    ),
                                    "max_length": int(generation.get("max_length", 225)),
                                    "no_repeat_ngram_size": int(
                                        generation.get("no_repeat_ngram_size", 3)
                                    ),
                                    "repetition_penalty": float(
                                        generation.get("repetition_penalty", 1.05)
                                    ),
                                }
                                if secondary_model == "whisper_fine_tuned"
                                else {}
                            )
                            try:
                                async with inference_lock:
                                    learning_result = await asyncio.to_thread(
                                        learning_store.evaluate_and_store,
                                        captured,
                                        secondary_model=secondary_model,
                                        secondary_transcriber=cache.get(secondary_model),
                                        secondary_options=secondary_options,
                                    )
                            except (RuntimeError, TypeError, ValueError) as exc:
                                learning_result = {
                                    "accepted": False,
                                    "reason": "secondary_inference_failed",
                                    "total_samples": learning_store.total_samples,
                                    "message": str(exc)[:160],
                                }
                            await websocket.send_json(
                                {
                                    "type": "learning",
                                    **{
                                        key: value
                                        for key, value in learning_result.items()
                                        if key != "audio_id"
                                    },
                                }
                            )
                    except (RuntimeError, TypeError, ValueError) as exc:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": f"Không thể xử lý đoạn âm thanh: {exc}",
                            }
                        )
            except WebSocketDisconnect:
                return
        finally:
            connection_gate.release()

    return app
