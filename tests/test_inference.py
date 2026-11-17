from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

import src.inference as inference
from src.inference import (
    Wav2Vec2Transcriber,
    WhisperTranscriber,
    build_wav2vec2_decoder,
    resolve_model_source,
)


def test_wav2vec2_decoder_loads_training_words_as_unigrams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"transcript": text}, ensure_ascii=False)
            for text in ("sáu giờ đi coi phòng", "người ta đi xuống")
        ),
        encoding="utf-8",
    )
    language_model = tmp_path / "vi_lm.bin"
    language_model.write_bytes(b"test")
    captured: dict[str, object] = {}

    def fake_build(labels, **kwargs):
        captured["labels"] = labels
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "pyctcdecode",
        SimpleNamespace(build_ctcdecoder=fake_build),
    )

    class Tokenizer:
        def get_vocab(self):
            return {"a": 0, "|": 1, "<pad>": 2}

    processor = SimpleNamespace(tokenizer=Tokenizer())
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=3))

    decoder, name, beam_width, _ = build_wav2vec2_decoder(
        processor,
        model,
        {
            "decoder": {
                "enabled": True,
                "language_model_path": str(language_model),
                "unigrams_manifest": str(manifest),
                "beam_width": 50,
            }
        },
    )

    assert decoder is not None
    assert name == "ctc_beam_search_4gram"
    assert beam_width == 50
    assert captured["unigrams"] == [
        "coi",
        "giờ",
        "người",
        "phòng",
        "sáu",
        "ta",
        "xuống",
        "đi",
    ]


def test_resolve_model_source_uses_declared_vietnamese_wav2vec2() -> None:
    config = {
        "wav2vec2_model": {"name": "org/wav2vec2-vietnamese"},
        "model": {"name": "openai/whisper-tiny"},
        "paths": {},
    }
    assert (
        resolve_model_source("wav2vec2_vietnamese", config)
        == "org/wav2vec2-vietnamese"
    )


def test_wav2vec2_loader_pins_the_declared_hub_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []

    class FakeModel:
        def eval(self):
            return self

    def fake_components(source: str, *, revision: str | None = None):
        calls.append((source, revision))
        return object(), FakeModel(), "cpu"

    monkeypatch.setattr(inference, "load_wav2vec2_components", fake_components)
    monkeypatch.setattr(
        inference,
        "build_wav2vec2_decoder",
        lambda processor, model, model_config: (None, "ctc_greedy", 1, []),
    )
    config = {
        "wav2vec2_model": {
            "name": "org/wav2vec2-vietnamese",
            "revision": "0123456789abcdef",
            "decoder": {"enabled": False},
        }
    }

    Wav2Vec2Transcriber.load(config)

    assert calls == [("org/wav2vec2-vietnamese", "0123456789abcdef")]


def test_resolve_model_source_does_not_fake_missing_finetuned_checkpoint(tmp_path: Path) -> None:
    config = {
        "model": {"name": "openai/whisper-tiny"},
        "paths": {"best_checkpoint": str(tmp_path / "missing")},
    }
    with pytest.raises(FileNotFoundError, match="fine-tuned"):
        resolve_model_source("whisper_fine_tuned", config)


class _FakeProcessor:
    class Tokenizer:
        all_special_ids = (1,)

    tokenizer = Tokenizer()

    def __call__(self, audio, sampling_rate, return_tensors):
        assert sampling_rate == 16000
        return {"input_features": torch.ones((1, 80, 4))}

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ["Xin chào Việt Nam"]


class _FakeModel:
    def eval(self):
        return self

    def generate(self, **kwargs):
        assert kwargs["language"] == "Vietnamese"
        assert kwargs["task"] == "transcribe"
        assert kwargs["no_repeat_ngram_size"] == 3
        assert kwargs["max_length"] == 225
        assert "max_new_tokens" not in kwargs
        assert kwargs["repetition_penalty"] == 1.05
        assert kwargs["return_dict_in_generate"] is True
        assert "output_scores" not in kwargs
        return type(
            "GenerationOutput",
            (),
            {
                "sequences": torch.tensor([[1, 2, 3]]),
                "scores": (torch.tensor([[0.0]]), torch.tensor([[0.0]])),
                "beam_indices": None,
            },
        )()

    def __call__(self, **kwargs):
        assert tuple(kwargs["input_features"].shape) == (1, 80, 4)
        assert kwargs["decoder_input_ids"].tolist() == [[1, 2]]
        assert kwargs["use_cache"] is False
        probabilities = torch.tensor(
            [[[0.05, 0.03, 0.90, 0.02], [0.05, 0.05, 0.10, 0.80]]]
        )
        return type("Output", (), {"logits": probabilities.log()})()

    def compute_transition_scores(self, *args, **kwargs):
        raise AssertionError("Không được dùng compute_transition_scores cho Whisper.")


class _FakeCodeProcessor(_FakeProcessor):
    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ["Xin chào lớp ấy tê hài bà đề"]


class _FakePhraseProcessor(_FakeProcessor):
    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ["Xin chờ tất cả các bạn nha"]


def test_transcriber_runs_local_components_and_returns_normalized_text() -> None:
    transcriber = WhisperTranscriber(
        processor=_FakeProcessor(), model=_FakeModel(), device="cpu", model_source="local-test"
    )
    result = transcriber.transcribe_array([0.0] * 8000, sample_rate=8000)
    assert result["transcript"] == "xin chào việt nam"
    assert result["model_source"] == "local-test"
    assert result["duration_seconds"] == 1.0
    assert result["inference_seconds"] >= 0
    assert result["confidence"] == pytest.approx((0.9 * 0.8) ** 0.5)
    assert (
        result["confidence_method"]
        == "teacher_forced_geometric_mean_token_probability"
    )


def test_whisper_transcriber_normalizes_declared_spoken_domain_code() -> None:
    transcriber = WhisperTranscriber(
        processor=_FakeCodeProcessor(),
        model=_FakeModel(),
        device="cpu",
        model_source="local-test",
        domain_codes=["ST23D"],
    )

    result = transcriber.transcribe_array([0.0] * 8000, sample_rate=8000)

    assert result["raw_transcript"] == "xin chào lớp ấy tê hài bà đề"
    assert result["transcript"] == "xin chào lớp st23d"
    assert result["text_normalization"] == [
        {"source": "ấy tê hài bà đề", "replacement": "st23d"}
    ]


def test_whisper_transcriber_applies_disclosed_configured_phrase_correction() -> None:
    transcriber = WhisperTranscriber(
        processor=_FakePhraseProcessor(),
        model=_FakeModel(),
        device="cpu",
        model_source="local-test",
        phrase_corrections={"xin chờ tất cả": "xin chào tất cả"},
    )

    result = transcriber.transcribe_array([0.0] * 8000, sample_rate=8000)

    assert result["raw_transcript"] == "xin chờ tất cả các bạn nha"
    assert result["transcript"] == "xin chào tất cả các bạn nha"
    assert result["text_normalization"] == [
        {"source": "xin chờ tất cả", "replacement": "xin chào tất cả"}
    ]


class _FakeWav2Vec2Processor:
    class Tokenizer:
        pad_token_id = 0

    tokenizer = Tokenizer()

    def __call__(self, audio, sampling_rate, return_tensors, padding):
        assert sampling_rate == 16000
        assert return_tensors == "pt"
        assert padding is True
        return {"input_values": torch.ones((1, 16000))}

    def batch_decode(self, token_ids):
        assert token_ids.tolist() == [[1, 0]]
        return ["Xin Chào Việt Nam"]


class _FakeWav2Vec2Model:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        assert tuple(kwargs["input_values"].shape) == (1, 16000)
        return type(
            "Output",
            (),
            {"logits": torch.tensor([[[0.0, 2.0], [3.0, 0.0]]])},
        )()


class _FakeCtcDecoder:
    def decode(self, logits, *, beam_width):
        assert tuple(logits.shape) == (2, 2)
        assert beam_width == 50
        return "Xin chào lớp este hai bade"


def test_wav2vec2_transcriber_uses_ctc_argmax_and_normalizes_text() -> None:
    transcriber = Wav2Vec2Transcriber(
        processor=_FakeWav2Vec2Processor(),
        model=_FakeWav2Vec2Model(),
        device="cpu",
        model_source="local-wav2vec2",
    )

    result = transcriber.transcribe_array([0.0] * 8000, sample_rate=8000)

    assert result["transcript"] == "xin chào việt nam"
    assert result["model_source"] == "local-wav2vec2"
    assert result["duration_seconds"] == 1.0
    assert result["inference_seconds"] >= 0
    assert result["confidence"] == pytest.approx(
        torch.softmax(torch.tensor([0.0, 2.0]), dim=-1).max().item()
    )
    assert result["confidence_method"] == "mean_non_blank_frame_probability"


def test_wav2vec2_transcriber_uses_language_model_and_normalizes_domain_code() -> None:
    transcriber = Wav2Vec2Transcriber(
        processor=_FakeWav2Vec2Processor(),
        model=_FakeWav2Vec2Model(),
        device="cpu",
        model_source="local-wav2vec2",
        decoder=_FakeCtcDecoder(),
        decoder_name="ctc_beam_search_4gram",
        beam_width=50,
        domain_codes=["ST23D"],
    )

    result = transcriber.transcribe_array([0.0] * 8000, sample_rate=8000)

    assert result["raw_transcript"] == "xin chào lớp este hai bade"
    assert result["transcript"] == "xin chào lớp st23d"
    assert result["text_normalization"] == [
        {"source": "este hai bade", "replacement": "st23d"}
    ]
    assert result["decoder"] == "ctc_beam_search_4gram"
