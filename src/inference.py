"""Suy luận Wav2Vec2/Whisper cục bộ, không thay model thiếu bằng kết quả giả."""

from __future__ import annotations

import os
import shutil
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .model_utils import load_wav2vec2_components, load_whisper_components
from .preprocessing import (
    load_audio,
    normalize_domain_codes,
    normalize_domain_phrases,
    normalize_transcript,
    preprocess_audio,
)


def _kenlm_compatible_path(language_model_path: str | Path) -> Path:
    """Trả đường dẫn ASCII cho KenLM native trên Windows, ưu tiên hard-link."""
    source = Path(language_model_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy language model 4-gram: {source}")
    if os.name != "nt" or str(source).isascii():
        return source
    cache_dir = Path(source.anchor) / "VietnameseASRCache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{source.stem}-{source.stat().st_size}{source.suffix}"
    if cached.is_file() and cached.stat().st_size == source.stat().st_size:
        return cached
    try:
        os.link(source, cached)
    except OSError:
        shutil.copy2(source, cached)
    return cached


def load_decoder_unigrams(manifest_path: str | Path) -> list[str]:
    """Build the decoder vocabulary from normalized training transcripts."""

    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest unigram: {path}")
    words: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Manifest unigram không hợp lệ ở dòng {line_number}: {path}"
                ) from exc
            transcript = normalize_transcript(
                str(record.get("transcript", "")), reject_empty=False
            )
            words.update(transcript.split())
    if not words:
        raise ValueError(f"Manifest unigram không có transcript hợp lệ: {path}")
    return sorted(words)


def build_wav2vec2_decoder(processor, model, model_config: Mapping[str, Any]):
    decoder_config = model_config.get("decoder", {})
    if not isinstance(decoder_config, Mapping) or not decoder_config.get("enabled", False):
        return None, "ctc_greedy", 1, []
    from pyctcdecode import build_ctcdecoder

    language_model_path = _kenlm_compatible_path(
        str(decoder_config.get("language_model_path", ""))
    )
    unigrams_manifest = str(decoder_config.get("unigrams_manifest", "")).strip()
    unigrams = (
        load_decoder_unigrams(unigrams_manifest) if unigrams_manifest else None
    )
    vocab_size = int(model.config.vocab_size)
    labels = [""] * vocab_size
    for token, token_id in processor.tokenizer.get_vocab().items():
        if 0 <= int(token_id) < vocab_size:
            labels[int(token_id)] = token
    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=str(language_model_path),
        unigrams=unigrams,
        alpha=float(decoder_config.get("alpha", 1.0)),
        beta=float(decoder_config.get("beta", 1.0)),
    )
    return (
        decoder,
        "ctc_beam_search_4gram",
        int(decoder_config.get("beam_width", 100)),
        [str(code) for code in decoder_config.get("domain_codes", [])],
    )


def resolve_model_source(model_choice: str, config: Mapping[str, Any]) -> str:
    choice = model_choice.casefold().replace("-", "_")
    if choice in {"wav2vec2", "wav2vec2_vietnamese"}:
        name = str(config.get("wav2vec2_model", {}).get("name", "")).strip()
        if not name:
            raise ValueError("Thiếu wav2vec2_model.name.")
        return name
    if choice in {"whisper_fine_tuned", "fine_tuned", "finetuned", "proposed"}:
        checkpoint = Path(str(config.get("paths", {}).get("best_checkpoint", "")))
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                "Chưa có checkpoint Whisper fine-tuned hợp lệ tại: "
                f"{checkpoint or '(chưa cấu hình)'}"
            )
        return str(checkpoint.resolve())
    raise ValueError(
        "model_choice chỉ nhận wav2vec2_vietnamese hoặc whisper_fine_tuned."
    )


class Wav2Vec2Transcriber:
    """Nhận dạng tiếng Việt bằng Wav2Vec2-CTC, có thể kết hợp LM 4-gram."""

    def __init__(
        self,
        *,
        processor,
        model,
        device: str,
        model_source: str,
        decoder=None,
        decoder_name: str = "ctc_greedy",
        beam_width: int = 1,
        domain_codes: list[str] | None = None,
        phrase_corrections: Mapping[str, str] | None = None,
    ):
        self.processor = processor
        self.model = model
        self.device = device
        self.model_source = model_source
        self.decoder = decoder
        self.decoder_name = decoder_name
        self.beam_width = beam_width
        self.domain_codes = list(domain_codes or [])
        self.phrase_corrections = dict(phrase_corrections or {})
        self.model.eval()

    @classmethod
    def load(cls, config: Mapping[str, Any]) -> Wav2Vec2Transcriber:
        source = resolve_model_source("wav2vec2_vietnamese", config)
        revision = str(config.get("wav2vec2_model", {}).get("revision", "")).strip()
        processor, model, device = load_wav2vec2_components(
            source,
            revision=revision or None,
        )
        decoder, decoder_name, beam_width, domain_codes = build_wav2vec2_decoder(
            processor, model, config.get("wav2vec2_model", {})
        )
        configured_codes = config.get("text_normalization", {}).get("domain_codes")
        if configured_codes is not None:
            domain_codes = [str(code) for code in configured_codes]
        phrase_corrections = config.get("text_normalization", {}).get(
            "phrase_corrections", {}
        )
        if not isinstance(phrase_corrections, Mapping):
            raise ValueError("text_normalization.phrase_corrections phải là mapping.")
        return cls(
            processor=processor,
            model=model,
            device=device,
            model_source=source,
            decoder=decoder,
            decoder_name=decoder_name,
            beam_width=beam_width,
            domain_codes=domain_codes,
            phrase_corrections=phrase_corrections,
        )

    def transcribe_array(
        self,
        audio,
        *,
        sample_rate: int,
        max_audio_seconds: float = 30.0,
    ) -> dict[str, Any]:
        import torch

        original = np.asarray(audio)
        duration = len(original) / sample_rate if sample_rate > 0 else 0.0
        samples = preprocess_audio(
            original,
            sample_rate,
            target_sample_rate=16_000,
            max_seconds=max_audio_seconds,
        )
        encoded = self.processor(
            samples,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in dict(encoded).items()
        }
        started = perf_counter()
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            frame_probabilities = torch.softmax(logits.float(), dim=-1).amax(dim=-1)
            blank_token_id = getattr(
                getattr(self.processor, "tokenizer", None), "pad_token_id", None
            )
            confidence_mask = torch.ones_like(predicted_ids, dtype=torch.bool)
            if blank_token_id is not None:
                confidence_mask = predicted_ids.ne(int(blank_token_id))
            selected_probabilities = frame_probabilities[confidence_mask]
            if selected_probabilities.numel() == 0:
                selected_probabilities = frame_probabilities.flatten()
            confidence = float(selected_probabilities.mean().clamp(0.0, 1.0).item())
        elapsed = perf_counter() - started
        if self.decoder is None:
            decoded = self.processor.batch_decode(predicted_ids)[0]
        else:
            decoded = self.decoder.decode(
                logits[0].float().cpu().numpy(), beam_width=self.beam_width
            )
        raw_transcript = normalize_transcript(decoded, reject_empty=False)
        transcript, replacements = normalize_domain_codes(
            raw_transcript, self.domain_codes
        )
        transcript, phrase_replacements = normalize_domain_phrases(
            transcript, self.phrase_corrections
        )
        replacements.extend(phrase_replacements)
        if not transcript:
            raise RuntimeError("Mô hình Wav2Vec2 không sinh được transcript.")
        return {
            "transcript": transcript,
            "raw_transcript": raw_transcript,
            "text_normalization": [
                {"source": source, "replacement": replacement}
                for source, replacement in replacements
            ],
            "decoder": self.decoder_name,
            "duration_seconds": duration,
            "inference_seconds": elapsed,
            "model_source": self.model_source,
            "device": self.device,
            "confidence": confidence,
            "confidence_method": "mean_non_blank_frame_probability",
        }

    def transcribe_file(
        self,
        audio_path: str | Path,
        *,
        max_audio_seconds: float = 30.0,
    ) -> dict[str, Any]:
        audio, sample_rate = load_audio(audio_path)
        return self.transcribe_array(
            audio,
            sample_rate=sample_rate,
            max_audio_seconds=max_audio_seconds,
        )


class WhisperTranscriber:
    def __init__(
        self,
        *,
        processor,
        model,
        device: str,
        model_source: str,
        domain_codes: list[str] | None = None,
        phrase_corrections: Mapping[str, str] | None = None,
    ):
        self.processor = processor
        self.model = model
        self.device = device
        self.model_source = model_source
        self.domain_codes = list(domain_codes or [])
        self.phrase_corrections = dict(phrase_corrections or {})
        self.model.eval()

    @classmethod
    def load(cls, model_choice: str, config: Mapping[str, Any]) -> WhisperTranscriber:
        source = resolve_model_source(model_choice, config)
        model_config = config["model"]
        processor, model, device = load_whisper_components(
            source,
            language=str(model_config.get("language", "Vietnamese")),
            task=str(model_config.get("task", "transcribe")),
            revision=str(model_config.get("revision", "")).strip() or None,
        )
        return cls(
            processor=processor,
            model=model,
            device=device,
            model_source=source,
            domain_codes=[
                str(code)
                for code in config.get("text_normalization", {}).get(
                    "domain_codes", []
                )
            ],
            phrase_corrections=config.get("text_normalization", {}).get(
                "phrase_corrections", {}
            ),
        )

    def transcribe_array(
        self,
        audio,
        *,
        sample_rate: int,
        num_beams: int = 1,
        max_audio_seconds: float = 30.0,
        max_length: int = 225,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.05,
    ) -> dict[str, Any]:
        import torch

        original = np.asarray(audio)
        duration = len(original) / sample_rate if sample_rate > 0 else 0.0
        samples = preprocess_audio(
            original,
            sample_rate,
            target_sample_rate=16_000,
            max_seconds=max_audio_seconds,
        )
        encoded = self.processor(samples, sampling_rate=16_000, return_tensors="pt")
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in dict(encoded).items()
        }
        started = perf_counter()
        with torch.inference_mode():
            generation_output = self.model.generate(
                **model_inputs,
                language="Vietnamese",
                task="transcribe",
                num_beams=num_beams,
                max_length=max_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                return_dict_in_generate=True,
            )
            predicted_ids = generation_output.sequences
            targets = predicted_ids[:, 1:]
            confidence_output = self.model(
                **model_inputs,
                decoder_input_ids=predicted_ids[:, :-1],
                use_cache=False,
            )
            token_scores = torch.log_softmax(
                confidence_output.logits.float(), dim=-1
            ).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            confidence_mask = torch.isfinite(token_scores)
            for token_id in getattr(self.processor.tokenizer, "all_special_ids", []):
                confidence_mask &= targets.ne(int(token_id))
            selected_scores = token_scores[confidence_mask]
            if selected_scores.numel() == 0:
                selected_scores = token_scores[torch.isfinite(token_scores)]
            confidence = (
                float(selected_scores.mean().exp().clamp(0.0, 1.0).item())
                if selected_scores.numel()
                else None
            )
        elapsed = perf_counter() - started
        decoded = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        raw_transcript = normalize_transcript(decoded, reject_empty=False)
        transcript, replacements = normalize_domain_codes(
            raw_transcript, self.domain_codes
        )
        transcript, phrase_replacements = normalize_domain_phrases(
            transcript, self.phrase_corrections
        )
        replacements.extend(phrase_replacements)
        if not transcript:
            raise RuntimeError("Mô hình không sinh được transcript.")
        return {
            "transcript": transcript,
            "raw_transcript": raw_transcript,
            "text_normalization": [
                {"source": source, "replacement": replacement}
                for source, replacement in replacements
            ],
            "duration_seconds": duration,
            "inference_seconds": elapsed,
            "model_source": self.model_source,
            "device": self.device,
            "confidence": confidence,
            "confidence_method": "teacher_forced_geometric_mean_token_probability",
        }

    def transcribe_file(
        self,
        audio_path: str | Path,
        *,
        num_beams: int = 1,
        max_audio_seconds: float = 30.0,
        max_length: int = 225,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.05,
    ) -> dict[str, Any]:
        audio, sample_rate = load_audio(audio_path)
        return self.transcribe_array(
            audio,
            sample_rate=sample_rate,
            num_beams=num_beams,
            max_audio_seconds=max_audio_seconds,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )


def load_transcriber(
    model_choice: str, config: Mapping[str, Any]
) -> Wav2Vec2Transcriber | WhisperTranscriber:
    """Factory duy nhất cho hai model mà đề tài yêu cầu."""
    choice = model_choice.casefold().replace("-", "_")
    if choice in {"wav2vec2", "wav2vec2_vietnamese"}:
        return Wav2Vec2Transcriber.load(config)
    if choice in {"whisper_fine_tuned", "fine_tuned", "finetuned", "proposed"}:
        return WhisperTranscriber.load("whisper_fine_tuned", config)
    raise ValueError(
        "model_choice chỉ nhận wav2vec2_vietnamese hoặc whisper_fine_tuned."
    )
