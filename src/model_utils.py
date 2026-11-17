"""Tiện ích cấu hình và model Whisper; import thư viện nặng theo nhu cầu."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def validate_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = ("name", "language", "task")
    missing = [key for key in required if not str(config.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Thiếu cấu hình model: {', '.join(missing)}")
    result = dict(config)
    name = str(result["name"])
    if name.endswith(".en"):
        raise ValueError("Checkpoint .en chỉ hỗ trợ tiếng Anh; cần bản Whisper đa ngôn ngữ.")
    if str(result["task"]).casefold() != "transcribe":
        raise ValueError("Bài toán ASR tiếng Việt phải dùng task='transcribe'.")
    if str(result["language"]).casefold() not in {"vietnamese", "vi", "tiếng việt"}:
        raise ValueError("Ngôn ngữ phải là Vietnamese/vi.")
    return result


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Dynamic padding riêng cho log-Mel features và token labels."""

    processor: Any
    decoder_start_token_id: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        start_token = self.decoder_start_token_id
        if start_token is None:
            start_token = self.processor.tokenizer.bos_token_id
        if labels.shape[1] and torch.all(labels[:, 0].eq(start_token)):
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def load_whisper_components(
    model_source: str,
    *,
    language: str = "Vietnamese",
    task: str = "transcribe",
    device: str | None = None,
    revision: str | None = None,
):
    """Nạp processor/model một lần với prompt ngôn ngữ tiếng Việt."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(
        model_source, language=language, task=task, revision=revision
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_source, revision=revision
    )
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target)
    return processor, model, target


def align_ctc_special_tokens(processor, model) -> None:
    """Bỏ BOS/EOS không dùng trong CTC và có thể nằm ngoài output vocab."""
    processor.tokenizer.bos_token = None
    processor.tokenizer.eos_token = None
    model.config.bos_token_id = None
    model.config.eos_token_id = None


def load_wav2vec2_components(
    model_source: str,
    *,
    device: str | None = None,
    revision: str | None = None,
):
    """Nạp acoustic model Wav2Vec2-CTC tiếng Việt và processor tương ứng."""
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_source, revision=revision)
    model = AutoModelForCTC.from_pretrained(model_source, revision=revision)
    align_ctc_special_tokens(processor, model)
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target)
    return processor, model, target
