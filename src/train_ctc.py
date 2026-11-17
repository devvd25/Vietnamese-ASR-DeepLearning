"""Fine-tuning Wav2Vec2-CTC trên các manifest ASR cục bộ."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import compute_corpus_metrics
from .preprocessing import normalize_transcript, prepare_asr_example
from .train import _limit_records, make_json_safe


def normalize_ctc_transcript(text: str, vocabulary: Mapping[str, int]) -> str:
    """Chuẩn hóa transcript và loại ký tự mà tokenizer CTC không hỗ trợ."""
    normalized = normalize_transcript(text)
    supported_characters = {
        token for token in vocabulary if len(token) == 1 and not token.startswith("<")
    }
    cleaned = "".join(
        character if character in supported_characters else " "
        for character in normalized
    )
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        raise ValueError("Transcript rỗng sau khi lọc theo vocabulary CTC.")
    return cleaned


@dataclass
class DataCollatorCTCWithPadding:
    """Dynamic padding riêng cho waveform và token labels CTC."""

    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        return batch


def build_ctc_compute_metrics(processor):
    """Tạo callback WER/CER cho logits CTC."""

    def compute_metrics(prediction):
        predictions = prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predicted_ids = np.asarray(predictions).argmax(axis=-1)
        labels = np.asarray(prediction.label_ids).copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        predicted_text = processor.tokenizer.batch_decode(predicted_ids)
        reference_text = processor.tokenizer.batch_decode(
            labels,
            group_tokens=False,
        )
        metrics = compute_corpus_metrics(reference_text, predicted_text)
        return {"wer": metrics["wer"], "cer": metrics["cer"]}

    return compute_metrics


class CTCManifestDataset:
    """Nạp và mã hóa waveform theo từng mẫu để giữ RAM ổn định."""

    def __init__(
        self,
        records: list[dict],
        *,
        processor,
        max_audio_seconds: float,
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.max_audio_seconds = max_audio_seconds
        self.vocabulary = processor.tokenizer.get_vocab()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        prepared = prepare_asr_example(
            self.records[index],
            target_sample_rate=16_000,
            max_seconds=self.max_audio_seconds,
        )
        inputs = self.processor(
            prepared["audio"]["array"],
            sampling_rate=16_000,
            return_tensors="pt",
        )
        transcript = normalize_ctc_transcript(
            prepared["transcript"],
            self.vocabulary,
        )
        return {
            "input_values": inputs.input_values[0],
            "labels": self.processor.tokenizer(transcript).input_ids,
        }


def build_ctc_training_arguments(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    cuda_available: bool,
):
    from transformers import TrainingArguments

    training = config.get("wav2vec2_training", config["training"])
    return TrainingArguments(
        output_dir=str(Path(output_dir)),
        do_train=True,
        do_eval=True,
        num_train_epochs=float(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        per_device_train_batch_size=int(training.get("train_batch_size", 1)),
        per_device_eval_batch_size=int(training.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        warmup_steps=int(training.get("warmup_steps", 0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        fp16=bool(cuda_available),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=int(training.get("eval_steps", 100)),
        save_steps=int(training.get("save_steps", training.get("eval_steps", 100))),
        logging_steps=int(training.get("logging_steps", 10)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        load_best_model_at_end=bool(training.get("load_best_model_at_end", True)),
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=int(training.get("seed", 42)),
        data_seed=int(training.get("seed", 42)),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=int(training.get("dataloader_num_workers", 0)),
    )


def run_wav2vec2_fine_tuning(
    config: Mapping[str, Any],
    *,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Fine-tune Wav2Vec2-CTC trên cùng manifest dùng cho Whisper."""
    import torch
    from transformers import Trainer, set_seed

    from .data_loader import load_jsonl_manifest
    from .model_utils import load_wav2vec2_components

    training = config.get("wav2vec2_training", config["training"])
    seed = int(training.get("seed", 42))
    set_seed(seed)
    processed_dir = Path(config["paths"]["processed_data"])
    train_records = load_jsonl_manifest(processed_dir / "train.jsonl")
    validation_records = load_jsonl_manifest(processed_dir / "validation.jsonl")
    train_records = _limit_records(
        train_records,
        config["data"].get("max_train_samples"),
        seed,
    )
    validation_records = _limit_records(
        validation_records,
        config["data"].get("max_validation_samples"),
        seed,
    )

    model_source = str(config["wav2vec2_model"]["name"])
    processor, model, _ = load_wav2vec2_components(
        model_source,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if bool(training.get("freeze_feature_encoder", True)):
        model.freeze_feature_encoder()
    model.config.ctc_loss_reduction = "mean"
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    max_audio_seconds = float(config["data"]["max_audio_seconds"])
    train_dataset = CTCManifestDataset(
        train_records,
        processor=processor,
        max_audio_seconds=max_audio_seconds,
    )
    validation_dataset = CTCManifestDataset(
        validation_records,
        processor=processor,
        max_audio_seconds=max_audio_seconds,
    )
    output_dir = Path(config["paths"]["wav2vec2_checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = build_ctc_training_arguments(
        config,
        output_dir=output_dir,
        cuda_available=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        processing_class=processor,
        compute_metrics=build_ctc_compute_metrics(processor),
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    best_dir = Path(config["paths"]["wav2vec2_best_checkpoint"])
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    result = {
        "status": "completed",
        "experiment": config.get("project", {}).get("experiment_name"),
        "model": model_source,
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "saved_best_directory": str(best_dir.resolve()),
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    reports_dir = Path(config["paths"]["wav2vec2_reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "training_result.json").open("w", encoding="utf-8") as handle:
        json.dump(
            make_json_safe(result),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    with (reports_dir / "trainer_log_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            make_json_safe(trainer.state.log_history),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    return result
