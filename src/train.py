"""Pipeline fine-tuning Whisper bằng Hugging Face Seq2SeqTrainer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import compute_corpus_metrics
from .preprocessing import normalize_transcript, prepare_asr_example, preprocess_audio


def make_json_safe(value: Any) -> Any:
    """Recursively replace non-finite numeric telemetry with JSON ``null``."""
    if isinstance(value, Mapping):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def build_training_arguments(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    cuda_available: bool,
):
    from transformers import Seq2SeqTrainingArguments

    training = config["training"]
    evaluation = config.get("evaluation", {})
    run_validation = bool(training.get("run_validation", True))
    load_best_model_at_end = run_validation and bool(
        training.get("load_best_model_at_end", True)
    )
    return Seq2SeqTrainingArguments(
        output_dir=str(Path(output_dir)),
        do_train=True,
        do_eval=run_validation,
        num_train_epochs=float(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        per_device_train_batch_size=int(training.get("train_batch_size", 4)),
        per_device_eval_batch_size=int(training.get("eval_batch_size", 4)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        warmup_steps=int(training.get("warmup_steps", 0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        fp16=bool(cuda_available),
        eval_strategy="steps" if run_validation else "no",
        save_strategy="steps",
        eval_steps=int(training.get("eval_steps", 100)),
        save_steps=int(training.get("save_steps", training.get("eval_steps", 100))),
        logging_steps=int(training.get("logging_steps", 10)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        predict_with_generate=True,
        generation_num_beams=int(evaluation.get("generation_num_beams", 1)),
        generation_max_length=int(evaluation.get("max_length", 225)),
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=int(training.get("seed", 42)),
        data_seed=int(training.get("seed", 42)),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=int(training.get("dataloader_num_workers", 0)),
    )


def configure_generation_for_evaluation(model, evaluation: Mapping[str, Any]) -> None:
    """Đồng bộ guardrail decode giữa validation và test inference."""
    model.generation_config.max_length = int(evaluation.get("max_length", 225))
    model.generation_config.no_repeat_ngram_size = int(
        evaluation.get("no_repeat_ngram_size", 0)
    )
    model.generation_config.repetition_penalty = float(
        evaluation.get("repetition_penalty", 1.0)
    )


def build_training_callbacks(*, load_best_model_at_end: bool) -> list[Any]:
    """Chỉ bật early stopping khi Trainer có theo dõi best checkpoint."""
    if not load_best_model_at_end:
        return []
    from transformers import EarlyStoppingCallback

    return [EarlyStoppingCallback(early_stopping_patience=2)]


def encode_asr_record(*args, **kwargs):
    record = args[0] if args else kwargs.pop("record")
    processor = kwargs.pop("processor")
    if kwargs:
        raise TypeError(f"Tham số không hỗ trợ: {sorted(kwargs)}")
    audio_data = record["audio"]
    samples = np.asarray(audio_data["array"], dtype=np.float32)
    sample_rate = int(audio_data["sampling_rate"])
    samples = preprocess_audio(samples, sample_rate, target_sample_rate=16_000)
    inputs = processor(samples, sampling_rate=16_000, return_tensors="pt")
    transcript = normalize_transcript(str(record["transcript"]))
    labels = processor.tokenizer(transcript).input_ids
    return {
        "input_features": inputs.input_features[0],
        "labels": labels,
    }


def build_compute_metrics(*args, **kwargs):
    processor = args[0] if args else kwargs["processor"]

    def compute_metrics(prediction):
        predictions = prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.asarray(prediction.label_ids).copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        predicted_text = processor.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        reference_text = processor.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        metrics = compute_corpus_metrics(reference_text, predicted_text)
        return {"wer": metrics["wer"], "cer": metrics["cer"]}

    return compute_metrics


def _limit_records(records: list[dict], limit: int | None, seed: int) -> list[dict]:
    if not limit or limit >= len(records):
        return records
    import random

    selected = list(records)
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def encode_records_as_dataset(
    records: list[dict],
    *,
    processor,
    max_audio_seconds: float,
):
    """Đọc audio theo từng record và tạo Hugging Face Dataset đã encode."""
    from datasets import Dataset

    dataset = Dataset.from_list(records)

    def encode(record: dict) -> dict:
        prepared = prepare_asr_example(
            record,
            target_sample_rate=16_000,
            max_seconds=max_audio_seconds,
        )
        return encode_asr_record(prepared, processor=processor)

    return dataset.map(
        encode,
        remove_columns=dataset.column_names,
        desc="Trích xuất log-Mel và token labels",
    )


class ASRManifestDataset:
    """Mã hóa từng mẫu khi DataLoader yêu cầu để tránh giữ toàn bộ log-Mel trong RAM."""

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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        prepared = prepare_asr_example(
            self.records[index],
            target_sample_rate=16_000,
            max_seconds=self.max_audio_seconds,
        )
        return encode_asr_record(prepared, processor=self.processor)


def run_fine_tuning(
    config: Mapping[str, Any],
    *,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Fine-tune thật; yêu cầu manifest đã tạo bởi prepare_data.py."""
    import json

    import torch
    from transformers import Seq2SeqTrainer, set_seed

    from .data_loader import load_jsonl_manifest
    from .model_utils import (
        DataCollatorSpeechSeq2SeqWithPadding,
        load_whisper_components,
    )

    seed = int(config["training"].get("seed", 42))
    set_seed(seed)
    processed_dir = Path(config["paths"]["processed_data"])
    train_records = load_jsonl_manifest(processed_dir / "train.jsonl")
    run_validation = bool(config["training"].get("run_validation", True))
    validation_records = (
        load_jsonl_manifest(processed_dir / "validation.jsonl")
        if run_validation
        else []
    )
    train_records = _limit_records(train_records, config["data"].get("max_train_samples"), seed)
    validation_records = _limit_records(
        validation_records, config["data"].get("max_validation_samples"), seed
    )
    model_config = config["model"]
    processor, model, _ = load_whisper_components(
        str(model_config["name"]),
        language=str(model_config["language"]),
        task=str(model_config["task"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.config.use_cache = False
    configure_generation_for_evaluation(model, config.get("evaluation", {}))
    train_dataset = ASRManifestDataset(
        train_records,
        processor=processor,
        max_audio_seconds=float(config["data"]["max_audio_seconds"]),
    )
    validation_dataset = (
        ASRManifestDataset(
            validation_records,
            processor=processor,
            max_audio_seconds=float(config["data"]["max_audio_seconds"]),
        )
        if run_validation
        else None
    )
    output_dir = Path(config["paths"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = build_training_arguments(
        config,
        output_dir=output_dir,
        cuda_available=torch.cuda.is_available(),
    )
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=build_compute_metrics(processor) if run_validation else None,
        callbacks=build_training_callbacks(
            load_best_model_at_end=training_args.load_best_model_at_end
        ),
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    validation_metrics = (
        trainer.evaluate(metric_key_prefix="validation") if run_validation else None
    )
    best_dir = Path(config["paths"]["best_checkpoint"])
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    result = {
        "status": "completed",
        "experiment": config.get("project", {}).get("experiment_name"),
        "model": str(model_config["name"]),
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "saved_best_directory": str(best_dir.resolve()),
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "training_result.json").open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(result), handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (reports_dir / "trainer_log_history.json").open("w", encoding="utf-8") as handle:
        json.dump(
            make_json_safe(trainer.state.log_history),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    from .evaluate import plot_available_results

    plot_available_results(dict(config))
    return result
