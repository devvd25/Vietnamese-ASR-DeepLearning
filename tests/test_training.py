from __future__ import annotations

import torch

from src.model_utils import DataCollatorSpeechSeq2SeqWithPadding, align_ctc_special_tokens
from src.train import (
    ASRManifestDataset,
    build_compute_metrics,
    build_training_arguments,
    build_training_callbacks,
    configure_generation_for_evaluation,
    encode_asr_record,
    make_json_safe,
)


class _FeatureExtractor:
    def pad(self, features, return_tensors):
        assert return_tensors == "pt"
        width = max(item["input_features"].shape[-1] for item in features)
        batch = torch.zeros((len(features), 80, width), dtype=torch.float32)
        for index, item in enumerate(features):
            value = torch.as_tensor(item["input_features"])
            batch[index, :, : value.shape[-1]] = value
        return {"input_features": batch}


class _Tokenizer:
    bos_token_id = 1

    def pad(self, features, return_tensors):
        assert return_tensors == "pt"
        width = max(len(item["input_ids"]) for item in features)
        ids = torch.zeros((len(features), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, item in enumerate(features):
            value = torch.tensor(item["input_ids"], dtype=torch.long)
            ids[index, : len(value)] = value
            mask[index, : len(value)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _Processor:
    feature_extractor = _FeatureExtractor()
    tokenizer = _Tokenizer()


def test_align_ctc_special_tokens_removes_unused_bos_and_eos() -> None:
    tokenizer = type("Tokenizer", (), {"bos_token": "<s>", "eos_token": "</s>"})()
    processor = type("Processor", (), {"tokenizer": tokenizer})()
    config = type("Config", (), {"bos_token_id": 110, "eos_token_id": 111})()
    model = type("Model", (), {"config": config})()

    align_ctc_special_tokens(processor, model)

    assert processor.tokenizer.bos_token is None
    assert processor.tokenizer.eos_token is None
    assert model.config.bos_token_id is None
    assert model.config.eos_token_id is None


def test_collator_dynamically_pads_and_masks_label_padding() -> None:
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=_Processor())
    batch = collator(
        [
            {"input_features": torch.ones(80, 3), "labels": [1, 10, 11]},
            {"input_features": torch.ones(80, 5), "labels": [1, 12]},
        ]
    )

    assert tuple(batch["input_features"].shape) == (2, 80, 5)
    assert batch["labels"].tolist() == [[10, 11], [12, -100]]


def test_training_arguments_enable_reproducible_best_checkpoint(tmp_path) -> None:
    config = {
        "training": {
            "epochs": 2,
            "learning_rate": 0.00005,
            "train_batch_size": 2,
            "eval_batch_size": 3,
            "gradient_accumulation_steps": 4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "seed": 42,
            "eval_steps": 25,
            "save_steps": 25,
            "logging_steps": 5,
            "save_total_limit": 2,
        },
        "evaluation": {
            "generation_num_beams": 1,
            "max_length": 225,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
        },
    }

    args = build_training_arguments(config, output_dir=tmp_path, cuda_available=True)

    assert args.fp16 is True
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "wer"
    assert args.greater_is_better is False
    assert args.per_device_train_batch_size == 2
    assert args.generation_max_length == 225


def test_training_arguments_allow_independent_saves_for_single_epoch(tmp_path) -> None:
    config = {
        "training": {
            "epochs": 1,
            "learning_rate": 0.00001,
            "train_batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "eval_steps": 2684,
            "save_steps": 500,
            "load_best_model_at_end": False,
        },
        "evaluation": {"max_length": 225},
    }

    args = build_training_arguments(config, output_dir=tmp_path, cuda_available=True)

    assert args.load_best_model_at_end is False
    assert args.eval_steps == 2684
    assert args.save_steps == 500


def test_training_arguments_can_disable_validation_until_requested(tmp_path) -> None:
    config = {
        "training": {
            "epochs": 1,
            "learning_rate": 0.0000005,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "run_validation": False,
            "load_best_model_at_end": False,
            "save_steps": 3000,
        },
        "evaluation": {"max_length": 225},
    }

    args = build_training_arguments(config, output_dir=tmp_path, cuda_available=True)

    assert args.do_eval is False
    assert str(args.eval_strategy) == "IntervalStrategy.NO"
    assert args.load_best_model_at_end is False


def test_early_stopping_is_only_enabled_when_best_model_tracking_is_enabled() -> None:
    assert build_training_callbacks(load_best_model_at_end=False) == []
    assert len(build_training_callbacks(load_best_model_at_end=True)) == 1


def test_validation_generation_uses_same_anti_repetition_guardrails() -> None:
    generation_config = type("GenerationConfig", (), {})()
    model = type("Model", (), {"generation_config": generation_config})()

    configure_generation_for_evaluation(
        model,
        {
            "max_length": 225,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
        },
    )

    assert generation_config.max_length == 225
    assert generation_config.no_repeat_ngram_size == 3
    assert generation_config.repetition_penalty == 1.05


class _Encoded:
    def __init__(self) -> None:
        self.input_features = [torch.ones(80, 4)]


class _TrainingTokenizer:
    pad_token_id = 0

    def __call__(self, text):
        return type("Tokens", (), {"input_ids": [1, 21, 22]})()

    def batch_decode(self, values, skip_special_tokens=True):
        mapping = {10: "xin chào", 20: "cảm ơn bạn", 30: "cảm ơn"}
        return [mapping[int(row[0])] for row in values]


class _TrainingProcessor:
    tokenizer = _TrainingTokenizer()

    def __call__(self, audio, sampling_rate, return_tensors):
        assert sampling_rate == 16000
        assert return_tensors == "pt"
        return _Encoded()


def test_encode_asr_record_creates_features_and_labels() -> None:
    encoded = encode_asr_record(
        {"audio": {"array": [0.0] * 160, "sampling_rate": 16000}, "transcript": "xin chào"},
        processor=_TrainingProcessor(),
    )
    assert tuple(encoded["input_features"].shape) == (80, 4)
    assert encoded["labels"] == [1, 21, 22]


def test_manifest_dataset_encodes_audio_lazily() -> None:
    class CountingProcessor(_TrainingProcessor):
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, audio, sampling_rate, return_tensors):
            self.calls += 1
            return super().__call__(audio, sampling_rate, return_tensors)

    processor = CountingProcessor()
    records = [
        {
            "audio": {"array": [0.0] * 160, "sampling_rate": 16_000},
            "transcript": "Xin chào",
        }
    ]

    dataset = ASRManifestDataset(
        records,
        processor=processor,
        max_audio_seconds=30.0,
    )

    assert len(dataset) == 1
    assert processor.calls == 0
    encoded = dataset[0]
    assert processor.calls == 1
    assert tuple(encoded["input_features"].shape) == (80, 4)
    assert encoded["labels"] == [1, 21, 22]


def test_compute_metrics_decodes_and_calculates_corpus_rates() -> None:
    compute = build_compute_metrics(_TrainingProcessor())
    prediction = type(
        "Prediction",
        (),
        {
            "predictions": torch.tensor([[10], [30]]).numpy(),
            "label_ids": torch.tensor([[10], [20]]).numpy(),
        },
    )()
    metrics = compute(prediction)
    assert metrics["wer"] == 0.2
    assert metrics["cer"] > 0


def test_make_json_safe_replaces_non_finite_training_values() -> None:
    value = {
        "loss": 1.25,
        "grad_norms": [float("nan"), float("inf"), float("-inf")],
    }

    assert make_json_safe(value) == {
        "loss": 1.25,
        "grad_norms": [None, None, None],
    }
