from __future__ import annotations

import torch

from src.train_ctc import (
    DataCollatorCTCWithPadding,
    build_ctc_compute_metrics,
    normalize_ctc_transcript,
)


def test_normalize_ctc_transcript_removes_tokens_missing_from_vocab() -> None:
    vocab = {
        character: index
        for index, character in enumerate("abcdeghinost23đểàịỉ ")
    }

    text = normalize_ctc_transcript("Địa chỉ: ST23D!", vocab)

    assert text == "địa chỉ st23d"


class _Tokenizer:
    pad_token_id = 0

    def pad(self, features, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        width = max(len(item["input_ids"]) for item in features)
        ids = torch.zeros((len(features), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, item in enumerate(features):
            values = torch.tensor(item["input_ids"], dtype=torch.long)
            ids[index, : len(values)] = values
            mask[index, : len(values)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def batch_decode(self, values, group_tokens=True):
        mapping = {1: "xin chào", 2: "cảm ơn", 3: "cảm ơn bạn"}
        return [mapping[int(row[0])] for row in values]


class _Processor:
    tokenizer = _Tokenizer()

    def pad(self, features, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        width = max(len(item["input_values"]) for item in features)
        values = torch.zeros((len(features), width), dtype=torch.float32)
        mask = torch.zeros((len(features), width), dtype=torch.long)
        for index, item in enumerate(features):
            row = torch.as_tensor(item["input_values"], dtype=torch.float32)
            values[index, : len(row)] = row
            mask[index, : len(row)] = 1
        return {"input_values": values, "attention_mask": mask}


def test_ctc_collator_pads_audio_and_masks_label_padding() -> None:
    collator = DataCollatorCTCWithPadding(processor=_Processor())

    batch = collator(
        [
            {"input_values": [0.1, 0.2], "labels": [1, 2]},
            {"input_values": [0.3, 0.4, 0.5], "labels": [3]},
        ]
    )

    assert tuple(batch["input_values"].shape) == (2, 3)
    assert batch["labels"].tolist() == [[1, 2], [3, -100]]


def test_ctc_metrics_use_argmax_and_do_not_group_reference_tokens() -> None:
    compute = build_ctc_compute_metrics(_Processor())
    prediction = type(
        "Prediction",
        (),
        {
            "predictions": torch.tensor(
                [[[0.0, 3.0, 1.0, 0.0]], [[0.0, 0.0, 3.0, 1.0]]]
            ).numpy(),
            "label_ids": torch.tensor([[1], [3]]).numpy(),
        },
    )()

    metrics = compute(prediction)

    assert metrics["wer"] == 0.2
    assert metrics["cer"] > 0
