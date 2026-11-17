from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pytest

import app.gradio_app as gradio_app
from app.gradio_app import (
    _available_models,
    build_demo,
    format_confidence_proxy,
    format_text_normalization,
    validate_audio_input,
)
from src.inference import Wav2Vec2Transcriber


def test_demo_rejects_missing_input() -> None:
    with pytest.raises(ValueError, match="tệp âm thanh"):
        validate_audio_input(None)


def test_demo_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "payload.txt"
    path.write_text("not audio", encoding="utf-8")
    with pytest.raises(ValueError, match="định dạng"):
        validate_audio_input(path)


def test_demo_rejects_oversized_upload_before_decoding(tmp_path: Path) -> None:
    path = tmp_path / "oversized.wav"
    with path.open("wb") as handle:
        handle.seek(100 * 1024 * 1024)
        handle.write(b"0")
    with pytest.raises(ValueError, match="100 MB"):
        validate_audio_input(path)


def test_demo_builds_with_installed_gradio_version() -> None:
    from gradio import utils

    demo = build_demo("config.yaml")
    assert demo is not None
    assert isinstance(
        any(
            demo._asr_theme.to_dict() == built_in.to_dict()
            for built_in in utils.BUILT_IN_THEMES.values()
        ),
        bool,
    )


def test_demo_tells_users_long_audio_is_chunked_without_a_duration_limit() -> None:
    demo = build_demo("config.yaml")
    rendered_config = json.dumps(demo.config, ensure_ascii=False)

    assert "Không giới hạn thời lượng" in rendered_config
    assert "chia thành các đoạn" in rendered_config


def test_demo_presents_the_asr_workbench_as_four_clear_sections() -> None:
    demo = build_demo("config.yaml")
    rendered_config = html.unescape(json.dumps(demo.config, ensure_ascii=False))

    expected_copy = (
        "Vietnamese ASR Benchmark",
        "On-device engine",
        "model khả dụng",
        "Dữ liệu đầu vào",
        "Chọn audio",
        "Cấu hình & thực thi",
        "Văn bản nhận dạng",
        "Bảng chỉ số kỹ thuật",
        "Lưu ý đánh giá",
        gradio_app.LABEL_LATENCY,
        gradio_app.LABEL_CONFIDENCE,
        gradio_app.LABEL_WER,
        gradio_app.LABEL_CER,
    )
    for text in expected_copy:
        assert text in rendered_config
    for number in range(1, 5):
        # Số thứ tự chỉ dành cho screen reader; huy hiệu hiển thị là aria-hidden.
        assert f"{number}. </span>" in rendered_config
    assert "SYSTEM READY" not in rendered_config


def test_space_runtime_does_not_claim_audio_stays_on_device() -> None:
    demo = build_demo("config.yaml", runtime="space")
    rendered_config = html.unescape(json.dumps(demo.config, ensure_ascii=False))

    assert "HF ZeroGPU" in rendered_config
    assert "tải lên Hugging Face Space" in rendered_config
    assert "Mô hình nhận dạng" in rendered_config
    assert "Model local" not in rendered_config
    assert "On-device engine" not in rendered_config
    assert "chạy trên máy này" not in rendered_config


def test_demo_rejects_unknown_runtime() -> None:
    with pytest.raises(ValueError, match="runtime"):
        build_demo("config.yaml", runtime="cloud")  # type: ignore[arg-type]


def test_hero_stats_come_from_config_not_marketing_copy() -> None:
    choices = [("Wav2Vec2", "wav2vec2_vietnamese")]

    hero = gradio_app._hero_html(choices, runtime="local", chunk_seconds=12.5)

    assert "<strong>1</strong><span>model khả dụng</span>" in hero
    assert "<strong>12.5 s</strong>" in hero
    assert "<strong>100 MB</strong>" in hero
    assert "PhoWhisper<small>" not in hero


def test_input_errors_keep_the_metric_card_grid() -> None:
    transcript, file_info, timing, metrics = gradio_app.transcribe_for_ui(
        None, "wav2vec2_vietnamese", ""
    )

    assert transcript == ""
    assert "Vui lòng chọn một tệp âm thanh" in file_info
    assert 'role="alert"' in file_info
    assert timing.count("metric-card") == 2
    assert metrics.count("metric-card") == 2


def test_describe_audio_reads_header_without_decoding(tmp_path: Path) -> None:
    import soundfile as sf

    path = tmp_path / "a&b.wav"
    sf.write(path, np.zeros(8000, dtype=np.float32), 16000)

    info = gradio_app.describe_audio_for_ui(str(path))

    assert "0.50 giây" in info
    assert "16,000 Hz gốc" in info
    assert "a&amp;b.wav" in info
    assert gradio_app.describe_audio_for_ui(None) == gradio_app.EMPTY_FILE_INFO


def test_performance_panel_formats_latency_rtf_confidence_and_reference_errors() -> None:
    output = {
        "transcript": "xin chào mọi người",
        "inference_seconds": 0.342,
        "device": "cuda:0",
        "model_source": "vinai/PhoWhisper-base",
        "confidence": 0.948,
        "confidence_method": "geometric_mean_token_probability",
        "decoder": "seq2seq_beam_search",
        "chunk_count": 1,
        "silent_chunk_count": 0,
        "text_normalization": [],
    }

    timing, metrics = gradio_app.format_performance_metrics(
        output,
        audio_duration=8.52,
        reference_text="xin chào mọi người",
    )

    assert "metric-card" in timing
    assert "0.342" in timing
    assert "RTF: 0.04x" in timing
    assert "94.80%" in timing
    assert "proxy chưa hiệu chuẩn" in timing
    assert "WER: 0.0000" in metrics
    assert "CER: 0.0000" in metrics
    assert metrics.count("metric-card") == 2
    # Màu đúng/sai luôn đi kèm chữ, không chỉ dựa vào màu.
    assert metrics.count("Khớp hoàn toàn") == 2
    assert "metric-success" in metrics


def test_demo_handler_delegates_to_the_injected_zero_gpu_runner() -> None:
    calls: list[tuple[object, str, str, str]] = []

    def zero_gpu_runner(audio_path, model_choice, reference_text, *, config_path):
        calls.append((audio_path, model_choice, reference_text, config_path))
        return "transcript", "file", "timing", "metrics"

    handler = gradio_app.make_transcribe_handler(
        "config.space.yaml", zero_gpu_runner
    )

    result = handler("audio.wav", "whisper_fine_tuned", "xin chào")

    assert result == ("transcript", "file", "timing", "metrics")
    assert calls == [
        (
            "audio.wav",
            "whisper_fine_tuned",
            "xin chào",
            "config.space.yaml",
        )
    ]


class _ChunkRecordingWav2Vec2(Wav2Vec2Transcriber):
    def __init__(self) -> None:
        self.chunk_lengths: list[int] = []

    def transcribe_array(
        self,
        audio,
        *,
        sample_rate: int,
        max_audio_seconds: float = 30.0,
    ):
        self.chunk_lengths.append(len(audio))
        chunk_number = len(self.chunk_lengths)
        return {
            "transcript": f"đoạn {chunk_number}",
            "raw_transcript": f"đoạn {chunk_number}",
            "text_normalization": [],
            "decoder": "ctc_greedy",
            "duration_seconds": len(audio) / sample_rate,
            "inference_seconds": 0.1,
            "model_source": "local-test",
            "device": "cpu",
            "confidence": 0.9,
            "confidence_method": "mean_non_blank_frame_probability",
        }


class _TrailingSilenceWav2Vec2(_ChunkRecordingWav2Vec2):
    def transcribe_array(self, audio, *, sample_rate: int, max_audio_seconds: float = 30.0):
        if len(self.chunk_lengths) == 2:
            self.chunk_lengths.append(len(audio))
            raise RuntimeError("Mô hình Wav2Vec2 không sinh được transcript.")
        return super().transcribe_array(
            audio,
            sample_rate=sample_rate,
            max_audio_seconds=max_audio_seconds,
        )


def test_demo_transcribes_audio_longer_than_training_limit_in_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "long.wav"
    audio_path.write_bytes(b"RIFF")
    transcriber = _ChunkRecordingWav2Vec2()
    config = {
        "data": {"max_audio_seconds": 30},
        "demo": {"chunk_seconds": 28},
        "evaluation": {
            "generation_num_beams": 1,
            "max_length": 225,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
        },
    }
    monkeypatch.setattr(
        gradio_app, "load_audio", lambda path: (np.zeros(650, dtype=np.float32), 10)
    )
    monkeypatch.setattr(gradio_app, "load_config", lambda path: config)
    monkeypatch.setattr(
        gradio_app, "_get_transcriber", lambda model_choice, config_path: transcriber
    )

    transcript, file_info, timing, metrics = gradio_app.transcribe_for_ui(
        audio_path, "wav2vec2_vietnamese", ""
    )

    assert transcript == "đoạn 1 đoạn 2 đoạn 3"
    assert transcriber.chunk_lengths == [280, 280, 90]
    assert "65.00 giây" in file_info
    assert "3 đoạn" in timing
    assert "Không tính WER/CER" in metrics


def test_demo_keeps_recognized_chunks_when_trailing_chunk_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "long-with-silence.wav"
    audio_path.write_bytes(b"RIFF")
    transcriber = _TrailingSilenceWav2Vec2()
    config = {
        "data": {"max_audio_seconds": 30},
        "demo": {"chunk_seconds": 28},
        "evaluation": {"generation_num_beams": 1},
    }
    monkeypatch.setattr(
        gradio_app, "load_audio", lambda path: (np.zeros(650, dtype=np.float32), 10)
    )
    monkeypatch.setattr(gradio_app, "load_config", lambda path: config)
    monkeypatch.setattr(
        gradio_app, "_get_transcriber", lambda model_choice, config_path: transcriber
    )

    transcript, _, timing, _ = gradio_app.transcribe_for_ui(
        audio_path, "wav2vec2_vietnamese", ""
    )

    assert transcript == "đoạn 1 đoạn 2"
    assert transcriber.chunk_lengths == [280, 280, 90]
    assert "3 đoạn" in timing
    assert "1 đoạn không có tiếng nói" in timing


def test_demo_labels_confidence_as_uncalibrated_proxy() -> None:
    text = format_confidence_proxy(0.83456, "geometric_mean_token_probability")

    assert "83.46%" in text
    assert "proxy chưa hiệu chuẩn" in text
    assert "xác suất câu đúng" in text


def test_format_text_normalization_discloses_domain_replacement() -> None:
    text = format_text_normalization(
        [{"source": "este hai bade", "replacement": "st23d"}]
    )

    assert "Chuẩn hóa miền" in text
    assert "`este hai bade` → `st23d`" in text


def test_demo_exposes_required_model_pair_when_whisper_checkpoint_exists(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    config = {
        "model": {
            "display_name": "PhoWhisper Base — ST23D supervised adaptation",
            "adaptation_disclosure": "Mẫu ST23D đã được dùng để supervised adaptation.",
        },
        "wav2vec2_model": {"name": "org/wav2vec2-vietnamese"},
        "paths": {"best_checkpoint": str(checkpoint)},
    }

    choices, status = _available_models(config)

    assert [value for _, value in choices] == [
        "wav2vec2_vietnamese",
        "whisper_fine_tuned",
    ]
    assert "Mẫu ST23D đã được dùng" in status
    assert choices[1][0] == "PhoWhisper Base — ST23D supervised adaptation"
