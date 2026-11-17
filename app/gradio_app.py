"""Demo Gradio; model chỉ được nạp khi người dùng thực sự suy luận."""

from __future__ import annotations

import html
import math
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import soundfile as sf

from src.cli import configure_utf8_stdio
from src.config import load_config
from src.inference import Wav2Vec2Transcriber, load_transcriber
from src.metrics import calculate_cer, calculate_wer
from src.preprocessing import (
    SUPPORTED_AUDIO_EXTENSIONS,
    load_audio,
    normalize_transcript,
)

MAX_AUDIO_FILE_BYTES = 100 * 1024 * 1024

Runtime = Literal["local", "space"]

RECOGNIZE_LABEL = "Nhận dạng tiếng nói"
RECOGNIZING_LABEL = "Đang nhận dạng…"
EMPTY_FILE_INFO = '<p class="file-meta file-meta-empty">Chưa chọn tệp âm thanh.</p>'

LABEL_LATENCY = "Thời gian suy luận"
LABEL_CONFIDENCE = "Độ tin cậy ước tính"
LABEL_WER = "WER (từ lỗi)"
LABEL_CER = "CER (ký tự lỗi)"

# Nhãn môi trường phải đúng sự thật: bản Space nhận audio qua upload lên Hugging Face.
RUNTIME_COPY: dict[str, dict[str, str]] = {
    "local": {
        "badge": "On-device engine",
        "notice": (
            "Mọi tác vụ suy luận chạy trên máy này; audio không được gửi tới "
            "dịch vụ ASR bên ngoài."
        ),
    },
    "space": {
        "badge": "HF ZeroGPU",
        "notice": (
            "Audio được tải lên Hugging Face Space này để suy luận trên ZeroGPU; "
            "không gửi tới dịch vụ ASR bên thứ ba."
        ),
    },
}


def format_confidence_proxy(confidence: float | None, method: str) -> str:
    """Định dạng proxy và nêu rõ giới hạn để không bị hiểu là độ chính xác."""
    method_labels = {
        "mean_non_blank_frame_probability": "trung bình xác suất khung CTC không rỗng",
        "geometric_mean_token_probability": "trung bình nhân xác suất token sinh",
        "teacher_forced_geometric_mean_token_probability": (
            "trung bình nhân xác suất token bằng teacher forcing"
        ),
    }
    if confidence is None or not math.isfinite(float(confidence)):
        value = "không khả dụng"
    else:
        value = f"{100 * float(confidence):.2f}%"
    method_label = method_labels.get(method, method)
    return (
        f"**Độ tin cậy ước lượng:** {value}  \n"
        f"**Cách tính:** {method_label}  \n"
        "*Đây là proxy chưa hiệu chuẩn, không phải xác suất câu đúng.*"
    )


def format_text_normalization(replacements: list[dict[str, str]] | None) -> str:
    """Công khai mọi thay thế hậu xử lý để kết quả demo không gây hiểu nhầm."""
    if not replacements:
        return ""
    details = "; ".join(
        f"`{item['source']}` → `{item['replacement']}`" for item in replacements
    )
    return f"**Chuẩn hóa miền:** {details}"


def _metric_card(
    label: str,
    value: str,
    detail: str,
    *,
    tone: str = "default",
    status: str | None = None,
) -> str:
    """Render one compact metric card with text as well as color semantics."""
    tones = {"default", "sky", "lilac", "success", "danger", "muted"}
    safe_tone = tone if tone in tones else "default"
    status_html = (
        f'<p class="metric-status">{html.escape(status)}</p>' if status else ""
    )
    return (
        f'<article class="metric-card metric-{safe_tone}">'
        f'<p class="metric-label">{html.escape(label)}</p>'
        f'<p class="metric-value">{html.escape(value)}</p>'
        f'<p class="metric-detail">{html.escape(detail)}</p>'
        f"{status_html}"
        "</article>"
    )


def empty_timing_html(note: str = "Chưa có thông tin thiết bị và kiến trúc.") -> str:
    """Placeholder cards kept identical in the idle and error states."""
    return (
        '<div class="metric-pair-content" aria-live="polite">'
        + _metric_card(LABEL_LATENCY, "—", "RTF: N/A", tone="muted")
        + _metric_card(
            LABEL_CONFIDENCE,
            "N/A",
            "proxy chưa hiệu chuẩn · Chưa chạy suy luận",
            tone="muted",
        )
        + f'<p class="run-metadata">{html.escape(note)}</p>'
        + "</div>"
    )


def empty_metrics_html(summary: str = "Không có metric.") -> str:
    return (
        '<div class="metric-pair-content" aria-live="polite">'
        + _metric_card(LABEL_WER, "N/A", "Chưa có transcript tham chiếu", tone="muted")
        + _metric_card(LABEL_CER, "N/A", "Chưa có transcript tham chiếu", tone="muted")
        + f'<p class="metric-summary">{html.escape(summary)}</p>'
        + "</div>"
    )


def _alert_html(message: str) -> str:
    return f'<p class="inline-alert" role="alert">{html.escape(message)}</p>'


def _file_meta_html(
    name: str, duration: float | None, sample_rate: int | None
) -> str:
    parts = [f"<strong>{html.escape(name)}</strong>"]
    if duration is not None:
        parts.append(f"{duration:.2f} giây")
    if sample_rate is not None:
        parts.append(f"{sample_rate:,} Hz gốc")
    else:
        parts.append("thời lượng được đo khi nhận dạng")
    return f'<p class="file-meta">{" · ".join(parts)}</p>'


def format_performance_metrics(
    output: dict[str, Any],
    *,
    audio_duration: float,
    reference_text: str,
) -> tuple[str, str]:
    """Format timing/confidence and reference errors as a four-card metric grid."""
    inference_seconds = float(output["inference_seconds"])
    rtf = inference_seconds / audio_duration if audio_duration > 0 else None
    rtf_text = f"RTF: {rtf:.2f}x" if rtf is not None else "RTF: N/A"

    confidence = output.get("confidence")
    if confidence is None or not math.isfinite(float(confidence)):
        confidence_value = "N/A"
        confidence_tone = "muted"
    else:
        confidence_value = f"{100 * float(confidence):.2f}%"
        confidence_tone = "lilac"

    method_labels = {
        "mean_non_blank_frame_probability": "Proxy CTC non-blank",
        "geometric_mean_token_probability": "Proxy token sinh",
        "teacher_forced_geometric_mean_token_probability": "Proxy teacher forcing",
    }
    confidence_method = method_labels.get(
        str(output.get("confidence_method", "unavailable")), "Proxy không khả dụng"
    )

    timing_cards = (
        '<div class="metric-pair-content" aria-live="polite">'
        + _metric_card(
            LABEL_LATENCY,
            f"{inference_seconds:.3f}s",
            rtf_text,
            tone="sky",
        )
        + _metric_card(
            LABEL_CONFIDENCE,
            confidence_value,
            f"{confidence_method} · proxy chưa hiệu chuẩn",
            tone=confidence_tone,
        )
    )

    metadata = [
        f"Thiết bị: {output.get('device', 'N/A')}",
        f"Kiến trúc: {output.get('model_source', 'N/A')}",
    ]
    if output.get("decoder"):
        metadata.append(f"Decoder: {output['decoder']}")
    chunk_count = int(output.get("chunk_count", 1))
    if chunk_count > 1:
        metadata.append(
            f"Phân đoạn: {chunk_count} đoạn, tối đa "
            f"{float(output['chunk_seconds']):g} giây/đoạn"
        )
    silent_chunk_count = int(output.get("silent_chunk_count", 0))
    if silent_chunk_count > 0:
        metadata.append(f"Khoảng lặng: {silent_chunk_count} đoạn không có tiếng nói")
    safe_metadata = "<span> • </span>".join(html.escape(item) for item in metadata)
    timing_cards += f'<p class="run-metadata">{safe_metadata}</p>'

    replacements = output.get("text_normalization") or []
    if replacements:
        details = "; ".join(
            f"{item['source']} → {item['replacement']}" for item in replacements
        )
        timing_cards += (
            '<p class="normalization-note"><strong>Chuẩn hóa miền:</strong> '
            f"{html.escape(details)}</p>"
        )
    timing_cards += "</div>"

    if reference_text and reference_text.strip():
        reference = normalize_transcript(reference_text)
        wer = calculate_wer(reference, str(output["transcript"]))
        cer = calculate_cer(reference, str(output["transcript"]))
        metrics_summary = "Đã đối chiếu với transcript tham chiếu."
        wer_card = _metric_card(
            LABEL_WER,
            f"{100 * wer:.2f}%",
            f"WER: {wer:.4f} · Chuẩn tham chiếu",
            tone="success" if wer == 0 else "danger",
            status="Khớp hoàn toàn" if wer == 0 else "Có sai lệch",
        )
        cer_card = _metric_card(
            LABEL_CER,
            f"{100 * cer:.2f}%",
            f"CER: {cer:.4f} · Chuẩn tham chiếu",
            tone="success" if cer == 0 else "danger",
            status="Khớp hoàn toàn" if cer == 0 else "Có sai lệch",
        )
        metrics_cards = (
            '<div class="metric-pair-content" aria-live="polite">'
            f"{wer_card}{cer_card}"
            f'<p class="metric-summary">{html.escape(metrics_summary)}</p>'
            "</div>"
        )
    else:
        metrics_cards = empty_metrics_html(
            "Không tính WER/CER vì chưa có transcript tham chiếu."
        )
    return timing_cards, metrics_cards


def validate_audio_input(audio_path: str | Path | None) -> Path:
    if audio_path is None or not str(audio_path).strip():
        raise ValueError("Vui lòng chọn một tệp âm thanh.")
    path = Path(audio_path).expanduser()
    if path.suffix.casefold() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Không hỗ trợ định dạng {path.suffix}; dùng một trong: {supported}.")
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp âm thanh: {path}")
    if path.stat().st_size == 0:
        raise ValueError("Tệp âm thanh rỗng.")
    if path.stat().st_size > MAX_AUDIO_FILE_BYTES:
        raise ValueError("Tệp âm thanh vượt giới hạn 100 MB.")
    return path.resolve()


def describe_audio_for_ui(audio_path: str | Path | None) -> str:
    """Hiện tên/thời lượng ngay khi chọn tệp, chỉ đọc header thay vì giải mã audio."""
    if audio_path is None or not str(audio_path).strip():
        return EMPTY_FILE_INFO
    try:
        path = validate_audio_input(audio_path)
    except (FileNotFoundError, ValueError) as exc:
        return _alert_html(f"Không thể xử lý: {exc}")
    try:
        info = sf.info(str(path))
    except Exception:  # noqa: BLE001 - codec chỉ có ở bước giải mã đầy đủ (vd. M4A).
        return _file_meta_html(path.name, None, None)
    return _file_meta_html(path.name, float(info.duration), int(info.samplerate))


@lru_cache(maxsize=2)
def _get_transcriber(model_choice: str, config_path: str):
    return load_transcriber(model_choice, load_config(config_path))


def _available_models(config: dict[str, Any]) -> tuple[list[tuple[str, str]], str]:
    choices = [("Wav2Vec2 — Vietnamese 250h", "wav2vec2_vietnamese")]
    checkpoint = Path(config["paths"]["best_checkpoint"])
    if checkpoint.is_dir() and (checkpoint / "config.json").is_file():
        model_config = config.get("model", {})
        display_name = str(
            model_config.get(
                "display_name", "PhoWhisper Base — VIVOS + Common Voice VI 21"
            )
        )
        choices.append(
            (display_name, "whisper_fine_tuned")
        )
        status = str(
            model_config.get(
                "adaptation_disclosure",
                "PhoWhisper Base fine-tuned đã sẵn sàng; có thể so sánh đủ hai model.",
            )
        )
    else:
        status = (
            "Chưa có checkpoint Whisper fine-tuned hợp lệ. Demo hiện chỉ có Wav2Vec2; "
            "chạy scripts/train_model.py để tạo model thứ hai."
        )
    return choices, status


def _transcribe_in_chunks(
    transcriber,
    audio,
    *,
    sample_rate: int,
    chunk_seconds: float,
    transcribe_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Suy luận audio dài theo từng cửa sổ và ghép kết quả theo thứ tự."""
    if sample_rate <= 0:
        raise ValueError("Sample rate phải là số dương.")
    if not 0 < chunk_seconds <= 30:
        raise ValueError("demo.chunk_seconds phải lớn hơn 0 và không quá 30 giây.")
    if len(audio) == 0:
        raise ValueError("Audio rỗng.")

    chunk_size = max(1, int(round(sample_rate * chunk_seconds)))
    outputs = []
    chunk_count = 0
    silent_chunk_count = 0
    for start in range(0, len(audio), chunk_size):
        chunk_count += 1
        chunk = audio[start : start + chunk_size]
        try:
            outputs.append(
                transcriber.transcribe_array(
                    chunk,
                    sample_rate=sample_rate,
                    max_audio_seconds=chunk_seconds,
                    **transcribe_kwargs,
                )
            )
        except RuntimeError as exc:
            if "không sinh được transcript" not in str(exc).casefold():
                raise
            silent_chunk_count += 1
    if not outputs:
        raise RuntimeError("Mô hình không sinh được transcript cho audio này.")

    durations = [float(output["duration_seconds"]) for output in outputs]
    confidences = [output.get("confidence") for output in outputs]
    weighted_confidence = None
    confidence_weight = sum(
        duration
        for duration, confidence in zip(durations, confidences)
        if confidence is not None
    )
    if confidence_weight > 0:
        weighted_confidence = sum(
            float(confidence) * duration
            for duration, confidence in zip(durations, confidences)
            if confidence is not None
        ) / confidence_weight

    combined = dict(outputs[0])
    combined.update(
        {
            "transcript": normalize_transcript(
                " ".join(output["transcript"] for output in outputs)
            ),
            "raw_transcript": normalize_transcript(
                " ".join(output["raw_transcript"] for output in outputs),
                reject_empty=False,
            ),
            "text_normalization": [
                replacement
                for output in outputs
                for replacement in output.get("text_normalization", [])
            ],
            "duration_seconds": len(audio) / sample_rate,
            "inference_seconds": sum(
                float(output["inference_seconds"]) for output in outputs
            ),
            "confidence": weighted_confidence,
            "chunk_count": chunk_count,
            "silent_chunk_count": silent_chunk_count,
            "chunk_seconds": chunk_seconds,
        }
    )
    return combined


def transcribe_for_ui(
    audio_path: str | Path | None,
    model_choice: str,
    reference_text: str,
    config_path: str = "config.yaml",
) -> tuple[str, str, str, str]:
    """Handler độc lập UI; trả transcript, file info, timing và metric tham chiếu."""
    try:
        path = validate_audio_input(audio_path)
        config = load_config(config_path)
        audio, sample_rate = load_audio(path)
        duration = len(audio) / sample_rate
        chunk_seconds = float(config.get("demo", {}).get("chunk_seconds", 28.0))
        transcriber = _get_transcriber(model_choice, str(Path(config_path).resolve()))
        if isinstance(transcriber, Wav2Vec2Transcriber):
            transcribe_kwargs: dict[str, Any] = {}
        else:
            transcribe_kwargs = {
                "num_beams": int(config["evaluation"]["generation_num_beams"]),
                "max_length": int(config["evaluation"].get("max_length", 225)),
                "no_repeat_ngram_size": int(
                    config["evaluation"].get("no_repeat_ngram_size", 3)
                ),
                "repetition_penalty": float(
                    config["evaluation"].get("repetition_penalty", 1.05)
                ),
            }
        output = _transcribe_in_chunks(
            transcriber,
            audio,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            transcribe_kwargs=transcribe_kwargs,
        )
        file_info = _file_meta_html(path.name, duration, sample_rate)
        timing, metric_text = format_performance_metrics(
            output,
            audio_duration=duration,
            reference_text=reference_text,
        )
        return output["transcript"], file_info, timing, metric_text
    except (FileNotFoundError, ValueError) as exc:
        # Giữ nguyên lưới thẻ chỉ số; lỗi chỉ hiện ở dòng thông tin tệp.
        return (
            "",
            _alert_html(f"Không thể xử lý: {exc}"),
            empty_timing_html("Chưa chạy suy luận."),
            empty_metrics_html(),
        )
    except Exception:  # noqa: BLE001 - UI boundary must never expose internals or crash.
        return (
            "",
            _alert_html(
                "Không thể xử lý: Lỗi nội bộ khi nhận dạng. Hãy xem log terminal và thử lại."
            ),
            empty_timing_html("Suy luận không hoàn tất."),
            empty_metrics_html(),
        )


def make_transcribe_handler(
    config_path: str,
    transcribe_fn: Callable[..., tuple[str, str, str, str]] = transcribe_for_ui,
) -> Callable[[str | Path | None, str, str], tuple[str, str, str, str]]:
    """Bind a local or cloud inference runner to one immutable config path."""

    def handler(
        audio_path: str | Path | None,
        model_choice: str,
        reference_text: str,
    ) -> tuple[str, str, str, str]:
        return transcribe_fn(
            audio_path,
            model_choice,
            reference_text,
            config_path=config_path,
        )

    return handler


CLAY_SHADOW = (
    "10px 10px 24px rgba(166, 112, 76, .16), -8px -8px 20px rgba(255, 255, 255, .95), "
    "inset 2px 2px 4px rgba(255, 255, 255, .8), inset -4px -4px 8px rgba(166, 112, 76, .08)"
)
CLAY_SHADOW_PRESSED = (
    "inset 4px 4px 10px rgba(120, 60, 20, .22), inset -3px -3px 8px rgba(255, 255, 255, .45)"
)
CLAY_SHADOW_INSET = (
    "inset 3px 3px 7px rgba(166, 112, 76, .12), inset -3px -3px 7px rgba(255, 255, 255, .9)"
)
CLAY_PAGE_BACKGROUND = (
    "radial-gradient(circle at 6% 10%, rgba(255, 159, 67, .22), transparent 30%), "
    "radial-gradient(circle at 94% 6%, rgba(251, 113, 133, .16), transparent 28%), "
    "radial-gradient(circle at 88% 72%, rgba(125, 211, 252, .18), transparent 32%), "
    "#FFF7EE"
)

# Gradio 6 follows the browser locale and has no built-in Vietnamese pack.
# Keep framework-owned Audio widget copy consistent with this Vietnamese interface.
VIETNAMESE_UI_JS = r"""
() => {
  const setText = (node, value) => {
    if (node && node.textContent !== value) node.textContent = value;
  };
  const setLabel = (node, value) => {
    if (node && node.getAttribute("aria-label") !== value) {
      node.setAttribute("aria-label", value);
    }
  };
  const localizeAudioWidget = () => {
    if (document.documentElement.lang !== "vi") {
      document.documentElement.lang = "vi";
    }

    const audio = document.querySelector("#asr-audio");
    if (!audio) return;

    const uploadText = audio.querySelector('[data-testid="upload-text"]');
    if (uploadText) {
      const textNodes = [...uploadText.childNodes].filter(
        (node) => node.nodeType === Node.TEXT_NODE && node.nodeValue.trim()
      );
      if (textNodes.length >= 2) {
        if (textNodes[0].nodeValue !== "Th\u1ea3 t\u1ec7p \u00e2m thanh v\u00e0o \u0111\u00e2y ") {
          textNodes[0].nodeValue = "Th\u1ea3 t\u1ec7p \u00e2m thanh v\u00e0o \u0111\u00e2y ";
        }
        const lastText = textNodes[textNodes.length - 1];
        if (lastText.nodeValue !== " Nh\u1ea5p \u0111\u1ec3 t\u1ea3i t\u1ec7p l\u00ean") {
          lastText.nodeValue = " Nh\u1ea5p \u0111\u1ec3 t\u1ea3i t\u1ec7p l\u00ean";
        }
      }
      setText(uploadText.querySelector(".or"), "\u2013 ho\u1eb7c \u2013");
    }

    setLabel(audio.querySelector("button.upload-container"), "T\u1ea3i t\u1ec7p \u00e2m thanh l\u00ean");
    setLabel(audio.querySelector('[data-testid="file-upload"]'), "Ch\u1ecdn t\u1ec7p \u00e2m thanh");

    const sourceButtons = audio.querySelectorAll('[data-testid="source-select"] button');
    setLabel(sourceButtons[0], "T\u1ea3i t\u1ec7p l\u00ean");
    setLabel(sourceButtons[1], "Ghi \u00e2m");

    if (audio.querySelector('[data-testid="microphone-waveform"]')) {
      setText(audio.querySelector("button.record-button"), "B\u1eaft \u0111\u1ea7u ghi \u00e2m");
      setText(audio.querySelector("button.stop-button"), "D\u1eebng ghi \u00e2m");
      setText(audio.querySelector("button.stop-button-paused"), "D\u1eebng ghi \u00e2m");
      setText(audio.querySelector("button.resume-button"), "Ti\u1ebfp t\u1ee5c ghi \u00e2m");
      setLabel(audio.querySelector("button.pause-button"), "T\u1ea1m d\u1eebng ghi \u00e2m");

      const microphoneSelect = audio.querySelector("select.mic-select");
      setLabel(microphoneSelect, "Ch\u1ecdn thi\u1ebft b\u1ecb \u0111\u1ea7u v\u00e0o");
      if (microphoneSelect?.disabled) {
        setText(microphoneSelect.querySelector('option[value=""]'), "Kh\u00f4ng t\u00ecm th\u1ea5y micr\u00f4");
      }

      const clearRecording = audio.querySelector(
        ".icon-button-wrapper.top-panel button.icon-button"
      );
      setLabel(clearRecording, "X\u00f3a b\u1ea3n ghi");
      if (clearRecording && clearRecording.title !== "X\u00f3a b\u1ea3n ghi") {
        clearRecording.title = "X\u00f3a b\u1ea3n ghi";
      }
    }

    document.querySelectorAll('button[aria-label="Copy"]').forEach(
      (button) => setLabel(button, "Sao ch\u00e9p")
    );
  };

  let pending = false;
  const scheduleLocalization = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      localizeAudioWidget();
    });
  };

  const observer = new MutationObserver(scheduleLocalization);
  observer.observe(document.documentElement, {
    subtree: true,
    childList: true,
    characterData: true,
    attributes: true,
    attributeFilter: ["aria-label"],
  });

  // Gradio rebuilds the Audio subtree after source and recording actions.
  // Run a second pass after each asynchronous render so framework-owned
  // recording labels cannot fall back to the browser locale.
  document.addEventListener("click", (event) => {
    if (event.target.closest("#asr-audio button")) {
      scheduleLocalization();
      setTimeout(localizeAudioWidget, 100);
    }
  }, true);
  localizeAudioWidget();
}
"""

CLAY_CSS = """
:root {
  color-scheme: light;
  --clay-bg: #FFF7EE;
  --clay-surface: #FFFDF9;
  --clay-tray: #FFFBF5;
  --clay-edge: rgba(166, 112, 76, .14);
  --clay-ink: #2D2433;
  --clay-ink-muted: #6B5E57;
  --clay-orange: #FF9F43;
  --clay-orange-deep: #FF7A59;
  --clay-on-orange: #3B1D0B;
  --clay-orange-ink: #9A3412;
  --clay-orange-tint: #FFE9D6;
  --clay-coral: #E8505B;
  --clay-sky-tint: #E0F4FF;
  --clay-sky-ink: #075985;
  --clay-lilac-tint: #F1ECFF;
  --clay-lilac-ink: #4C1D95;
  --clay-mint-tint: #E3FBF1;
  --clay-mint-ink: #065F46;
  --clay-rose-tint: #FFE8EC;
  --clay-rose-ink: #9F1239;
  --clay-butter-tint: #FFF6D6;
  --clay-butter-ink: #78350F;
  --clay-focus: #7C2D12;
  --clay-radius-xl: 28px;
  --clay-radius-lg: 22px;
  --clay-radius-md: 16px;
  --clay-shadow: SHADOW;
  --clay-shadow-soft: 6px 6px 14px rgba(166, 112, 76, .12), -5px -5px 12px rgba(255, 255, 255, .9),
    inset 1px 1px 3px rgba(255, 255, 255, .8);
  --clay-shadow-pressed: PRESSED;
  --clay-shadow-inset: INSET;
  --clay-heading: "Nunito", "Be Vietnam Pro", system-ui, sans-serif;
}
.gradio-container {
  width: min(100%, 1200px) !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
  padding: 0 !important;
  overflow-x: clip;
}
/* Gradio gắn tiền tố ".gradio-container .contain" cho selector trong @media, nên
   lề responsive phải đặt trên #asr-root; .main chỉ giữ gutter tối thiểu 16px. */
.main.fillable { padding: 8px 16px 48px !important; }
@media (min-width: 701px) { #asr-root { padding-inline: 8px !important; } }
footer { display: none !important; }
#asr-root { gap: 24px !important; color: var(--clay-ink); }
#asr-root .html-container { padding: 0 !important; }
/* Lớp loading ẩn của Gradio vẫn tô nền lên mép chữ; chỉ hiện khi đang chạy. */
#asr-root .wrap.hide { opacity: 0 !important; pointer-events: none; }
/* overflow:auto của khối Markdown không viền cắt mất nét đầu dòng ở tọa độ lẻ pixel. */
#asr-root .block.hide-container { overflow: visible !important; }
#asr-root h1, #asr-root h2, #asr-root .metric-value, #asr-root .hero-stat strong {
  font-family: var(--clay-heading);
  color: var(--clay-ink);
}
#asr-root p { color: inherit; }
#asr-root .sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
}

/* Hero */
#asr-root .hero {
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(0, .9fr);
  align-items: center;
  gap: 40px;
  padding: 36px 4px 12px;
}
#asr-root .hero-topline { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 22px; }
#asr-root .eyebrow-chip, #asr-root .status-pill {
  display: inline-flex; align-items: center; gap: 8px;
  min-height: 34px; margin: 0; padding: 6px 14px;
  border: 1.5px solid var(--clay-edge); border-radius: 9999px;
  font-size: .875rem; font-weight: 700; line-height: 1.3; white-space: nowrap;
  box-shadow: var(--clay-shadow-soft);
}
#asr-root .eyebrow-chip { background: var(--clay-orange-tint); color: var(--clay-orange-ink); }
#asr-root .status-pill { background: var(--clay-surface); color: var(--clay-ink); }
#asr-root .status-pill-sky { background: var(--clay-sky-tint); color: var(--clay-sky-ink); }
#asr-root .chip-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; flex: 0 0 auto; }
#asr-root .hero h1 {
  margin: 0 0 16px;
  font-size: clamp(2.5rem, 5vw, 3.25rem);
  font-weight: 800;
  line-height: 1.08;
  letter-spacing: -.02em;
  text-wrap: balance;
}
#asr-root .hero h1 .accent { color: var(--clay-coral); }
#asr-root .hero-subtitle {
  max-width: 560px; margin: 0;
  color: var(--clay-ink-muted); font-size: 1.0625rem; line-height: 1.65; text-wrap: pretty;
}
#asr-root .hero-actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 26px; }
#asr-root .clay-link {
  display: inline-flex; align-items: center; justify-content: center;
  min-height: 48px; padding: 12px 22px;
  border-radius: var(--clay-radius-md);
  font-size: 1rem; font-weight: 700; text-decoration: none;
  transition: transform 160ms ease, box-shadow 160ms ease;
  touch-action: manipulation;
}
#asr-root .clay-link-primary {
  background: linear-gradient(135deg, var(--clay-orange), var(--clay-orange-deep));
  color: var(--clay-on-orange); box-shadow: var(--clay-shadow);
}
#asr-root .clay-link-ghost {
  border: 2px solid var(--clay-orange); background: var(--clay-surface);
  color: var(--clay-orange-ink); box-shadow: var(--clay-shadow-soft);
}
#asr-root .clay-link:hover { transform: translateY(-1px); }
#asr-root .clay-link:active { transform: translateY(1px); box-shadow: var(--clay-shadow-pressed); }
#asr-root .hero-stats { display: flex; flex-wrap: wrap; gap: 28px; margin: 28px 0 0; padding: 0; }
#asr-root .hero-stat { display: flex; flex-direction: column; gap: 2px; }
#asr-root .hero-stat strong { font-size: 1.75rem; font-weight: 800; line-height: 1.1; font-variant-numeric: tabular-nums; }
#asr-root .hero-stat:nth-child(1) strong { color: var(--clay-orange-ink); }
#asr-root .hero-stat:nth-child(2) strong { color: var(--clay-sky-ink); }
#asr-root .hero-stat:nth-child(3) strong { color: var(--clay-lilac-ink); }
#asr-root .hero-stat span { color: var(--clay-ink-muted); font-size: .875rem; }
#asr-root .hero-visual {
  position: relative;
  padding: 44px 28px;
  border: 1.5px solid var(--clay-edge);
  border-radius: var(--clay-radius-xl);
  background: linear-gradient(145deg, #FFE3CC 0%, #FDE2EA 52%, #E4EEFF 100%);
  box-shadow: var(--clay-shadow);
}
#asr-root .hero-visual svg { display: block; width: 100%; height: auto; }
#asr-root .float-chip {
  position: absolute;
  display: inline-flex; align-items: center; gap: 8px;
  padding: 8px 14px;
  border: 1.5px solid var(--clay-edge); border-radius: var(--clay-radius-md);
  background: var(--clay-surface); color: var(--clay-ink);
  font-size: .875rem; font-weight: 700; line-height: 1.25;
  box-shadow: var(--clay-shadow-soft);
}
#asr-root .float-chip small { display: block; color: var(--clay-ink-muted); font-size: .8125rem; font-weight: 500; }
#asr-root .float-chip-icon { width: 28px; height: 28px; border-radius: 9px; flex: 0 0 auto; }
#asr-root .float-chip-top { top: -16px; left: -14px; }
#asr-root .float-chip-top .float-chip-icon { background: linear-gradient(135deg, #38BDF8, #7DD3FC); }
#asr-root .float-chip-bottom { right: -14px; bottom: -16px; }
#asr-root .float-chip-bottom .float-chip-icon { background: linear-gradient(135deg, #FB7185, #FDA4AF); }

/* Section cards */
#asr-root .clay-card {
  min-width: 0 !important;
  gap: 16px !important;
  padding: 26px !important;
  border: 1.5px solid var(--clay-edge) !important;
  border-radius: var(--clay-radius-xl) !important;
  background: var(--clay-surface) !important;
  box-shadow: var(--clay-shadow) !important;
  overflow-wrap: anywhere;
}
#asr-root .section-heading { display: flex; align-items: center; gap: 14px; margin: 0 0 4px; }
#asr-root .step-badge {
  display: inline-grid; place-items: center;
  width: 42px; height: 42px; flex: 0 0 auto;
  border-radius: 14px;
  font-family: var(--clay-heading); font-size: 1.25rem; font-weight: 800;
  box-shadow: var(--clay-shadow-soft);
}
#asr-root .step-orange { background: var(--clay-orange-tint); color: var(--clay-orange-ink); }
#asr-root .step-lilac { background: var(--clay-lilac-tint); color: var(--clay-lilac-ink); }
#asr-root .step-sky { background: var(--clay-sky-tint); color: var(--clay-sky-ink); }
#asr-root .step-butter { background: var(--clay-butter-tint); color: var(--clay-butter-ink); }
#asr-root .section-heading h2 { margin: 0; font-size: 1.25rem; font-weight: 800; line-height: 1.3; }
#asr-root .section-heading p { margin: 2px 0 0; color: var(--clay-ink-muted); font-size: .875rem; line-height: 1.5; }

/* Input column */
#asr-input { gap: 24px !important; }
#asr-root .file-meta, #asr-root .inline-alert {
  margin: 0; padding: 10px 14px;
  border-radius: 14px;
  font-size: .9375rem; line-height: 1.5;
}
#asr-root .file-meta { background: var(--clay-tray); color: var(--clay-ink-muted); box-shadow: var(--clay-shadow-inset); }
#asr-root .file-meta strong { color: var(--clay-ink); }
#asr-root .inline-alert { background: var(--clay-rose-tint); color: var(--clay-rose-ink); font-weight: 600; }
#asr-root .limits-note, #asr-root .limits-note p {
  color: var(--clay-ink-muted) !important;
  font-size: .875rem !important;
  line-height: 1.55 !important;
}
#asr-root .controls-column { justify-content: flex-start; }
#asr-root .controls-column > * { flex-grow: 0 !important; }
#asr-root .clay-notice {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 14px 16px;
  border: 1.5px solid var(--clay-edge);
  border-radius: var(--clay-radius-lg);
  font-size: .875rem; line-height: 1.55;
}
#asr-root .clay-notice p { margin: 0; }
#asr-root .clay-notice svg { flex: 0 0 auto; margin-top: 1px; }
#asr-root .notice-butter { background: var(--clay-butter-tint); color: var(--clay-butter-ink); }
#asr-root .notice-sky { background: var(--clay-sky-tint); color: var(--clay-sky-ink); }
#asr-root .checkpoint-label { display: block; margin-bottom: 2px; font-size: .9375rem; font-weight: 800; }
#asr-root .primary-action {
  min-height: 56px !important;
  margin-top: auto;
  font-size: 1.0625rem !important;
  font-weight: 800 !important;
  transition: transform 160ms ease, box-shadow 160ms ease, opacity 160ms ease;
  touch-action: manipulation;
}
#asr-root .primary-action:hover { transform: translateY(-1px); }
#asr-root .primary-action:active { transform: translateY(1px); }
#asr-root .primary-action:disabled { opacity: .7; cursor: progress; transform: none; }

/* Output */
#asr-root .result-box textarea {
  min-height: 140px !important;
  color: var(--clay-ink) !important;
  font-size: 1.0625rem !important;
  line-height: 1.7 !important;
}

/* Metrics */
#asr-root .metric-grid { flex-wrap: wrap !important; align-items: stretch !important; gap: 16px !important; }
#asr-root .metric-pair { min-width: min(100%, 420px) !important; }
#asr-root .metric-pair-content {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  height: 100%;
}
#asr-root .metric-card {
  display: flex; flex-direction: column; gap: 6px;
  min-width: 0; min-height: 150px;
  padding: 18px;
  border: 1.5px solid var(--clay-edge);
  border-radius: var(--clay-radius-lg);
  background: var(--clay-surface);
  color: var(--clay-ink);
  box-shadow: var(--clay-shadow-soft);
}
#asr-root .metric-card p { margin: 0 !important; }
#asr-root .metric-label { color: inherit; font-size: .9375rem; font-weight: 700; line-height: 1.35; }
#asr-root .metric-value {
  margin-top: 4px !important;
  font-size: clamp(1.6rem, 3vw, 1.9rem);
  font-weight: 800; line-height: 1.15;
  color: inherit !important;
  font-variant-numeric: tabular-nums;
}
#asr-root .metric-detail { color: inherit; opacity: .85; font-size: .875rem; line-height: 1.45; font-variant-numeric: tabular-nums; }
#asr-root .metric-status {
  align-self: flex-start;
  margin-top: auto !important;
  padding: 3px 12px;
  border-radius: 9999px;
  background: rgba(255, 255, 255, .75);
  font-size: .875rem; font-weight: 800;
}
#asr-root .metric-sky { background: var(--clay-sky-tint); color: var(--clay-sky-ink); }
#asr-root .metric-lilac { background: var(--clay-lilac-tint); color: var(--clay-lilac-ink); }
#asr-root .metric-success { background: var(--clay-mint-tint); color: var(--clay-mint-ink); }
#asr-root .metric-danger { background: var(--clay-rose-tint); color: var(--clay-rose-ink); }
#asr-root .metric-muted { background: #FBF5EE; color: var(--clay-ink-muted); }
#asr-root .run-metadata, #asr-root .normalization-note, #asr-root .metric-summary {
  grid-column: 1 / -1;
  margin: 0 !important;
  color: var(--clay-ink-muted);
  font-size: .875rem; line-height: 1.55;
  overflow-wrap: anywhere;
}
#asr-root .normalization-note {
  padding: 10px 14px; border-radius: 14px;
  background: var(--clay-orange-tint); color: var(--clay-orange-ink);
}

/* Accessibility */
#asr-root button, #asr-root input, #asr-root textarea, #asr-root select { touch-action: manipulation; }
#asr-root [data-testid="source-select"] button,
#asr-root button[aria-label="Copy"],
#asr-root button[aria-label^="Sao"] {
  width: 44px !important;
  min-width: 44px !important;
  height: 44px !important;
  min-height: 44px !important;
  display: inline-flex !important;
  align-items: center;
  justify-content: center;
}
#asr-root a:focus-visible, #asr-root button:focus-visible, #asr-root input:focus-visible,
#asr-root textarea:focus-visible, #asr-root select:focus-visible {
  outline: 3px solid var(--clay-focus) !important;
  outline-offset: 3px;
}
#asr-root .skip-link {
  position: absolute; left: -9999px; top: 8px;
  padding: 10px 14px; border-radius: 12px;
  background: var(--clay-surface); color: var(--clay-ink); z-index: 1000;
}
#asr-root .skip-link:focus-visible { left: 8px; }

@media (max-width: 860px) {
  #asr-root .hero { grid-template-columns: 1fr; gap: 32px; }
  #asr-root .hero-visual { max-width: 460px; }
}
@media (max-width: 700px) {
  #asr-root {
    gap: 18px !important;
    --clay-shadow: 5px 5px 12px rgba(166, 112, 76, .14), -4px -4px 10px rgba(255, 255, 255, .9),
      inset 1px 1px 3px rgba(255, 255, 255, .8);
    --clay-shadow-soft: 3px 3px 8px rgba(166, 112, 76, .1), -3px -3px 7px rgba(255, 255, 255, .9);
  }
  #asr-root .hero { padding: 24px 0 4px; }
  #asr-root .hero h1 { font-size: clamp(2rem, 10vw, 2.5rem); }
  #asr-root .hero-visual { display: none; }
  #asr-root .hero-stats { gap: 20px; }
  #asr-root .clay-card { padding: 18px !important; border-radius: var(--clay-radius-lg) !important; }
  #asr-input { flex-direction: column !important; }
  #asr-root .metric-pair-content { grid-template-columns: 1fr; }
  #asr-root .metric-card { min-height: 0; }
  #asr-root .run-metadata, #asr-root .normalization-note, #asr-root .metric-summary { grid-column: 1; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
  }
  #asr-root .clay-link:hover, #asr-root .primary-action:hover { transform: none; }
}
""".replace("SHADOW", CLAY_SHADOW).replace("PRESSED", CLAY_SHADOW_PRESSED).replace(
    "INSET", CLAY_SHADOW_INSET
)

# Thanh sóng âm trang trí trong hero; chiều cao cố định để render giống nhau mọi lần.
_WAVE_HEIGHTS = (28, 52, 86, 64, 112, 74, 128, 96, 58, 104, 136, 82, 118, 66, 92, 44, 76, 36)
_WAVE_COLORS = ("#FF9F43", "#FB7185", "#38BDF8", "#A78BFA")


def build_theme():
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue="orange",
        secondary_hue="rose",
        neutral_hue="stone",
        font=[
            gr.themes.GoogleFont("Be Vietnam Pro", weights=(400, 500, 600, 700)),
            gr.themes.GoogleFont("Nunito", weights=(700, 800)),
            "system-ui",
            "sans-serif",
        ],
    ).set(
        body_background_fill=CLAY_PAGE_BACKGROUND,
        body_text_color="#2D2433",
        body_text_color_subdued="#6B5E57",
        block_background_fill="#FFFBF5",
        block_border_color="rgba(166, 112, 76, .14)",
        block_border_width="1.5px",
        block_radius="22px",
        block_shadow=CLAY_SHADOW_INSET,
        block_label_background_fill="#FFE9D6",
        block_label_text_color="#9A3412",
        block_label_border_width="0px",
        block_label_radius="12px",
        block_title_text_color="#6B5E57",
        input_background_fill="#FFFFFF",
        input_background_fill_focus="#FFFFFF",
        input_border_color="rgba(166, 112, 76, .18)",
        input_border_color_focus="#FF9F43",
        input_radius="16px",
        input_shadow="none",
        button_large_radius="16px",
        button_medium_radius="14px",
        button_primary_background_fill="linear-gradient(135deg, #FF9F43, #FF7A59)",
        button_primary_background_fill_hover="linear-gradient(135deg, #FFAD5C, #FF8A6B)",
        button_primary_border_color="transparent",
        button_primary_border_color_hover="transparent",
        button_primary_text_color="#3B1D0B",
        button_primary_text_color_hover="#3B1D0B",
        button_primary_shadow=CLAY_SHADOW,
        button_primary_shadow_hover=CLAY_SHADOW,
        button_primary_shadow_active=CLAY_SHADOW_PRESSED,
        color_accent="#FF9F43",
        color_accent_soft="#FFE9D6",
    )
    # Claymorphism trên nền tối khó giữ tương phản, nên chế độ tối dùng lại token sáng.
    for name in list(vars(theme)):
        if name.endswith("_dark") and hasattr(theme, name[: -len("_dark")]):
            setattr(theme, name, getattr(theme, name[: -len("_dark")]))
    return theme


def _section_heading(number: int, title: str, description: str, tone: str) -> str:
    return (
        '<div class="section-heading">'
        f'<span class="step-badge step-{tone}" aria-hidden="true">{number}</span>'
        f'<div><h2><span class="sr-only">{number}. </span>{html.escape(title)}</h2>'
        f"<p>{description}</p></div>"
        "</div>"
    )


def _hero_html(
    model_choices: list[tuple[str, str]],
    *,
    runtime: Runtime,
    chunk_seconds: float,
) -> str:
    bars = "".join(
        f'<rect x="{8 + index * 17}" y="{80 - height / 2:g}" width="10" '
        f'height="{height}" rx="5" fill="{_WAVE_COLORS[index % len(_WAVE_COLORS)]}"/>'
        for index, height in enumerate(_WAVE_HEIGHTS)
    )
    model_values = {value for _, value in model_choices}
    seq2seq_chip = (
        '<span class="float-chip float-chip-bottom">'
        '<span class="float-chip-icon"></span>'
        "<span>PhoWhisper<small>Seq2Seq</small></span></span>"
        if "whisper_fine_tuned" in model_values
        else ""
    )
    model_count = len(model_choices)
    return f"""
    <header class="hero">
      <a class="skip-link" href="#asr-input">Bỏ qua đến nội dung chính</a>
      <div class="hero-copy">
        <div class="hero-topline">
          <p class="eyebrow-chip"><span class="chip-dot" aria-hidden="true"></span>Vietnamese ASR Benchmark</p>
          <span class="status-pill">{html.escape(RUNTIME_COPY[runtime]["badge"])}</span>
          <span class="status-pill status-pill-sky">{model_count} model khả dụng</span>
        </div>
        <h1>Vietnamese AI <span class="accent">Speech Recognition</span></h1>
        <p class="hero-subtitle">Nhận dạng tiếng nói tiếng Việt và so sánh Wav2Vec2 CTC với PhoWhisper Seq2Seq trên cùng một tệp âm thanh, kèm WER/CER khi có transcript tham chiếu.</p>
        <nav class="hero-actions" aria-label="Lối tắt">
          <a class="clay-link clay-link-primary" href="#asr-input">Bắt đầu nhận dạng</a>
          <a class="clay-link clay-link-ghost" href="#asr-metrics">Xem chỉ số</a>
        </nav>
        <div class="hero-stats">
          <div class="hero-stat"><strong>{model_count}</strong><span>model khả dụng</span></div>
          <div class="hero-stat"><strong>{chunk_seconds:g} s</strong><span>mỗi phân đoạn audio</span></div>
          <div class="hero-stat"><strong>{MAX_AUDIO_FILE_BYTES // (1024 * 1024)} MB</strong><span>dung lượng tệp tối đa</span></div>
        </div>
      </div>
      <div class="hero-visual" aria-hidden="true">
        <span class="float-chip float-chip-top"><span class="float-chip-icon"></span><span>Wav2Vec2<small>CTC</small></span></span>
        <svg viewBox="0 0 316 160" focusable="false">{bars}</svg>
        {seq2seq_chip}
      </div>
    </header>
    """


_NOTICE_ICON = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" aria-hidden="true">'
    '<circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16.5h.01"/></svg>'
)


def build_demo(
    config_path: str = "config.yaml",
    *,
    transcribe_fn: Callable[..., tuple[str, str, str, str]] = transcribe_for_ui,
    runtime: Runtime = "local",
):
    import gradio as gr

    if runtime not in RUNTIME_COPY:
        raise ValueError(f"runtime phải là một trong: {', '.join(RUNTIME_COPY)}.")
    config = load_config(config_path)
    model_choices, checkpoint_status = _available_models(config)
    chunk_seconds = float(config.get("demo", {}).get("chunk_seconds", 28))
    theme = build_theme()

    with gr.Blocks(title="Vietnamese AI Speech Recognition", fill_width=True) as demo:
        with gr.Column(elem_id="asr-root"):
            gr.HTML(_hero_html(model_choices, runtime=runtime, chunk_seconds=chunk_seconds))
            with gr.Row(equal_height=True, elem_id="asr-input"):
                with gr.Column(scale=7, elem_classes="clay-card"):
                    gr.HTML(
                        _section_heading(
                            1,
                            "Dữ liệu đầu vào",
                            "Chọn audio từ máy hoặc ghi âm trực tiếp.",
                            "orange",
                        )
                    )
                    audio = gr.Audio(
                        label="Tệp hoặc bản ghi âm",
                        sources=["upload", "microphone"],
                        type="filepath",
                        elem_id="asr-audio",
                    )
                    file_info = gr.Markdown(EMPTY_FILE_INFO)
                    gr.Markdown(
                        "Tệp hỗ trợ: WAV, FLAC, OGG, MP3, M4A (tùy codec hệ thống). "
                        "**Không giới hạn thời lượng** trong demo; audio dài được chia thành "
                        f"các đoạn tối đa {chunk_seconds:g} giây. "
                        "Kích thước tệp tối đa: 100 MB.",
                        elem_classes="limits-note",
                    )
                    reference = gr.Textbox(
                        label="Transcript tham chiếu (không bắt buộc)",
                        placeholder="Nhập nội dung đúng nếu muốn tính WER/CER cho mẫu này…",
                        lines=2,
                    )
                with gr.Column(scale=5, elem_classes=["clay-card", "controls-column"]):
                    gr.HTML(
                        _section_heading(
                            2,
                            "Cấu hình & thực thi",
                            "Chọn mô hình rồi chạy nhận dạng.",
                            "lilac",
                        )
                    )
                    model = gr.Dropdown(
                        choices=model_choices,
                        value="wav2vec2_vietnamese",
                        label="Mô hình nhận dạng",
                        interactive=len(model_choices) > 1,
                    )
                    gr.HTML(
                        f"""
                        <div class="clay-notice notice-butter" role="note">
                          {_NOTICE_ICON}
                          <div>
                            <span class="checkpoint-label">Lưu ý đánh giá</span>
                            <p>{html.escape(checkpoint_status)}</p>
                          </div>
                        </div>
                        """
                    )
                    recognize = gr.Button(
                        RECOGNIZE_LABEL,
                        variant="primary",
                        size="lg",
                        elem_classes="primary-action",
                    )
            with gr.Column(elem_id="asr-output", elem_classes="clay-card"):
                gr.HTML(
                    _section_heading(
                        3,
                        "Văn bản nhận dạng",
                        "Kết quả được ghép theo đúng thứ tự các phân đoạn audio.",
                        "sky",
                    )
                )
                transcript = gr.Textbox(
                    label="Văn bản nhận dạng",
                    placeholder="Kết quả sẽ hiển thị tại đây sau khi nhận dạng.",
                    lines=5,
                    buttons=["copy"],
                    interactive=False,
                    elem_classes="result-box",
                )
            with gr.Column(elem_id="asr-metrics", elem_classes="clay-card"):
                gr.HTML(
                    _section_heading(
                        4,
                        "Bảng chỉ số kỹ thuật",
                        "Số liệu hiệu năng và sai số trên mẫu hiện tại.",
                        "butter",
                    )
                )
                with gr.Row(equal_height=True, elem_classes="metric-grid"):
                    timing = gr.Markdown(empty_timing_html(), elem_classes="metric-pair")
                    metrics = gr.Markdown(empty_metrics_html(), elem_classes="metric-pair")
            gr.HTML(
                f"""
                <aside class="clay-notice notice-sky" aria-label="Lưu ý kỹ thuật">
                  {_NOTICE_ICON}
                  <p><strong>Lưu ý kỹ thuật.</strong> Độ tin cậy hiển thị là proxy chưa hiệu chuẩn để tham khảo, không phải xác suất transcript đúng. {html.escape(RUNTIME_COPY[runtime]["notice"])}</p>
                </aside>
                """
            )

        audio.change(
            fn=describe_audio_for_ui,
            inputs=audio,
            outputs=file_info,
            queue=False,
            show_progress="hidden",
        )
        recognize.click(
            fn=lambda: gr.Button(value=RECOGNIZING_LABEL, interactive=False),
            outputs=recognize,
            queue=False,
            show_progress="hidden",
        ).then(
            fn=make_transcribe_handler(config_path, transcribe_fn),
            inputs=[audio, model, reference],
            outputs=[transcript, file_info, timing, metrics],
            show_progress="full",
            show_progress_on=[transcript],
        ).then(
            # .then chạy cả khi bước trước lỗi, nên nút luôn được mở lại.
            fn=lambda: gr.Button(value=RECOGNIZE_LABEL, interactive=True),
            outputs=recognize,
            queue=False,
            show_progress="hidden",
        )
    demo._asr_theme = theme
    demo._asr_css = CLAY_CSS
    demo._asr_js = VIETNAMESE_UI_JS
    return demo


def launch_demo(
    config_path: str = "config.yaml",
    *,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
):
    demo = build_demo(config_path)
    return demo.queue(default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        max_file_size="100mb",
        theme=demo._asr_theme,
        css=demo._asr_css,
        js=demo._asr_js,
    )


def main() -> None:
    configure_utf8_stdio()
    launch_demo()


if __name__ == "__main__":
    main()
