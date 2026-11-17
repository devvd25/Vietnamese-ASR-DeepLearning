"""Hugging Face ZeroGPU entry point for the Vietnamese ASR demo."""

from __future__ import annotations

from pathlib import Path

import spaces

from app.gradio_app import (
    _get_transcriber,
    build_demo,
    transcribe_for_ui,
)
from src.space_runtime import ensure_language_model


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LANGUAGE_MODEL_PATH = (
    PROJECT_ROOT / "models/language_models/vietnamese_4gram/vi_lm_4grams.bin"
)

# ZeroGPU supports model initialization at module scope.  Moving both cached
# models here lets the platform provision them with the GPU only for decorated
# inference calls while avoiding a reload for every request.
ensure_language_model(LANGUAGE_MODEL_PATH)
for _model_choice in ("wav2vec2_vietnamese", "whisper_fine_tuned"):
    _transcriber = _get_transcriber(_model_choice, str(CONFIG_PATH))
    _transcriber.model.to("cuda")
    _transcriber.device = "cuda"


@spaces.GPU(duration=120)
def zero_gpu_transcribe(
    audio_path,
    model_choice,
    reference_text,
    *,
    config_path,
):
    return transcribe_for_ui(
        audio_path,
        model_choice,
        reference_text,
        config_path=config_path,
    )


demo = build_demo(
    str(CONFIG_PATH), transcribe_fn=zero_gpu_transcribe, runtime="space"
)
demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(
        max_file_size="100mb",
        theme=demo._asr_theme,
        css=demo._asr_css,
        js=demo._asr_js,
    )
