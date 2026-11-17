"""Đánh giá một checkpoint PhoWhisper trên VIVOS, Common Voice và mẫu ST23D.

Dùng cùng pipeline suy luận với demo (chuẩn hóa mã miền + quy tắc cụm từ trong
``--config``), chỉ thay checkpoint. Prediction được lưu theo từng split để
``scripts/review_promotions.py`` có thể so sánh ghép cặp.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import configure_utf8_stdio
from src.config import load_config
from src.evaluate import run_model_evaluation
from src.metrics import calculate_cer, calculate_wer


def _config_for(base: dict, checkpoint: Path, processed: Path, output: Path) -> dict:
    config = copy.deepcopy(base)
    config["paths"].update(
        {
            "best_checkpoint": str(checkpoint),
            "processed_data": str(processed),
            "metrics_dir": str(output / "tables"),
            "predictions_dir": str(output / "predictions"),
            "figures_dir": str(output / "figures"),
            "reports_dir": str(output / "reports"),
        }
    )
    config["data"]["max_test_samples"] = None
    return config


def evaluate_st23d(config: dict, audio: Path, reference: str, output: Path) -> dict:
    from src.inference import load_transcriber

    generation = config.get("evaluation", {})
    transcriber = load_transcriber("whisper_fine_tuned", config)
    result = transcriber.transcribe_file(
        audio,
        num_beams=int(generation.get("generation_num_beams", 1)),
        max_audio_seconds=float(config["data"]["max_audio_seconds"]),
        max_length=int(generation.get("max_length", 225)),
        no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 3)),
        repetition_penalty=float(generation.get("repetition_penalty", 1.05)),
    )
    metrics = {
        "reference": reference,
        "raw_prediction": result["raw_transcript"],
        "prediction": result["transcript"],
        "text_normalization": result.get("text_normalization", []),
        "wer": calculate_wer(reference, result["transcript"]),
        "cer": calculate_cer(reference, result["transcript"]),
        "model_source": transcriber.model_source,
        "trained_on_sample": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "st23d_sample_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--processed-data", default="data/processed/vivos_cv21")
    parser.add_argument("--splits", nargs="+", default=["test", "common_voice_test"])
    parser.add_argument("--st23d-audio", default="data/evaluation/st23d_sample.wav")
    parser.add_argument(
        "--st23d-reference", default="data/evaluation/st23d_sample_reference.txt"
    )
    parser.add_argument("--skip-st23d", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    output = Path(args.output_dir)
    summary: dict[str, dict] = {}
    if not args.skip_st23d:
        reference = Path(args.st23d_reference).read_text(encoding="utf-8").strip()
        config = _config_for(base, checkpoint, Path(args.processed_data), output / "st23d")
        st23d = evaluate_st23d(config, Path(args.st23d_audio), reference, output / "st23d")
        summary["st23d"] = {key: st23d[key] for key in ("prediction", "wer", "cer")}
        print(f"st23d: WER={st23d['wer']:.4f} → {st23d['prediction']}", flush=True)
    for split in args.splits:
        config = _config_for(base, checkpoint, Path(args.processed_data), output / split)
        metrics = run_model_evaluation(config, model_choice="whisper_fine_tuned", split=split)["metrics"]
        summary[split] = {
            key: metrics[key] for key in ("num_samples", "wer", "cer", "ser", "wer_ci95")
        }
        print(f"{split}: n={metrics['num_samples']} WER={metrics['wer']:.4f}", flush=True)
    (output / "summary.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), **summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
