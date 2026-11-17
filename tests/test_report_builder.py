from pathlib import Path

from docx import Document


def test_report_builder_uses_kaggle_adaptation_training_artifacts():
    from scripts.build_report import TRAINING_REPORTS_DIR

    assert TRAINING_REPORTS_DIR == Path("outputs/kaggle_cv21/reports")


def test_report_discloses_st23d_domain_normalization(tmp_path):
    from scripts.build_report import build_report

    output_path = tmp_path / "report.docx"
    build_report(output_path)
    text = "\n".join(paragraph.text for paragraph in Document(output_path).paragraphs)

    assert "ST23D" in text
    assert "Common Voice VI 21.0" in text
    assert "không dùng transcript tham chiếu để sửa đầu ra" in text


def test_report_records_fpt_experiment_and_non_promotion_decision(tmp_path):
    from scripts.build_report import build_report

    output_path = tmp_path / "report.docx"
    build_report(output_path)
    text = "\n".join(paragraph.text for paragraph in Document(output_path).paragraphs)

    assert "FPT Open Speech" in text
    assert "không thay checkpoint demo" in text
    assert "40.675" in text


def test_report_records_kaggle_speech_experiment_and_license_limit(tmp_path):
    from scripts.build_report import build_report

    output_path = tmp_path / "report.docx"
    build_report(output_path)
    text = "\n".join(paragraph.text for paragraph in Document(output_path).paragraphs)

    assert "Vietnamese Speech Recognition Data" in text
    assert "374" in text
    assert "license `unknown`" in text
    assert "Không promote" in text


def test_report_discloses_st23d_supervised_adaptation_and_training_fit(tmp_path):
    from scripts.build_report import build_report

    output_path = tmp_path / "report.docx"
    build_report(output_path)
    text = "\n".join(paragraph.text for paragraph in Document(output_path).paragraphs)

    assert "Supervised adaptation cho demo ST23D" in text
    assert "training fit" in text
    assert "checkpoint-16" in text
    assert "không phải test độc lập" in text
    assert "checkpoint bước 16 được dùng cho demo" in text
    assert "WER/CER 0 trên mẫu này là training fit" in text


def test_report_toc_uses_one_paragraph_per_section_and_lists_ablation_experiments():
    from scripts.build_report import _add_toc

    document = Document()
    _add_toc(document)
    entries = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]

    assert "13 Thí nghiệm bổ sung với FPT Open Speech" in entries
    assert "14 Thí nghiệm Kaggle Vietnamese Speech Recognition Data" in entries
    assert "15 Supervised adaptation cho demo ST23D" in entries
    assert "16 Hạn chế" in entries
    assert "17 Kết luận và hướng phát triển" in entries
    assert all("\n" not in entry for entry in entries)


def test_report_describes_unlimited_demo_duration_with_chunked_inference(
    tmp_path,
) -> None:
    from scripts.build_report import build_report

    output_path = tmp_path / "report.docx"
    build_report(output_path)
    text = "\n".join(paragraph.text for paragraph in Document(output_path).paragraphs)

    assert "không giới hạn tổng thời lượng" in text
    assert "cửa sổ 28 giây" in text
