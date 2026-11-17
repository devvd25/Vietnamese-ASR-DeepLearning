"""Tạo báo cáo DOCX từ các artifact thực nghiệm đã chạy thật."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_REPORTS_DIR = Path("outputs/kaggle_cv21/reports")
KAGGLE_OUTPUTS_DIR = Path("outputs/kaggle_cv21")
KAGGLE_SPEECH_OUTPUTS_DIR = Path("outputs/kaggle_speech")
ST23D_ADAPTATION_OUTPUTS_DIR = Path("outputs/st23d_demo_candidate")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import configure_utf8_stdio


def _load_json(relative_path: str) -> dict[str, Any] | list[dict[str, Any]]:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu artifact bắt buộc: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_predictions(
    model_choice: str, base_dir: Path = Path("outputs/predictions")
) -> list[dict[str, str]]:
    path = PROJECT_ROOT / base_dir / f"{model_choice}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu prediction: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt_rate(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}".replace(".", ",")


def _fmt_seconds(value: Any) -> str:
    return f"{float(value):.3f}".replace(".", ",") + " s"


def _set_run_font(run, size: float = 11, bold: bool | None = None) -> None:
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def _shade_cell(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def _repeat_table_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    properties.append(header)


def _add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instruction, separate, text, end):
        run._r.append(element)
    _set_run_font(run, 9)


def _add_toc(document: Document) -> None:
    entries = (
        "1 Giới thiệu",
        "2 Phát biểu bài toán và mục tiêu",
        "3 Tổng quan phương pháp",
        "4 Cơ sở lý thuyết",
        "5 Dữ liệu VIVOS và Common Voice VI 21",
        "6 Tiền xử lý dữ liệu",
        "7 Kiến trúc hai model",
        "8 Thiết lập fine-tuning",
        "9 Thiết kế thực nghiệm",
        "10 Kết quả",
        "11 Phân tích lỗi",
        "12 Demo Gradio",
        "13 Thí nghiệm bổ sung với FPT Open Speech",
        "14 Thí nghiệm Kaggle Vietnamese Speech Recognition Data",
        "15 Supervised adaptation cho demo ST23D",
        "16 Hạn chế",
        "17 Kết luận và hướng phát triển",
        "Tài liệu tham khảo",
    )
    for entry in entries:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.8)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(entry)
        _set_run_font(run, 10.5)


def _add_paragraph(document: Document, text: str, *, indent: bool = True):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.line_spacing = 1.15
    paragraph.paragraph_format.space_after = Pt(6)
    if indent:
        paragraph.paragraph_format.first_line_indent = Cm(1)
    run = paragraph.add_run(text)
    _set_run_font(run)
    return paragraph


def _add_equation(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(text)
    _set_run_font(run, 11)
    run.italic = True


def _add_table(document: Document, headers: list[str], rows: list[list[str]], caption: str) -> None:
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption_paragraph.add_run(caption)
    _set_run_font(caption_run, 10, bold=True)
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    header_row = table.rows[0]
    _repeat_table_header(header_row)
    for cell, value in zip(header_row.cells, headers):
        _shade_cell(cell, "D9EAF7")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(value)
        _set_run_font(run, 9, bold=True)
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(str(value))
            _set_run_font(run, 9)
    document.add_paragraph()


def _add_figure(document: Document, relative_path: str, caption: str) -> None:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture = paragraph.add_run().add_picture(str(path), width=Cm(15.5))
    picture._inline.docPr.set("descr", caption)
    picture._inline.docPr.set("title", caption)
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption_paragraph.add_run(caption)
    _set_run_font(caption_run, 9)
    caption_run.italic = True


def _add_reference(document: Document, index: int, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Cm(0.8)
    paragraph.paragraph_format.first_line_indent = Cm(-0.8)
    paragraph.paragraph_format.space_after = Pt(4)
    run = paragraph.add_run(f"[{index}] {text}")
    _set_run_font(run, 9)


def _configure_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.2)
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(11)
    for style_name, size in (("Title", 18), ("Heading 1", 14), ("Heading 2", 12)):
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header_run = header.add_run("Đồ án Deep Learning - STT 19")
    _set_run_font(header_run, 9)
    _add_page_number(section.footer.paragraphs[0])


def _best_validation_metrics(logs: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    rows = [row for row in logs if row.get("eval_wer") is not None]
    if not rows:
        return None, None
    best = min(rows, key=lambda row: float(row["eval_wer"]))
    return float(best["eval_wer"]), float(best["eval_cer"])


def _representative_errors(rows: list[dict[str, str]], count: int = 3) -> list[dict[str, str]]:
    errors = [row for row in rows if row.get("error_note") != "Khớp hoàn toàn"]
    errors.sort(key=lambda row: float(row.get("wer", 0.0)), reverse=True)
    selected: list[dict[str, str]] = []
    seen_categories: set[str] = set()
    for row in errors:
        category = row.get("error_note", "")
        if category in seen_categories and len(selected) < count - 1:
            continue
        selected.append(row)
        seen_categories.add(category)
        if len(selected) == count:
            break
    return selected


def build_report(output_path: Path) -> Path:
    dataset = _load_json(KAGGLE_OUTPUTS_DIR / "tables/dataset_statistics.json")
    wav = _load_json("outputs/tables/wav2vec2_vietnamese_metrics.json")
    whisper = _load_json(KAGGLE_OUTPUTS_DIR / "tables/adapted_vivos_test_metrics.json")
    cv_baseline = _load_json(
        KAGGLE_OUTPUTS_DIR
        / "tables/baseline_vivos_checkpoint_common_voice_test_metrics.json"
    )
    cv_adapted = _load_json(
        KAGGLE_OUTPUTS_DIR / "tables/adapted_common_voice_test_metrics.json"
    )
    fpt_experiment = _load_json(
        "outputs/kaggle_fpt/tables/fpt_experiment_comparison.json"
    )
    kaggle_speech_experiment = _load_json(
        KAGGLE_SPEECH_OUTPUTS_DIR
        / "tables/kaggle_speech_experiment_comparison.json"
    )
    st23d_adaptation = _load_json(
        ST23D_ADAPTATION_OUTPUTS_DIR
        / "tables/st23d_supervised_adaptation_comparison.json"
    )
    training = _load_json(TRAINING_REPORTS_DIR / "training_result.json")
    logs = _load_json(TRAINING_REPORTS_DIR / "trainer_log_history.json")
    assert isinstance(dataset, dict)
    assert isinstance(wav, dict)
    assert isinstance(whisper, dict)
    assert isinstance(cv_baseline, dict)
    assert isinstance(cv_adapted, dict)
    assert isinstance(fpt_experiment, dict)
    assert isinstance(kaggle_speech_experiment, dict)
    assert isinstance(st23d_adaptation, dict)
    assert isinstance(training, dict)
    assert isinstance(logs, list)
    wav_rows = _load_predictions("wav2vec2_vietnamese")
    whisper_rows = _load_predictions(
        "adapted_vivos_test", KAGGLE_OUTPUTS_DIR / "predictions"
    )
    if wav.get("num_samples") != whisper.get("num_samples"):
        raise ValueError("Hai model chưa được đánh giá trên cùng số mẫu test.")
    if int(wav["num_samples"]) != int(dataset["test"]["num_records"]):
        raise ValueError("Bảng kết quả chưa bao phủ toàn bộ test set VIVOS.")
    completed_epochs = float(training["train_metrics"].get("epoch", 0.0))

    document = Document()
    _configure_document(document)

    title = document.add_paragraph()
    title.style = document.styles["Title"]
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Cm(3)
    run = title.add_run("BÁO CÁO ĐỒ ÁN HỌC PHẦN DEEP LEARNING")
    _set_run_font(run, 16, bold=True)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_before = Cm(1.5)
    run = subtitle.add_run(
        "NHẬN DẠNG TIẾNG NÓI TIẾNG VIỆT TỰ ĐỘNG CHUYỂN THÀNH VĂN BẢN BẰNG DEEP LEARNING"
    )
    _set_run_font(run, 18, bold=True)
    english = document.add_paragraph()
    english.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = english.add_run("Automatic Speech Recognition for Vietnamese Language Using Deep Learning")
    _set_run_font(run, 12)
    run.italic = True
    code = document.add_paragraph()
    code.alignment = WD_ALIGN_PARAGRAPH.CENTER
    code.paragraph_format.space_before = Cm(2)
    run = code.add_run("Mã đề tài: STT 19")
    _set_run_font(run, 12, bold=True)
    models = document.add_paragraph()
    models.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = models.add_run(
        "Wav2Vec2-Vietnamese + LM 4-gram và PhoWhisper Base fine-tuned trên VIVOS + Common Voice VI 21.0"
    )
    _set_run_font(run, 11)
    document.add_page_break()

    document.add_heading("Tóm tắt", level=1)
    _add_paragraph(
        document,
        "Đồ án xây dựng pipeline nhận dạng tiếng nói tiếng Việt chạy cục bộ, gồm chuẩn bị "
        "dữ liệu, tiền xử lý audio, fine-tuning, đánh giá và demo. Wav2Vec2-Vietnamese "
        "250h dùng CTC beam search kết hợp LM 4-gram được so sánh với PhoWhisper Base được fine-tune trên toàn bộ "
        f"{int(training['train_samples']):,} mẫu hỗn hợp VIVOS và Common Voice VI 21.0. Trên {int(wav['num_samples'])} "
        f"mẫu test độc lập, Wav2Vec2 đạt WER {_fmt_rate(wav['wer'])} và CER "
        f"{_fmt_rate(wav['cer'])}; Whisper đạt WER {_fmt_rate(whisper['wer'])} và CER "
        f"{_fmt_rate(whisper['cer'])}. Trên 546 mẫu Common Voice test, bước thích nghi giảm WER "
        f"từ {_fmt_rate(cv_baseline['wer'])} xuống {_fmt_rate(cv_adapted['wer'])} và CER từ "
        f"{_fmt_rate(cv_baseline['cer'])} xuống {_fmt_rate(cv_adapted['cer'])}. Kết quả được tạo bằng inference thật và lưu kèm "
        "prediction từng mẫu để kiểm tra lại. FPT Open Speech và Vietnamese Speech Recognition Data "
        "được dùng trong các thí nghiệm bổ sung; các candidate này không được promote "
        "vì không cải thiện đồng nhất trên test độc lập. Riêng cho mẫu ST23D, "
        "PhoWhisper được supervised adaptation và checkpoint bước 16 được dùng cho demo; "
        "WER/CER 0 trên mẫu này là training fit, không phải test độc lập.",
    )
    keywords = document.add_paragraph()
    keywords.paragraph_format.space_after = Pt(10)
    run = keywords.add_run(
        "Từ khóa: nhận dạng tiếng nói, tiếng Việt, Wav2Vec2, CTC, Whisper, fine-tuning, WER, CER."
    )
    _set_run_font(run, 11)
    document.add_heading("Mục lục", level=1)
    _add_toc(document)
    document.add_page_break()

    document.add_heading("1 Giới thiệu", level=1)
    _add_paragraph(
        document,
        "Nhận dạng tiếng nói tự động chuyển tín hiệu âm thanh thành văn bản. Bài toán tiếng "
        "Việt đòi hỏi mô hình phân biệt hệ thống nguyên âm, phụ âm và thanh điệu; một lỗi dấu "
        "có thể làm thay đổi từ và nghĩa. Đồ án tập trung vào hai kiến trúc học sâu được giảng "
        "viên chỉ định: Wav2Vec2-Vietnamese và Whisper fine-tuned.",
    )
    _add_paragraph(
        document,
        "Sản phẩm không gọi API nhận dạng bên ngoài. Cùng một pipeline tiền xử lý và cùng test "
        "set được dùng cho hai model. Thiết kế này giúp truy vết kết quả từ audio, reference, "
        "prediction đến WER/CER, đồng thời tách rõ quick experiment khỏi thí nghiệm cuối.",
    )

    document.add_heading("2 Phát biểu bài toán và mục tiêu", level=1)
    _add_paragraph(
        document,
        "Đầu vào là một tệp chứa phát ngôn tiếng Việt. Demo không giới hạn tổng thời lượng; "
        "audio dài được suy luận tuần tự theo cửa sổ 28 giây rồi ghép transcript theo thứ tự. "
        "Mục tiêu gồm xây dựng pipeline dữ liệu tái lập, fine-tune model thực sự, "
        "so sánh hai kiến trúc bằng WER/CER, phân tích lỗi từng mẫu và cung cấp demo Gradio.",
    )

    document.add_heading("3 Tổng quan phương pháp", level=1)
    _add_paragraph(
        document,
        "Wav2Vec2 học biểu diễn từ waveform chưa gán nhãn bằng bài toán tương phản trong không "
        "gian tiềm ẩn, sau đó fine-tune có giám sát cho ASR [1]. Checkpoint tiếng Việt dùng trong "
        "đồ án được pretrain trên khoảng 13.000 giờ audio YouTube tiếng Việt và fine-tune trên "
        "250 giờ VLSP [2]. Whisper là Transformer encoder-decoder được huấn luyện bằng 680.000 "
        "giờ dữ liệu đa ngôn ngữ và đa nhiệm có giám sát yếu [3].",
    )
    _add_paragraph(
        document,
        "Hai hướng khác nhau ở cơ chế giải mã. Wav2Vec2 dùng CTC để căn chỉnh ngầm giữa frame "
        "âm thanh và ký hiệu đầu ra. Whisper sinh token tự hồi quy, nên có khả năng mô hình hóa "
        "quan hệ ngôn ngữ mạnh hơn nhưng suy luận tuần tự và nhạy với lỗi lặp token.",
    )

    document.add_heading("4 Cơ sở lý thuyết", level=1)
    document.add_heading("4.1 CTC", level=2)
    _add_paragraph(
        document,
        "CTC mở rộng bảng chữ cái với ký hiệu blank và cộng xác suất của mọi đường căn chỉnh "
        "có thể rút gọn thành transcript mục tiêu. Beam search giữ nhiều giả thuyết CTC và kết hợp "
        "điểm acoustic với LM 4-gram; sau đó gom token lặp liên tiếp và loại blank [2], [4].",
    )
    _add_equation(document, "P(y|x) = sum_{pi in B^-1(y)} product_t p(pi_t|x)")
    document.add_heading("4.2 Transformer encoder-decoder", level=2)
    _add_paragraph(
        document,
        "Whisper chuyển audio 16 kHz thành log-Mel spectrogram. Encoder tạo biểu diễn theo thời "
        "gian; decoder dùng self-attention và cross-attention để dự đoán token kế tiếp [3], [5].",
    )
    _add_equation(document, "Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V")
    document.add_heading("4.3 Chỉ số đánh giá", level=2)
    _add_paragraph(
        document,
        "WER và CER dùng khoảng cách Levenshtein. Với WER, S là số thay thế, D là số thiếu, I "
        "là số thừa và N là số từ trong reference. CER áp dụng cùng công thức trên ký tự.",
    )
    _add_equation(document, "WER = (S + D + I) / N")
    _add_equation(document, "CER = (Sc + Dc + Ic) / Nc")

    document.add_heading("5 Dữ liệu VIVOS và Common Voice VI 21", level=1)
    _add_paragraph(
        document,
        "VIVOS 1.0 là corpus tiếng Việt đọc khoảng 15 giờ, thu trong môi trường yên tĩnh bằng "
        "microphone chất lượng cao [6]. Dataset có giấy phép CC BY-NC-SA 4.0 và giới hạn mục "
        "đích học thuật. Đồ án giữ nguyên test split chính thức; validation được tách theo speaker "
        "từ train để ngăn speaker leakage.",
    )
    _add_paragraph(
        document,
        "Common Voice VI 21.0 được tải từ mirror Kaggle `minhquangphamle/common-voice-vi-21`, "
        "có giấy phép CC0-1.0. Pipeline dùng `validated.tsv`, vô hiệu hóa diễn giải dấu ngoặc kép "
        "của CSV để một dấu ngoặc kép thiếu không thể ghép nhiều hàng, đồng thời loại transcript dài "
        "bất thường. Dữ liệu được tách theo speaker; Common Voice test được giữ độc lập để đo khả năng "
        "thích nghi ngoài miền VIVOS.",
    )
    dataset_rows = []
    for split in ("train", "validation", "test", "common_voice_test"):
        row = dataset[split]
        dataset_rows.append(
            [
                split,
                f"{int(row['num_records']):,}",
                _fmt_rate(row["total_duration_hours"], 3),
                _fmt_rate(row["mean_duration_seconds"], 3),
                str(int(row["rejected_records"])),
            ]
        )
    _add_table(
        document,
        ["Split", "Số mẫu", "Giờ", "Thời lượng TB (s)", "Bị loại"],
        dataset_rows,
        "Bảng 1. Thống kê dữ liệu hỗn hợp sau khi chuẩn bị",
    )
    _add_figure(
        document,
        "outputs/kaggle_cv21/figures/audio_duration_distribution.png",
        "Hình 1. Phân phối thời lượng audio theo split",
    )

    document.add_heading("6 Tiền xử lý dữ liệu", level=1)
    _add_paragraph(
        document,
        "Pipeline kiểm tra file tồn tại, dữ liệu rỗng, sample rate, NaN/Inf và giới hạn thời "
        "lượng. Audio stereo được lấy trung bình kênh và resample polyphase về mono 16 kHz. "
        "Transcript được chuẩn hóa Unicode NFC, chuyển chữ thường và gom khoảng trắng; dấu tiếng "
        "Việt được giữ nguyên. Các quy tắc này được dùng lại trong train, evaluate và demo.",
    )

    document.add_heading("7 Kiến trúc hai model", level=1)
    document.add_heading("7.1 Wav2Vec2-Vietnamese 250h", level=2)
    _add_paragraph(
        document,
        "Model `nguyenvulebinh/wav2vec2-base-vietnamese-250h` có khoảng 95 triệu tham số, nhận "
        "waveform 16 kHz và trả logits CTC [2]. Đồ án dùng beam search kết hợp LM 4-gram do "
        "chính tác giả checkpoint phát hành; alpha=1, beta=1 và beam width=100 được chọn trên validation.",
    )
    document.add_heading("7.2 PhoWhisper Base fine-tuned", level=2)
    _add_paragraph(
        document,
        "PhoWhisper Base được khởi tạo từ `vinai/PhoWhisper-base`, một checkpoint kiến trúc Whisper "
        "đã thích nghi tiếng Việt đa giọng, sau đó tiếp tục cập nhật trọng "
        "số bằng train split VIVOS, sau đó thích nghi thêm trên hỗn hợp VIVOS và Common Voice VI 21.0. Dataset mã hóa lười theo từng batch để tránh giữ toàn bộ "
        "log-Mel trong RAM. Dynamic padding áp dụng riêng cho input feature và label; token "
        "padding được đổi thành -100 để không đóng góp vào loss.",
    )

    document.add_heading("8 Thiết lập fine-tuning", level=1)
    config_rows = [
        ["Checkpoint khởi tạo", str(training["model"])],
        ["Train / validation", f"{training['train_samples']:,} / {training['validation_samples']:,}"],
        ["Epoch", _fmt_rate(completed_epochs, 0)],
        ["Learning rate", "2e-6"],
        ["Batch train / eval", "1 / 1"],
        ["Gradient accumulation", "4"],
        ["Weight decay", "0,01"],
        ["Warmup steps", "250"],
        ["Chu kỳ lưu/đánh giá", "3.689 optimizer steps"],
        ["Gradient checkpointing", "Có"],
        ["Precision", "FP16"],
        ["Thiết bị", "NVIDIA GeForce RTX 4050 Laptop GPU 6 GB"],
        ["Seed", "42"],
    ]
    _add_table(document, ["Tham số", "Giá trị"], config_rows, "Bảng 2. Siêu tham số huấn luyện")
    best_validation_wer, best_validation_cer = _best_validation_metrics(logs)
    _add_paragraph(
        document,
        f"Huấn luyện {completed_epochs:.0f} epoch trên toàn bộ train split, hoàn tất trong "
        f"{_fmt_seconds(training['train_metrics']['train_runtime'])}. Checkpoint có WER validation "
        f"thấp nhất được lưu làm model của đồ án. Validation WER là {_fmt_rate(best_validation_wer)} và "
        f"CER là {_fmt_rate(best_validation_cer)}. Validation và test dùng "
        "cùng giới hạn 225 token, no-repeat 3-gram và repetition penalty 1,05.",
    )
    _add_figure(
        document,
        "outputs/kaggle_cv21/figures/training_validation_loss.png",
        "Hình 2. Loss huấn luyện và validation từ Trainer log",
    )

    document.add_heading("9 Thiết kế thực nghiệm", level=1)
    _add_paragraph(
        document,
        "Cả hai model được đánh giá tuần tự trên đúng 760 mẫu test VIVOS, cùng audio preprocessing, "
        "reference và transcript normalization. Wav2Vec2 dùng CTC beam search + LM 4-gram; Whisper dùng beam theo cấu hình. "
        "WER/CER được tính theo micro-average trên toàn corpus. Thời gian suy luận đo phần forward "
        "và generate, chưa phải benchmark latency production vì không loại hoàn toàn ảnh hưởng "
        "warm-up, cache và tiến trình nền. Riêng tác động của bước thích nghi được đo bằng cùng 546 "
        "mẫu Common Voice test trước và sau fine-tuning; tập này không tham gia train hoặc chọn checkpoint.",
    )

    document.add_heading("10 Kết quả", level=1)
    result_rows = [
        [
            "Wav2Vec2-Vietnamese 250h",
            str(wav["num_samples"]),
            _fmt_rate(wav["wer"]),
            _fmt_rate(wav["cer"]),
            str(wav["exact_match_count"]),
            _fmt_seconds(wav["mean_inference_seconds"]),
        ],
        [
            "PhoWhisper Base VIVOS + Common Voice VI 21",
            str(whisper["num_samples"]),
            _fmt_rate(whisper["wer"]),
            _fmt_rate(whisper["cer"]),
            str(whisper["exact_match_count"]),
            _fmt_seconds(whisper["mean_inference_seconds"]),
        ],
    ]
    _add_table(
        document,
        ["Model", "Mẫu", "WER", "CER", "Khớp đúng", "Suy luận TB"],
        result_rows,
        "Bảng 3. So sánh trên toàn bộ test set VIVOS",
    )
    _add_figure(
        document,
        "outputs/kaggle_cv21/figures/wer_cer_comparison.png",
        "Hình 3. So sánh WER và CER trên cùng test set",
    )
    delta_wer = float(whisper["wer"]) - float(wav["wer"])
    delta_cer = float(whisper["cer"]) - float(wav["cer"])
    better = "Wav2Vec2" if delta_wer > 0 else "Whisper"
    _add_paragraph(
        document,
        f"{better} có WER thấp hơn trong thí nghiệm này. Chênh lệch Whisper trừ Wav2Vec2 là "
        f"{_fmt_rate(delta_wer)} WER và {_fmt_rate(delta_cer)} CER. Đây không phải so sánh đồng "
        "mức về ngân sách dữ liệu: Wav2Vec2 đã được fine-tune trước trên 250 giờ VLSP, trong khi "
        "Whisper của đồ án dùng 18,20 giờ dữ liệu train hỗn hợp VIVOS và Common Voice.",
    )
    _add_table(
        document,
        ["Checkpoint PhoWhisper", "Mẫu", "WER", "CER", "Khớp đúng"],
        [
            [
                "Trước thích nghi (VIVOS)",
                str(cv_baseline["num_samples"]),
                _fmt_rate(cv_baseline["wer"]),
                _fmt_rate(cv_baseline["cer"]),
                str(cv_baseline["exact_match_count"]),
            ],
            [
                "Sau thích nghi VIVOS + CV21",
                str(cv_adapted["num_samples"]),
                _fmt_rate(cv_adapted["wer"]),
                _fmt_rate(cv_adapted["cer"]),
                str(cv_adapted["exact_match_count"]),
            ],
        ],
        "Bảng 4. Hiệu quả thích nghi trên Common Voice test độc lập",
    )
    _add_paragraph(
        document,
        "Bước thích nghi giảm WER Common Voice 0,0241 tuyệt đối (khoảng 13,7% tương đối), giảm CER "
        "0,0115 tuyệt đối và tăng số câu khớp hoàn toàn từ 199 lên 227. Trên VIVOS, WER giữ nguyên "
        "0,0892; CER thay đổi từ 0,0394 thành 0,0403. Kết quả cho thấy độ tổng quát tăng mà không làm "
        "giảm WER trên miền dữ liệu cũ.",
    )
    _add_paragraph(
        document,
        "Demo còn hiển thị proxy độ tin cậy chưa hiệu chuẩn. Wav2Vec2 dùng trung bình xác suất "
        "frame CTC không blank; Whisper dùng trung bình nhân xác suất token bằng teacher forcing. Hai giá trị không "
        "được diễn giải là xác suất toàn câu đúng và không dùng thay cho WER/CER.",
    )

    document.add_heading("11 Phân tích lỗi", level=1)
    categories = sorted(
        set(Counter(row["error_note"] for row in wav_rows))
        | set(Counter(row["error_note"] for row in whisper_rows))
    )
    wav_counts = Counter(row["error_note"] for row in wav_rows)
    whisper_counts = Counter(row["error_note"] for row in whisper_rows)
    category_rows = [
        [category, str(wav_counts[category]), str(whisper_counts[category])]
        for category in categories
    ]
    _add_table(
        document,
        ["Nhóm lỗi tự động", "Wav2Vec2", "Whisper"],
        category_rows,
        "Bảng 5. Phân bố nhãn lỗi theo mẫu VIVOS",
    )
    duration_rows = []
    for label in ("<3s", "3-6s", ">=6s"):
        for model_name, metrics in (("Wav2Vec2", wav), ("Whisper", whisper)):
            row = metrics["duration_bins"][label]
            duration_rows.append(
                [model_name, label, str(row["num_samples"]), _fmt_rate(row["wer"]), _fmt_rate(row["cer"])]
            )
    _add_table(
        document,
        ["Model", "Nhóm thời lượng", "Mẫu", "WER", "CER"],
        duration_rows,
        "Bảng 6. WER/CER theo thời lượng audio VIVOS",
    )
    _add_paragraph(
        document,
        "Nhãn lỗi được suy ra từ căn chỉnh Levenshtein và kiểm tra dấu, nên chỉ phản ánh hiện "
        "tượng quan sát được trong transcript. Không quy lỗi cho tiếng ồn hoặc giọng vùng miền "
        "khi VIVOS không cung cấp bằng chứng tương ứng và audio được thu trong môi trường yên tĩnh.",
    )
    for subsection, (model_name, rows) in enumerate(
        (("Wav2Vec2", wav_rows), ("Whisper", whisper_rows)), start=1
    ):
        document.add_heading(f"11.{subsection} Ví dụ lỗi của {model_name}", level=2)
        for row in _representative_errors(rows):
            heading = document.add_paragraph()
            run = heading.add_run(
                f"{row['audio_id']} - WER {_fmt_rate(row['wer'])}, CER {_fmt_rate(row['cer'])}"
            )
            _set_run_font(run, 10, bold=True)
            _add_paragraph(document, f"Reference: {row['reference']}", indent=False)
            _add_paragraph(document, f"Prediction: {row['prediction']}", indent=False)
            _add_paragraph(document, f"Nhận xét: {row['error_note']}.", indent=False)

    document.add_heading("12 Demo Gradio", level=1)
    _add_paragraph(
        document,
        "Giao diện cho phép tải file hoặc ghi âm, phát lại audio, chọn một trong hai model, xem "
        "transcript, thiết bị và thời gian suy luận. Nếu nhập transcript tham chiếu, ứng dụng tính "
        "WER/CER cho mẫu. Model được cache theo lựa chọn và audio không được gửi đến dịch vụ ASR "
        "bên ngoài. Nút nhận dạng có trạng thái tiến trình; lỗi file được trả bằng thông báo tiếng "
        "Việt thay vì làm dừng ứng dụng. Demo không giới hạn tổng thời lượng; audio dài "
        "được chia thành cửa sổ 28 giây để kiểm soát bộ nhớ GPU và ghép kết quả theo thứ tự. "
        "Với mã lớp ST23D, quy tắc miền chỉ kích hoạt sau ngữ cảnh "
        "`lớp` hoặc `mã` và giao diện công khai cụm nhận dạng gốc → cụm chuẩn hóa. Transcript tham "
        "chiếu chỉ dùng để tính WER/CER, không dùng transcript tham chiếu để sửa đầu ra.",
    )
    _add_figure(
        document,
        "outputs/reports/gradio_smoke.png",
        "Hình 4. Giao diện demo được kiểm tra bằng trình duyệt tự động",
    )

    document.add_heading("13 Thí nghiệm bổ sung với FPT Open Speech", level=1)
    fpt_data = fpt_experiment["dataset"]
    wav_fpt = fpt_experiment["wav2vec2"]
    whisper_fpt = fpt_experiment["phowhisper"]
    fpt_train_samples = f"{int(fpt_data['train_samples']):,}".replace(",", ".")
    _add_paragraph(
        document,
        "FPT Open Speech được tải từ mirror Kaggle và đối chiếu với bản công bố chính thức trên "
        "Mendeley Data. Sau lọc thời lượng tối đa 30 giây, 25.919 mẫu FPT được thêm riêng vào train; "
        f"tổng train tăng lên {fpt_train_samples} mẫu "
        f"({float(fpt_data['train_hours']):.3f} giờ). Validation 1.811 mẫu, VIVOS test 760 mẫu và "
        "Common Voice test 546 mẫu được giữ nguyên, không trộn vào train.",
    )
    _add_table(
        document,
        ["Model", "Tập đánh giá", "WER trước", "WER sau FPT", "CER trước", "CER sau FPT"],
        [
            [
                "Wav2Vec2",
                "VIVOS",
                _fmt_rate(wav_fpt["vivos"]["baseline_wer"]),
                _fmt_rate(wav_fpt["vivos"]["candidate_wer"]),
                _fmt_rate(wav_fpt["vivos"]["baseline_cer"]),
                _fmt_rate(wav_fpt["vivos"]["candidate_cer"]),
            ],
            [
                "Wav2Vec2",
                "Common Voice",
                _fmt_rate(wav_fpt["common_voice"]["baseline_wer"]),
                _fmt_rate(wav_fpt["common_voice"]["candidate_wer"]),
                _fmt_rate(wav_fpt["common_voice"]["baseline_cer"]),
                _fmt_rate(wav_fpt["common_voice"]["candidate_cer"]),
            ],
            [
                "PhoWhisper",
                "VIVOS",
                _fmt_rate(whisper_fpt["vivos"]["baseline_wer"]),
                _fmt_rate(whisper_fpt["vivos"]["candidate_wer"]),
                _fmt_rate(whisper_fpt["vivos"]["baseline_cer"]),
                _fmt_rate(whisper_fpt["vivos"]["candidate_cer"]),
            ],
            [
                "PhoWhisper",
                "Common Voice",
                _fmt_rate(whisper_fpt["common_voice"]["baseline_wer"]),
                _fmt_rate(whisper_fpt["common_voice"]["candidate_wer"]),
                _fmt_rate(whisper_fpt["common_voice"]["baseline_cer"]),
                _fmt_rate(whisper_fpt["common_voice"]["candidate_cer"]),
            ],
        ],
        "Bảng 7. So sánh checkpoint trước và sau khi thêm FPT vào train",
    )
    _add_paragraph(
        document,
        "Wav2Vec2 sau FPT cải thiện nhẹ trên VIVOS nhưng WER Common Voice tăng nhẹ; trên mẫu ST23D "
        f"độc lập, WER tăng từ {_fmt_rate(wav_fpt['st23d']['baseline_wer'])} lên "
        f"{_fmt_rate(wav_fpt['st23d']['candidate_wer'])}. PhoWhisper sau FPT xấu hơn trên validation, "
        "VIVOS, Common Voice và mẫu ST23D. Vì vậy hai candidate được giữ làm artifact thí nghiệm nhưng "
        "không thay checkpoint demo. Kết quả này cũng cho thấy không thể bảo đảm độ chính xác 100% trên "
        "audio chưa thấy; dự án không đưa test hoặc transcript tham chiếu vào quá trình sửa prediction.",
    )

    document.add_heading(
        "14 Thí nghiệm Kaggle Vietnamese Speech Recognition Data", level=1
    )
    kaggle_data = kaggle_speech_experiment["dataset"]
    combined_data = kaggle_speech_experiment["combined_data"]
    wav_kaggle = kaggle_speech_experiment["wav2vec2"]
    whisper_kaggle = kaggle_speech_experiment["phowhisper"]
    _add_paragraph(
        document,
        "Dataset Kaggle `tuvo6804/vietnamese-speech-recognition-data` có "
        f"{int(kaggle_data['labeled_audio_files']):,} FLAC có nhãn/"
        f"{float(kaggle_data['labeled_hours']):.6f} giờ và "
        f"{int(kaggle_data['unlabeled_public_test_files_unused'])} public-test không nhãn không dùng. "
        f"Pipeline chia cố định thành {int(kaggle_data['train_files']):,} train và "
        f"{int(kaggle_data['holdout_files'])} holdout. Metadata Kaggle ghi license `unknown`; "
        "dữ liệu chỉ được giữ cho thí nghiệm học thuật nội bộ, không phân phối lại và cần xác minh "
        "quyền trước mục đích khác. Dataset không có speaker ID tin cậy nên không tuyên bố holdout "
        "speaker-disjoint.",
    )
    _add_paragraph(
        document,
        f"Manifest kết hợp có {int(combined_data['train_samples']):,} train/"
        f"{float(combined_data['train_hours']):.6f} giờ, "
        f"{int(combined_data['validation_samples']):,} validation, "
        f"{int(combined_data['vivos_test_samples'])} VIVOS test, "
        f"{int(combined_data['common_voice_test_samples'])} Common Voice test và "
        f"{int(combined_data['kaggle_holdout_samples'])} Kaggle holdout. "
        "Kaggle holdout không trùng ID, đường dẫn hoặc transcript với phần Kaggle train và không "
        "trùng transcript chính xác với các split VIVOS/Common Voice hiện hữu.",
    )
    kaggle_rows: list[list[str]] = []
    for model_name, model_result in (
        ("Wav2Vec2", wav_kaggle),
        ("PhoWhisper", whisper_kaggle),
    ):
        for split_key, split_label in (
            ("vivos_test", "VIVOS"),
            ("common_voice_test", "Common Voice"),
            ("kaggle_holdout", "Kaggle holdout"),
        ):
            row = model_result["benchmarks"][split_key]
            kaggle_rows.append(
                [
                    model_name,
                    split_label,
                    _fmt_rate(row["baseline_wer"]),
                    _fmt_rate(row["candidate_wer"]),
                    _fmt_rate(row["baseline_cer"]),
                    _fmt_rate(row["candidate_cer"]),
                ]
            )
    _add_table(
        document,
        ["Model", "Tập đánh giá", "WER trước", "WER candidate", "CER trước", "CER candidate"],
        kaggle_rows,
        "Bảng 8. So sánh checkpoint trước và sau thích nghi Kaggle",
    )
    _add_paragraph(
        document,
        "Micro-average trên ba corpus độc lập cho thấy Wav2Vec2 giảm WER từ "
        f"{_fmt_rate(wav_kaggle['micro_average_three_corpora']['baseline_wer'])} xuống "
        f"{_fmt_rate(wav_kaggle['micro_average_three_corpora']['candidate_wer'])}; PhoWhisper giảm từ "
        f"{_fmt_rate(whisper_kaggle['micro_average_three_corpora']['baseline_wer'])} xuống "
        f"{_fmt_rate(whisper_kaggle['micro_average_three_corpora']['candidate_wer'])}. "
        "Dù tổng thể tốt hơn, Wav2Vec2 tăng WER Common Voice và làm mẫu ST23D giảm từ 0 xuống "
        f"{_fmt_rate(wav_kaggle['benchmarks']['st23d']['candidate_wer'])}; PhoWhisper tăng lỗi trên "
        "VIVOS và Common Voice, còn ST23D không cải thiện. Không promote hai candidate Kaggle. "
        "Kết quả không hỗ trợ tuyên bố "
        "độ chính xác 100% trên audio chưa thấy.",
    )

    document.add_heading("15 Supervised adaptation cho demo ST23D", level=1)
    adaptation = st23d_adaptation["adaptation_protocol"]
    independent = st23d_adaptation["independent_benchmarks"]
    micro = st23d_adaptation["micro_average_three_independent_corpora"]
    training_fit = st23d_adaptation["st23d_training_fit"]
    _add_paragraph(
        document,
        "Mẫu ST23D được dùng có giám sát để thích nghi PhoWhisper, vì vậy kết quả trên "
        "mẫu này là training fit, không phải test độc lập. Ba chiến lược được thử; cấu hình "
        f"triển khai chọn `{st23d_adaptation['promoted_checkpoint']}` tại optimizer step "
        f"{int(adaptation['selected_optimizer_step'])}, là checkpoint sớm nhất đạt mục tiêu với "
        "mức thay đổi trọng số nhỏ. Label giữ cách đọc tự nhiên của mã lớp trước khi quy tắc "
        "miền chuẩn hóa thành ST23D.",
    )
    st23d_rows: list[list[str]] = []
    for split_key, split_label in (
        ("vivos_test", "VIVOS test"),
        ("common_voice_test", "Common Voice test"),
        ("kaggle_holdout", "Kaggle holdout"),
    ):
        row = independent[split_key]
        st23d_rows.append(
            [
                split_label,
                str(int(row["num_samples"])),
                _fmt_rate(row["baseline_wer"]),
                _fmt_rate(row["candidate_wer"]),
                _fmt_rate(row["baseline_cer"]),
                _fmt_rate(row["candidate_cer"]),
            ]
        )
    st23d_rows.extend(
        [
            [
                "Micro-average độc lập",
                str(int(micro["num_samples"])),
                _fmt_rate(micro["baseline_wer"]),
                _fmt_rate(micro["candidate_wer"]),
                _fmt_rate(micro["baseline_cer"]),
                _fmt_rate(micro["candidate_cer"]),
            ],
            [
                "ST23D training fit",
                "1",
                "0,1000",
                _fmt_rate(training_fit["phowhisper_wer"]),
                "0,0353",
                _fmt_rate(training_fit["phowhisper_cer"]),
            ],
        ]
    )
    _add_table(
        document,
        ["Phạm vi", "Mẫu", "WER trước", "WER sau", "CER trước", "CER sau"],
        st23d_rows,
        "Bảng 9. PhoWhisper trước và sau supervised adaptation ST23D",
    )
    correction = adaptation["configured_phrase_correction"]
    _add_paragraph(
        document,
        "Pipeline demo khai báo quy tắc nguyên-token `"
        f"{correction['source']}` → `{correction['replacement']}` và công khai mọi lần thay "
        "thế trên giao diện. Quy tắc không dùng transcript tham chiếu ở thời điểm suy luận. "
        f"Trên {int(micro['num_samples']):,} mẫu độc lập, micro-average WER thay đổi từ "
        f"{_fmt_rate(micro['baseline_wer'])} thành {_fmt_rate(micro['candidate_wer'])} và CER từ "
        f"{_fmt_rate(micro['baseline_cer'])} thành {_fmt_rate(micro['candidate_cer'])}. "
        "Cả Wav2Vec2 và PhoWhisper đạt WER/CER 0/0 trên mẫu ST23D đã thích nghi, nhưng "
        "kết quả này không chứng minh độ chính xác 100% trên audio chưa thấy.",
    )

    document.add_heading("16 Hạn chế", level=1)
    limitations = [
        "VIVOS và Common Voice mở rộng độ đa dạng nhưng vẫn chưa đại diện đầy đủ hội thoại tự nhiên, tiếng ồn và mọi vùng miền.",
        "Hai model không có cùng ngân sách pretraining và supervised fine-tuning, nên kết quả chỉ mô tả hiệu năng triển khai cụ thể.",
        "PhoWhisper Base, batch size 1 và một epoch được chọn theo giới hạn VRAM 6 GB và ngân sách tính toán; chưa khảo sát nhiều epoch hoặc bản Small.",
        "Proxy độ tin cậy chưa được calibration bằng tập riêng và không phải xác suất transcript đúng.",
        "Demo đã xử lý audio dài theo cửa sổ 28 giây nhưng chưa có VAD hoặc chunk overlap, nên từ nằm đúng biên có thể kém chính xác.",
        "Vietnamese Speech Recognition Data có license metadata unknown và không cung cấp speaker ID tin cậy; holdout chỉ được xác minh tách record/transcript, không chứng minh speaker-disjoint.",
        "Thí nghiệm Kaggle chạy một epoch và một seed; cải thiện trên holdout cùng miền không đại diện mọi giọng nói hoặc điều kiện âm học tiếng Việt.",
        "Mẫu ST23D đã được dùng cho supervised adaptation; WER/CER 0 trên mẫu này chỉ đo độ khớp dữ liệu huấn luyện và không phải bằng chứng tổng quát hóa.",
    ]
    for item in limitations:
        paragraph = document.add_paragraph(style="List Bullet")
        run = paragraph.add_run(item)
        _set_run_font(run, 11)

    document.add_heading("17 Kết luận và hướng phát triển", level=1)
    _add_paragraph(
        document,
        "Đồ án đã hoàn thiện một pipeline Deep Learning có huấn luyện thực, checkpoint local, "
        "đánh giá trên test độc lập, phân tích lỗi và demo. Kết quả cho thấy việc chọn kiến trúc "
        "và quy mô dữ liệu fine-tuning ảnh hưởng trực tiếp đến WER/CER. Hướng tiếp theo là tăng "
        "dữ liệu hội thoại và nhiễu có nhãn, khảo sát PhoWhisper Small, thêm augmentation có kiểm soát, calibration "
        "độ tin cậy và nâng cấp phân đoạn audio dài bằng VAD cùng chunk overlap.",
    )

    document.add_heading("Tài liệu tham khảo", level=1)
    references = [
        "A. Baevski, H. Zhou, A. Mohamed, and M. Auli, 'wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations,' 2020. https://arxiv.org/abs/2006.11477",
        "T. B. Nguyen, 'Vietnamese end-to-end speech recognition using wav2vec 2.0,' model card, 2021. https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        "A. Radford et al., 'Robust Speech Recognition via Large-Scale Weak Supervision,' 2022. https://arxiv.org/abs/2212.04356",
        "A. Graves et al., 'Connectionist Temporal Classification,' ICML, 2006. https://www.cs.toronto.edu/~graves/icml_2006.pdf",
        "A. Vaswani et al., 'Attention Is All You Need,' 2017. https://arxiv.org/abs/1706.03762",
        "H.-T. Luong and H.-Q. Vu, 'VIVOS: Vietnamese Speech Corpus for ASR,' Zenodo, 2016. https://doi.org/10.5281/zenodo.7068130",
        "H.-T. Luong and H.-Q. Vu, 'A non-expert Kaldi recipe for Vietnamese Speech Recognition System,' WLSI/OIAF4HLT, 2016. https://aclanthology.org/W16-5207/",
        "T.-T. Le, L. T. Nguyen, and D. Q. Nguyen, 'PhoWhisper: Automatic Speech Recognition for Vietnamese,' ICLR Tiny Papers, 2024. https://huggingface.co/vinai/PhoWhisper-base",
        "Mozilla Common Voice, 'Common Voice Dataset,' Vietnamese locale, version 21.0, CC0-1.0. https://commonvoice.mozilla.org/vi/datasets",
        "M. Q. P. Le, 'Common Voice Vietnamese 21.0,' Kaggle dataset mirror. https://www.kaggle.com/datasets/minhquangphamle/common-voice-vi-21",
        "FPT.AI, 'FPT Open Speech Dataset,' Mendeley Data, version 4. https://data.mendeley.com/datasets/k9sxg2twv4/4",
        "Binhhhhhhhhh, 'FPT Open Speech Dataset FOSD Vietnamese,' Kaggle dataset mirror. https://www.kaggle.com/datasets/binhhhhhhhhh/fpt-open-speech-dataset-fosd-vietnamese",
        "Tu Vo, 'Vietnamese speech recognition_data,' Kaggle dataset. https://www.kaggle.com/datasets/tuvo6804/vietnamese-speech-recognition-data",
    ]
    for index, reference in enumerate(references, start=1):
        _add_reference(document, index, reference)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
    return output_path


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "Vietnamese_ASR_Project_Report.docx",
    )
    args = parser.parse_args()
    print(build_report(args.output.resolve()))


if __name__ == "__main__":
    main()
