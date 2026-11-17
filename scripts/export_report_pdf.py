"""Xuất PDF từ DOCX báo cáo mà không phụ thuộc Microsoft Word/LibreOffice."""

from __future__ import annotations

import argparse
import html
import sys
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.document import Document as DocumentObject
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import configure_utf8_stdio


def _register_fonts() -> tuple[str, str, str]:
    fonts_dir = Path("C:/Windows/Fonts")
    regular = fonts_dir / "times.ttf"
    bold = fonts_dir / "timesbd.ttf"
    italic = fonts_dir / "timesi.ttf"
    if not (regular.is_file() and bold.is_file() and italic.is_file()):
        return "Times-Roman", "Times-Bold", "Times-Italic"
    pdfmetrics.registerFont(TTFont("TimesNewRoman", regular))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", bold))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Italic", italic))
    pdfmetrics.registerFontFamily(
        "TimesNewRoman",
        normal="TimesNewRoman",
        bold="TimesNewRoman-Bold",
        italic="TimesNewRoman-Italic",
        boldItalic="TimesNewRoman-Bold",
    )
    return "TimesNewRoman", "TimesNewRoman-Bold", "TimesNewRoman-Italic"


def _iter_blocks(document: DocumentObject) -> Iterator[DocxParagraph | DocxTable]:
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield DocxParagraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield DocxTable(child, document)


def _paragraph_markup(paragraph: DocxParagraph) -> str:
    fragments: list[str] = []
    for run in paragraph.runs:
        text = html.escape(run.text).replace("\n", "<br/>")
        if not text:
            continue
        if run.bold:
            text = f"<b>{text}</b>"
        if run.italic:
            text = f"<i>{text}</i>"
        fragments.append(text)
    return "".join(fragments) or html.escape(paragraph.text).replace("\n", "<br/>")


def _image_flowables(document: DocumentObject, paragraph: DocxParagraph) -> list[Image]:
    images: list[Image] = []
    for blip in paragraph._p.xpath(".//a:blip"):
        relationship_id = blip.get(qn("r:embed"))
        if not relationship_id:
            continue
        part = document.part.related_parts.get(relationship_id)
        if part is None or not hasattr(part, "blob"):
            continue
        image = Image(BytesIO(part.blob))
        image._restrictSize(16.0 * cm, 11.0 * cm)
        image.hAlign = "CENTER"
        images.append(image)
    return images


def _styles():
    regular, bold, italic = _register_fonts()
    sample = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "BodyVietnamese",
            parent=sample["BodyText"],
            fontName=regular,
            fontSize=10.5,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "center": ParagraphStyle(
            "CenterVietnamese",
            parent=sample["BodyText"],
            fontName=regular,
            fontSize=10.5,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "title": ParagraphStyle(
            "TitleVietnamese",
            parent=sample["Title"],
            fontName=bold,
            fontSize=17,
            leading=22,
            alignment=TA_CENTER,
            spaceBefore=20,
            spaceAfter=20,
        ),
        "h1": ParagraphStyle(
            "Heading1Vietnamese",
            parent=sample["Heading1"],
            fontName=bold,
            fontSize=13.5,
            leading=17,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=12,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2Vietnamese",
            parent=sample["Heading2"],
            fontName=bold,
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#244A73"),
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "CaptionVietnamese",
            parent=sample["BodyText"],
            fontName=italic,
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            spaceAfter=7,
        ),
        "table_caption": ParagraphStyle(
            "TableCaptionVietnamese",
            parent=sample["BodyText"],
            fontName=bold,
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "BulletVietnamese",
            parent=sample["BodyText"],
            fontName=regular,
            fontSize=10.5,
            leading=14,
            leftIndent=14,
            firstLineIndent=-8,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "TableVietnamese",
            parent=sample["BodyText"],
            fontName=regular,
            fontSize=8.3,
            leading=10,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "TableHeaderVietnamese",
            parent=sample["BodyText"],
            fontName=bold,
            fontSize=8.3,
            leading=10,
            alignment=TA_CENTER,
        ),
    }


def _table_flowable(table: DocxTable, styles: dict[str, ParagraphStyle]) -> Table:
    values: list[list[Paragraph]] = []
    for row_index, row in enumerate(table.rows):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        values.append(
            [Paragraph(html.escape(cell.text).replace("\n", "<br/>"), style) for cell in row.cells]
        )
    column_count = max(len(row) for row in values)
    usable_width = A4[0] - 4.2 * cm
    first_header = table.cell(0, 0).text.strip()
    if first_header == "Model" and column_count == 6:
        ratios = [2.0, 0.75, 0.8, 0.8, 1.0, 1.15]
    elif first_header == "Nhóm lỗi tự động" and column_count == 3:
        ratios = [2.0, 1.0, 1.0]
    else:
        ratios = [1.0] * column_count
    ratio_total = sum(ratios)
    flowable = Table(
        values,
        colWidths=[usable_width * ratio / ratio_total for ratio in ratios],
        repeatRows=1,
        hAlign="CENTER",
    )
    flowable.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#7A8A99")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return flowable


def _draw_page(canvas, document) -> None:
    canvas.saveState()
    regular, _, _ = _register_fonts()
    canvas.setFont(regular, 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, str(document.page))
    if document.page > 1:
        canvas.drawRightString(A4[0] - 2.1 * cm, A4[1] - 1.2 * cm, "Đồ án Deep Learning - STT 19")
    canvas.restoreState()


def export_pdf(input_path: Path, output_path: Path) -> Path:
    document = Document(input_path)
    styles = _styles()
    story = []
    for block in _iter_blocks(document):
        if isinstance(block, DocxTable):
            story.extend((_table_flowable(block, styles), Spacer(1, 7)))
            continue
        if block._p.xpath('.//w:br[@w:type="page"]'):
            story.append(PageBreak())
            if not block.text.strip() and not block._p.xpath(".//a:blip"):
                continue
        images = _image_flowables(document, block)
        if images:
            story.extend(images)
            continue
        plain_text = block.text.strip()
        text = _paragraph_markup(block).strip()
        if not text:
            story.append(Spacer(1, 3))
            continue
        style_name = block.style.name if block.style is not None else "Normal"
        if style_name == "Title":
            style = styles["title"]
        elif style_name == "Heading 1":
            style = styles["h1"]
        elif style_name == "Heading 2":
            style = styles["h2"]
        elif style_name.startswith("List Bullet"):
            story.append(Paragraph(text, styles["bullet"], bulletText="•"))
            continue
        elif block.alignment == WD_ALIGN_PARAGRAPH.CENTER:
            if plain_text.startswith("Bảng "):
                style = styles["table_caption"]
            elif plain_text.startswith("Hình "):
                style = styles["caption"]
            else:
                style = styles["center"]
        else:
            style = styles["body"]
        story.append(Paragraph(text, style))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2.1 * cm,
        leftMargin=2.1 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.7 * cm,
        title="Vietnamese ASR Deep Learning - STT 19",
        author="Vietnamese ASR Deep Learning Project",
        subject="Wav2Vec2-Vietnamese and Whisper fine-tuned on VIVOS",
    )
    pdf.build(story, onFirstPage=_draw_page, onLaterPages=_draw_page)
    return output_path


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "docs" / "Vietnamese_ASR_Project_Report.docx")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "pdf" / "Vietnamese_ASR_Project_Report.pdf")
    args = parser.parse_args()
    print(export_pdf(args.input.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
