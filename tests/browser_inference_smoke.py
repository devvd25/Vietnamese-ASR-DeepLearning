from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

AUDIO_PATH = Path(os.environ["ASR_SMOKE_AUDIO"])
SCREENSHOT_DIR = Path(os.environ.get("ASR_SMOKE_SCREENSHOT_DIR", "outputs/browser_smoke"))
DEMO_URL = os.environ.get("ASR_SMOKE_URL", "http://127.0.0.1:7860")
REFERENCE = (
    "xin chào chào mừng đến với lớp st23d mình là đặng công vũ "
    "xin chào tất cả các bạn nha"
)


def normalized_value(page) -> str:
    textareas = page.locator("textarea")
    assert textareas.count() >= 2
    output = textareas.last
    output.wait_for(state="visible", timeout=180_000)
    return output.input_value().strip().lower()


with sync_playwright() as playwright:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    console_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text) if message.type == "error" else None,
    )
    page.goto(DEMO_URL, wait_until="networkidle")
    assert page.get_by_text("Vietnamese ASR Benchmark", exact=True).count() == 1
    assert page.get_by_text("On-device engine", exact=True).count() == 1
    assert page.get_by_text("SYSTEM READY", exact=True).count() == 0
    assert page.get_by_role("heading", name="1. Dữ liệu đầu vào").count() == 1
    assert page.get_by_role("heading", name="2. Cấu hình & thực thi").count() == 1
    assert page.get_by_role("heading", name="3. Văn bản nhận dạng").count() == 1
    assert page.get_by_role("heading", name="4. Bảng chỉ số kỹ thuật").count() == 1
    technical_font_sizes = page.locator(
        ".status-pill, .eyebrow-chip, .checkpoint-label, .metric-label, .metric-status"
    ).evaluate_all(
        "elements => elements.map(element => parseFloat(getComputedStyle(element).fontSize))"
    )
    assert technical_font_sizes
    assert min(technical_font_sizes) >= 12, technical_font_sizes
    assert page.get_by_text("Không giới hạn thời lượng", exact=False).count() >= 1
    page.locator('input[type="file"]').first.set_input_files(str(AUDIO_PATH))
    page.locator("audio").first.wait_for(state="attached", timeout=30_000)
    page.wait_for_timeout(1_000)
    page.get_by_label("Transcript tham chiếu (không bắt buộc)").fill(REFERENCE)
    page.get_by_role("button", name="Nhận dạng tiếng nói").click()
    page.get_by_text("WER: 0.0000", exact=False).wait_for(timeout=180_000)
    assert normalized_value(page) == REFERENCE
    assert page.get_by_text("ST23D", exact=False).count() >= 1
    page.screenshot(path=str(SCREENSHOT_DIR / "wav2vec2_desktop.png"), full_page=True)

    model_select = page.get_by_role("combobox", name="Mô hình nhận dạng")
    model_select.click()
    page.get_by_role(
        "option", name="PhoWhisper Base — ST23D supervised adaptation"
    ).click()
    page.get_by_role("button", name="Nhận dạng tiếng nói").click()
    page.get_by_text("xin chờ tất cả", exact=False).wait_for(timeout=180_000)
    page.get_by_text("WER: 0.0000", exact=False).wait_for(timeout=180_000)
    whisper_text = normalized_value(page)
    assert whisper_text == REFERENCE
    assert page.get_by_text("không phải test độc lập", exact=False).count() >= 1
    assert page.get_by_text("ét tê hai ba đê", exact=False).count() >= 1
    page.screenshot(path=str(SCREENSHOT_DIR / "phowhisper_desktop.png"), full_page=True)

    page.set_viewport_size({"width": 375, "height": 812})
    page.wait_for_timeout(500)
    assert page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"
    )
    page.screenshot(path=str(SCREENSHOT_DIR / "phowhisper_mobile_375.png"), full_page=True)
    page.emulate_media(reduced_motion="reduce")
    page.add_style_tag(content="html { font-size: 20px !important; }")
    assert page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches")
    assert page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"
    )
    page.screenshot(
        path=str(SCREENSHOT_DIR / "phowhisper_mobile_large_text.png"), full_page=True
    )
    page.set_viewport_size({"width": 812, "height": 375})
    page.wait_for_timeout(300)
    assert page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"
    )
    page.screenshot(
        path=str(SCREENSHOT_DIR / "phowhisper_landscape_812.png"), full_page=True
    )
    visible_text = page.locator("body").inner_text()
    for emoji in ("🎙️", "🚀", "🤖", "✨", "📊", "⚡", "❌", "✅"):
        assert emoji not in visible_text
    audio_ready_state = page.locator("audio").first.evaluate("element => element.readyState")
    assert audio_ready_state >= 2
    known_gradio_audio_messages = (
        "Failed to load resource: net::ERR_CONTENT_LENGTH_MISMATCH",
        "Waveform load error: TypeError: Failed to fetch",
    )
    unexpected_console_errors = [
        message
        for message in console_errors
        if message not in known_gradio_audio_messages
    ]
    assert not unexpected_console_errors, unexpected_console_errors
    browser.close()

print(
    "Browser inference smoke test OK: Wav2Vec2 WER=0; "
    "PhoWhisper adapted WER=0; disclosure visible; mobile width=375px"
)
