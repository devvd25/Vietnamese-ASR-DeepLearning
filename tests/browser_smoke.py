from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import configure_utf8_stdio

SCREENSHOT = PROJECT_ROOT / "outputs" / "reports" / "gradio_smoke.png"
configure_utf8_stdio()


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    console_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text) if message.type == "error" else None,
    )
    page.goto("http://127.0.0.1:7860")
    page.wait_for_load_state("networkidle")
    assert page.get_by_role("heading", name="Vietnamese AI Speech Recognition").count() >= 1
    assert page.get_by_text("Chọn audio", exact=False).count() >= 1
    assert page.get_by_text("PhoWhisper VietSpeech 10h đã sẵn sàng", exact=False).count() >= 1
    assert page.get_by_text("proxy chưa hiệu chuẩn", exact=False).count() >= 1
    model_select = page.get_by_role("combobox", name="Mô hình nhận dạng")
    model_select.click()
    assert page.get_by_role(
        "option", name="PhoWhisper Base — VietSpeech 10h"
    ).count() == 1
    page.keyboard.press("Escape")
    recognize = page.get_by_role("button", name="Nhận dạng tiếng nói")
    assert recognize.is_visible()
    recognize.click()
    page.get_by_text("Vui lòng chọn một tệp âm thanh", exact=False).wait_for(timeout=15_000)
    page.screenshot(path=str(SCREENSHOT), full_page=True)
    page.set_viewport_size({"width": 375, "height": 812})
    page.reload(wait_until="networkidle")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    page.set_viewport_size({"width": 812, "height": 375})
    page.reload(wait_until="networkidle")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    assert not console_errors, console_errors
    browser.close()

print(f"Browser smoke test OK; screenshot={SCREENSHOT}")
