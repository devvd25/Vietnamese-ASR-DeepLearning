from __future__ import annotations

import os

from playwright.sync_api import sync_playwright


DEMO_URL = os.environ.get("ASR_UI_URL", "http://127.0.0.1:7860")
EXPECTED_RUNTIME_BADGE = os.environ.get("ASR_EXPECTED_RUNTIME_BADGE")
BROWSER_LOCALE = os.environ.get("ASR_BROWSER_LOCALE", "en-US")
MIN_TOUCH_TARGET_PX = 44


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(
        locale=BROWSER_LOCALE,
        viewport={"width": 375, "height": 812},
    )
    response = page.goto(DEMO_URL, wait_until="networkidle", timeout=120_000)
    assert response and response.ok
    assert page.get_by_role("combobox", name="Mô hình nhận dạng").is_visible()
    if EXPECTED_RUNTIME_BADGE:
        assert page.get_by_text(EXPECTED_RUNTIME_BADGE, exact=True).is_visible()

    upload_copy = page.get_by_test_id("upload-text").inner_text()
    assert "Thả tệp âm thanh vào đây" in upload_copy
    assert "hoặc" in upload_copy
    assert "Nhấp để tải tệp lên" in upload_copy
    assert "Déposez" not in upload_copy
    assert "Click to Upload" not in upload_copy
    assert page.locator("html").get_attribute("lang") == "vi"

    for accessible_name in ("Tải tệp lên", "Ghi âm", "Sao chép"):
        control = page.get_by_role("button", name=accessible_name).first
        assert control.is_visible(), accessible_name
        box = control.bounding_box()
        assert box is not None
        assert box["width"] >= MIN_TOUCH_TARGET_PX, (accessible_name, box)
        assert box["height"] >= MIN_TOUCH_TARGET_PX, (accessible_name, box)

    page.get_by_role("button", name="Ghi \u00e2m").click()
    audio = page.locator("#asr-audio")
    audio.get_by_role(
        "button", name="B\u1eaft \u0111\u1ea7u ghi \u00e2m"
    ).wait_for(state="visible", timeout=5_000)
    assert audio.get_by_role("button", name="X\u00f3a b\u1ea3n ghi").is_visible()
    assert audio.get_by_role(
        "combobox", name="Ch\u1ecdn thi\u1ebft b\u1ecb \u0111\u1ea7u v\u00e0o"
    ).is_visible()
    empty_microphone_option = audio.locator('select.mic-select option[value=""]')
    assert empty_microphone_option.inner_text() == "Kh\u00f4ng t\u00ecm th\u1ea5y micr\u00f4"
    recording_copy = audio.text_content()
    for french_text in ("Enregistrer", "Arr\u00eater", "Reprendre", "Aucun microphone"):
        assert french_text not in recording_copy

    assert page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1"
    )
    browser.close()

print("Browser UI accessibility OK: icon controls >= 44px; no mobile overflow")
