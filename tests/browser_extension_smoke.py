from __future__ import annotations

import os
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "extension"
SCREENSHOT_DIR = ROOT / "outputs" / "extension"
YOUTUBE_FIXTURE = """
<!doctype html>
<html lang="vi">
  <head><meta charset="utf-8"><title>Video kiểm thử</title></head>
  <body style="margin:0;background:#111">
    <div id="movie_player" style="position:relative;width:960px;height:540px;background:#222"></div>
  </body>
</html>
"""


with tempfile.TemporaryDirectory(prefix="asr-extension-") as profile:
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            profile,
            headless=True,
            channel="chromium",
            viewport={"width": 1280, "height": 720},
            args=[
                f"--disable-extensions-except={EXTENSION}",
                f"--load-extension={EXTENSION}",
            ],
        )
        service_worker = (
            context.service_workers[0]
            if context.service_workers
            else context.wait_for_event("serviceworker", timeout=30_000)
        )
        extension_id = service_worker.url.split("/")[2]

        popup = context.new_page()
        popup.set_viewport_size({"width": 420, "height": 720})
        popup_errors: list[str] = []
        popup.on(
            "console",
            lambda message: popup_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        popup.goto(f"chrome-extension://{extension_id}/popup.html")
        popup.wait_for_load_state("networkidle")
        assert popup.get_by_role("heading", name="Phụ đề YouTube realtime").is_visible()
        assert popup.get_by_label("Mô hình nhận dạng").is_visible()
        assert popup.get_by_label("Độ trễ phụ đề").is_visible()
        learning = popup.get_by_label("Học có kiểm soát")
        assert learning.is_visible()
        assert learning.is_checked()
        for label in ("Bắt đầu phụ đề", "Dừng", "Xóa phụ đề đang hiển thị"):
            button = popup.get_by_role("button", name=label)
            assert button.is_visible()
            box = button.bounding_box()
            assert box and box["height"] >= 44, (label, box)
        assert popup.locator("html").get_attribute("lang") == "vi"
        assert popup.locator('meta[name="theme-color"][media*="light"]').count() == 1
        assert popup.locator('meta[name="theme-color"][media*="dark"]').count() == 1
        assert popup.evaluate(
            "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
        )
        if os.environ.get("ASR_SKIP_SOCKET_SMOKE") != "1":
            websocket_ready = popup.evaluate(
                """() => new Promise((resolve, reject) => {
              const socket = new WebSocket(
                'ws://127.0.0.1:8765/ws/captions?model=wav2vec2_vietnamese&sample_rate=16000&chunk_seconds=4'
              );
              const timer = setTimeout(() => reject(new Error('WebSocket timeout')), 5000);
              socket.onmessage = (event) => {
                clearTimeout(timer);
                const payload = JSON.parse(event.data);
                socket.close(1000, 'browser smoke complete');
                resolve(payload);
              };
              socket.onerror = () => {
                clearTimeout(timer);
                reject(new Error('WebSocket connection failed'));
              };
            })"""
            )
            assert websocket_ready["type"] == "ready"
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        popup.screenshot(path=str(SCREENSHOT_DIR / "popup.png"), full_page=True)

        youtube = context.new_page()
        youtube.set_viewport_size({"width": 1280, "height": 720})
        youtube.route(
            "https://www.youtube.com/watch?v=asr-extension-test",
            lambda route: route.fulfill(
                status=200,
                content_type="text/html; charset=utf-8",
                body=YOUTUBE_FIXTURE,
            ),
        )
        youtube.goto("https://www.youtube.com/watch?v=asr-extension-test")
        youtube.wait_for_load_state("domcontentloaded")
        service_worker.evaluate(
            """async () => {
              const tabs = await chrome.tabs.query({url: 'https://www.youtube.com/*'});
              if (!tabs.length) throw new Error('YouTube fixture tab missing');
              await chrome.tabs.sendMessage(tabs[0].id, {
                target: 'content', type: 'caption', sequence: 1,
                text: 'xin chào phụ đề trực tiếp'
              });
            }"""
        )
        youtube.wait_for_function(
            """() => document.querySelector('#vietnamese-asr-live-caption-host')
              ?.shadowRoot?.querySelector('.caption')?.textContent
              === 'xin chào phụ đề trực tiếp'""",
            timeout=10_000,
        )
        assert youtube.evaluate(
            """() => document.querySelector('#vietnamese-asr-live-caption-host')
              .shadowRoot.querySelector('.caption').getAttribute('role')"""
        ) == "status"
        service_worker.evaluate(
            """async () => {
              const tabs = await chrome.tabs.query({url: 'https://www.youtube.com/*'});
              await chrome.tabs.sendMessage(tabs[0].id, {
                target: 'content', type: 'caption', sequence: 2,
                text: 'nội dung tiếp theo'
              });
            }"""
        )
        youtube.wait_for_function(
            """() => document.querySelector('#vietnamese-asr-live-caption-host')
              ?.shadowRoot?.querySelector('.caption')?.textContent
              === 'xin chào phụ đề trực tiếp nội dung tiếp theo'""",
            timeout=10_000,
        )
        assert youtube.evaluate(
            """() => document.querySelector('#vietnamese-asr-live-caption-host')
              .shadowRoot.querySelector('.caption').getClientRects().length"""
        ) == 1
        service_worker.evaluate(
            """async () => {
              const tabs = await chrome.tabs.query({url: 'https://www.youtube.com/*'});
              await chrome.tabs.sendMessage(tabs[0].id, {
                target: 'content', type: 'caption', sequence: 3,
                text: 'đây là một câu phụ đề rất dài dùng để kiểm tra giao diện gọn gàng dễ đọc và không che phần điều khiển video youtube'
              });
            }"""
        )
        compact_layout = youtube.evaluate(
            """() => {
              const player = document.querySelector('#movie_player').getBoundingClientRect();
              const caption = document.querySelector('#vietnamese-asr-live-caption-host')
                .shadowRoot.querySelector('.caption');
              const rect = caption.getBoundingClientRect();
              const style = getComputedStyle(caption);
              return {
                width: rect.width,
                height: rect.height,
                fontSize: parseFloat(style.fontSize),
                lineClamp: style.webkitLineClamp,
                borderRadius: style.borderRadius,
                clearance: player.bottom - rect.bottom,
              };
            }"""
        )
        assert compact_layout["width"] <= 760
        assert compact_layout["height"] <= 82
        assert compact_layout["fontSize"] <= 24
        assert compact_layout["lineClamp"] == "2"
        assert compact_layout["borderRadius"] == "12px"
        assert compact_layout["clearance"] >= 65
        youtube.screenshot(path=str(SCREENSHOT_DIR / "youtube_overlay.png"), full_page=True)
        service_worker.evaluate(
            """async () => {
              const tabs = await chrome.tabs.query({url: 'https://www.youtube.com/*'});
              await chrome.tabs.sendMessage(tabs[0].id, {
                target: 'content', type: 'silence', sequence: 4,
                start_seconds: 4, end_seconds: 8
              });
            }"""
        )
        youtube.wait_for_function(
            """() => document.querySelector('#vietnamese-asr-live-caption-host')
              ?.shadowRoot?.querySelector('.caption')?.textContent === ''""",
            timeout=10_000,
        )
        assert not popup_errors, popup_errors
        context.close()

print(
    "Browser extension smoke OK: popup accessible; "
    "YouTube caption overlay rendered in Shadow DOM"
)
