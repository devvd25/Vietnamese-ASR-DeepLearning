from __future__ import annotations

import json
from pathlib import Path


def test_youtube_extension_manifest_declares_realtime_capture_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    extension = root / "extension"
    manifest = json.loads((extension / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == 3
    assert int(manifest["minimum_chrome_version"]) >= 116
    assert {"activeTab", "offscreen", "storage", "tabCapture"}.issubset(
        manifest["permissions"]
    )
    assert manifest["background"] == {
        "service_worker": "service-worker.js",
        "type": "module",
    }
    assert manifest["action"]["default_popup"] == "popup.html"
    assert manifest["content_scripts"][0]["matches"] == [
        "https://www.youtube.com/*"
    ]
    assert "ws://127.0.0.1:8765" in manifest["content_security_policy"][
        "extension_pages"
    ]
    for filename in (
        "audio-worklet.js",
        "content.js",
        "offscreen.html",
        "offscreen.js",
        "popup.css",
        "popup.html",
        "popup.js",
        "service-worker.js",
    ):
        assert (extension / filename).is_file(), filename
