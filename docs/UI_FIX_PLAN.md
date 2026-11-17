# Kế hoạch làm lại giao diện demo ASR — Claymorphism + Vibrant & Block-based

**Phạm vi:** `app/gradio_app.py` (+ `deploy/huggingface-space/app.py`, `tests/test_demo_validation.py`)
**Ảnh tham khảo:** thẻ "Pet Grooming & Spa — Claymorphism + Vibrant & Block-based" (nền kem, khối bo tròn dày,
bóng "đất sét" nổi phồng, CTA cam, chữ nhấn màu san hô, chip nổi trên ảnh, hàng số liệu lớn dưới CTA).
**Bằng chứng hiện trạng:** đo computed style trên Space live `devvd25-vietnamese-asr-stt19.hf.space` (Gradio 6.30.0).

Nguyên tắc: **giữ nguyên tính trung thực khoa học** của một công cụ benchmark (WER/CER, proxy chưa hiệu chuẩn),
chỉ đổi lớp trình bày sang phong cách clay vui mắt. Màu vibrant dùng cho khối/nhấn, **không** làm sai nghĩa màu trạng thái.

---

## 0. Việc bắt buộc trước khi đổi style (độc lập phong cách)

Hiện CSS tùy biến gần như không có tác dụng, nên mọi style clay sẽ cũng bị nuốt nếu không sửa gốc:

| Phần tử | Mong muốn | Thực tế đo được | Nguyên nhân |
| :--- | :--- | :--- | :--- |
| `.hero h1` | 32–48px, 800 | **26px**, 600 | `.gradio-container-6-30-0 .prose h1` (0,2,1) thắng `.hero h1` (0,1,1) |
| `.section-heading h2` | ~17px | **22px** | như trên |
| `.product-kicker`, `.technical-name` | màu brand | **đen** | `.prose p/span` ghi đè |
| `.status-pill`, `.metric-pair` | viền 1px | **0px** | reset border của Gradio |
| Font | Be Vietnam Pro | Segoe UI | chưa nạp font |

- **0.1 Specificity:** bọc toàn bộ nội dung trong `gr.Column(elem_id="asr-root")`, scope mọi selector dưới `#asr-root`.
- **0.2 Nội dung trung thực:** bỏ pill tĩnh `SYSTEM READY`; nhãn `ON-DEVICE ENGINE` và câu "100% cục bộ, audio không gửi ra ngoài"
  chỉ đúng khi chạy local. Thêm `build_demo(..., runtime="local" | "space")`; Space hiển thị `HF ZEROGPU` +
  "Audio được tải lên Hugging Face Space để suy luận, không gửi tới dịch vụ ASR bên thứ ba".
- **0.3 Đường lỗi:** nhánh `except` của `transcribe_for_ui` đang trả chuỗi thô vào ô metric → mất 4 thẻ. Đưa
  `empty_timing`/`empty_metrics` ra module-level và trả lại chúng khi lỗi; lỗi hiển thị bằng khối tone danger trong `file_info`.
- **0.4 Trạng thái chạy:** khóa nút khi đang suy luận (`interactive=False`, nhãn "Đang nhận dạng…"), `.then()` mở lại;
  cập nhật `file_info` ngay ở `audio.change`.

---

## 1. Design tokens Clay

### 1.1 Bảng màu (đã kiểm tra tương phản trên nền kem `#FFF7EE`)

| Token | Hex | Dùng cho | Ghi chú tương phản |
| :--- | :--- | :--- | :--- |
| `--clay-bg` | `#FFF7EE` | nền trang (kem ấm như ảnh mẫu) | — |
| `--clay-surface` | `#FFFDF9` | mặt khối/card | — |
| `--clay-ink` | `#2D2433` | chữ chính | ~14:1 |
| `--clay-ink-muted` | `#6B5E57` | mô tả, caption | ~6:1 (AA) |
| `--clay-orange` | `#FF9F43` | CTA chính, chip eyebrow | chữ trên nền này dùng `--clay-on-orange` |
| `--clay-orange-deep` | `#FF7A59` | điểm cuối gradient CTA, hover | — |
| `--clay-on-orange` | `#3B1D0B` | chữ trên nút cam | ~7.5:1 trên `#FF9F43`, ~6:1 trên `#FF7A59` |
| `--clay-coral` | `#E8505B` | chữ nhấn trong H1 ("Speech Recognition") | ~3.5:1 → **chỉ dùng cho chữ ≥ 24px** |
| `--clay-sky` | `#7DD3FC` / tint `#E0F4FF` | khối thời gian suy luận, panel lưu ý | chữ trên tint: `#075985` |
| `--clay-lilac` | `#C4B5FD` / tint `#F1ECFF` | khối độ tin cậy | chữ: `#4C1D95` |
| `--clay-mint` | `#6EE7B7` / tint `#E3FBF1` | **chỉ** trạng thái khớp (WER/CER = 0) | chữ: `#065F46` |
| `--clay-rose` | `#FB7185` / tint `#FFE8EC` | **chỉ** trạng thái lỗi/sai lệch | chữ: `#9F1239` |
| `--clay-butter` | `#FDE68A` / tint `#FFF6D6` | cảnh báo (checkpoint ST23D) | chữ: `#78350F` |

Quy tắc: mỗi khối có **một** màu pastel riêng (block-based); xanh mint/hồng rose dành riêng cho kết quả đúng/sai
và luôn đi kèm chữ ("Khớp hoàn toàn" / "Có sai lệch"), không chỉ dựa vào màu.

### 1.2 Hình khối & bóng "đất sét"

```css
--clay-radius-xl: 28px;   /* section card */
--clay-radius-lg: 22px;   /* metric card, input, dropzone */
--clay-radius-md: 16px;   /* button */
--clay-radius-pill: 9999px;
--clay-shadow:            /* khối nổi phồng */
  10px 10px 24px rgba(166, 112, 76, .16),
  -8px -8px 20px rgba(255, 255, 255, .95),
  inset 2px 2px 4px rgba(255, 255, 255, .80),
  inset -4px -4px 8px rgba(166, 112, 76, .08);
--clay-shadow-pressed:    /* nút khi nhấn: lõm xuống */
  inset 4px 4px 10px rgba(120, 60, 20, .22),
  inset -3px -3px 8px rgba(255, 255, 255, .45);
--clay-shadow-inset:      /* ô nhập liệu: lõm nhẹ */
  inset 3px 3px 7px rgba(166, 112, 76, .12),
  inset -3px -3px 7px rgba(255, 255, 255, .90);
```

- Thêm viền `1.5px solid rgba(166,112,76,.10)` cho mọi khối: claymorphism dễ bị "khối hòa vào nền" khi chỉ có bóng.
- Nền trang: 2–3 đốm blur tĩnh (radial-gradient cam/hồng/mint ở góc, opacity ≤ .35) như ảnh mẫu;
  **không** dùng `backdrop-filter` và không animate để giữ hiệu năng.
- Mobile (< 700px): giảm offset bóng còn ~50% để khối không bị "dày" trên màn hẹp.

### 1.3 Chữ

- Tiêu đề: **Nunito** 800 (bo tròn, hợp clay, có subset tiếng Việt). Thân bài: **Be Vietnam Pro** 400/500.
- Nạp qua `gr.themes.GoogleFont(...)` trong theme (không `@import` tay).
- Thang chữ: H1 40–52px/800, line-height 1.08 · H2 20px/800 · body 16px · caption 14px (không còn chữ 13px) · số liệu 30px tabular-nums.

### 1.4 Đưa token vào theme Gradio (tên biến đã kiểm tra trên Gradio 6.30)

```python
theme = gr.themes.Base(
    primary_hue="orange", neutral_hue="stone",
    font=[gr.themes.GoogleFont("Be Vietnam Pro"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#FFF7EE",
    block_background_fill="#FFFDF9",
    block_radius="22px", block_border_width="1.5px", block_shadow="<--clay-shadow>",
    input_background_fill="#FFFBF5", input_radius="16px", input_shadow="<--clay-shadow-inset>",
    button_large_radius="16px",
    button_primary_background_fill="linear-gradient(135deg, #FF9F43, #FF7A59)",
    button_primary_text_color="#3B1D0B",
    button_primary_shadow="<--clay-shadow>", button_primary_shadow_active="<--clay-shadow-pressed>",
)
```

Nhờ đó Audio, Dropdown, Textbox tự mang style clay; CSS tay chỉ còn cho hero, thẻ metric, chip.
Dark mode: **ép light** (claymorphism tối rất khó giữ tương phản) — thêm `js` bỏ class `dark`, đặt `color-scheme: light`.

---

## 2. Bố cục mới (block-based)

```
┌───────────────────────────────────────────────────────────────────────────┐
│ (● Vietnamese ASR Benchmark)  ← chip cam nhạt                [HF ZEROGPU] │
│                                                    ┌────────────────────┐ │
│ Vietnamese AI                                      │ [Wav2Vec2 CTC] ←chip│ │
│ Speech Recognition  ← chữ san hô                   │  ▂▅▇▅▃▆▇▄▂ (sóng   │ │
│ So sánh Wav2Vec2 CTC và PhoWhisper Seq2Seq         │  âm SVG trên khối  │ │
│ trên cùng một tệp âm thanh.                        │  pastel)           │ │
│ [ Bắt đầu nhận dạng ]  [ Xem chỉ số ]              │  [PhoWhisper] ←chip│ │
│  2 model   ·  28 s/đoạn  ·  100 MB                 └────────────────────┘ │
├──────────────────────────────────────────┬────────────────────────────────┤
│ ① Dữ liệu đầu vào                        │ ② Cấu hình & thực thi          │
│ ╭ dropzone clay lõm, icon mic/upload ╮   │ Model local  [ Wav2Vec2 ▾ ]    │
│ ╰ WAV/FLAC/OGG/MP3/M4A · ≤100 MB ────╯   │ ╭ Lưu ý đánh giá (butter) ─╮  │
│ sample.wav · 8.52 s · 16 kHz             │ ╰ WER 0 trên ST23D là ... ─╯  │
│ Transcript tham chiếu (không bắt buộc)   │ [ NHẬN DẠNG TIẾNG NÓI ] cam    │
├──────────────────────────────────────────┴────────────────────────────────┤
│ ③ Văn bản nhận dạng            (ô lõm, chữ 17px, nút Sao chép)            │
├───────────────────────────────────────────────────────────────────────────┤
│ ④ Bảng chỉ số kỹ thuật                                                    │
│ ╭ sky ───────╮ ╭ lilac ─────╮ ╭ mint/rose ─╮ ╭ mint/rose ─╮               │
│ │Thời gian   │ │Độ tin cậy  │ │WER         │ │CER         │               │
│ │0.342 s     │ │94.8 %      │ │0.00 %      │ │0.00 %      │               │
│ │RTF 0.04x   │ │proxy chưa  │ │Khớp hoàn   │ │Khớp hoàn   │               │
│ ╰────────────╯ ╰ hiệu chuẩn ╯ ╰ toàn ──────╯ ╰ toàn ──────╯               │
│ Thiết bị: cuda · Kiến trúc: vinai/PhoWhisper-base · 1 đoạn                │
├───────────────────────────────────────────────────────────────────────────┤
│ Lưu ý kỹ thuật (sky tint, theo runtime)                                   │
└───────────────────────────────────────────────────────────────────────────┘
```

Chi tiết từng khối:

1. **Hero**
   - Chip eyebrow bo tròn cam nhạt (giống "#1 Pet Spa in Town") thay cho kicker chữ hoa trơn.
   - H1 giữ nguyên chuỗi `Vietnamese AI Speech Recognition` (test browser dùng), bọc `Speech Recognition` trong `<span class="accent">` màu san hô.
   - Hai CTA dạng link neo: `Bắt đầu nhận dạng` → `#asr-input` (cam), `Xem chỉ số` → `#asr-metrics` (viền cam, nền kem).
   - Hàng số liệu lớn kiểu "5,000+ / 4.9 / 10+" nhưng **dùng số thật lấy từ config**: số model khả dụng (`_available_models`),
     `demo.chunk_seconds`, `MAX_AUDIO_FILE_BYTES`. Không bịa số người dùng/đánh giá.
   - Khối minh họa bên phải: SVG sóng âm inline trên khối pastel bo 28px, hai chip nổi tên model ở góc (như "Full Spa / Styling").
     Ẩn khối này ở < 700px.
2. **Section card:** bo 28px, `--clay-shadow`, tiêu đề có huy hiệu số tròn ①②③④ màu pastel thay cho "1." trơn;
   bỏ dòng nhãn tiếng Anh UPPERCASE trùng nghĩa (`AUDIO INPUT`, `ENGINE CONTROLS`…) — giữ 1 dòng mô tả 14px.
3. **Hàng nhập liệu:** `scale=7 / 5`, `equal_height=True`, nút CTA dính đáy cột phải; dòng định dạng hỗ trợ chuyển xuống
   ngay dưới dropzone; `file_info` thành 1 dòng meta, bỏ khung lồng 2 lớp.
4. **Panel checkpoint:** đổi từ xanh "success" sang **butter** (cảnh báo), tiêu đề "Lưu ý đánh giá".
5. **Nút "Nhận dạng tiếng nói":** cao 56px, gradient cam, chữ `#3B1D0B` 700; hover nhích lên 1px + bóng đậm hơn;
   active dùng `--clay-shadow-pressed` + `translateY(1px)`; trạng thái đang chạy: chữ "Đang nhận dạng…" và mờ 70%.
6. **Thẻ metric:** grid `repeat(auto-fit, minmax(190px, 1fr))` → 4 thẻ một hàng desktop, 2×2 tablet, 1 cột mobile.
   Nhãn câu thường ngắn (hết gãy dòng ở 1024px). WER/CER ở trạng thái chưa có tham chiếu dùng nền kem trung tính.
7. **Ô kết quả:** clay lõm (`input_shadow`), placeholder "Kết quả sẽ hiển thị tại đây sau khi nhận dạng", chữ 17px / 1.7.

---

## 3. Thứ tự triển khai

| Bước | Nội dung | File |
| :--- | :--- | :--- |
| 1 | Mục 0 (specificity, runtime, đường lỗi, khóa nút) | `app/gradio_app.py`, `deploy/huggingface-space/app.py` |
| 2 | Theme clay qua `.set()` + font + ép light | `app/gradio_app.py` |
| 3 | CSS token + section card + nút + nền đốm blur | `app/gradio_app.py` |
| 4 | Hero mới (chip, accent, CTA neo, số liệu thật, SVG) | `app/gradio_app.py` |
| 5 | Layout hàng nhập liệu, panel butter, `file_info` 1 dòng | `app/gradio_app.py` |
| 6 | Thẻ metric pastel + nhãn chữ trạng thái | `format_performance_metrics`, `_metric_card` |
| 7 | Test + kiểm tra trực quan | `tests/` |

## 4. Test & kiểm tra

- Sửa `test_demo_presents_the_asr_workbench_as_four_clear_sections` theo copy mới (bỏ `SYSTEM READY`, `ON-DEVICE ENGINE`
  chỉ còn ở runtime local, nhãn metric đổi sang câu thường). Giữ: "Không giới hạn thời lượng", "chia thành các đoạn",
  "proxy chưa hiệu chuẩn", `WER: 0.0000`, heading `Vietnamese AI Speech Recognition`, combobox `Model local`,
  nút `Nhận dạng tiếng nói`, chữ `Chọn audio`.
- Test mới: nhánh lỗi vẫn trả HTML có `metric-card`; `runtime="space"` không chứa "100% cục bộ"; hàng số hero lấy từ config.
- Chạy `pytest tests/test_demo_validation.py -v`, `tests/browser_smoke.py`, `tests/browser_inference_smoke.py`.
- Chạy local `python scripts/launch_demo.py`, đo lại computed style và chụp 1440 / 1024 / 375px.

## 5. Tiêu chí nghiệm thu

- [ ] Computed style khớp token: H1 ≥ 40px Nunito, nền `#FFF7EE`, card bo 28px có bóng clay, nút gradient cam.
- [ ] Mọi cặp chữ/nền đạt WCAG AA (chữ san hô chỉ xuất hiện ở cỡ ≥ 24px); focus ring 3px nhìn rõ trên nền kem.
- [ ] Màu mint/rose chỉ dùng cho đúng/sai, luôn kèm chữ; panel checkpoint là tone cảnh báo.
- [ ] Không còn câu "100% cục bộ" trên bản Space; không còn pill trạng thái tĩnh; không có số liệu bịa trên hero.
- [ ] Lỗi đầu vào không làm mất lưới thẻ chỉ số; nút bị khóa trong lúc chạy.
- [ ] Không cuộn ngang ở 375px; bóng đổ giảm trên mobile; `prefers-reduced-motion` tắt hiệu ứng nhấc nút.
- [ ] Toàn bộ test pass.
